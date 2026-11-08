"""CRT signal bot, phase 1. Sends Telegram alerts only. Places no trades.

Flow per pair and CRT timeframe (daily, 4H):
1. Weekly DXY bias filters direction on USD pairs.
2. Find a CRT range: candle 1 high/low (CRT H / CRT L).
3. Name the model: 2 candle, 3 candle, inside bar, multiple candle.
4. Detect the sweep of CRT H or CRT L.
5. Detect the MSS on the lower timeframe. Alert you.

Run:  python bot.py        (normal run)
      python bot.py test   (sends a test message)
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- CONFIG
CONFIG = {
    # pair name -> Yahoo tickers, tried in order
    "pairs": {
        "XAUUSD": ["GC=F"],
        "XAGUSD": ["SI=F"],
        "EURUSD": ["EURUSD=X"],
        "GBPUSD": ["GBPUSD=X"],
        "USDJPY": ["USDJPY=X"],
        "GBPJPY": ["GBPJPY=X"],
        "AUDUSD": ["AUDUSD=X"],
        "USDCAD": ["USDCAD=X"],
        "NZDUSD": ["NZDUSD=X"],
        "EURJPY": ["EURJPY=X"],
        "EURGBP": ["EURGBP=X"],
        "AUDJPY": ["AUDJPY=X"],
        "NZDJPY": ["NZDJPY=X"],
        "USDCHF": ["USDCHF=X"],
        "CADJPY": ["CADJPY=X"],
    },
    "dxy": ["DX-Y.NYB"],
    # "auto" = bot decides from last closed weekly DXY candles.
    # "buy" or "sell" = you set the DXY weekly bias by hand.
    "dxy_bias": "auto",
    # CRT timeframe -> lower timeframe used for the MSS
    "crt_timeframes": {"1D": "1H", "4H": "15M"},
    # "inside" = sweep candle must close back inside the CRT range
    # "wick"   = a wick beyond the level is enough
    "sweep_confirm": "inside",
    # CRT models the bot may alert on. All four:
    # ["2 candle", "3 candle", "Inside bar", "Multiple candle"]
    "models": ["3 candle", "Inside bar"],
    # Rules from the Rowdy FX book:
    # sweep candle must close in the trade direction (down-close for a sell)
    "require_sweep_candle_direction": True,
    # candle 1 body must be at least this share of its range (long body, small wicks)
    "min_body_ratio": 0.5,
    # Skip entry alerts whose RR is below this. RR is measured if you enter on a
    # retest of the broken MSS level, with stop at the sweep and target at the
    # opposite CRT level. Set 0 to turn the filter off.
    "min_rr": 2.0,
    # Higher timeframe bias of each market itself (replaces the DXY rule):
    # "struct" = daily swing structure (higher highs/lows = bullish, lower = bearish)
    # "week"   = direction of the last closed weekly candle
    # "ma"     = last daily close above/below the 50-day average
    # "none"   = no bias filter
    "bias_mode": "struct",
    # False = scanner mode: the bias is shown on every alert (agrees / against) but does not block it.
    # True  = alerts only fire in the direction of the bias.
    "bias_filter": False,
    # stop: "extreme" = deepest sweep point, "protected" = last swing before the break
    "stop_mode": "extreme",
    "max_between": 5,       # max candles between candle 1 and the sweep candle
    "lookback_sweeps": 2,   # how many recent CRT candles may be the sweep candle
    "swing_n": 2,           # bars each side for a swing high/low
    "alert_on_sweep": False,  # also alert on a sweep before the MSS shows
    "tz": "America/New_York",  # daily candle closes 17:00 in this zone
    "display_tz": "Africa/Lagos",  # time zone shown in alerts
    # "oanda" = OANDA prices (matches TradingView OANDA charts). Needs OANDA_TOKEN.
    # Without a token the bot falls back to Yahoo.
    "source": "oanda",
    "oanda_host": "api-fxpractice.oanda.com",  # live account: api-fxtrade.oanda.com
    "oanda_price": "B",  # B = bid (TradingView default), M = mid, A = ask
    "history": "60d",
}

STATE_FILE = "state.json"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
DELTA = {
    "15M": pd.Timedelta(minutes=15),
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1),
}
TF_NAME = {"1D": "Daily", "4H": "4H", "1H": "1H", "15M": "15M"}
# Stop buffer beyond the stop level (about 1.5 pips). Price units.
STOP_BUFFER = {"EURUSD": 0.00015, "GBPUSD": 0.00020, "USDJPY": 0.015, "GBPJPY": 0.030,
               "AUDUSD": 0.00015, "USDCAD": 0.00020, "NZDUSD": 0.00020,
               "XAUUSD": 0.50, "XAGUSD": 0.05, "EURJPY": 0.025, "EURGBP": 0.00015,
               "AUDJPY": 0.030, "NZDJPY": 0.035, "USDCHF": 0.00020, "CADJPY": 0.035}


# ------------------------------------------------------------------- DATA
def fetch_1h(tickers, interval="1h"):
    """Download candles from Yahoo ("1h" or "15m"). Index = New York clock, no timezone."""
    import yfinance as yf

    for t in tickers:
        try:
            df = yf.download(
                t,
                period=CONFIG["history"],
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:  # network or Yahoo error
            print(f"  fetch error {t}: {e}")
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.columns = ["open", "high", "low", "close"]
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df.index = idx.tz_convert(CONFIG["tz"]).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    return None


def oanda_get(inst, params):
    """Fetch OANDA candles. Retries 3 times and reports OANDA's own error text."""
    headers = {"Authorization": f"Bearer {os.environ['OANDA_TOKEN']}"}
    url = f"https://{CONFIG['oanda_host']}/v3/instruments/{inst}/candles"
    last = ""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.ok:
                return r.json()["candles"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(last)


def fetch_oanda(pair, gran="H1"):
    """OANDA candles (gran H1 or M15). Same index format as fetch_1h."""
    if not os.environ.get("OANDA_TOKEN"):
        return None
    price = CONFIG["oanda_price"]
    key = {"M": "mid", "B": "bid", "A": "ask"}[price]
    try:
        candles = oanda_get(f"{pair[:3]}_{pair[3:]}",
                            {"granularity": gran, "count": 3000, "price": price})
    except Exception as e:
        print(f"  OANDA error {pair} {gran}: {e}")
        return None
    rows = [
        (c["time"], float(c[key]["o"]), float(c[key]["h"]),
         float(c[key]["l"]), float(c[key]["c"]))
        for c in candles
        if key in c
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    df.index = df.index.tz_convert(CONFIG["tz"]).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_oanda_1h(pair):
    return fetch_oanda(pair, "H1")


def fetch_pair(pair, tickers, gran="H1"):
    """Returns (dataframe, source name). gran is H1 or M15."""
    if CONFIG["source"] == "oanda":
        df = fetch_oanda(pair, gran)
        if df is not None:
            return df, "OANDA"
        print(f"  {pair}: OANDA {gran} failed or no token, using Yahoo")
    df = fetch_1h(tickers, "1h" if gran == "H1" else "15m")
    return df, "Yahoo (levels may differ from TradingView)"


def resample(df, tf):
    """Build 1H, 4H or daily candles from finer data, aligned to the 17:00 New York close."""
    if tf == "15M":
        return df
    if tf == "1H":
        step = df.index.to_series().diff().median()
        if pd.isna(step) or step >= pd.Timedelta(hours=1):
            return df
        return df.resample("1h").agg(AGG).dropna()
    if tf == "4H":
        r = df.resample("4h", offset="1h")  # 01,05,09,13,17,21
    else:
        r = df.resample("24h", offset="17h")  # 17:00 to 17:00 (24h, not 1D: pandas 3 ignores offset on 1D)
    return r.agg(AGG).dropna()


def drop_forming(df, tf, now):
    """Remove the last candle if it is still open."""
    if len(df) and df.index[-1] + DELTA[tf] > now:
        return df.iloc[:-1]
    return df


# ------------------------------------------------------------------- BIAS
def usd_role(pair):
    if pair.endswith("USD"):
        return "quote"
    if pair.startswith("USD"):
        return "base"
    return None


def allowed_directions(pair, dxy_bias):
    """USD base follows DXY. USD quote goes opposite. No USD = both ways."""
    role = usd_role(pair)
    if role is None:
        return {"buy", "sell"}
    if dxy_bias is None:
        return set()
    opposite = "sell" if dxy_bias == "buy" else "buy"
    return {dxy_bias} if role == "base" else {opposite}


def weekly_bias(dxy_1h, now):
    """Bias from the last two CLOSED weekly DXY candles. Fixed for the week."""
    if CONFIG["dxy_bias"] in ("buy", "sell"):
        return CONFIG["dxy_bias"]
    daily = resample(dxy_1h, "1D")
    if daily.empty:
        return None

    def week_key(ts):
        k = ts + pd.Timedelta(hours=7)  # Sun 17:00 belongs to next Monday
        return (k - pd.Timedelta(days=k.weekday())).normalize()

    keys = [week_key(t) for t in daily.index]
    weekly = daily.groupby(keys).agg(AGG)
    closed = weekly[weekly.index < week_key(now)]
    if len(closed) < 2:
        return None
    w0, w1 = closed.iloc[-2], closed.iloc[-1]
    inside = w0.low <= w1.close <= w0.high
    if w1.low < w0.low and inside:
        return "buy"   # swept the low, closed back inside
    if w1.high > w0.high and inside:
        return "sell"  # swept the high, closed back inside
    return "buy" if w1.close > w1.open else "sell"


# ---------------------------------------------------------- CRT DETECTION
def week_key(ts):
    """Monday that starts the trading week (week opens Sunday 17:00 New York)."""
    k = ts + pd.Timedelta(hours=7)
    return (k - pd.Timedelta(days=k.weekday())).normalize()


def bias_from_daily(daily, mode, now):
    """Higher timeframe bias of one market. +1 bullish, -1 bearish, 0 neutral.

    daily must hold CLOSED daily candles only.
    """
    if mode == "struct":
        d = daily.iloc[-120:]
        n = 2
        hi, lo = d.high.values, d.low.values
        highs, lows = [], []
        for k in range(n, len(d) - n):
            if hi[k] > hi[k - n : k].max() and hi[k] > hi[k + 1 : k + n + 1].max():
                highs.append(hi[k])
            if lo[k] < lo[k - n : k].min() and lo[k] < lo[k + 1 : k + n + 1].min():
                lows.append(lo[k])
        if len(highs) < 2 or len(lows) < 2:
            return 0
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return 1
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return -1
        return 0
    if mode == "ma":
        if len(daily) < 50:
            return 0
        return 1 if daily.close.iloc[-1] > daily.close.iloc[-50:].mean() else -1
    if mode == "week":
        d = daily.iloc[-60:]
        if d.empty:
            return 0
        keys = [week_key(t) for t in d.index]
        weekly = d.groupby(keys).agg(AGG)
        closed = weekly[weekly.index < week_key(now)]
        if closed.empty:
            return 0
        w = closed.iloc[-1]
        return 1 if w.close > w.open else -1
    return 0


def mss_stop(ltf, direction, mss, mode, buffer=0.0):
    """Stop price. 'extreme' = deepest sweep point. 'protected' = last swing
    low (high) before the break, if there is one. A small buffer goes beyond it."""
    sg = 1 if direction == "buy" else -1
    stop = mss["extreme"]
    if mode == "protected":
        n = CONFIG["swing_n"]
        lo = ltf.low.values if sg == 1 else -ltf.high.values
        p, q = mss["p"], mss["q"]
        for k in range(q - 1, p, -1):
            if k - n < 0 or k + n > q:
                continue
            if lo[k] < lo[k - n : k].min() and lo[k] < lo[k + 1 : k + n + 1].min():
                stop = lo[k] * sg
                break
    return float(stop - sg * buffer)


def pair_bias(df, now):
    """(+1/-1/0 or None, text). None = no bias filter."""
    mode = CONFIG["bias_mode"]
    if mode == "none":
        return None, ""
    daily = drop_forming(resample(df, "1D"), "1D", now)
    b = bias_from_daily(daily, mode, now)
    return b, {1: "bullish", -1: "bearish", 0: "neutral"}[b]


def bias_note(b, text, direction):
    """Bias text for one alert: says whether it agrees with the trade direction."""
    if b is None:
        return ""
    if b == 0:
        return "neutral (no clear daily structure)"
    agrees = (b == 1) == (direction == "buy")
    return f"{text}, {'AGREES with' if agrees else 'AGAINST'} this {direction}"


def directions_for(b):
    return {None: {"buy", "sell"}, 1: {"buy"}, -1: {"sell"}, 0: set()}[b]


def find_setup(c, i):
    """Candle i is the sweep candle. Look back for candle 1 (the CRT range).

    Models (book definitions):
    - 3 candle:  candle 2 sweeps and closes back inside, still short of the 50% line.
                 Candle 3 is the move.
    - 2 candle:  candle 2 sweeps and already closed past the 50% line.
    - Inside bar: candle 2 sits fully inside candle 1, candle 3 sweeps.
    - Multiple candle: several candles close inside, then one sweeps.
    """
    s = c.iloc[i]
    # widest range first, so an inside bar is judged against its mother candle
    for m in range(CONFIG["max_between"] + 1, 0, -1):
        j = i - m
        if j < 0:
            continue
        c1 = c.iloc[j]
        H, L = c1.high, c1.low
        if H <= L:
            continue
        ratio = abs(c1.close - c1.open) / (H - L)
        if ratio < CONFIG["min_body_ratio"]:
            continue
        between = c.iloc[j + 1 : i]
        if len(between) and not ((between.close <= H) & (between.close >= L)).all():
            continue

        closes_inside = L <= s.close <= H
        need_close = CONFIG["sweep_confirm"] == "inside"
        if s.low < L and s.high <= H and (closes_inside or not need_close):
            direction = "buy"
        elif s.high > H and s.low >= L and (closes_inside or not need_close):
            direction = "sell"
        else:
            continue
        buy = direction == "buy"
        dir_ok = bool(s.close > s.open) if buy else bool(s.close < s.open)
        if CONFIG["require_sweep_candle_direction"] and not dir_ok:
            continue

        mid = (H + L) / 2
        if m == 1:
            past_mid = s.close >= mid if buy else s.close <= mid
            model = "2 candle" if past_mid else "3 candle"
        elif m == 2:
            b = between.iloc[0]
            model = "Inside bar" if (b.high <= H and b.low >= L) else "Multiple candle"
        else:
            model = "Multiple candle"
        if model not in CONFIG["models"]:
            continue
        return {
            "model": model,
            "direction": direction,
            "H": float(H),
            "L": float(L),
            "c1_time": c.index[j],
            "times": list(c.index[j : i + 1]),
            "sweep_time": c.index[i],
            "c1_ratio": float(ratio),
            "dir_ok": dir_ok,
        }
    return None


def find_mss(ltf, direction, H, L, start):
    """After the sweep, find an LTF candle that closes past the last swing."""
    n = CONFIG["swing_n"]
    sg = 1 if direction == "buy" else -1
    # sells are handled as a mirror image of buys
    hi = ltf.high.values if sg == 1 else -ltf.low.values
    lo = ltf.low.values if sg == 1 else -ltf.high.values
    cl = ltf.close.values * sg
    level = L if sg == 1 else -H
    pos0 = int(ltf.index.searchsorted(start))
    breach = [k for k in range(pos0, len(ltf)) if lo[k] < level]
    if not breach:
        return None
    b = breach[0]
    p = b + int(np.argmin(lo[b:]))  # deepest point of the sweep
    ref = None
    for k in range(p - 1, n - 1, -1):
        if k + n >= len(hi):
            continue
        if hi[k] > hi[k - n : k].max() and hi[k] > hi[k + 1 : k + n + 1].max():
            ref = hi[k]
            break
    if ref is None:
        return None
    for q in range(p + 1, len(ltf)):
        if cl[q] > ref:
            return {
                "time": ltf.index[q],
                "level": float(ref * sg),
                "extreme": float(lo[p] * sg),
                "p": p,
                "q": q,
            }
    return None


# ----------------------------------------------------------------- OUTPUT
def fmt(pair, x):
    if "JPY" in pair:
        d = 3
    elif pair == "XAUUSD":
        d = 2
    elif pair == "XAGUSD":
        d = 3
    else:
        d = 5
    return f"{x:.{d}f}"


def calc_rr(direction, entry, stop, target):
    sg = 1 if direction == "buy" else -1
    risk = (entry - stop) * sg
    reward = (target - entry) * sg
    return reward / risk if risk > 0 and reward > 0 else 0.0


def setup_prices(setup, mss):
    buy = setup["direction"] == "buy"
    return mss.get("stop", mss["extreme"]), (setup["H"] if buy else setup["L"])


def to_local(ts):
    """Bot candle time (New York clock) -> your clock (Lagos), no tz attached."""
    t = pd.Timestamp(ts).tz_localize(CONFIG["tz"], ambiguous=True,
                                     nonexistent="shift_forward")
    return t.tz_convert(CONFIG["display_tz"]).tz_localize(None)


def local_time(ts):
    try:
        return to_local(ts).strftime("%a %d %b %H:%M")
    except Exception:
        return str(ts)


def build_message(pair, tf, ltf_tf, setup, mss, price, bias, source="", closes_at=None):
    buy = setup["direction"] == "buy"
    head = f"{'🟢 BUY' if buy else '🔴 SELL'} {pair} | {TF_NAME[tf]} CRT ({setup['model']})"
    lines = [head, f"CRT H: {fmt(pair, setup['H'])}   CRT L: {fmt(pair, setup['L'])}"]
    names = {0: "Candle 1 (range)"}
    for k, t in enumerate(setup["times"]):
        last = k == len(setup["times"]) - 1
        name = "Candle %d (sweep)" % (k + 1) if last else names.get(k, "Candle %d" % (k + 1))
        lines.append(f"{name}: {local_time(t)}")
    if tf == "4H":
        hr = pd.Timestamp(setup["sweep_time"]).hour
        lines.append(f"Sweep candle opened {hr:02d}:00 New York time "
                     f"({'INSIDE' if hr in (1, 5, 9) else 'outside'} the book's 1/5/9am window)")
    if closes_at is not None:
        lines.append(f"Sweep candle closes {local_time(closes_at)}. Valid only if it closes back inside CRT H/L.")
    if mss:
        stop, target = setup_prices(setup, mss)
        word = "above" if buy else "below"
        lines.append(f"Sweep {'low' if buy else 'high'}: {fmt(pair, mss['extreme'])}")
        lines.append(f"MSS on {ltf_tf}: closed {word} {fmt(pair, mss['level'])} (candle {local_time(mss['time'])})")
        lines.append(f"LIMIT entry (retest of MSS level): {fmt(pair, mss['level'])}")
        lines.append(f"Stop: {fmt(pair, stop)}")
        lines.append(f"Target: {fmt(pair, target)} (CRT {'H' if buy else 'L'})")
        rr = calc_rr(setup["direction"], mss["level"], stop, target)
        lines.append(f"RR from entry: {rr:.1f}   Price now: {fmt(pair, price)}")
        lines.append("Cancel the order if price reaches the target before the retest.")
    else:
        lines.append(f"CRT confirmed: swept, then closed back inside the range. Waiting for the MSS on {ltf_tf}.")
    if bias:
        lines.append(f"HTF bias ({CONFIG['bias_mode']}): {bias}")
    lines.append("Times are Lagos time (UTC+1)")
    if source:
        lines.append(f"Data: {source}")
    return "\n".join(lines)


def draw_candles(ax, df):
    from matplotlib.patches import Rectangle

    for k, (o, h, l, c) in enumerate(zip(df.open, df.high, df.low, df.close)):
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([k, k], [l, h], color=col, lw=1, zorder=2)
        body = max(abs(c - o), (h - l) * 0.02)
        ax.add_patch(Rectangle((k - 0.3, min(o, c)), 0.6, body, color=col, zorder=3))


def set_time_axis(ax, df, step):
    pos = list(range(0, len(df), step))
    ax.set_xticks(pos)
    ax.set_xticklabels([to_local(df.index[k]).strftime("%d %b\n%H:%M") for k in pos], fontsize=7)
    ax.set_xlim(-1, len(df) + 4)
    ax.grid(alpha=0.2)


def make_chart(pair, tf, ltf_tf, crt, ltf, setup, mss, forming, path):
    """Picture of what the bot sees: candles 1-3, CRT H/L, sweep, MSS."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buy = setup["direction"] == "buy"
    H, L = setup["H"], setup["L"]
    times = setup["times"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9.5), dpi=110)

    # top: the CRT timeframe, candles numbered
    a = max(crt.index.get_loc(times[0]) - 4, 0)
    top = crt.iloc[a:]
    draw_candles(ax1, top)
    ax1.axhline(H, color="#1565c0", ls="--", lw=1)
    ax1.axhline(L, color="#6a1b9a", ls="--", lw=1)
    ax1.text(len(top) + 0.5, H, " CRT H", color="#1565c0", fontsize=8, va="center")
    ax1.text(len(top) + 0.5, L, " CRT L", color="#6a1b9a", fontsize=8, va="center")
    rng = H - L
    for k, t in enumerate(times):
        x = top.index.get_loc(t)
        ax1.text(x, top.high.iloc[x] + rng * 0.04, str(k + 1), ha="center",
                 fontsize=11, fontweight="bold", color="black")
    sx = top.index.get_loc(times[-1])
    ext = top.low.iloc[sx] if buy else top.high.iloc[sx]
    ax1.plot([sx], [ext], marker="x", color="black", ms=8, zorder=5)
    ax1.set_ylim(min(L, top.low.min()) - rng * 0.15, max(H, top.high.max()) + rng * 0.2)
    note = " (sweep candle still open)" if forming else ""
    ax1.set_title(f"{'BUY' if buy else 'SELL'} {pair} | {TF_NAME[tf]} CRT ({setup['model']}){note}", fontsize=10)
    set_time_axis(ax1, top, max(len(top) // 6, 1))

    # bottom: lower timeframe, sweep and MSS
    start = int(ltf.index.searchsorted(setup["sweep_time"]))
    bot = ltf.iloc[max(start - 12, 0):]
    if len(bot) > 110:
        bot = bot.iloc[-110:]
    if len(bot):
        draw_candles(ax2, bot)
        swept, swept_name = (L, "CRT L") if buy else (H, "CRT H")
        ax2.axhline(swept, color="#6a1b9a" if buy else "#1565c0", ls="--", lw=1)
        ax2.text(len(bot) + 0.5, swept, f" {swept_name}", fontsize=8, va="center")
        if mss:
            ax2.axhline(mss["level"], color="#ef6c00", ls=":", lw=1.2)
            ax2.text(0, mss["level"], " MSS level = limit entry", color="#ef6c00", fontsize=8, va="bottom")
            stop_px = mss.get("stop", mss["extreme"])
            ax2.axhline(stop_px, color="black", ls=":", lw=1)
            ax2.text(0, stop_px, " stop", fontsize=8, va="top")
            if mss["time"] in bot.index:
                mx = bot.index.get_loc(mss["time"])
                ax2.annotate("MSS", xy=(mx, bot.close.iloc[mx]), xytext=(mx, mss["level"]),
                             ha="center", fontsize=9, color="#ef6c00",
                             arrowprops=dict(arrowstyle="->", color="#ef6c00"))
        set_time_axis(ax2, bot, max(len(bot) // 6, 1))
        ax2.set_title(f"{ltf_tf} view (Lagos time)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def send(text, photo=None):
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[no Telegram keys] " + text)
        return False
    if photo:
        try:
            with open(photo, "rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{tok}/sendPhoto",
                    data={"chat_id": chat, "caption": text[:1000]},
                    files={"photo": f},
                    timeout=60,
                )
            if r.ok:
                return True
        except Exception as e:
            print(f"  photo error: {e}")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=20,
        )
        return r.ok
    except Exception as e:
        print(f"  telegram error: {e}")
        return False


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"sent": {}}


def save_state(state, now):
    cutoff = (now - pd.Timedelta(days=30)).isoformat()
    state["sent"] = {k: v for k, v in state["sent"].items() if v >= cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


# -------------------------------------------------------------------- RUN
def alert(pair, tf, ltf_tf, crt, ltf, setup, mss, price, bias, source, now):
    """Build the text and chart, then send. Returns True if sent."""
    end = setup["sweep_time"] + DELTA[tf]
    forming = end > now
    closes_at = end if (forming and CONFIG["sweep_confirm"] == "inside") else None
    msg = build_message(pair, tf, ltf_tf, setup, mss, price, bias, source, closes_at)
    photo = None
    try:
        path = os.path.join(tempfile.gettempdir(), "crt_chart.png")
        photo = make_chart(pair, tf, ltf_tf, crt, ltf, setup, mss, forming, path)
    except Exception as e:
        print(f"  chart error: {e}")
    return send(msg, photo)


def run():
    now = pd.Timestamp.now(tz=CONFIG["tz"]).tz_localize(None)
    state = load_state()
    need15 = "15M" in CONFIG["crt_timeframes"].values()

    for pair, tickers in CONFIG["pairs"].items():
        df, source = fetch_pair(pair, tickers, "H1")
        if df is None:
            print(f"{pair}: no data")
            continue
        df15 = fetch_pair(pair, tickers, "M15")[0] if need15 else None
        b, bias_txt = pair_bias(df, now)
        dirs = directions_for(b) if CONFIG["bias_filter"] else {"buy", "sell"}
        print(f"{pair}: HTF bias {bias_txt or 'off'}")
        if not dirs:
            print(f"{pair}: skipped, neutral bias")
            continue
        price = float((df15 if df15 is not None else df).close.iloc[-1])
        for tf, ltf_tf in CONFIG["crt_timeframes"].items():
            base = df15 if ltf_tf == "15M" else df
            if base is None:
                print(f"{pair} {tf}: no {ltf_tf} data")
                continue
            crt = drop_forming(resample(df, tf), tf, now)  # closed candles only
            ltf = drop_forming(resample(base, ltf_tf), ltf_tf, now)
            n = len(crt)
            for i in range(n - 1, max(n - 1 - CONFIG["lookback_sweeps"], 0), -1):
                setup = find_setup(crt, i)
                if not setup or setup["direction"] not in dirs:
                    continue
                key = f"{pair}|{tf}|{setup['c1_time']}|{setup['direction']}"
                btxt = bias_note(b, bias_txt, setup["direction"])
                mss = find_mss(
                    ltf, setup["direction"], setup["H"], setup["L"], setup["sweep_time"]
                )
                if mss:
                    mss["stop"] = mss_stop(ltf, setup["direction"], mss, CONFIG["stop_mode"],
                                           STOP_BUFFER.get(pair, 0.0))
                    stop, target = setup_prices(setup, mss)
                    rr = calc_rr(setup["direction"], mss["level"], stop, target)
                    if rr < CONFIG["min_rr"]:
                        print(f"  {pair} {tf}: skipped, RR {rr:.1f} below {CONFIG['min_rr']}")
                    elif key not in state["sent"]:
                        if alert(pair, tf, ltf_tf, crt, ltf, setup, mss, price, btxt, source, now):
                            state["sent"][key] = now.isoformat()
                elif CONFIG["alert_on_sweep"] and key + "|sweep" not in state["sent"]:
                    if alert(pair, tf, ltf_tf, crt, ltf, setup, None, price, btxt, source, now):
                        state["sent"][key + "|sweep"] = now.isoformat()
            print(f"{pair} {tf}: checked")
    save_state(state, now)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = send("CRT bot connected. Telegram works.")
        print("sent" if ok else "FAILED: check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
    else:
        run()
