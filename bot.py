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
    },
    "dxy": ["DX-Y.NYB"],
    # "auto" = bot decides from last closed weekly DXY candles.
    # "buy" or "sell" = you set the DXY weekly bias by hand.
    "dxy_bias": "auto",
    # CRT timeframe -> lower timeframe used for the MSS
    "crt_timeframes": {"1D": "4H", "4H": "1H"},
    # "inside" = sweep candle must close back inside the CRT range
    # "wick"   = a wick beyond the level is enough
    "sweep_confirm": "inside",
    # CRT models the bot may alert on. All four:
    # ["2 candle", "3 candle", "Inside bar", "Multiple candle"]
    "models": ["3 candle", "Inside bar"],
    # Skip entry alerts whose RR is below this. RR is measured if you enter on a
    # retest of the broken MSS level, with stop at the sweep and target at the
    # opposite CRT level. Set 0 to turn the filter off.
    "min_rr": 2.0,
    "max_between": 5,       # max candles between candle 1 and the sweep candle
    "lookback_sweeps": 2,   # how many recent CRT candles may be the sweep candle
    "swing_n": 2,           # bars each side for a swing high/low
    "alert_on_sweep": True,   # also alert on a sweep before the MSS shows
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
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1),
}
TF_NAME = {"1D": "Daily", "4H": "4H", "1H": "1H"}


# ------------------------------------------------------------------- DATA
def fetch_1h(tickers):
    """Download 1H candles. Index = New York wall-clock time, no timezone."""
    import yfinance as yf

    for t in tickers:
        try:
            df = yf.download(
                t,
                period=CONFIG["history"],
                interval="1h",
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


def fetch_oanda_1h(pair):
    """OANDA 1H candles. Same index format as fetch_1h."""
    tok = os.environ.get("OANDA_TOKEN")
    if not tok:
        return None
    price = CONFIG["oanda_price"]
    key = {"M": "mid", "B": "bid", "A": "ask"}[price]
    inst = f"{pair[:3]}_{pair[3:]}"
    try:
        r = requests.get(
            f"https://{CONFIG['oanda_host']}/v3/instruments/{inst}/candles",
            headers={"Authorization": f"Bearer {tok}"},
            params={"granularity": "H1", "count": 1000, "price": price},
            timeout=30,
        )
        r.raise_for_status()
        candles = r.json()["candles"]
    except Exception as e:
        print(f"  OANDA error {pair}: {e}")
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


def fetch_pair(pair, tickers):
    """Returns (dataframe, source name)."""
    if CONFIG["source"] == "oanda":
        df = fetch_oanda_1h(pair)
        if df is not None:
            return df, "OANDA"
        print(f"  {pair}: OANDA failed or no token, using Yahoo")
    df = fetch_1h(tickers)
    return df, "Yahoo (levels may differ from TradingView)"


def resample(df, tf):
    """Build 4H or daily candles aligned to the 17:00 New York close."""
    if tf == "1H":
        return df
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
def find_setup(c, i):
    """Candle i is the sweep candle. Look back for candle 1 (the CRT range)."""
    s = c.iloc[i]
    # widest range first, so an inside bar is judged against its mother candle
    for m in range(CONFIG["max_between"] + 1, 0, -1):
        j = i - m
        if j < 0:
            continue
        c1 = c.iloc[j]
        H, L = c1.high, c1.low
        between = c.iloc[j + 1 : i]
        if len(between) and not ((between.close <= H) & (between.close >= L)).all():
            continue
        if m == 1:
            model = "2 candle"
        elif m == 2:
            b = between.iloc[0]
            model = "Inside bar" if (b.high <= H and b.low >= L) else "3 candle"
        else:
            model = "Multiple candle"
        if model not in CONFIG["models"]:
            continue

        closes_inside = L <= s.close <= H
        need_close = CONFIG["sweep_confirm"] == "inside"
        if s.low < L and s.high <= H and (closes_inside or not need_close):
            direction = "buy"
        elif s.high > H and s.low >= L and (closes_inside or not need_close):
            direction = "sell"
        else:
            continue
        return {
            "model": model,
            "direction": direction,
            "H": float(H),
            "L": float(L),
            "c1_time": c.index[j],
            "times": list(c.index[j : i + 1]),
            "sweep_time": c.index[i],
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
    return mss["extreme"], (setup["H"] if buy else setup["L"])


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
    if closes_at is not None:
        lines.append(f"Sweep candle closes {local_time(closes_at)}. Valid only if it closes back inside CRT H/L.")
    if mss:
        stop, target = setup_prices(setup, mss)
        word = "above" if buy else "below"
        lines.append(f"Sweep {'low' if buy else 'high'}: {fmt(pair, stop)}")
        lines.append(f"MSS on {ltf_tf}: closed {word} {fmt(pair, mss['level'])} (candle {local_time(mss['time'])})")
        lines.append(f"Stop: beyond {fmt(pair, stop)}")
        lines.append(f"Target: {fmt(pair, target)} (CRT {'H' if buy else 'L'})")
        rr_lvl = calc_rr(setup["direction"], mss["level"], stop, target)
        rr_now = calc_rr(setup["direction"], price, stop, target)
        lines.append(f"Price now: {fmt(pair, price)}")
        lines.append(f"RR from MSS level: {rr_lvl:.1f}   RR at price now: {rr_now:.1f}")
        lines.append("Next: find your POI on the chart.")
    else:
        lines.append(f"CRT confirmed: swept, then closed back inside the range. Waiting for the MSS on {ltf_tf}.")
    if usd_role(pair):
        lines.append(f"DXY weekly bias: {bias}")
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
    if len(bot):
        draw_candles(ax2, bot)
        swept, swept_name = (L, "CRT L") if buy else (H, "CRT H")
        ax2.axhline(swept, color="#6a1b9a" if buy else "#1565c0", ls="--", lw=1)
        ax2.text(len(bot) + 0.5, swept, f" {swept_name}", fontsize=8, va="center")
        if mss:
            ax2.axhline(mss["level"], color="#ef6c00", ls=":", lw=1.2)
            ax2.text(0, mss["level"], " MSS level", color="#ef6c00", fontsize=8, va="bottom")
            ax2.axhline(mss["extreme"], color="black", ls=":", lw=1)
            ax2.text(0, mss["extreme"], " sweep extreme / stop", fontsize=8, va="top")
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
    bias = None
    if any(usd_role(p) for p in CONFIG["pairs"]):
        dxy = fetch_1h(CONFIG["dxy"])
        if dxy is not None:
            bias = weekly_bias(dxy, now)
        print(f"DXY weekly bias: {bias}")

    for pair, tickers in CONFIG["pairs"].items():
        df, source = fetch_pair(pair, tickers)
        if df is None:
            print(f"{pair}: no data")
            continue
        dirs = allowed_directions(pair, bias)
        if not dirs:
            print(f"{pair}: skipped, no DXY bias")
            continue
        price = float(df.close.iloc[-1])
        for tf, ltf_tf in CONFIG["crt_timeframes"].items():
            crt = drop_forming(resample(df, tf), tf, now)  # closed candles only
            ltf = drop_forming(resample(df, ltf_tf), ltf_tf, now)
            n = len(crt)
            for i in range(n - 1, max(n - 1 - CONFIG["lookback_sweeps"], 0), -1):
                setup = find_setup(crt, i)
                if not setup or setup["direction"] not in dirs:
                    continue
                key = f"{pair}|{tf}|{setup['c1_time']}|{setup['direction']}"
                mss = find_mss(
                    ltf, setup["direction"], setup["H"], setup["L"], setup["sweep_time"]
                )
                if mss:
                    stop, target = setup_prices(setup, mss)
                    rr = calc_rr(setup["direction"], mss["level"], stop, target)
                    if rr < CONFIG["min_rr"]:
                        print(f"  {pair} {tf}: skipped, RR {rr:.1f} below {CONFIG['min_rr']}")
                    elif key not in state["sent"]:
                        if alert(pair, tf, ltf_tf, crt, ltf, setup, mss, price, bias, source, now):
                            state["sent"][key] = now.isoformat()
                elif CONFIG["alert_on_sweep"] and key + "|sweep" not in state["sent"]:
                    if alert(pair, tf, ltf_tf, crt, ltf, setup, None, price, bias, source, now):
                        state["sent"][key + "|sweep"] = now.isoformat()
            print(f"{pair} {tf}: checked")
    save_state(state, now)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = send("CRT bot connected. Telegram works.")
        print("sent" if ok else "FAILED: check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
    else:
        run()
