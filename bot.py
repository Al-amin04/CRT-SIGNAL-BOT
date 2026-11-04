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

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- CONFIG
CONFIG = {
    # pair name -> Yahoo tickers, tried in order
    "pairs": {
        "XAUUSD": ["XAUUSD=X", "GC=F"],
        "XAGUSD": ["XAGUSD=X", "SI=F"],
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
    "max_between": 5,       # max candles between candle 1 and the sweep candle
    "lookback_sweeps": 2,   # how many recent CRT candles may be the sweep candle
    "swing_n": 2,           # bars each side for a swing high/low
    "alert_on_sweep": False,  # also alert on a sweep before the MSS shows
    "tz": "America/New_York",  # daily candle closes 17:00 in this zone
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


def resample(df, tf):
    """Build 4H or daily candles aligned to the 17:00 New York close."""
    if tf == "1H":
        return df
    if tf == "4H":
        r = df.resample("4h", offset="1h")  # 01,05,09,13,17,21
    else:
        r = df.resample("1D", offset="17h")  # 17:00 to 17:00
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


def build_message(pair, tf, ltf_tf, setup, mss, price, bias):
    buy = setup["direction"] == "buy"
    head = f"{'🟢 BUY' if buy else '🔴 SELL'} {pair} | {TF_NAME[tf]} CRT ({setup['model']})"
    lines = [head, f"CRT H: {fmt(pair, setup['H'])}   CRT L: {fmt(pair, setup['L'])}"]
    if mss:
        stop, target = mss["extreme"], (setup["H"] if buy else setup["L"])
        word = "above" if buy else "below"
        lines.append(f"Sweep {'low' if buy else 'high'}: {fmt(pair, stop)}")
        lines.append(f"MSS on {ltf_tf}: closed {word} {fmt(pair, mss['level'])}")
        lines.append(f"Stop: beyond {fmt(pair, stop)}")
        lines.append(f"Target: {fmt(pair, target)} (CRT {'H' if buy else 'L'})")
        risk, reward = abs(price - stop), abs(target - price)
        if risk > 0:
            lines.append(f"Price now: {fmt(pair, price)}   RR about {reward / risk:.1f}")
        lines.append("Next: find your POI on the chart.")
    else:
        lines.append(f"Sweep seen. Waiting for the MSS on {ltf_tf}.")
    if usd_role(pair):
        lines.append(f"DXY weekly bias: {bias}")
    return "\n".join(lines)


def send(text):
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[no Telegram keys] " + text)
        return False
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
        df = fetch_1h(tickers)
        if df is None:
            print(f"{pair}: no data")
            continue
        dirs = allowed_directions(pair, bias)
        if not dirs:
            print(f"{pair}: skipped, no DXY bias")
            continue
        price = float(df.close.iloc[-1])
        for tf, ltf_tf in CONFIG["crt_timeframes"].items():
            crt = resample(df, tf)
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
                    if key in state["sent"]:
                        break
                    msg = build_message(pair, tf, ltf_tf, setup, mss, price, bias)
                    if send(msg):
                        state["sent"][key] = now.isoformat()
                    break
                if CONFIG["alert_on_sweep"] and key + "|sweep" not in state["sent"]:
                    msg = build_message(pair, tf, ltf_tf, setup, None, price, bias)
                    if send(msg):
                        state["sent"][key + "|sweep"] = now.isoformat()
                    break
            print(f"{pair} {tf}: checked")
    save_state(state, now)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = send("CRT bot connected. Telegram works.")
        print("sent" if ok else "FAILED: check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
    else:
        run()
