"""Backtest of the CRT bot rules on past OANDA data. Sends the summary to Telegram.

Uses the same detection code as bot.py (CRT range, model, sweep, MSS).
It walks forward candle by candle, so it never uses future data to find a setup.

Mechanical entry rules (your real entry is by eye, so this is an estimate):
- Alert fires when the sweep candle has closed AND the MSS has closed.
- Entry: limit order on a retest of the broken MSS level.
- Stop: the sweep extreme. Target: the opposite CRT level.
- Order is cancelled if price hits the stop or target before the retest,
  or if there is no retest within ENTRY_WAIT_BARS hours.
- If stop and target fall in the same 1H candle, the stop counts first.
- Spread, slippage and commission are ignored. Real results will be worse.

Run:  python backtest.py            (needs OANDA_TOKEN; BACKTEST_YEARS optional)
"""
import csv
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import requests

import bot

YEARS = float(os.environ.get("BACKTEST_YEARS") or 2)
ENTRY_WAIT_BARS = 24      # hours to wait for the retest
MAX_HOLD_BARS = 240       # hours before a trade is closed at market
RR_STEPS = [0, 1.5, 2, 3]
ALL_MODELS = ["2 candle", "3 candle", "Inside bar", "Multiple candle"]
YOUR_MODELS = ["3 candle", "Inside bar"]


# ------------------------------------------------------------------- DATA
def load_oanda(pair, years):
    """Page back through OANDA 1H candles. Same index format as bot.fetch_1h."""
    tok = os.environ.get("OANDA_TOKEN")
    if not tok:
        return None
    key = {"M": "mid", "B": "bid", "A": "ask"}[bot.CONFIG["oanda_price"]]
    inst = f"{pair[:3]}_{pair[3:]}"
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(365 * years))
    rows = []
    while True:
        r = requests.get(
            f"https://{bot.CONFIG['oanda_host']}/v3/instruments/{inst}/candles",
            headers={"Authorization": f"Bearer {tok}"},
            params={
                "granularity": "H1",
                "count": 5000,
                "price": bot.CONFIG["oanda_price"],
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=60,
        )
        r.raise_for_status()
        cs = r.json()["candles"]
        if not cs:
            break
        for c in cs:
            if c.get("complete") and key in c:
                m = c[key]
                rows.append((c["time"], float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"])))
        last = pd.to_datetime(cs[-1]["time"], utc=True)
        if len(cs) < 5000 or last >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2):
            break
        start = last + pd.Timedelta(hours=1)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    df.index = df.index.tz_convert(bot.CONFIG["tz"]).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_dxy():
    df = None
    try:
        import yfinance as yf

        raw = yf.download("DX-Y.NYB", period="730d", interval="1h", progress=False, auto_adjust=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close"]].dropna()
            df.columns = ["open", "high", "low", "close"]
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df.index = idx.tz_convert(bot.CONFIG["tz"]).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        print(f"DXY error: {e}")
    return df


# ------------------------------------------------------------- SIMULATION
def week_key(ts):
    k = ts + pd.Timedelta(hours=7)
    return (k - pd.Timedelta(days=k.weekday())).normalize()


def simulate(h1, direction, level, stop, target, t0):
    """Walk 1H candles from t0. Returns dict(R, outcome, entry_time, rr) or None."""
    sg = 1 if direction == "buy" else -1
    hi = h1.high.values if sg == 1 else -h1.low.values
    lo = h1.low.values if sg == 1 else -h1.high.values
    cl = h1.close.values * sg
    lev, stp, tgt = level * sg, stop * sg, target * sg
    if not (stp < lev < tgt):
        return None
    rr = (tgt - lev) / (lev - stp)
    pos = int(h1.index.searchsorted(t0))
    end = min(pos + ENTRY_WAIT_BARS, len(h1))
    fill = None
    for k in range(pos, end):
        if lo[k] > lev:                 # not touched yet
            if hi[k] >= tgt:
                return None             # ran to target with no retest
            continue
        if lo[k] <= stp:                # touched, then stopped, same candle
            return {"R": -1.0, "outcome": "loss", "entry_time": h1.index[k], "rr": rr}
        if hi[k] >= tgt:
            return None                 # order of events unknown
        fill = k
        break
    if fill is None:
        return None
    last = min(fill + 1 + MAX_HOLD_BARS, len(h1))
    for m in range(fill + 1, last):
        if lo[m] <= stp:
            return {"R": -1.0, "outcome": "loss", "entry_time": h1.index[fill], "rr": rr}
        if hi[m] >= tgt:
            return {"R": rr, "outcome": "win", "entry_time": h1.index[fill], "rr": rr}
    if last - 1 <= fill:
        return None
    r = (cl[last - 1] - lev) / (lev - stp)
    return {"R": r, "outcome": "timeout", "entry_time": h1.index[fill], "rr": rr}


def find_trades(pair, h1, dxy, bias_cache):
    trades = []
    now = h1.index[-1] + pd.Timedelta(hours=1)
    for tf, ltf_tf in bot.CONFIG["crt_timeframes"].items():
        crt = bot.drop_forming(bot.resample(h1, tf), tf, now)
        ltf = bot.drop_forming(bot.resample(h1, ltf_tf), ltf_tf, now)
        dlt, ldt = bot.DELTA[tf], bot.DELTA[ltf_tf]
        for i in range(8, len(crt)):
            setup = bot.find_setup(crt, i)
            if not setup:
                continue
            s0 = setup["sweep_time"]
            pos0 = int(ltf.index.searchsorted(s0))
            limit = s0 + 3 * dlt        # same window the live bot allows
            mss = None
            for k in range(pos0, len(ltf)):
                if ltf.index[k] + ldt > limit:
                    break
                window = ltf.iloc[max(0, k - 200) : k + 1]
                m = bot.find_mss(window, setup["direction"], setup["H"], setup["L"], s0)
                if m:
                    mss, mss_close = m, ltf.index[k] + ldt
                    break
            if not mss:
                continue
            stop, target = bot.setup_prices(setup, mss)
            alert_t = max(mss_close, s0 + dlt)
            res = simulate(h1, setup["direction"], mss["level"], stop, target, alert_t)
            if not res:
                continue
            # DXY weekly bias that was known at the time
            wk = week_key(s0)
            if dxy is not None and wk not in bias_cache:
                bias_cache[wk] = bot.weekly_bias(dxy[dxy.index < s0], s0)
            bias = bias_cache.get(wk)
            role = bot.usd_role(pair)
            if role is None:
                bias_ok = True
            elif bias is None:
                bias_ok = None          # unknown, excluded from the "with bias" view
            else:
                bias_ok = setup["direction"] in bot.allowed_directions(pair, bias)
            res.update(pair=pair, tf=tf, model=setup["model"], direction=setup["direction"],
                       alert_time=alert_t, bias_ok=bias_ok)
            trades.append(res)
    return trades


# ------------------------------------------------------------------ STATS
def stats(trades):
    if not trades:
        return "no trades"
    ts = sorted(trades, key=lambda t: t["entry_time"])
    R = np.array([t["R"] for t in ts])
    wins, losses = R[R > 0], R[R <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    streak = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        streak = max(streak, cur)
    return (f"{len(R)} trades | win {100 * len(wins) / len(R):.0f}% | avg {R.mean():+.2f}R | "
            f"total {R.sum():+.1f}R | PF {pf:.1f} | maxDD {dd:.1f}R | worst streak {streak}")


def pick(trades, models=None, min_rr=0, bias=False, **eq):
    out = []
    for t in trades:
        if models and t["model"] not in models:
            continue
        if t["rr"] < min_rr:
            continue
        if bias and t["bias_ok"] is not True:
            continue
        if any(t[k] != v for k, v in eq.items()):
            continue
        out.append(t)
    return out


def report(trades, pairs, years, has_dxy):
    L = [f"CRT BACKTEST | {years:g}y OANDA {bot.CONFIG['oanda_price']} | pairs: {len(pairs)}",
         "Entry = retest of MSS level, stop = sweep, target = opposite CRT level.",
         "Spread and slippage ignored. Real results are worse.", ""]
    L.append("YOUR MODELS (3 candle + inside bar), by min RR:")
    for rr in RR_STEPS:
        L.append(f"RR>={rr}: " + stats(pick(trades, YOUR_MODELS, rr)))
    if has_dxy:
        L += ["", "Same, with DXY weekly bias filter:"]
        for rr in RR_STEPS:
            L.append(f"RR>={rr}: " + stats(pick(trades, YOUR_MODELS, rr, bias=True)))
    else:
        L += ["", "DXY data failed: bias filter not tested."]
    L += ["", "BY MODEL (RR>=2, no bias filter):"]
    for m in ALL_MODELS:
        L.append(f"{m}: " + stats(pick(trades, [m], 2)))
    L += ["", "BY TIMEFRAME (your models, RR>=2):"]
    for tf in bot.CONFIG["crt_timeframes"]:
        L.append(f"{bot.TF_NAME[tf]}: " + stats(pick(trades, YOUR_MODELS, 2, tf=tf)))
    L += ["", "BY PAIR (your models, RR>=2):"]
    for p in pairs:
        L.append(f"{p}: " + stats(pick(trades, YOUR_MODELS, 2, pair=p)))
    L += ["", "Small samples mean little. Under about 30 trades, treat results as noise."]
    return "\n".join(L)


# ----------------------------------------------------------------- OUTPUT
def tg(text=None, doc=None):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    try:
        if text:
            for i in range(0, len(text), 3800):
                requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text[i : i + 3800]}, timeout=30)
        if doc:
            with open(doc, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{tok}/sendDocument",
                              data={"chat_id": chat}, files={"document": f}, timeout=60)
    except Exception as e:
        print(f"telegram error: {e}")


def save_csv(trades, path):
    cols = ["pair", "tf", "model", "direction", "alert_time", "entry_time", "rr", "R", "outcome", "bias_ok"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in sorted(trades, key=lambda x: x["entry_time"]):
            w.writerow([t[c] if c not in ("alert_time", "entry_time") else
                        pd.Timestamp(t[c]).strftime("%Y-%m-%d %H:%M") for c in cols])


def main(loader=None, dxy=None, dxy_loaded=False):
    loader = loader or load_oanda
    bot.CONFIG["dxy_bias"] = "auto"
    bot.CONFIG["models"] = ALL_MODELS
    if not dxy_loaded:
        dxy = load_dxy()
    trades, used, cache = [], [], {}
    for pair in bot.CONFIG["pairs"]:
        try:
            h1 = loader(pair, YEARS)
        except Exception as e:
            print(f"{pair}: data error {e}")
            continue
        if h1 is None or len(h1) < 500:
            print(f"{pair}: not enough data")
            continue
        t = find_trades(pair, h1, dxy, cache)
        print(f"{pair}: {len(h1)} candles, {len(t)} simulated trades")
        trades += t
        used.append(pair)
    if not used:
        print("No data. Is OANDA_TOKEN set?")
        tg("Backtest failed: no data. Check OANDA_TOKEN.")
        return
    text = report(trades, used, YEARS, dxy is not None)
    print(text)
    path = os.path.join(tempfile.gettempdir(), "crt_backtest_trades.csv")
    save_csv(trades, path)
    tg(text=text, doc=path)


if __name__ == "__main__":
    main()
