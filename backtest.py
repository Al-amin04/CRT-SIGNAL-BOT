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

YEARS = float(os.environ.get("BACKTEST_YEARS") or 5)
ENTRY_WAIT_BARS = 24      # hours to wait for the retest
MAX_HOLD_BARS = 240       # hours before a trade is closed at market
TIMING_HOURS = (1, 5, 9)   # New York hour a 4H sweep candle opens (book's 'time and price')
MIN_BODY = 0.5
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
        cs = bot.oanda_get(inst, {
            "granularity": "H1",
            "count": 5000,
            "price": bot.CONFIG["oanda_price"],
            "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
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
            stop, target_full = bot.setup_prices(setup, mss)
            target_half = (setup["H"] + setup["L"]) / 2
            alert_t = max(mss_close, s0 + dlt)
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
            for tname, tgt in (("full", target_full), ("half", target_half)):
                res = simulate(h1, setup["direction"], mss["level"], stop, tgt, alert_t)
                if not res:
                    continue
                res.update(pair=pair, tf=tf, model=setup["model"], direction=setup["direction"],
                           alert_time=alert_t, bias_ok=bias_ok, target=tname,
                           c1_ratio=setup["c1_ratio"], dir_ok=setup["dir_ok"],
                           hour=s0.hour)
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


def pick(trades, models=None, target="full", direction=False, quality=False,
         timing=False, bias=False, **eq):
    """Filter simulated trades. The flags are the book rules being tested."""
    out = []
    for t in trades:
        if t["target"] != target:
            continue
        if models and t["model"] not in models:
            continue
        if direction and not t["dir_ok"]:
            continue
        if quality and t["c1_ratio"] < MIN_BODY:
            continue
        if timing and t["tf"] == "4H" and t["hour"] not in TIMING_HOURS:
            continue
        if bias and t["bias_ok"] is not True:
            continue
        if any(t[k] != v for k, v in eq.items()):
            continue
        out.append(t)
    return out


def report(trades, pairs, years, has_dxy):
    L = [f"CRT BACKTEST v2 | {years:g}y OANDA {bot.CONFIG['oanda_price']} | pairs: {len(pairs)}",
         "Models per the book: 3 candle = candle 2 sweeps, candle 3 is the move.",
         "Entry = retest of MSS level, stop = sweep, target = full range ('full')",
         "or the 50% line ('half'). No min RR. Spread ignored: real results are worse.",
         "", "YOUR MODELS (3 candle + inside bar), one rule added at a time:"]
    variants = [
        ("A baseline", {}),
        ("B +candle2 direction", {"direction": True}),
        ("C +candle1 body>=50%", {"quality": True}),
        ("D = B+C", {"direction": True, "quality": True}),
        ("E = D+1/5/9am NY (4H)", {"direction": True, "quality": True, "timing": True}),
    ]
    if has_dxy:
        variants.append(("F = D+DXY bias", {"direction": True, "quality": True, "bias": True}))
    for name, kw in variants:
        L.append(name)
        for tg_ in ("full", "half"):
            L.append(f"  {tg_}: " + stats(pick(trades, YOUR_MODELS, tg_, **kw)))
    if not has_dxy:
        L.append("DXY data failed: bias filter not tested.")
    D = {"direction": True, "quality": True}
    L += ["", "BY MODEL (rules B+C):"]
    for m in ALL_MODELS:
        for tg_ in ("full", "half"):
            L.append(f"{m} {tg_}: " + stats(pick(trades, [m], tg_, **D)))
    L += ["", "BY TIMEFRAME (your models, B+C):"]
    for tf in bot.CONFIG["crt_timeframes"]:
        for tg_ in ("full", "half"):
            L.append(f"{bot.TF_NAME[tf]} {tg_}: " + stats(pick(trades, YOUR_MODELS, tg_, tf=tf, **D)))
    L += ["", "BY PAIR (your models, B+C, full target):"]
    for p in pairs:
        L.append(f"{p}: " + stats(pick(trades, YOUR_MODELS, "full", pair=p, **D)))
    L += ["", "Under about 30 trades, treat results as noise. Many variants on",
          "one data set can flatter any rule: do not pick the best row."]
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
    cols = ["pair", "tf", "model", "direction", "target", "alert_time", "entry_time", "rr", "R", "outcome", "bias_ok", "c1_ratio", "dir_ok", "hour"]
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
    bot.CONFIG["require_sweep_candle_direction"] = False   # tested afterwards as a filter
    bot.CONFIG["min_body_ratio"] = 0.0                      # tested afterwards as a filter
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
