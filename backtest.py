"""Backtest v4 of the CRT bot rules on past OANDA data. Sends the summary to Telegram.

Uses the same detection code as bot.py (CRT range, model, sweep, MSS).
It walks forward candle by candle, so it never uses future data to find a setup.

Rules tested (your plan):
- Higher timeframe bias of each market itself (no DXY): 3 definitions compared.
- Alert fires when the sweep candle has closed AND the MSS candle has closed.
- Entry: limit order on a RETEST of the level the MSS broke (24h wait, else cancelled).
- Stop: beyond the sweep extreme, or beyond the last swing before the break ("protected").
- Target: the opposite CRT level. Minimum RR 1:2 (counted from the entry).
- If stop and target fall in the same 1H candle, the stop counts first.
- Spreads are assumed. Real results will differ.
- Tuning period and a hold-out period (last 18 months) are reported separately.

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
EXTRA = (os.environ.get("BACKTEST_EXTRA") or "no").lower() == "yes"
ENTRY_WAIT_BARS = 24      # hours to wait for the retest
MAX_HOLD_BARS = 240       # hours before a trade is closed at market
MIN_BODY = 0.5            # candle 1 body share (book rule)
HOLDOUT_DAYS = 548        # last ~18 months are kept apart
MIN_TRADES = 30           # a rule set needs this many tuning trades to be chosen
BIASES = ("none", "struct", "week", "ma")
STOPS = ("extreme", "protected")
YOUR_MODELS = ["3 candle", "Inside bar"]
EXTRA_PAIRS = {"EURJPY": ["EURJPY=X"], "EURGBP": ["EURGBP=X"], "AUDJPY": ["AUDJPY=X"],
               "NZDJPY": ["NZDJPY=X"], "USDCHF": ["USDCHF=X"], "CADJPY": ["CADJPY=X"]}
# Assumed typical spreads in price units. These are guesses, not your broker's numbers.
SPREADS = {"EURUSD": 0.00010, "GBPUSD": 0.00015, "USDJPY": 0.012, "GBPJPY": 0.025,
           "AUDUSD": 0.00012, "USDCAD": 0.00015, "NZDUSD": 0.00018,
           "XAUUSD": 0.35, "XAGUSD": 0.04, "EURJPY": 0.020, "EURGBP": 0.00015,
           "AUDJPY": 0.025, "NZDJPY": 0.030, "USDCHF": 0.00015, "CADJPY": 0.030}


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


# ------------------------------------------------------------- SIMULATION
def simulate(h1, direction, level, stop, target, t0):
    """Walk 1H candles from t0. Returns a trade dict, or {"skip": reason}.

    mfe = best move in R after entry before the stop was hit (measured past the target too).
    """
    sg = 1 if direction == "buy" else -1
    hi = h1.high.values if sg == 1 else -h1.low.values
    lo = h1.low.values if sg == 1 else -h1.high.values
    cl = h1.close.values * sg
    lev, stp, tgt = level * sg, stop * sg, target * sg
    if not (stp < lev < tgt):
        return {"skip": "geometry"}
    risk = lev - stp
    rr = (tgt - lev) / risk
    pos = int(h1.index.searchsorted(t0))
    end = min(pos + ENTRY_WAIT_BARS, len(h1))
    fill = None
    for k in range(pos, end):
        if lo[k] > lev:                 # not touched yet
            if hi[k] >= tgt:
                return {"skip": "ran"}  # ran to target with no retest
            continue
        if lo[k] <= stp:                # touched, then stopped, same candle
            return {"R": -1.0, "outcome": "loss", "entry_time": h1.index[k], "rr": rr, "mfe": 0.0}
        if hi[k] >= tgt:
            return {"skip": "amb"}      # order of events unknown
        fill = k
        break
    if fill is None:
        return {"skip": "nofill"}
    last = min(fill + 1 + MAX_HOLD_BARS, len(h1))
    result, mfe = None, 0.0
    for m in range(fill + 1, last):
        if lo[m] <= stp:
            if result is None:
                result = ("loss", -1.0)
            break
        mfe = max(mfe, (hi[m] - lev) / risk)
        if result is None and hi[m] >= tgt:
            result = ("win", rr)
    if result is None:
        if last - 1 <= fill:
            return {"skip": "short"}
        result = ("timeout", (cl[last - 1] - lev) / risk)
    return {"R": result[1], "outcome": result[0], "entry_time": h1.index[fill],
            "rr": rr, "mfe": mfe}


def find_trades(pair, h1):
    """All setups with an MSS, simulated for both stop styles.

    Returns (trades, signals). Signals record whether each MSS got a retest fill.
    """
    trades, signals = [], []
    now = h1.index[-1] + pd.Timedelta(hours=1)
    daily = bot.drop_forming(bot.resample(h1, "1D"), "1D", now)
    daily_end = daily.index + pd.Timedelta(days=1)
    buf = bot.STOP_BUFFER.get(pair, 0.0)
    spread = SPREADS.get(pair, 0.0)
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
            mss = window = None
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
            alert_t = max(mss_close, s0 + dlt)
            # higher timeframe bias known at the sweep candle's open
            d_slice = daily.iloc[: int(np.searchsorted(daily_end.values, np.datetime64(s0), side="right"))]
            bias = {md: bot.bias_from_daily(d_slice, md, s0) for md in ("struct", "week", "ma")}
            target = setup["H"] if setup["direction"] == "buy" else setup["L"]
            base = dict(pair=pair, tf=tf, model=setup["model"], direction=setup["direction"],
                        alert_time=alert_t, c1_ratio=setup["c1_ratio"], dir_ok=setup["dir_ok"],
                        hour=s0.hour, bias_struct=bias["struct"], bias_week=bias["week"],
                        bias_ma=bias["ma"])
            for sm in STOPS:
                stop = bot.mss_stop(window, setup["direction"], mss, sm, buf)
                res = simulate(h1, setup["direction"], mss["level"], stop, target, alert_t)
                if sm == "extreme":
                    signals.append(dict(base, reason=res.get("skip", "filled")))
                if "skip" in res:
                    continue
                res["R"] -= spread / abs(mss["level"] - stop)     # entry cost in R
                trades.append(dict(res, **base, stop_mode=sm))
    return trades, signals


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
            f"total {R.sum():+.1f}R | PF {pf:.1f} | maxDD {dd:.1f}R | streak {streak}")


def allowed(t, bias):
    if bias == "none":
        return True
    return t["bias_" + bias] == (1 if t["direction"] == "buy" else -1)


def pick(rows, bias="none", stop=None, min_rr=0, since=None, until=None, **eq):
    """Filter trades or signals. Book rules (candle 2 direction, candle 1 body) are on."""
    out = []
    for t in rows:
        if t["model"] not in YOUR_MODELS or not t["dir_ok"] or t["c1_ratio"] < MIN_BODY:
            continue
        if stop is not None and t["stop_mode"] != stop:
            continue
        if t.get("rr", 99) < min_rr:
            continue
        if since is not None and t["alert_time"] < since:
            continue
        if until is not None and t["alert_time"] >= until:
            continue
        if not allowed(t, bias):
            continue
        if any(t[k] != v for k, v in eq.items()):
            continue
        out.append(t)
    return out


def avg_r(rows):
    return np.mean([t["R"] for t in rows]) if rows else -9


def mfe_line(trades):
    if not trades:
        return "no trades"
    m = np.array([t["mfe"] for t in trades])
    parts = [f"+{x:g}R {100 * (m >= x).mean():.0f}%" for x in (0.5, 1, 1.5, 2, 3)]
    return f"{len(m)} trades, price reached: " + " | ".join(parts)


def report(trades, signals, pairs, years):
    split = max(t["alert_time"] for t in trades + signals) - pd.Timedelta(days=HOLDOUT_DAYS) if (trades or signals) else None
    RR = 2.0
    L = [f"CRT BACKTEST v4 | {years:g}y OANDA {bot.CONFIG['oanda_price']} | pairs: {len(pairs)}",
         "Models: 3 candle + inside bar. Book rules on (candle 2 direction, candle 1 body >= 50%).",
         "Entry = limit on retest of the MSS level (24h wait). Target = opposite CRT level.",
         f"Min RR {RR:g} counted from the entry. Stop has a ~1.5 pip buffer. R is net of ASSUMED spreads.",
         f"TUNING = before {split:%Y-%m-%d}. HOLD-OUT = after (about 18 months). Choose from tuning only.", ""]

    sig_t = pick(signals, until=split)
    n = len(sig_t)
    if n:
        c = {k: sum(1 for t in sig_t if t["reason"] == k) for k in ("filled", "ran", "nofill", "geometry", "amb", "short")}
        L.append(f"FILL RATE (tuning, {n} MSS signals, no RR filter): filled {100 * c['filled'] / n:.0f}%, "
                 f"ran to target with no retest {100 * c['ran'] / n:.0f}%, never retested {100 * c['nofill'] / n:.0f}%, "
                 f"MSS level already past the target {100 * c['geometry'] / n:.0f}%, "
                 f"unclear {100 * (c['amb'] + c['short']) / n:.0f}%")
    L += ["", f"TUNING PERIOD, RR >= {RR:g}: bias x stop"]
    rows = {}
    for b in BIASES:
        for sm in STOPS:
            sel = pick(trades, b, sm, RR, until=split)
            rows[(b, sm)] = sel
            L.append(f"  {b}/{sm}: " + stats(sel))
    L += ["", "RR THRESHOLD CONTEXT (tuning, no bias, extreme stop):"]
    for r in (0, 1.5, 2, 3):
        L.append(f"  RR >= {r:g}: " + stats(pick(trades, "none", "extreme", r, until=split)))

    elig = {k: v for k, v in rows.items() if len(v) >= MIN_TRADES}
    base_hold = pick(trades, "none", "extreme", RR, since=split)
    if elig:
        best = max(elig, key=lambda k: avg_r(elig[k]))
        b, sm = best
        L += ["", f"CHOSEN BY A FIXED RULE (best average R on tuning with >= {MIN_TRADES} trades): bias={b}, stop={sm}",
              "  tuning:   " + stats(rows[best]),
              "  HOLD-OUT: " + stats(pick(trades, b, sm, RR, since=split)),
              "BASELINE (no bias, extreme stop) on HOLD-OUT: " + stats(base_hold)]
        allsel = pick(trades, b, sm, RR)
        if allsel:
            rrs = np.mean([t["rr"] for t in allsel])
            L += ["", f"BREAK-EVEN: planned RR averages {rrs:.1f}, so the win rate must beat {100 / (1 + rrs):.0f}% before costs."]
        L += ["", "HOW FAR TRADES RUN before the stop (chosen rule, all data):", "  " + mfe_line(allsel),
              "", "CHOSEN RULE BY MODEL (all data):"]
        for m in YOUR_MODELS:
            L.append(f"  {m}: " + stats(pick(trades, b, sm, RR, model=m)))
        L += ["", "CHOSEN RULE BY TIMEFRAME (all data):"]
        for tf in bot.CONFIG["crt_timeframes"]:
            L.append(f"  {bot.TF_NAME[tf]}: " + stats(pick(trades, b, sm, RR, tf=tf)))
        L += ["", "CHOSEN RULE BY PAIR (all data, small samples):"]
        for p in pairs:
            L.append(f"  {p}: " + stats(pick(trades, b, sm, RR, pair=p)))
    else:
        L += ["", f"No rule set reached {MIN_TRADES} trades in the tuning period. Too few to conclude.",
              "BASELINE (no bias, extreme stop) on HOLD-OUT: " + stats(base_hold)]
    L += ["", "Rule: pick from TUNING, read HOLD-OUT once. Under about 30 trades is noise."]
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
    cols = ["pair", "tf", "model", "direction", "stop_mode", "alert_time", "entry_time", "rr", "R",
            "outcome", "mfe", "c1_ratio", "dir_ok", "hour", "bias_struct", "bias_week", "bias_ma"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in sorted(trades, key=lambda x: x["entry_time"]):
            w.writerow([pd.Timestamp(t[c]).strftime("%Y-%m-%d %H:%M") if c in ("alert_time", "entry_time")
                        else t[c] for c in cols])


def main(loader=None):
    loader = loader or load_oanda
    bot.CONFIG["models"] = YOUR_MODELS
    bot.CONFIG["require_sweep_candle_direction"] = False   # applied afterwards as a filter
    bot.CONFIG["min_body_ratio"] = 0.0                      # applied afterwards as a filter
    names = list(bot.CONFIG["pairs"]) + (list(EXTRA_PAIRS) if EXTRA else [])
    trades, signals, used = [], [], []
    for pair in names:
        try:
            h1 = loader(pair, YEARS)
        except Exception as e:
            print(f"{pair}: data error {e}")
            continue
        if h1 is None or len(h1) < 500:
            print(f"{pair}: not enough data")
            continue
        t, s = find_trades(pair, h1)
        print(f"{pair}: {len(h1)} candles, {len(s)} MSS signals, {len(t)} simulated trades")
        trades += t
        signals += s
        used.append(pair)
    if not used:
        print("No data. Is OANDA_TOKEN set?")
        tg("Backtest failed: no data. Check OANDA_TOKEN.")
        return
    text = report(trades, signals, used, YEARS)
    print(text)
    path = os.path.join(tempfile.gettempdir(), "crt_backtest_trades.csv")
    save_csv(trades, path)
    tg(text=text, doc=path)


if __name__ == "__main__":
    main()
