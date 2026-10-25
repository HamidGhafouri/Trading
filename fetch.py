#!/usr/bin/env python3
"""Trading Desk data job (runs in GitHub Actions). Pulls daily bars, computes indicators, writes data/latest.json.
Indicators: SMA 5/50/100/200, RSI(2) Wilder, ATR(14) Wilder, 52-week high, 1/12/24-month returns, 60-day vol,
trading days to next earnings. 'close' is the latest price (today's partial bar during market hours)."""
import json, math, sys, datetime as dt, time
import numpy as np, pandas as pd, yfinance as yf

UNIVERSE = sorted(set("""SPY QQQ IWM DIA XLK XLF XLE XLV XLP XLI XLY XLU XLB XLRE XLC VTI QUAL MTUM SGOV IBIT ETHA
AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA JPM V MA LLY UNH XOM CVX COST WMT HD PG JNJ ABBV MRK ORCL CRM AMD NFLX
GS CAT GE BAC KO PEP LIN TMO MU ADBE""".split()))
EARNINGS = set("""AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA JPM V MA LLY UNH XOM CVX COST WMT HD PG JNJ ABBV MRK ORCL
CRM AMD NFLX GS CAT GE BAC KO PEP LIN TMO MU ADBE""".split())


def wilder(series, n):
    return series.ewm(alpha=1 / n, adjust=False).mean()


def indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    out = {"close": float(c.iloc[-1]), "open": float(df["Open"].iloc[-1]), "high": float(h.iloc[-1]),
           "low": float(l.iloc[-1]), "bar_date": str(df.index[-1].date()), "bars": int(len(c))}
    for n in (5, 50, 100, 200):
        out[f"sma{n}"] = float(c.iloc[-n:].mean()) if len(c) >= n else None
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), 2), wilder((-d).clip(lower=0), 2)
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] > 0 else float("inf")
    out["rsi2"] = float(100 - 100 / (1 + rs)) if math.isfinite(rs) else 100.0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = float(wilder(tr, 14).iloc[-1])
    out["high52"] = float(h.iloc[-252:].max())
    for k, n in (("ret_1m", 21), ("ret_12m", 252), ("ret_24m", 504)):
        out[k] = float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None
    lr = np.log(c / c.shift()).iloc[-60:]
    out["vol60"] = float(lr.std() * math.sqrt(252)) if len(lr) >= 60 else None
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def trading_days_until(target):
    import pandas.tseries.offsets as off
    today = dt.date.today()
    if target < today:
        return None
    return int(np.busday_count(today, target))


def main():
    raw = yf.download(UNIVERSE + ["^VIX"], period="3y", interval="1d", auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    syms, errors = {}, []
    for t in UNIVERSE:
        try:
            df = raw[t].dropna(subset=["Close"])
            if len(df) < 210:
                errors.append(f"{t}: only {len(df)} bars"); continue
            syms[t] = indicators(df)
        except Exception as e:
            errors.append(f"{t}: {e}")
    vix = None
    try:
        vix = round(float(raw["^VIX"]["Close"].dropna().iloc[-1]), 2)
    except Exception as e:
        errors.append(f"VIX: {e}")
    for t in sorted(EARNINGS & set(syms)):
        try:
            cal = yf.Ticker(t).calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            nxt = min(dates) if dates else None
            if nxt:
                n = trading_days_until(nxt)
                syms[t]["next_earnings"] = str(nxt)
                syms[t]["next_earnings_days"] = n if n is not None and n <= 30 else None
            time.sleep(0.2)
        except Exception as e:
            errors.append(f"{t} earnings: {e}")
    out = {"generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "source": "yfinance (Yahoo Finance), adjusted daily bars",
           "market": {"vix": vix}, "symbols": syms, "errors": errors}
    import os
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/latest.json", "w"), indent=0)
    print(f"{len(syms)} symbols, VIX {vix}, {len(errors)} errors")
    if len(syms) < len(UNIVERSE) * 0.8:
        sys.exit(1)


if __name__ == "__main__":
    main()
