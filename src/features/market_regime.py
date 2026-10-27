"""Market-regime features (same value for every symbol on a given date).

    mkt_trend      1 if benchmark close > SMA200 and SMA50 > SMA200, else 0
    mkt_ret_20d    benchmark 20-day return
    mkt_vol_20     benchmark 20-day realised vol (annualised)
    mkt_vol_pctile percentile of today's mkt_vol_20 within the PRIOR year
    breadth_50     share of universe symbols trading above their SMA50
"""
from __future__ import annotations

import pandas as pd

from src.features import technical as ta


def regime_features(benchmark: pd.DataFrame | None, closes: pd.DataFrame, sma50: pd.DataFrame) -> pd.DataFrame:
    idx = closes.index
    out = pd.DataFrame(index=idx)
    if benchmark is not None and not benchmark.empty:
        c = benchmark["close"].reindex(idx)
        s50, s200 = ta.sma(c, 50), ta.sma(c, 200)
        trend = ((c > s200) & (s50 > s200)).astype(float)
        out["mkt_trend"] = trend.where(s200.notna())
        out["mkt_ret_20d"] = c.pct_change(20)
        vol = ta.realized_vol(c, 20)
        out["mkt_vol_20"] = vol
        # Trailing percentile: rank of today's vol vs the previous 252 values.
        out["mkt_vol_pctile"] = vol.rolling(253, min_periods=126).apply(
            lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    else:
        for col in ("mkt_trend", "mkt_ret_20d", "mkt_vol_20", "mkt_vol_pctile"):
            out[col] = float("nan")
    above = (closes > sma50).astype(float).where(sma50.notna())
    out["breadth_50"] = above.mean(axis=1, skipna=True).astype(float)
    return out
