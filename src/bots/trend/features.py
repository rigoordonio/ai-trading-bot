"""TREND_V1 features: sustained trends on DAILY bars.

Shared pipeline columns plus one trend-specific feature:

    trend_strength = (SMA50 / SMA200 - 1) / realised_vol_60
        moving-average spread in units of annualised volatility. Uses only
        values of the same row (no look-ahead). NaN when volatility is 0/NaN.
"""
import numpy as np
import pandas as pd

FEATURES_VERSION = "trend-f1"

REQUIRED = (
    "ret_5d", "ret_20d", "ret_60d",
    "sma_20", "sma_50", "sma_100", "sma_200", "ema_20", "ema_50",
    "rsi_14", "macd", "atr_14", "realized_vol_20", "realized_vol_60",
    "dist_52w_high", "dist_sma_50", "dist_sma_200",   # price/SMA50 - 1, price/SMA200 - 1
    "trend_strength", "risk_adj_mom_60",
)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    vol = out["realized_vol_60"].where(out["realized_vol_60"] > 0)
    out["trend_strength"] = (out["sma_50"] / out["sma_200"] - 1.0) / vol
    out["trend_strength"] = out["trend_strength"].replace([np.inf, -np.inf], np.nan)
    return out
