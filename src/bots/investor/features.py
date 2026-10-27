"""INVESTOR_V1 features: durable trends on WEEKLY candles.

Computed in weekly space (one row per COMPLETED week), then prefixed ``w_``
and made available from the first trading day after the week ends.

    drawdown_52    close / max(close over the last 52 weeks) - 1
    trend_strength (SMA10w / SMA40w - 1) / realised_vol_26w

Fundamental features: the architecture allows them (config `fundamentals`),
but they stay disabled until a reliable source is configured. Nothing here
fabricates fundamental data.
"""
import numpy as np
import pandas as pd

FEATURES_VERSION = "investor-f1"

REQUIRED = (
    "w_ret_4d", "w_ret_12d", "w_ret_26d", "w_ret_52d",     # 4/12/26/52-WEEK returns
    "w_sma_10", "w_sma_20", "w_sma_40",                    # 10/20/40-week moving averages
    "w_rsi_14", "w_realized_vol_26", "w_dist_52w_high",
    "w_trend_strength", "w_drawdown_52", "w_risk_adj_mom_52",
)


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    close = out["close"]
    peak = close.groupby(level="symbol").transform(lambda s: s.rolling(52, min_periods=52).max())
    out["drawdown_52"] = close / peak - 1.0
    vol = out["realized_vol_26"].where(out["realized_vol_26"] > 0)
    ts = (out["sma_10"] / out["sma_40"] - 1.0) / vol
    out["trend_strength"] = ts.replace([np.inf, -np.inf], np.nan)
    return out
