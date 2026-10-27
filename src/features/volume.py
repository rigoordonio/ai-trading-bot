"""Volume / liquidity features."""
from __future__ import annotations

import pandas as pd


def volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    v = df["volume"]
    mean = v.rolling(window, min_periods=window).mean()
    sd = v.rolling(window, min_periods=window).std()
    out = pd.DataFrame(index=df.index)
    out[f"avg_volume_{window}"] = mean
    out["volume_ratio"] = v / mean
    out["volume_zscore"] = (v - mean) / sd.where(sd > 0)
    out[f"dollar_volume_{window}"] = (v * df["close"]).rolling(window, min_periods=window).mean()
    return out
