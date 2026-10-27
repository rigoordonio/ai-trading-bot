"""Volatility features."""
from __future__ import annotations

import pandas as pd

from src.features import technical as ta


def volatility_features(df: pd.DataFrame, atr_window=14, vol_windows=(20, 60), bb_cfg=None,
                        periods_per_year: int = 252) -> pd.DataFrame:
    bb_cfg = bb_cfg or {"window": 20, "num_std": 2.0}
    out = pd.DataFrame(index=df.index)
    out[f"atr_{atr_window}"] = ta.atr(df["high"], df["low"], df["close"], atr_window)
    out["atr_pct"] = out[f"atr_{atr_window}"] / df["close"]
    for n in vol_windows:
        out[f"realized_vol_{n}"] = ta.realized_vol(df["close"], n, periods_per_year=periods_per_year)
    out = out.join(ta.bollinger(df["close"], bb_cfg["window"], bb_cfg["num_std"]))
    return out
