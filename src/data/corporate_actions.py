"""Explicit corporate-action handling.

Why this matters: a 4-for-1 split makes the *quoted* price fall 75% overnight.
A strategy fed unadjusted prices sees a crash that never happened (and an RSI
of ~0). Dividends matter too: ignoring them understates long-run returns.

``adjusted`` mode (default)
    factor_t = adj_close_t / close_t
    open/high/low/close are multiplied by factor_t -> a total-return series.
    The original quoted prices are kept as ``raw_*`` columns for audit.
    Volume is divided by the cumulative *split* ratio only, if the vendor has
    not already split-adjusted it.

Backtests execute on adjusted prices. That is equivalent to assuming
dividends are reinvested on the ex-date, which is documented in the README.
For live/paper order quantities the raw (quoted) price is what matters; the
latest bar always has factor 1, so the two coincide for current prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjustment_factor(df: pd.DataFrame) -> pd.Series:
    f = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    return f.ffill().bfill().fillna(1.0)


def split_volume_factor(df: pd.DataFrame) -> pd.Series:
    """Cumulative product of split ratios occurring AFTER each date."""
    ratios = df.get("splits", pd.Series(0.0, index=df.index)).replace(0, 1.0).fillna(1.0)
    # product over s > t  (reverse cumprod, shifted by one)
    rev = ratios[::-1].cumprod()[::-1]
    return rev.shift(-1).fillna(1.0)


def apply_adjustment(df: pd.DataFrame, mode: str = "adjusted", volume_split_adjusted: bool = True) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[f"raw_{col}"] = out[col]
    if mode == "raw":
        out["adj_factor"] = 1.0
        return out
    if mode != "adjusted":
        raise ValueError(f"unknown price_adjustment mode '{mode}'")
    factor = adjustment_factor(out)
    out["adj_factor"] = factor
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    if not volume_split_adjusted:
        out["volume"] = out["volume"] * split_volume_factor(out)
    return out
