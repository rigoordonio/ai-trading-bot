"""Return and momentum features."""
from __future__ import annotations

import pandas as pd

from src.features import technical as ta


def momentum_features(df: pd.DataFrame, return_windows=(1, 5, 20, 60), rsi_window=14,
                      macd_cfg=None, roc_window=10) -> pd.DataFrame:
    c = df["close"]
    macd_cfg = macd_cfg or {"fast": 12, "slow": 26, "signal": 9}
    out = pd.DataFrame(index=df.index)
    for n in return_windows:
        out[f"ret_{n}d"] = c.pct_change(n)
    out[f"rsi_{rsi_window}"] = ta.rsi(c, rsi_window)
    out = out.join(ta.macd(c, macd_cfg["fast"], macd_cfg["slow"], macd_cfg["signal"]))
    out[f"roc_{roc_window}"] = ta.roc(c, roc_window)
    return out


def risk_adjusted_momentum(ret: pd.Series, vol: pd.Series) -> pd.Series:
    """Momentum per unit of volatility (a Sharpe-like ranking score)."""
    return ret / vol.where(vol > 0)
