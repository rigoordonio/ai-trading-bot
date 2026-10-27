"""Strategy A - trend following.

Hypothesis: assets in an established uptrend tend to keep trending
(time-series momentum).

Conditions (score = share of conditions true):
    close > SMA(fast)
    SMA(fast) > SMA(slow)
    return over `momentum_window` days > 0
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, conditions_score


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def __init__(self, params=None, entry_threshold: float = 1.0):
        super().__init__({"fast_sma": 50, "slow_sma": 200, "momentum_window": 20, **(params or {})}, entry_threshold)
        p = self.params
        self.required_features = (self.c("close"), self.c(f"sma_{p['fast_sma']}"), self.c(f"sma_{p['slow_sma']}"),
                                  self.c(f"ret_{p['momentum_window']}d"))

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        fast, slow = panel[self.c(f"sma_{p['fast_sma']}")], panel[self.c(f"sma_{p['slow_sma']}")]
        mom = panel[self.c(f"ret_{p['momentum_window']}d")]
        c1 = (panel[self.c("close")] > fast).where(fast.notna())
        c2 = (fast > slow).where(slow.notna())
        c3 = (mom > 0).where(mom.notna())
        return conditions_score(c1, c2, c3).rename(self.name)
