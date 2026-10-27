"""Long-horizon components (designed for WEEKLY features, prefix ``w_``).

LongTermMomentumStrategy  ("12-1 momentum")
    Hypothesis: 12-month winners keep outperforming over the next 3-12 months,
    but the most recent month tends to mean-revert, so it is skipped:
        mom_12_1 = (1 + ret_52w) / (1 + ret_4w) - 1
    Score = cross-sectional percentile rank, 0 if mom_12_1 <= 0.

LowVolatilityStrategy
    Hypothesis: lower-volatility stocks deliver better risk-adjusted returns
    than high-volatility ones (low-vol anomaly).
    Score = percentile rank of -realised_vol.

DrawdownFilterStrategy
    Hypothesis: long-term buys work poorly in assets in deep drawdowns.
    Score = 1 if close is within `max_drawdown` of its 52-week high,
    linearly down to 0 at 2x that distance.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class LongTermMomentumStrategy(Strategy):
    name = "long_term_momentum"

    def __init__(self, params=None, entry_threshold: float = 0.6):
        super().__init__({"long_window": 52, "skip_window": 4, "prefix": "w_", **(params or {})}, entry_threshold)
        p = self.params
        self.required_features = (self.c(f"ret_{p['long_window']}d"), self.c(f"ret_{p['skip_window']}d"))

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        long_r, skip_r = panel[self.c(f"ret_{p['long_window']}d")], panel[self.c(f"ret_{p['skip_window']}d")]
        mom = (1 + long_r) / (1 + skip_r) - 1
        rank = mom.groupby(level="date").rank(pct=True)
        return rank.where(mom > 0, 0.0).where(mom.notna()).rename(self.name)


class LowVolatilityStrategy(Strategy):
    name = "low_volatility"

    def __init__(self, params=None, entry_threshold: float = 0.5):
        super().__init__({"vol_window": 26, "prefix": "w_", **(params or {})}, entry_threshold)
        self.required_features = (self.c(f"realized_vol_{self.params['vol_window']}"),)

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        vol = panel[self.c(f"realized_vol_{self.params['vol_window']}")]
        return (-vol).groupby(level="date").rank(pct=True).where(vol.notna()).rename(self.name)


class DrawdownFilterStrategy(Strategy):
    name = "drawdown_filter"

    def __init__(self, params=None, entry_threshold: float = 0.5):
        super().__init__({"max_drawdown": 0.20, "prefix": "w_", **(params or {})}, entry_threshold)
        self.required_features = (self.c("dist_52w_high"),)

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        dd = -panel[self.c("dist_52w_high")]          # positive = below the high
        m = self.params["max_drawdown"]
        return (1 - ((dd - m) / m).clip(lower=0, upper=1)).where(dd.notna()).rename(self.name)
