"""Strategy C - cross-sectional momentum.

Hypothesis: assets that outperformed peers over the past 1-3 months tend to
keep outperforming over the next weeks (relative momentum).

For each date, rank symbols on each of: 20d return, 60d return and
(optionally) 60d risk-adjusted momentum. Score = average percentile rank.
Symbols outside the top `top_n` are capped at 0.5 so they cannot pass entry
thresholds above 0.5. Absolute-momentum filter: score 0 if 60d return <= 0.

Ranks use only same-date values -> no look-ahead.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, params=None, entry_threshold: float = 0.67):
        super().__init__({"windows": [20, 60], "risk_adjusted": True, "top_n": 3,
                          "require_positive_momentum": True, **(params or {})}, entry_threshold)
        cols = [self.c(f"ret_{w}d") for w in self.params["windows"]]
        if self.params["risk_adjusted"]:
            cols.append(self.c(f"risk_adj_mom_{max(self.params['windows'])}"))
        self.rank_columns = cols
        self.required_features = tuple(cols)

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        ranks = pd.concat([panel[c].groupby(level="date").rank(pct=True) for c in self.rank_columns], axis=1)
        valid = ranks.notna().all(axis=1)
        composite = ranks.mean(axis=1).where(valid)
        # re-rank the composite so the score is a clean percentile in (0, 1]
        score = composite.groupby(level="date").rank(pct=True)
        order = composite.groupby(level="date").rank(ascending=False, method="first")
        score = score.where(order <= p["top_n"], score.clip(upper=0.5))
        if p["require_positive_momentum"]:
            longest = self.c(f"ret_{max(p['windows'])}d")
            score = score.where(panel[longest] > 0, 0.0).where(valid)
        return score.rename(self.name)
