"""Volume confirmation (swing component).

Hypothesis: up-moves on above-average volume carry more information
(institutional participation) than up-moves on thin volume.

Score = share of: volume_ratio >= min_ratio, 1-day return > 0,
close > SMA(short). Stateless.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, conditions_score


class VolumeConfirmationStrategy(Strategy):
    name = "volume_confirmation"

    def __init__(self, params=None, entry_threshold: float = 1.0):
        super().__init__({"min_ratio": 1.3, "short_sma": 20, **(params or {})}, entry_threshold)
        self.required_features = (self.c("close"), self.c("volume_ratio"), self.c("ret_1d"),
                                  self.c(f"sma_{self.params['short_sma']}"))

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        vr, r1 = panel[self.c("volume_ratio")], panel[self.c("ret_1d")]
        sma = panel[self.c(f"sma_{p['short_sma']}")]
        return conditions_score((vr >= p["min_ratio"]).where(vr.notna()), (r1 > 0).where(r1.notna()),
                                (panel[self.c("close")] > sma).where(sma.notna())).rename(self.name)
