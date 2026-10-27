"""Strategy D - breakout (Donchian-style).

Hypothesis: a close above the prior N-day high on strong volume, within an
uptrend, marks the start of a new leg up.

Entry: close > prior `lookback`-day high (today excluded), volume_ratio >=
`volume_ratio_min`, and close > SMA50 if `require_trend`.
Exit (stateful): close < prior `exit_lookback`-day low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, by_symbol, conditions_score


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self, params=None, entry_threshold: float = 1.0):
        super().__init__({"lookback": 55, "exit_lookback": 20, "volume_ratio_min": 1.5, "require_trend": True,
                          **(params or {})}, entry_threshold)
        p = self.params
        self.required_features = ("close", f"prior_high_{p['lookback']}", f"prior_low_{p['exit_lookback']}",
                                  "volume_ratio", "sma_50")

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        hi = panel[f"prior_high_{p['lookback']}"]
        conds = [(panel["close"] > hi).where(hi.notna()),
                 (panel["volume_ratio"] >= p["volume_ratio_min"]).where(panel["volume_ratio"].notna())]
        if p["require_trend"]:
            conds.append((panel["close"] > panel["sma_50"]).where(panel["sma_50"].notna()))
        strength = conditions_score(*conds)
        parts = []
        for sym, g in by_symbol(panel):
            st = strength.xs(sym, level="symbol").reindex(g.index).to_numpy(dtype=float)
            close = g["close"].to_numpy(dtype=float)
            exit_lvl = g[f"prior_low_{p['exit_lookback']}"].to_numpy(dtype=float)
            score = np.full(len(g), np.nan)
            in_pos = False
            for i in range(len(g)):
                if in_pos:
                    if np.isfinite(exit_lvl[i]) and close[i] < exit_lvl[i]:
                        in_pos, score[i] = False, 0.0
                    else:
                        score[i] = 1.0
                    continue
                if not np.isfinite(st[i]):
                    continue
                if st[i] >= self.entry_threshold:
                    in_pos, score[i] = True, 1.0
                else:
                    score[i] = st[i] * 0.99
            idx = pd.MultiIndex.from_arrays([g.index, [sym] * len(g)], names=["date", "symbol"])
            parts.append(pd.Series(score, index=idx))
        return pd.concat(parts).reindex(panel.index).rename(self.name)
