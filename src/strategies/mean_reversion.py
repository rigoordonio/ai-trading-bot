"""Strategy B - mean reversion (buy short-term oversold dips).

Hypothesis: sharp short-term declines in liquid assets partially revert,
especially when the long-term trend is still up.

Entry strength = share of: RSI < threshold, z-score(close vs 20d mean) <
zscore_threshold, and (optionally) close < lower Bollinger band.
Optional filter: close > SMA200 (only buy dips in uptrends).

STATEFUL: after entry the score stays 1 until close recovers to the 20-day
mean or `max_holding_days` sessions pass. The state machine walks forward in
time and only uses data up to each date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, by_symbol, conditions_score


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    required_features = ("close", "rsi_14", "bb_lower", "zscore_20", "sma_20", "sma_200")

    def __init__(self, params=None, entry_threshold: float = 0.5):
        super().__init__({"rsi_threshold": 30, "zscore_threshold": -2.0, "use_bollinger": True,
                          "exit_on_mean": True, "max_holding_days": 10, "require_long_term_uptrend": True,
                          **(params or {})}, entry_threshold)

    def entry_strength(self, panel: pd.DataFrame) -> pd.Series:
        p = self.params
        conds = [(panel["rsi_14"] < p["rsi_threshold"]).where(panel["rsi_14"].notna()),
                 (panel["zscore_20"] < p["zscore_threshold"]).where(panel["zscore_20"].notna())]
        if p["use_bollinger"]:
            conds.append((panel["close"] < panel["bb_lower"]).where(panel["bb_lower"].notna()))
        strength = conditions_score(*conds)
        if p["require_long_term_uptrend"]:
            uptrend = (panel["close"] > panel["sma_200"]).astype(float).where(panel["sma_200"].notna())
            strength = strength.where(uptrend.isna() | (uptrend == 1.0), 0.0).where(uptrend.notna())
        return strength

    def score(self, panel: pd.DataFrame) -> pd.Series:
        self.check_features(panel)
        p = self.params
        strength = self.entry_strength(panel)
        parts = []
        for sym, g in by_symbol(panel):
            st = strength.xs(sym, level="symbol").reindex(g.index).to_numpy(dtype=float)
            close, mean = g["close"].to_numpy(dtype=float), g["sma_20"].to_numpy(dtype=float)
            score = np.full(len(g), np.nan)
            in_pos, held = False, 0
            for i in range(len(g)):
                if in_pos:
                    held += 1
                    if (p["exit_on_mean"] and close[i] >= mean[i]) or held >= p["max_holding_days"]:
                        in_pos = False
                        score[i] = 0.0          # exit today; no re-entry on the same bar
                    else:
                        score[i] = 1.0
                    continue
                if not np.isfinite(st[i]):
                    continue                    # undefined (warm-up / missing data)
                if st[i] >= self.entry_threshold:
                    in_pos, held = True, 0
                    score[i] = 1.0
                else:
                    score[i] = st[i] * 0.99     # partial evidence, always below the entry level
            idx = pd.MultiIndex.from_arrays([g.index, [sym] * len(g)], names=["date", "symbol"])
            parts.append(pd.Series(score, index=idx))
        return pd.concat(parts).reindex(panel.index).rename(self.name)
