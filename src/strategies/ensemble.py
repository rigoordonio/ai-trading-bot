"""Signal ensemble: weighted combination of strategy scores.

    combined = sum_i(w_i * s_i) / sum_i(w_i  for components defined on that row)

Weights and thresholds come from strategies.yaml. If a component is NaN on a
row (warm-up, missing model) its weight is removed and the rest are
renormalised; if fewer than ``min_components`` are defined, the combined
score is NaN -> HOLD (no trade on thin evidence).
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import SignalFrame, Strategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.long_term import DrawdownFilterStrategy, LongTermMomentumStrategy, LowVolatilityStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.ml_strategy import MLStrategy, centre_probability
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.volume_confirmation import VolumeConfirmationStrategy

RULE_STRATEGIES: dict[str, type[Strategy]] = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "volume_confirmation": VolumeConfirmationStrategy,
    "long_term_momentum": LongTermMomentumStrategy,
    "low_volatility": LowVolatilityStrategy,
    "drawdown_filter": DrawdownFilterStrategy,
}


def build_strategy(name: str, scfg, strategy_type: str | None = None) -> Strategy:
    kind = strategy_type or name
    if kind not in RULE_STRATEGIES:
        raise KeyError(f"unknown strategy type '{kind}'")
    strat = RULE_STRATEGIES[kind](dict(scfg.params), scfg.entry_threshold)
    strat.name = name               # component name (a bot may run two momentum variants)
    return strat


def build_rule_strategies(cfg) -> dict[str, Strategy]:
    return {n: build_strategy(n, s) for n, s in cfg.strategies.strategies.items()
            if s.enabled and n in RULE_STRATEGIES}


class EnsembleStrategy:
    name = "ensemble"

    def __init__(self, strategies: dict[str, Strategy], ensemble_cfg, ml: MLStrategy | None = None,
                 min_components: int = 2, required_components: tuple[str, ...] = ()):
        self.strategies = strategies
        self.cfg = ensemble_cfg
        self.ml = ml
        self.min_components = min_components
        # A required component that is undefined on a row makes the combined
        # score NaN -> HOLD. Used for ML: model missing/failed => NO TRADE,
        # never "trade on the remaining rules".
        self.required_components = tuple(required_components)

    def component_scores(self, panel: pd.DataFrame) -> pd.DataFrame:
        cols = {name: s.score(panel).clip(0, 1) for name, s in self.strategies.items()}
        out = pd.DataFrame(cols, index=panel.index)
        if self.ml is not None:
            prob = self.ml.score(panel)
            out["ml_probability"] = prob
            out["ml"] = centre_probability(prob, self.cfg.ml_probability_floor, self.cfg.ml_probability_cap)
        return out

    def combine(self, comps: pd.DataFrame) -> pd.Series:
        weights = {k: v for k, v in self.cfg.weights.items() if k in comps.columns and v > 0}
        if not weights:
            raise ValueError("no ensemble weights match the available components")
        num = sum(comps[k].fillna(0.0) * w for k, w in weights.items())
        den = sum(comps[k].notna().astype(float) * w for k, w in weights.items())
        n_def = sum(comps[k].notna().astype(int) for k in weights)
        combined = (num / den.where(den > 0))
        combined = combined.where(n_def >= min(self.min_components, len(weights)))
        for req in self.required_components:
            if req not in comps.columns:
                return combined * float("nan")
            combined = combined.where(comps[req].notna())
        return combined

    def signal_frame(self, panel: pd.DataFrame) -> SignalFrame:
        comps = self.component_scores(panel)
        comps.insert(0, "score", self.combine(comps))
        c = self.cfg
        return SignalFrame("ensemble", comps, c.buy_threshold, c.hold_threshold, c.exit_threshold,
                           c.reduce_fraction, meta={"weights": dict(c.weights)})
