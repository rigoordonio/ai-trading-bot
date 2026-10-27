"""Strategy interface and the SignalFrame contract.

A strategy maps the feature panel to a SCORE in [0, 1] per (date, symbol).
It never sees the portfolio, never sizes positions and never creates
orders. That separation is what lets the risk engine stay in charge.

A ``SignalFrame`` bundles scores with the thresholds that turn a score into
an action, given whether the symbol is currently held:

    not held : score >= buy_threshold              -> BUY       else HOLD (stay flat)
    held     : score <  exit_threshold             -> EXIT
               exit <= score < hold_threshold      -> REDUCE (once)
               otherwise                           -> HOLD
    NaN score (missing data)                       -> HOLD (never trade on missing data)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.errors import FeatureSchemaError

BUY, HOLD, REDUCE, EXIT = "BUY", "HOLD", "REDUCE", "EXIT"


@dataclass
class SignalFrame:
    name: str
    scores: pd.DataFrame                 # index (date, symbol); column "score" + component columns
    buy_threshold: float
    hold_threshold: float
    exit_threshold: float
    reduce_fraction: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "score" not in self.scores.columns:
            raise ValueError("SignalFrame.scores needs a 'score' column")
        if not (self.exit_threshold <= self.hold_threshold <= self.buy_threshold):
            raise ValueError("thresholds must satisfy exit <= hold <= buy")

    def decide(self, score: float, held: bool, reduced: bool = False) -> str:
        if score is None or not np.isfinite(score):
            return HOLD
        if not held:
            return BUY if score >= self.buy_threshold else HOLD
        if score < self.exit_threshold:
            return EXIT
        if score < self.hold_threshold:
            return HOLD if reduced else REDUCE
        return HOLD

    def on(self, date: pd.Timestamp) -> pd.DataFrame:
        try:
            return self.scores.xs(date, level="date")
        except KeyError:
            return pd.DataFrame(columns=self.scores.columns)


class Strategy(ABC):
    """Base class. Subclasses implement ``score``."""

    name: str = "base"
    required_features: tuple[str, ...] = ()

    def __init__(self, params: dict[str, Any] | None = None, entry_threshold: float = 0.5):
        self.params = dict(params or {})
        # Column prefix, e.g. "w_" to run the same hypothesis on weekly features.
        self.prefix: str = self.params.pop("prefix", "")
        self.entry_threshold = entry_threshold

    def c(self, column: str) -> str:
        """Feature column name with this strategy's prefix."""
        return f"{self.prefix}{column}"

    def check_features(self, panel: pd.DataFrame) -> None:
        missing = [c for c in self.required_features if c not in panel.columns]
        if missing:
            raise FeatureSchemaError(f"{self.name}: feature panel is missing {missing}; enable them in the "
                                     "feature config of this strategy/bot")

    @abstractmethod
    def score(self, panel: pd.DataFrame) -> pd.Series:
        """Series indexed like ``panel`` with values in [0, 1] (NaN = undefined)."""

    def signal_frame(self, panel: pd.DataFrame) -> SignalFrame:
        """Stand-alone use: hold while score >= entry_threshold."""
        s = self.score(panel).clip(0, 1)
        t = self.entry_threshold
        return SignalFrame(self.name, pd.DataFrame({"score": s, self.name: s}), t, t, t,
                           meta={"strategy": self.name, "params": self.params})


def by_symbol(panel: pd.DataFrame):
    """Iterate (symbol, frame indexed by date)."""
    for sym, g in panel.groupby(level="symbol", sort=False):
        yield sym, g.droplevel("symbol")


def conditions_score(*conds: pd.Series) -> pd.Series:
    """Fraction of boolean conditions that are true; NaN if any input is NaN."""
    df = pd.concat(conds, axis=1)
    valid = df.notna().all(axis=1)
    return df.astype(float).mean(axis=1).where(valid)
