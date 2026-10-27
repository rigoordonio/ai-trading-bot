"""ML strategy: wraps model probabilities as a strategy score.

It only accepts OUT-OF-SAMPLE predictions (from walk-forward validation, or
from a model whose training data ended before the prediction dates). Using
in-sample predictions in a backtest is a classic way to fool yourself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import ModelInferenceError
from src.strategies.base import Strategy


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, predictions: pd.Series, params=None, entry_threshold: float = 0.55,
                 trained_until: pd.Timestamp | None = None):
        super().__init__(params, entry_threshold)
        if not isinstance(predictions.index, pd.MultiIndex):
            raise ValueError("predictions must be indexed by (date, symbol)")
        vals = predictions.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if ((finite < 0) | (finite > 1)).any():
            raise ModelInferenceError("probabilities outside [0, 1]")
        self.predictions = predictions.astype(float)
        self.trained_until = pd.Timestamp(trained_until) if trained_until is not None else None

    def score(self, panel: pd.DataFrame) -> pd.Series:
        s = self.predictions.reindex(panel.index)
        if self.trained_until is not None:
            dates = s.index.get_level_values("date")
            s = s.where(dates > self.trained_until)   # never score inside the training window
        return s.rename(self.name)


def centre_probability(p: pd.Series, floor: float = 0.5, cap: float = 0.7) -> pd.Series:
    """Map a probability to a [0, 1] ensemble score: <= floor -> 0, >= cap -> 1."""
    if cap <= floor:
        raise ValueError("cap must exceed floor")
    return ((p - floor) / (cap - floor)).clip(0.0, 1.0)
