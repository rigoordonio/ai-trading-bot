"""TREND_V1 strategy logic: momentum, trend following, moving-average
alignment and breakout confirmation.

Candidate rule (a HYPOTHESIS, objectively backtested, not assumed):
    Close > SMA50 AND SMA50 > SMA200 AND 60D momentum > threshold
implemented by the `long_trend` trend_following component.
"""
import pandas as pd

ALLOWED_COMPONENTS = frozenset({"trend_following", "momentum", "breakout"})
SCORE_NAME = "MEDIUM_TERM_TREND_SCORE"


def bullish_structure(panel: pd.DataFrame, momentum_threshold: float = 0.0) -> pd.Series:
    """The candidate rule as a boolean per row (NaN inputs -> False)."""
    ok = (panel["close"] > panel["sma_50"]) & (panel["sma_50"] > panel["sma_200"]) \
        & (panel["ret_60d"] > momentum_threshold)
    return ok.fillna(False)
