"""Resampling validated DAILY bars into higher timeframes.

One data source, one validation path: weekly candles are always built from
already validated, adjusted daily bars. A week's candle is labelled with the
last trading date that falls in that week (Mon-Fri), so a holiday Friday
gives a Thursday label.

The candle of the CURRENT week is only complete after its last session.
Consumers must apply the availability rule in src/features/weekly.py
(usable from the first trading day after the label), never the label itself.
"""
from __future__ import annotations

import pandas as pd

from src.errors import DataValidationError

WEEKLY_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "adj_close": "last",
              "volume": "sum", "dividends": "sum", "splits": "max"}


def infer_timeframe(index: pd.DatetimeIndex) -> str:
    """'daily' or 'weekly' from the median spacing of a bar index."""
    if len(index) < 3:
        return "unknown"
    gap = pd.Series(index).diff().dropna().dt.days.median()
    if gap <= 4:
        return "daily"
    if 5 <= gap <= 9:
        return "weekly"
    return "unknown"


def assert_daily(symbol: str, df: pd.DataFrame) -> None:
    """Bots always receive daily bars and resample themselves. Feeding a bot
    pre-aggregated bars would silently change every feature's meaning."""
    tf = infer_timeframe(pd.DatetimeIndex(df.index))
    if tf not in ("daily", "unknown"):
        raise DataValidationError(f"{symbol}: expected DAILY bars, got {tf} bars",
                                  context={"symbol": symbol, "rule": "timeframe"})


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    cols = {k: v for k, v in WEEKLY_AGG.items() if k in daily.columns}
    period = daily.index.to_period("W-FRI")
    g = daily.groupby(period)
    weekly = g.agg(cols)
    weekly.index = g.apply(lambda x: x.index.max()).values      # label = last trading date in the week
    weekly.index = pd.DatetimeIndex(weekly.index, name="date")
    return weekly
