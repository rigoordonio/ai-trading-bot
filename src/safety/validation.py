"""Point-of-decision validation (fail closed).

These run on the CURRENT observation right before a signal can become an
order. Historical cleaning lives in src/data/validator.py; this module answers
a narrower question: "can I trust the numbers I am about to trade on?"

Nothing here repairs data. Invalid -> raise -> no trade for that symbol.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.errors import DataValidationError, FeatureSchemaError, ModelInferenceError, StaleMarketDataError


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_current_bar(symbol: str, bars: pd.DataFrame, as_of: pd.Timestamp, max_staleness_days: int = 4,
                         max_abs_return: float = 0.40) -> pd.Series:
    """Validate the latest bar used for a decision. Returns it, or raises."""
    if bars is None or bars.empty:
        raise DataValidationError(f"{symbol}: no market data", context={"symbol": symbol})
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataValidationError(f"{symbol}: index is not a DatetimeIndex", context={"symbol": symbol})
    if idx.tz is not None:
        raise DataValidationError(f"{symbol}: unexpected timezone {idx.tz} (expected tz-naive exchange dates)",
                                  context={"symbol": symbol})
    upto = bars.loc[:as_of]
    if upto.empty:
        raise DataValidationError(f"{symbol}: no bar on or before {as_of.date()}", context={"symbol": symbol})
    if upto.index.has_duplicates:
        raise DataValidationError(f"{symbol}: duplicate timestamps in recent data", context={"symbol": symbol})
    bar = upto.iloc[-1]
    ts = upto.index[-1]
    prev = upto["close"].iloc[-2] if len(upto) >= 2 else None
    validate_bar_values(symbol, bar, ts, as_of, prev, max_staleness_days, max_abs_return)
    return bar


def validate_bar_values(symbol: str, bar: pd.Series | dict, bar_date: pd.Timestamp, as_of: pd.Timestamp,
                        prev_close: float | None = None, max_staleness_days: int = 4,
                        max_abs_return: float = 0.40) -> None:
    """Checks on ONE bar (no history needed): staleness, finiteness, positivity,
    OHLC consistency and jump vs the previous close. Raises on failure."""
    ctx = {"symbol": symbol, "bar_date": str(pd.Timestamp(bar_date).date())}
    if getattr(pd.Timestamp(bar_date), "tz", None) is not None:
        raise DataValidationError(f"{symbol}: unexpected timezone on bar timestamp", context=ctx)
    age = (pd.Timestamp(as_of).normalize() - pd.Timestamp(bar_date).normalize()).days
    if age < 0:
        raise DataValidationError(f"{symbol}: bar dated after decision time", context=ctx)
    if age > max_staleness_days:
        raise StaleMarketDataError(f"{symbol}: latest bar {ctx['bar_date']} is {age} days old "
                                   f"(max {max_staleness_days})", context={**ctx, "age_days": age})
    vals: dict[str, float] = {}
    for name in ("open", "high", "low", "close", "volume"):
        raw = bar.get(name) if hasattr(bar, "get") else None
        if not _finite(raw):
            raise DataValidationError(f"{symbol}: {name} is missing/NaN/inf on {ctx['bar_date']}", context=ctx)
        vals[name] = float(raw)  # type: ignore[arg-type]  # _finite() guarantees a real number
    o, h, lo, c, v = vals["open"], vals["high"], vals["low"], vals["close"], vals["volume"]
    if min(o, h, lo, c) <= 0:
        raise DataValidationError(f"{symbol}: non-positive price on {ctx['bar_date']}", context=ctx)
    if v < 0:
        raise DataValidationError(f"{symbol}: negative volume on {ctx['bar_date']}", context=ctx)
    if h < lo or h < o or h < c or lo > o or lo > c:
        raise DataValidationError(f"{symbol}: inconsistent OHLC on {ctx['bar_date']} (o={o}, h={h}, l={lo}, c={c})",
                                  context=ctx)
    if prev_close is not None and _finite(prev_close) and prev_close > 0 and abs(c / prev_close - 1) > max_abs_return:
        raise DataValidationError(f"{symbol}: extreme discontinuity {c / prev_close - 1:+.1%} on {ctx['bar_date']} "
                                  "(unadjusted split? bad tick?)", context=ctx)


def validate_feature_row(symbol: str, row: pd.Series, required: Iterable[str], as_of: pd.Timestamp | None = None,
                         row_date: pd.Timestamp | None = None) -> None:
    """All required features present and finite for the current row."""
    required = list(required)
    missing = [c for c in required if c not in row.index]
    if missing:
        raise FeatureSchemaError(f"{symbol}: missing features {missing}", context={"symbol": symbol})
    vals = pd.to_numeric(row[required], errors="coerce").to_numpy(dtype=float)
    bad = [c for c, v in zip(required, vals, strict=True) if not np.isfinite(v)]
    if bad:
        raise FeatureSchemaError(f"{symbol}: NaN/inf in features {bad}", context={"symbol": symbol, "features": bad})
    if as_of is not None and row_date is not None and pd.Timestamp(row_date) > pd.Timestamp(as_of):
        raise FeatureSchemaError(f"{symbol}: feature row dated after decision time (look-ahead)")


def assert_feature_schema(frame: pd.DataFrame, expected_columns: Sequence[str], expected_version: str | None = None,
                          actual_version: str | None = None) -> None:
    """Exact column list AND order, numeric dtypes, matching feature-set version."""
    if expected_version is not None and actual_version != expected_version:
        raise FeatureSchemaError(f"feature set version {actual_version!r} != model's {expected_version!r}")
    cols = list(frame.columns)
    if cols != list(expected_columns):
        missing = [c for c in expected_columns if c not in cols]
        extra = [c for c in cols if c not in expected_columns]
        raise FeatureSchemaError(f"feature columns differ from model schema (missing={missing}, extra={extra}, "
                                 f"order_ok={not missing and not extra})")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise FeatureSchemaError(f"non-numeric feature columns: {non_numeric}")
    arr = frame.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise FeatureSchemaError("infinite values in model input")


def validate_probabilities(p, n_expected: int | None = None) -> np.ndarray:
    """Model output must be finite and within [0, 1]. Never default to bullish."""
    arr = np.asarray(p, dtype=float).ravel()
    if n_expected is not None and arr.size != n_expected:
        raise ModelInferenceError(f"model returned {arr.size} predictions, expected {n_expected}")
    if not np.isfinite(arr).all():
        raise ModelInferenceError("model returned NaN/inf probabilities")
    if (arr < 0).any() or (arr > 1).any():
        raise ModelInferenceError(f"model returned probabilities outside [0, 1] (min={arr.min()}, max={arr.max()})")
    return arr
