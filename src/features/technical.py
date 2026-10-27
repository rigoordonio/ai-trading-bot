"""Core technical indicators implemented directly in pandas.

pandas-ta is not used: it is unmaintained for recent pandas/numpy releases,
and writing the formulas out keeps every calculation inspectable and tested.

LOOK-AHEAD RULE: every function only uses values at or before row t to
produce row t (trailing rolling windows, recursive EWMs, ``shift(+n)``).
A ``shift(-n)`` (future data) must never appear in this package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used by RSI and ATR): EMA with alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), n)
    loss = wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss != 0, 100.0)          # no losses at all -> RSI 100
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def roc(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def realized_vol(close: pd.Series, n: int, annualise: bool = True, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    lr = np.log(close / close.shift(1))
    vol = lr.rolling(n, min_periods=n).std()
    return vol * np.sqrt(periods_per_year) if annualise else vol


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_pct": (close - lower) / (upper - lower),
        "zscore_20": (close - mid) / sd,
    })


def rolling_high(s: pd.Series, n: int, include_today: bool = False) -> pd.Series:
    """Highest value over the PRIOR n bars (today excluded by default, so a
    breakout 'close > prior high' can actually be observed at today's close)."""
    out = s.rolling(n, min_periods=n).max()
    return out if include_today else out.shift(1)


def rolling_low(s: pd.Series, n: int, include_today: bool = False) -> pd.Series:
    out = s.rolling(n, min_periods=n).min()
    return out if include_today else out.shift(1)


def price_structure(close: pd.Series, high: pd.Series, sma50: pd.Series, sma200: pd.Series,
                    window_52w: int = 252) -> pd.DataFrame:
    hi52 = high.rolling(window_52w, min_periods=window_52w).max()
    return pd.DataFrame({
        "dist_52w_high": close / hi52 - 1.0,
        "dist_sma_50": close / sma50 - 1.0,
        "dist_sma_200": close / sma200 - 1.0,
    })
