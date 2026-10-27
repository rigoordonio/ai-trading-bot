"""Market-data provider abstraction.

Every provider returns the SAME schema so the rest of the platform never
knows (or cares) where prices came from:

    index : DatetimeIndex named "date" (tz-naive, normalised to midnight)
    columns: open, high, low, close, adj_close, volume, dividends, splits

``close`` is the price as quoted by the vendor; ``adj_close`` additionally
reflects dividends/splits. Corporate-action adjustment is applied later and
explicitly (see ``corporate_actions.py``), never silently inside a provider.

Swapping vendors = writing one new subclass of ``MarketDataProvider``.
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_setup import get_logger

log = get_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
STANDARD_COLUMNS = PRICE_COLUMNS + ["dividends", "splits"]


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a vendor frame to the standard schema."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    out = out.rename(columns={"adj_close": "adj_close", "adjclose": "adj_close", "stock_splits": "splits"})
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    for col in ("dividends", "splits"):
        if col not in out.columns:
            out[col] = 0.0
    missing = [c for c in PRICE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"provider data missing columns: {missing}")
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[STANDARD_COLUMNS].astype(float)
    return out.sort_index()


class MarketDataProvider(ABC):
    """Interface every data vendor implements."""

    name: str = "base"
    #: True if the vendor's ``volume`` is already split-adjusted (Yahoo is).
    volume_split_adjusted: bool = True

    @abstractmethod
    def get_history(self, symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None,
                    interval: str = "1d") -> pd.DataFrame:
        """Return daily bars in the standard schema (may be empty)."""

    def get_latest(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        end = pd.Timestamp.today().normalize()
        return self.get_history(symbol, end - pd.Timedelta(days=lookback_days), end)


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance via yfinance. Research/prototyping only: no SLA, data can
    be revised, and the free feed has no delisted symbols (survivorship bias)."""

    name = "yfinance"
    volume_split_adjusted = True

    def get_history(self, symbol, start, end=None, interval="1d"):
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError("yfinance is not installed: pip install yfinance") from exc
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        # yfinance's `end` is exclusive -> add one day so `end` is inclusive.
        raw = yf.Ticker(symbol).history(
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            actions=True,
            timeout=20,               # never hang the pipeline on a slow vendor
            raise_errors=True,        # surface failures to the retry/circuit-breaker layer
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        return standardise(raw)


class CSVProvider(MarketDataProvider):
    """Reads ``{directory}/{SYMBOL}.csv`` with a ``date`` column + standard columns.
    Use it to plug in any vendor export (Polygon, Tiingo, Norgate, ...)."""

    name = "csv"

    def __init__(self, directory: str | Path, volume_split_adjusted: bool = True):
        self.directory = Path(directory)
        self.volume_split_adjusted = volume_split_adjusted

    def get_history(self, symbol, start, end=None, interval="1d"):
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = standardise(df)
        df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df


class SyntheticProvider(MarketDataProvider):
    """Deterministic, offline, *fake* prices for tests and demos.

    A shared market factor with bull/bear regimes drives every symbol (so the
    universe is correlated like real equities), plus idiosyncratic noise.
    Series are generated on a fixed calendar, so asking for different date
    ranges always returns identical values for overlapping dates.

    Never draw investment conclusions from synthetic data.
    """

    name = "synthetic"
    volume_split_adjusted = False
    CAL_START, CAL_END = "2000-01-03", "2030-12-31"

    def __init__(self, seed: int = 7, splits: dict[str, tuple[str, float]] | None = None,
                 dividend_yield: float = 0.015):
        self.seed = seed
        self.splits = splits or {}
        self.dividend_yield = dividend_yield
        self._cache: dict[str, pd.DataFrame] = {}
        self._calendar = pd.bdate_range(self.CAL_START, self.CAL_END, name="date")
        self._market = self._market_returns()

    def _market_returns(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = len(self._calendar)
        # Two-state Markov regime: bull (positive drift, low vol) / bear.
        regime = np.zeros(n, dtype=int)
        for t in range(1, n):
            stay = 0.995 if regime[t - 1] == 0 else 0.98
            regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
        mu = np.where(regime == 0, 0.0006, -0.0010)
        sigma = np.where(regime == 0, 0.009, 0.022)
        return mu + sigma * rng.standard_normal(n)

    def _generate(self, symbol: str) -> pd.DataFrame:
        sym_seed = zlib.crc32(symbol.encode()) ^ self.seed
        rng = np.random.default_rng(sym_seed)
        n = len(self._calendar)
        beta = 0.7 + 0.8 * rng.random()
        idio_vol = 0.004 + 0.012 * rng.random()
        alpha = rng.normal(0.0001, 0.0002)
        ret = alpha + beta * self._market + idio_vol * rng.standard_normal(n)
        start_price = float(rng.uniform(20, 300))
        close = start_price * np.exp(np.cumsum(ret))
        gap = 0.003 * rng.standard_normal(n)
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(gap)
        hi = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(n)))
        lo = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(n)))
        base_vol = rng.uniform(5e6, 6e7)
        volume = base_vol * np.exp(0.35 * rng.standard_normal(n) + 8 * np.abs(ret))

        df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
                          index=self._calendar)
        # Quarterly dividends -> adj_close < close historically (total-return factor).
        df["dividends"] = 0.0
        q_idx = np.arange(40, n, 63)
        df.iloc[q_idx, df.columns.get_loc("dividends")] = close[q_idx] * self.dividend_yield / 4
        df["splits"] = 0.0
        # Total-return adjustment factor (Yahoo convention: latest factor = 1).
        div_frac = (df["dividends"] / df["close"].shift(1)).fillna(0.0).to_numpy()
        factor = np.cumprod((1 - div_frac)[::-1])[::-1]
        factor = np.concatenate([factor[1:], [1.0]])
        df["adj_close"] = df["close"] * factor

        # Optional stock split: raw prices BEFORE the split date are scaled up
        # (what an unadjusted feed shows); adj_close stays continuous.
        if symbol in self.splits:
            date, ratio = self.splits[symbol]
            pre = df.index < pd.Timestamp(date)
            for col in ("open", "high", "low", "close"):
                df.loc[pre, col] = df.loc[pre, col] * ratio
            df.loc[pre, "dividends"] = df.loc[pre, "dividends"] * ratio
            df.loc[pre, "volume"] = df.loc[pre, "volume"] / ratio
            df.loc[df.index[~pre][0], "splits"] = ratio  # first session on/after the split date
        df.index.name = "date"
        return df[STANDARD_COLUMNS]

    def get_history(self, symbol, start, end=None, interval="1d"):
        if symbol not in self._cache:
            self._cache[symbol] = self._generate(symbol)
        df = self._cache[symbol]
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        out = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= end_ts)].copy()
        if not out.empty:
            # Like a real vendor, adjust relative to the LAST bar delivered
            # (latest factor = 1). Adjusted returns are unaffected.
            out["adj_close"] *= out["close"].iloc[-1] / out["adj_close"].iloc[-1]
        return out


def get_provider(name: str, **kwargs) -> MarketDataProvider:
    name = name.lower()
    if name == "yfinance":
        return YFinanceProvider()
    if name == "csv":
        return CSVProvider(kwargs.get("directory", "data/raw/csv"))
    if name == "synthetic":
        log.warning("Using SYNTHETIC market data: results are not evidence of real-world performance.")
        return SyntheticProvider(**{k: v for k, v in kwargs.items() if k in {"seed", "splits", "dividend_yield"}})
    raise ValueError(f"unknown data provider '{name}'")
