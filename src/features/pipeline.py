"""Feature pipeline: ONE implementation used by backtesting, ML training and
paper trading, so research and "production" can never drift apart.

Input : {symbol: adjusted OHLCV DataFrame}
Output: long panel, MultiIndex (date, symbol), one row per symbol per day.

Timing convention: a row dated t uses information up to and including the
CLOSE of day t. Anything derived from it may only be traded at t+1.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.errors import ConfigurationError
from src.features import technical as ta
from src.features.market_regime import regime_features
from src.features.momentum import momentum_features, risk_adjusted_momentum
from src.features.volatility import volatility_features
from src.features.volume import volume_features

#: Bump when ANY feature formula changes. Stored with every trained model; a
#: model refuses features produced under a different version.
FEATURE_SET_VERSION = "1.0"

# Price-LEVEL columns: useful for rules, but non-stationary -> excluded from ML.
LEVEL_COLUMNS = {"open", "high", "low", "close", "volume", "bb_mid", "bb_upper", "bb_lower", "macd",
                 "macd_signal", "macd_hist"}
LEVEL_PREFIXES = ("sma_", "ema_", "avg_volume_", "dollar_volume_", "prior_high_",
                  "prior_low_", "raw_", "adj_")


@dataclass
class FeatureConfig:
    return_windows: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    sma_windows: list[int] = field(default_factory=lambda: [20, 50, 200])
    ema_windows: list[int] = field(default_factory=lambda: [12, 26])
    rsi_window: int = 14
    macd: dict[str, int] = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    roc_window: int = 10
    atr_window: int = 14
    realized_vol_windows: list[int] = field(default_factory=lambda: [20, 60])
    volume_window: int = 20
    bollinger: dict[str, float] = field(default_factory=lambda: {"window": 20, "num_std": 2.0})
    high_52w_window: int = 252
    breakout_window: int = 55
    extra_breakout_windows: list[int] = field(default_factory=lambda: [20])
    dist_high_windows: list[int] = field(default_factory=list)   # close vs N-day high (incl. today)
    periods_per_year: int = 252                                  # 52 for weekly bars

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FeatureConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigurationError(f"unknown feature settings: {sorted(unknown)}")
        return cls(**d)

    def version(self) -> str:
        """Formula version + hash of the parameters -> identifies the schema."""
        digest = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]
        return f"{FEATURE_SET_VERSION}+{digest}"


class FeaturePipeline:
    def __init__(self, config: FeatureConfig | None = None, benchmark: str = "SPY"):
        self.cfg = config or FeatureConfig()
        self.benchmark = benchmark

    @property
    def version(self) -> str:
        return self.cfg.version()

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every feature is defined."""
        return max(max(self.cfg.sma_windows), 200, self.cfg.high_52w_window, max(self.cfg.return_windows)) + 1

    def symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        close = df["close"]
        out = pd.DataFrame(index=df.index)
        out["open"], out["high"], out["low"], out["close"], out["volume"] = (
            df["open"], df["high"], df["low"], close, df["volume"])
        for n in c.sma_windows:
            out[f"sma_{n}"] = ta.sma(close, n)
        for n in c.ema_windows:
            out[f"ema_{n}"] = ta.ema(close, n)
        out = out.join(momentum_features(df, c.return_windows, c.rsi_window, c.macd, c.roc_window))
        out = out.join(volatility_features(df, c.atr_window, c.realized_vol_windows, c.bollinger,
                                           c.periods_per_year))
        out = out.join(volume_features(df, c.volume_window))
        s50 = out["sma_50"] if "sma_50" in out else ta.sma(close, 50)
        s200 = out["sma_200"] if "sma_200" in out else ta.sma(close, 200)
        if "sma_50" not in out:
            out["sma_50"] = s50       # always available: used by regime / breadth / risk rules
        if "sma_200" not in out:
            out["sma_200"] = s200
        out = out.join(ta.price_structure(close, df["high"], s50, s200, c.high_52w_window))
        for n in c.dist_high_windows:
            out[f"dist_high_{n}"] = close / df["high"].rolling(n, min_periods=n).max() - 1.0
        for n in sorted({c.breakout_window, *c.extra_breakout_windows}):
            out[f"prior_high_{n}"] = ta.rolling_high(df["high"], n)
            out[f"prior_low_{n}"] = ta.rolling_low(df["low"], n)
        vols = c.realized_vol_windows
        for n in c.return_windows:
            if n >= 20 or (c.periods_per_year < 252 and n >= 4):
                vol = out[f"realized_vol_{min(vols, key=lambda v: abs(v - n))}"]
                vol_n = vol * (n / c.periods_per_year) ** 0.5     # vol scaled to the return horizon
                out[f"risk_adj_mom_{n}"] = risk_adjusted_momentum(out[f"ret_{n}d"], vol_n)
        # MACD scaled by price so it is comparable across symbols (ML-safe)
        out["macd_pct"] = out["macd_hist"] / close
        return out

    def compute(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not prices:
            raise ValueError("no price data supplied to FeaturePipeline")
        per_symbol = {s: self.symbol_features(df) for s, df in prices.items() if not df.empty}
        closes = pd.DataFrame({s: f["close"] for s, f in per_symbol.items()}).sort_index()
        sma50 = pd.DataFrame({s: f["sma_50"] for s, f in per_symbol.items()}).reindex(closes.index)
        regime = regime_features(prices.get(self.benchmark), closes, sma50)
        frames = []
        for s, f in per_symbol.items():
            g = f.join(regime, how="left")
            g["symbol"] = s
            frames.append(g)
        panel = pd.concat(frames)
        panel.index.name = "date"
        panel = panel.set_index("symbol", append=True).sort_index()
        return panel


def ml_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Scale-free features suitable for cross-sectional ML models."""
    cols = []
    for c in panel.columns:
        if c in LEVEL_COLUMNS or c.startswith(LEVEL_PREFIXES) or re.fullmatch(r"atr_\d+", c) or c == "symbol":
            continue
        if not pd.api.types.is_numeric_dtype(panel[c]):
            continue
        cols.append(c)
    return cols


def pipeline_from_config(cfg) -> FeaturePipeline:
    return FeaturePipeline(FeatureConfig.from_dict(cfg.strategies.features), benchmark=cfg.settings.universe.benchmark)


def wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    """(date, symbol) panel column -> dates x symbols frame."""
    return panel[column].unstack("symbol")
