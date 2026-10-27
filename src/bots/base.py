"""BaseTradingBot: the contract every strategy bot implements.

A bot = features + strategy components + ML model + thresholds + risk
parameters + capital sleeve. Concrete bots live in src/bots/<type>/ and
declare their CONTRACT (timeframe, allowed strategy components, allowed
prediction horizon, bot-specific features). The YAML in config/bots/
supplies parameters; the contract is checked when the bot is built, so a
config edit cannot silently turn the swing bot into a copy of the investor.

A bot only produces scores (SignalFrame) and TradingSignal objects. It
cannot size, approve or send orders: its sleeve RiskEngine (level 1), the
account-wide portfolio risk manager (level 2), the OrderValidator and the
OrderManager do that, in that order. Bots never hold a broker reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.bots.signal import SignalType, TradingSignal
from src.config import BotConfig, EnsembleConfig, RiskConfig
from src.data.resampling import assert_daily
from src.errors import ConfigurationError, FeatureCalculationError
from src.features.pipeline import FeatureConfig, FeaturePipeline
from src.features.weekly import WeeklyFeaturePipeline
from src.strategies.base import SignalFrame, Strategy
from src.strategies.ensemble import EnsembleStrategy, build_strategy
from src.strategies.ml_strategy import MLStrategy


@dataclass
class BotVersions:
    bot_id: str
    strategy_version: str
    feature_version: str
    model_version: str | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class BaseTradingBot:
    # ---- contract, overridden by each concrete bot -----------------------
    BOT_TYPE: ClassVar[str] = "generic"
    TIMEFRAME: ClassVar[str | None] = None                 # "daily" | "weekly" | None = any
    ALLOWED_COMPONENTS: ClassVar[frozenset[str] | None] = None
    TARGET_HORIZON_RANGE: ClassVar[tuple[int, int] | None] = None   # trading days, inclusive
    ML_FEATURE_PREFIX: ClassVar[str | None] = None          # e.g. "w_": model sees weekly data only
    BOT_FEATURES_VERSION: ClassVar[str] = ""                 # version of add_bot_features()

    def __init__(self, bot_id: str, bcfg: BotConfig, risk_cfg: RiskConfig, universe_symbols: list[str],
                 benchmark: str = "SPY"):
        self.bot_id = bot_id
        self.cfg = bcfg
        self.risk_cfg = risk_cfg
        self.benchmark = benchmark
        self.symbols = [s.upper() for s in (bcfg.symbols or universe_symbols)]
        self.daily_pipeline = FeaturePipeline(FeatureConfig.from_dict(bcfg.features), benchmark)
        self.weekly_pipeline = (WeeklyFeaturePipeline(FeatureConfig.from_dict(bcfg.weekly_features), benchmark,
                                                      post=self.add_weekly_features)
                                if bcfg.weekly_features else None)
        self.components: dict[str, Strategy] = {
            name: build_strategy(name, c, c.type) for name, c in bcfg.components.items() if c.enabled}
        self.ml_cfg = bcfg.ml if (bcfg.ml and bcfg.ml.enabled) else None
        self.validate_contract()

    # ------------------------------------------------------------- contract
    def validate_contract(self) -> None:
        c, name = self.cfg, f"{self.bot_id} ({self.BOT_TYPE})"
        if c.bot_type != self.BOT_TYPE and self.BOT_TYPE != "generic":
            raise ConfigurationError(f"{name}: config bot_type {c.bot_type!r} does not match {self.BOT_TYPE!r}")
        if self.TIMEFRAME and c.timeframe != self.TIMEFRAME:
            raise ConfigurationError(f"{name}: must run on {self.TIMEFRAME} data, config says {c.timeframe}")
        if self.ALLOWED_COMPONENTS is not None:
            bad = sorted({x.type for x in c.components.values() if x.enabled} - self.ALLOWED_COMPONENTS)
            if bad:
                raise ConfigurationError(f"{name}: strategy components {bad} are not part of this bot's logic "
                                         f"(allowed: {sorted(self.ALLOWED_COMPONENTS)})")
        if self.ml_cfg is not None:
            if self.TARGET_HORIZON_RANGE:
                lo, hi = self.TARGET_HORIZON_RANGE
                if not lo <= self.ml_cfg.target_horizon_days <= hi:
                    raise ConfigurationError(f"{name}: ML target horizon {self.ml_cfg.target_horizon_days}d "
                                             f"outside this bot's range [{lo}, {hi}]")
            if self.ML_FEATURE_PREFIX:
                bad = [f for f in self.ml_cfg.features if not f.startswith(self.ML_FEATURE_PREFIX)]
                if bad:
                    raise ConfigurationError(f"{name}: model may only use {self.ML_FEATURE_PREFIX}* features, "
                                             f"got {bad}")
        if c.fundamentals.enabled:
            raise ConfigurationError(f"{name}: fundamentals enabled but no reliable fundamentals provider is "
                                     "configured; refusing to run rather than fabricate data")

    # ------------------------------------------------------------- metadata
    @property
    def timeframe(self) -> str:
        return self.cfg.timeframe

    @property
    def feature_version(self) -> str:
        v = self.daily_pipeline.version
        if self.weekly_pipeline:
            v += "|" + self.weekly_pipeline.version
        if self.BOT_FEATURES_VERSION:
            v += "|" + self.BOT_FEATURES_VERSION
        return v

    @property
    def versions(self) -> BotVersions:
        return BotVersions(self.bot_id, self.cfg.strategy_version, self.feature_version,
                           self.ml_cfg.model_version if self.ml_cfg else None)

    @property
    def capital_allocation(self) -> float:
        return self.cfg.capital_allocation

    @property
    def rebalance_frequency(self) -> str:
        return self.cfg.rebalance_frequency

    @property
    def expected_horizon_days(self) -> tuple[int, int]:
        return self.cfg.holding_period.in_trading_days()

    @property
    def ml_features(self) -> list[str]:
        return list(self.ml_cfg.features) if self.ml_cfg else []

    @property
    def signal_features(self) -> list[str]:
        """Features that must be finite before this bot may open a position."""
        cols = {c for s in self.components.values() for c in s.required_features}
        return sorted(cols)

    # ------------------------------------------------------------- features
    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = ()       # columns the bot's design depends on

    def add_weekly_features(self, weekly_panel: pd.DataFrame) -> pd.DataFrame:
        """Bot-specific features in WEEKLY space (rows = completed weeks,
        unprefixed names; ``w_`` is added afterwards). Override."""
        return weekly_panel

    def add_bot_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Bot-specific derived features (override). Must only use columns of
        the same row, i.e. information available at that row's timestamp."""
        return panel

    def compute_features(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Shared framework, bot-specific configuration. Input is always
        validated DAILY bars; weekly bots resample internally. Benchmark data
        is used for regime features even if the bot does not trade it."""
        for s, df in prices.items():
            assert_daily(s, df)
        panel = self.daily_pipeline.compute(prices)
        if self.weekly_pipeline is not None:
            weekly = self.weekly_pipeline.compute(prices)
            panel = panel.join(weekly.drop(columns=[c for c in ("week_end",) if c in weekly]), how="left")
        panel = self.add_bot_features(panel)
        missing = [c for c in self.REQUIRED_FEATURES if c not in panel.columns]
        if missing:
            raise FeatureCalculationError(f"{self.bot_id}: feature configuration no longer produces {missing}")
        keep = panel.index.get_level_values("symbol").isin(self.symbols)
        return panel[keep]

    # --------------------------------------------------------------- signal
    def ensemble_config(self) -> EnsembleConfig:
        weights = {name: c.weight for name, c in self.cfg.components.items() if c.enabled}
        if self.ml_cfg:
            weights["ml"] = self.ml_cfg.weight
        t = self.cfg.thresholds
        return EnsembleConfig(weights=weights, buy_threshold=t.buy, hold_threshold=t.hold, exit_threshold=t.exit,
                              reduce_fraction=t.reduce_fraction,
                              ml_probability_floor=self.ml_cfg.probability_floor if self.ml_cfg else 0.5,
                              ml_probability_cap=self.ml_cfg.probability_cap if self.ml_cfg else 0.7)

    def signal_frame(self, panel: pd.DataFrame, ml_predictions: pd.Series | None = None,
                     trained_until: pd.Timestamp | None = None) -> SignalFrame:
        """If the bot has an ML component, predictions are REQUIRED: rows
        without a valid prediction get a NaN score (-> HOLD, no new trade)."""
        ml = None
        if self.ml_cfg is not None:
            preds = ml_predictions if ml_predictions is not None else pd.Series(
                float("nan"), index=panel.index, dtype=float)
            ml = MLStrategy(preds, entry_threshold=self.ml_cfg.probability_floor, trained_until=trained_until)
        ens = EnsembleStrategy(self.components, self.ensemble_config(), ml=ml,
                               required_components=("ml",) if self.ml_cfg else ())
        sf = ens.signal_frame(panel)
        sf.name = self.bot_id
        sf.meta.update(self.versions.as_dict())
        return sf

    def make_signal(self, symbol: str, data_timestamp: pd.Timestamp, action: str, score: float | None,
                    model_probability: float | None, reason: str = "") -> TradingSignal:
        """Standard signal object handed to the portfolio layer."""
        v = self.versions
        return TradingSignal.create(
            bot_id=self.bot_id, strategy_version=v.strategy_version, symbol=symbol,
            data_timestamp=pd.Timestamp(data_timestamp), timeframe=self.timeframe,
            signal_type=SignalType.from_action(action), signal_score=score, model_probability=model_probability,
            expected_horizon=self.expected_horizon_days, feature_set_version=v.feature_version,
            model_version=v.model_version, reason=reason)
