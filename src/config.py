"""Configuration loading and schema validation.

YAML files hold behaviour (universe, parameters, limits); .env holds secrets.
Every section is validated by a pydantic model with ``extra="forbid"`` so a
typo in a YAML key fails loudly instead of being silently ignored.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------- settings
class UniverseConfig(_Strict):
    symbols: list[str]
    benchmark: str = "SPY"
    membership_file: str | None = None
    sectors: dict[str, str] = Field(default_factory=dict)

    @field_validator("symbols")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("universe.symbols must not be empty")
        return [s.upper() for s in v]


class ValidationConfig(_Strict):
    max_daily_abs_return: float = 0.5
    volume_zscore_threshold: float = 8.0
    stale_run_length: int = 5
    max_missing_business_days: int = 3
    fail_on_error: bool = False


class DataConfig(_Strict):
    provider: Literal["yfinance", "csv", "synthetic"] = "yfinance"
    start: str = "2012-01-01"
    end: str | None = None
    interval: str = "1d"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    retries: int = 3
    retry_backoff_seconds: float = 2.0
    price_adjustment: Literal["adjusted", "raw"] = "adjusted"
    paper_history_days: int = 2200        # paper runs: calendar days of history (weekly SMA200 needs ~4 years)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class DatabaseConfig(_Strict):
    url_env: str = "DATABASE_URL"
    default_url: str = "sqlite:///data/trading.db"


class CostConfig(_Strict):
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_pct: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0
    impact_coefficient: float = 10.0
    max_participation: float = 0.05


class BacktestConfig(_Strict):
    initial_capital: float = 100_000
    start: str = "2015-01-01"
    end: str | None = None
    execution_delay_bars: int = 1
    costs: CostConfig = Field(default_factory=CostConfig)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"

    @field_validator("execution_delay_bars")
    @classmethod
    def _delay(cls, v: int) -> int:
        if v < 1:
            raise ValueError("execution_delay_bars must be >= 1 (no same-bar execution: look-ahead)")
        return v


class SplitConfig(_Strict):
    train: tuple[str, str]
    validation: tuple[str, str]
    test: tuple[str, str | None]

    @model_validator(mode="after")
    def _chronological(self) -> "SplitConfig":
        if not (self.train[1] < self.validation[0] and self.validation[1] < self.test[0]):
            raise ValueError("ML splits must be chronological and non-overlapping: train < validation < test")
        return self


class MLConfig(_Strict):
    target_horizon_days: int = 5
    target_type: Literal["classification", "regression"] = "classification"
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting"])
    selection_metric: str = "roc_auc"
    split: SplitConfig
    purge_days: int | None = None
    random_state: int = 42
    artifacts_dir: str = "models/artifacts"


class WalkForwardConfig(_Strict):
    train_years: int = 4
    validation_years: int = 1
    test_years: int = 1
    step_years: int = 1


class LoggingConfig(_Strict):
    level_env: str = "LOG_LEVEL"
    default_level: str = "INFO"
    json_logs: bool = Field(False, alias="json")


class Settings(_Strict):
    mode: Literal["backtest", "paper"] = "backtest"
    universe: UniverseConfig
    data: DataConfig = Field(default_factory=DataConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    ml: MLConfig
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("mode", mode="before")
    @classmethod
    def _no_live(cls, v: Any) -> Any:
        if str(v).lower() == "live":
            raise ValueError("Live trading is not supported by this platform. Use 'backtest' or 'paper'.")
        return v

    @property
    def database_url(self) -> str:
        return os.getenv(self.database.url_env, self.database.default_url)


# ------------------------------------------------------------------- strategies
class StrategyConfig(_Strict):
    enabled: bool = True
    entry_threshold: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)


class EnsembleConfig(_Strict):
    weights: dict[str, float]
    buy_threshold: float = 0.6
    hold_threshold: float = 0.4
    exit_threshold: float = 0.25
    reduce_fraction: float = 0.5
    ml_probability_floor: float = 0.5
    ml_probability_cap: float = 0.7

    @model_validator(mode="after")
    def _ordered(self) -> "EnsembleConfig":
        if not (self.exit_threshold <= self.hold_threshold <= self.buy_threshold):
            raise ValueError("ensemble thresholds must satisfy exit <= hold <= buy")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("ensemble weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("ensemble weights must not all be zero")
        return self


class StrategiesFile(_Strict):
    features: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, StrategyConfig]
    ensemble: EnsembleConfig


# ------------------------------------------------------------------------- risk
class PositionSizingConfig(_Strict):
    method: Literal["fixed_fractional", "volatility", "atr"] = "volatility"
    fixed_fraction: float = 0.10
    target_position_vol: float = 0.15
    vol_lookback_feature: str = "realized_vol_20"
    atr_risk_per_trade: float = 0.01
    atr_stop_multiple: float = 2.0


class LimitsConfig(_Strict):
    max_position_weight: float = 0.15
    max_gross_exposure: float = 0.95
    min_cash_reserve: float = 0.05
    max_sector_weight: float = 0.40
    max_concurrent_positions: int = 6
    max_portfolio_volatility: float = 0.25
    min_trade_value: float = 500
    min_price: float = 5.0
    min_avg_dollar_volume: float = 0.0

    @model_validator(mode="after")
    def _consistent(self) -> "LimitsConfig":
        if self.max_gross_exposure + self.min_cash_reserve > 1.0 + 1e-9:
            raise ValueError("max_gross_exposure + min_cash_reserve must be <= 1 (long-only, no leverage)")
        for name in ("max_position_weight", "max_gross_exposure", "min_cash_reserve", "max_sector_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


class StopsConfig(_Strict):
    enabled: bool = True
    fixed_stop_pct: float | None = 0.10
    trailing_atr_multiple: float | None = 3.0
    max_holding_days: int | None = None


class DrawdownConfig(_Strict):
    max_portfolio_drawdown: float = 0.20
    liquidate_on_kill_switch: bool = False
    max_daily_loss: float | None = 0.04
    kill_switch_reset: str | dict[str, int] = "manual"


class CircuitBreakerConfig(_Strict):
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 300
    half_open_max_calls: int = 1


class SafetyConfig(_Strict):
    max_order_notional: float = 25_000
    max_signal_age_days: int = 4
    max_data_staleness_days: int = 4
    max_bar_abs_return: float = 0.40
    allow_risk_reducing_orders_when_halted: bool = False
    reconciliation_cash_tolerance: float = 1.0
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    @field_validator("max_order_notional")
    @classmethod
    def _positive(cls, v: float) -> float:
        if not v > 0:
            raise ValueError("max_order_notional must be > 0")
        return v


class PortfolioLimitsConfig(_Strict):
    max_gross_exposure: float = 0.95
    max_symbol_weight: float = 0.25
    max_bot_weight: float = 1.0           # one bot's positions / account equity (strategy concentration)
    max_total_positions: int = 15
    max_portfolio_drawdown: float = 0.20
    max_daily_loss: float | None = 0.05


class RiskConfig(_Strict):
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    portfolio: PortfolioLimitsConfig = Field(default_factory=PortfolioLimitsConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    stops: StopsConfig = Field(default_factory=StopsConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)


# ------------------------------------------------------------------------- bots
class BotComponentConfig(_Strict):
    type: str
    weight: float
    entry_threshold: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class BotMLConfig(_Strict):
    enabled: bool = True
    weight: float = 0.25
    model_version: str
    target_horizon_days: int
    target_threshold: float = 0.0
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting"])
    probability_floor: float = 0.5
    probability_cap: float = 0.7
    features: list[str]

    @field_validator("target_horizon_days")
    @classmethod
    def _h(cls, v: int) -> int:
        if v < 1:
            raise ValueError("target_horizon_days must be >= 1")
        return v


class BotThresholds(_Strict):
    buy: float
    hold: float
    exit: float
    reduce_fraction: float = 0.5


class HoldingPeriod(_Strict):
    min: int
    max: int
    unit: Literal["trading_days", "weeks", "months"] = "trading_days"

    @model_validator(mode="after")
    def _check(self) -> "HoldingPeriod":
        if not 0 < self.min <= self.max:
            raise ValueError("holding_period needs 0 < min <= max")
        return self

    def in_trading_days(self) -> tuple[int, int]:
        f = {"trading_days": 1, "weeks": 5, "months": 21}[self.unit]
        return self.min * f, self.max * f


class FundamentalsConfig(_Strict):
    """Optional fundamental features. Disabled unless a RELIABLE source is
    configured; no provider in this project supplies fundamentals yet, so
    enabling it fails configuration instead of inventing numbers."""
    enabled: bool = False
    source: str | None = None
    features: list[str] = Field(default_factory=list)


_TIMEFRAME_ALIASES = {"1d": "daily", "d": "daily", "daily": "daily", "1w": "weekly", "w": "weekly",
                      "weekly": "weekly"}


class BotConfig(_Strict):
    enabled: bool = True
    bot_type: Literal["swing", "trend", "investor"]
    description: str = ""
    strategy_version: str
    timeframe: Literal["daily", "weekly"] = "daily"
    holding_period: HoldingPeriod
    capital_allocation: float
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"
    symbols: list[str] | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    weekly_features: dict[str, Any] | None = None
    fundamentals: FundamentalsConfig = Field(default_factory=FundamentalsConfig)
    components: dict[str, BotComponentConfig]
    ml: BotMLConfig | None = None
    thresholds: BotThresholds
    risk: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timeframe", mode="before")
    @classmethod
    def _tf(cls, v: Any) -> Any:
        return _TIMEFRAME_ALIASES.get(str(v).strip().lower(), v)

    @model_validator(mode="after")
    def _check(self) -> "BotConfig":
        if not 0 < self.capital_allocation <= 1:
            raise ValueError("capital_allocation must be in (0, 1]")
        if self.timeframe == "weekly" and not self.weekly_features:
            raise ValueError("weekly bots need a weekly_features block")
        t = self.thresholds
        if not t.exit <= t.hold <= t.buy:
            raise ValueError("thresholds must satisfy exit <= hold <= buy")
        return self


class BotsFile(_Strict):
    bots: dict[str, BotConfig]

    @model_validator(mode="after")
    def _alloc(self) -> "BotsFile":
        total = sum(b.capital_allocation for b in self.bots.values() if b.enabled)
        if total > 1.0 + 1e-9:
            raise ValueError(f"enabled bots allocate {total:.0%} of capital (> 100%)")
        return self


# ---------------------------------------------------------------------- loading
class AppConfig(BaseModel):
    settings: Settings
    strategies: StrategiesFile
    risk: RiskConfig
    bots: BotsFile | None = None

    def bot_risk_config(self, bot_id: str) -> RiskConfig:
        """Bot risk = risk.yaml with the bot's overrides merged on top."""
        from src.errors import ConfigurationError

        if self.bots is None or bot_id not in self.bots.bots:
            raise ConfigurationError(f"unknown bot {bot_id!r}")
        base = self.risk.model_dump(by_alias=True)
        return RiskConfig.model_validate(deep_merge(base, self.bots.bots[bot_id].risk))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_bots(cdir: Path) -> dict[str, Any] | None:
    """One file per bot in config/bots/*.yaml (each with a ``bot_id``).
    A legacy single config/bots.yaml is still read if the folder is absent."""
    from src.errors import ConfigurationError

    bdir = cdir / "bots"
    if bdir.is_dir():
        bots: dict[str, Any] = {}
        for f in sorted(bdir.glob("*.yaml")):
            raw = _read_yaml(f)
            bot_id = str(raw.pop("bot_id", "") or "").strip()
            if not bot_id:
                raise ConfigurationError(f"{f.name}: missing bot_id")
            if bot_id in bots:
                raise ConfigurationError(f"{f.name}: duplicate bot_id {bot_id}")
            bots[bot_id] = raw
        return {"bots": bots} if bots else None
    legacy = cdir / "bots.yaml"
    return _read_yaml(legacy) if legacy.exists() else None


def load_config(config_dir: str | Path | None = None, overrides: dict[str, dict] | None = None) -> AppConfig:
    """Load and validate settings.yaml, strategies.yaml, risk.yaml and config/bots/*.yaml.

    ``overrides`` lets tests/scripts patch values, e.g.
    ``{"settings": {"data": {"provider": "synthetic"}}}``.
    """
    if os.getenv("TRADING_BOT_IGNORE_DOTENV") != "1":     # set by the test suite: tests never read your .env
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    raw = {
        "settings": _read_yaml(cdir / "settings.yaml"),
        "strategies": _read_yaml(cdir / "strategies.yaml"),
        "risk": _read_yaml(cdir / "risk.yaml"),
        "bots": _read_bots(cdir),
    }
    for key, patch in (overrides or {}).items():
        raw[key] = deep_merge(raw.get(key) or {}, patch)
    try:
        return AppConfig(
            settings=Settings.model_validate(raw["settings"]),
            strategies=StrategiesFile.model_validate(raw["strategies"]),
            risk=RiskConfig.model_validate(raw["risk"]),
            bots=BotsFile.model_validate(raw["bots"]) if raw.get("bots") else None,
        )
    except Exception as exc:  # pydantic.ValidationError -> our taxonomy, original chained
        from src.errors import ConfigurationError

        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(p: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
