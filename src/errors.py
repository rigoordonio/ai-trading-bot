"""Error taxonomy and severity model.

Rule: catch an error only where the code knows how to handle it. In the
trading path, "handling" almost always means: record it, emit a risk/system
event, and produce NO TRADE (fail closed). Never ``except: pass``.

Severity -> effect
    INFO      informational
    WARNING   non-trading issue (e.g. dashboard), trading continues
    ERROR     a bad record / a single blocked order; trading continues
    CRITICAL  trading-critical failure -> TRADING_ENABLED = False until a
              validated recovery (see src/safety/health.py)
"""
from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradingSystemError(Exception):
    """Base class. ``severity`` decides whether trading is halted."""

    severity: Severity = Severity.ERROR
    component: str = "system"

    def __init__(self, message: str, *, context: dict | None = None, severity: Severity | None = None):
        super().__init__(message)
        self.context = dict(context or {})
        if severity is not None:
            self.severity = severity


# --- configuration ---------------------------------------------------------
class ConfigurationError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "config"


# --- market data -----------------------------------------------------------
class MarketDataError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "market_data"


class StaleMarketDataError(MarketDataError):
    pass


class DataValidationError(MarketDataError):
    pass


# --- features / model ------------------------------------------------------
class FeatureCalculationError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "features"


class FeatureSchemaError(FeatureCalculationError):
    pass


class ModelInferenceError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "model"


class ModelCompatibilityError(ModelInferenceError):
    pass


# --- risk / portfolio ------------------------------------------------------
class RiskEngineError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "risk_engine"


class RiskLimitViolation(RiskEngineError):
    """A limit was hit. Not a malfunction: the order is blocked, trading continues."""
    severity = Severity.ERROR


class PortfolioStateError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "portfolio"


# --- broker / orders -------------------------------------------------------
class BrokerError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "broker"


class OrderValidationError(BrokerError):
    severity = Severity.ERROR


class OrderRejectedError(BrokerError):
    severity = Severity.ERROR


class DuplicateOrderError(OrderValidationError):
    pass


class TradingDisabledError(BrokerError):
    """Raised by submit_order when TRADING_ENABLED is false."""
    severity = Severity.WARNING


class BrokerTimeoutError(BrokerError):
    """The request may or may not have reached the broker: state is UNKNOWN."""


class ReconciliationError(BrokerError):
    pass


class CircuitOpenError(BrokerError):
    severity = Severity.WARNING
    component = "circuit_breaker"


# --- persistence -----------------------------------------------------------
class DatabaseError(TradingSystemError):
    severity = Severity.CRITICAL
    component = "database"


class LiveTradingNotAllowedError(ConfigurationError):
    pass
