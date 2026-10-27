"""BrokerAdapter interface + the factory that keeps real money OFF.

Strategy, risk and order-management code depend only on this interface, so a
broker's PAPER API can be added later without touching them. There is no live
adapter in this codebase, and ``create_broker`` refuses "live" outright.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.errors import LiveTradingNotAllowedError
from src.execution.orders import Order
from src.logging_setup import get_logger

log = get_logger(__name__)

ALLOWED_MODES = {"BACKTEST", "PAPER"}


@dataclass
class AccountInfo:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"


@dataclass
class BrokerPosition:
    symbol: str
    quantity: int
    avg_price: float


@dataclass
class BrokerOrder:
    broker_order_id: str
    client_order_id: str          # = our idempotency key
    symbol: str
    side: str
    quantity: int
    status: str                   # ACKNOWLEDGED / PARTIALLY_FILLED / FILLED / CANCELLED / REJECTED
    filled_quantity: int = 0
    avg_fill_price: float | None = None
    submitted_at: str = ""
    reject_reason: str | None = None
    internal_order_id: str | None = None
    fills: list[dict] = field(default_factory=list)   # itemised fills (PaperBroker); Alpaca reports totals
    filled_at: str | None = None


class BrokerAdapter(ABC):
    name = "base"
    is_paper = True

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_positions(self) -> dict[str, BrokerPosition]: ...

    @abstractmethod
    def get_market_data(self, symbol: str, start=None, end=None) -> pd.DataFrame: ...

    @abstractmethod
    def submit_order(self, order: Order) -> BrokerOrder: ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> BrokerOrder: ...

    @abstractmethod
    def get_orders(self, status: str | None = None) -> list[BrokerOrder]: ...

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Lookup by idempotency key: THE tool for resolving UNKNOWN orders."""

    def submission_window_open(self) -> tuple[bool, str]:
        """Can new orders be submitted right now? Brokers with time windows
        (e.g. Alpaca opening-auction orders) override this."""
        return True, ""

    def sync(self, as_of: pd.Timestamp, bars: dict[str, pd.Series]) -> None:
        """Advance the broker to ``as_of`` (simulated brokers fill pending
        orders here; real paper APIs fill on their own, so this is a no-op)."""
        return None

    def mark(self, closes: dict[str, float]) -> None:
        """Optional: give a simulated broker the latest closes."""
        return None


def trading_mode() -> str:
    return os.getenv("TRADING_MODE", "BACKTEST").strip().upper()


def assert_safe_mode(mode: str | None = None) -> str:
    m = (mode or trading_mode()).upper()
    if m == "LIVE":
        raise LiveTradingNotAllowedError("TRADING_MODE=LIVE is not supported: real-money execution is disabled")
    if m not in ALLOWED_MODES:
        raise LiveTradingNotAllowedError(f"unknown TRADING_MODE {m!r}; allowed: {sorted(ALLOWED_MODES)}")
    return m


def startup_banner(mode: str | None = None) -> str:
    m = assert_safe_mode(mode)
    banner = "\n".join(["=" * 32, f"TRADING MODE: {m}", "REAL MONEY: DISABLED", "=" * 32])
    log.info("TRADING MODE: %s | REAL MONEY: DISABLED", m)
    print(banner, flush=True)
    return banner


def create_broker(mode: str | None = None, broker: str | None = None, **kwargs) -> BrokerAdapter:
    """``broker``: "simulated" (built-in PaperBroker, default) or "alpaca_paper".
    Defaults to the BROKER env var. Live brokers do not exist here."""
    m = assert_safe_mode(mode)
    name = (broker or os.getenv("BROKER") or "simulated").strip().lower()
    if name in ("simulated", "paper", "sim"):
        from src.execution.paper_broker import PaperBroker

        return PaperBroker(**kwargs)
    if name == "alpaca_paper":
        if m != "PAPER":
            raise LiveTradingNotAllowedError("the Alpaca adapter is only used in PAPER mode")
        from src.execution.alpaca_paper_broker import AlpacaPaperBroker

        allowed = {"trading_enabled", "known_symbols", "time_in_force", "session", "clock", "api_key", "api_secret",
                   "base_url"}
        return AlpacaPaperBroker(**{k: v for k, v in kwargs.items() if k in allowed})
    raise LiveTradingNotAllowedError(f"unknown broker {name!r}; allowed: simulated, alpaca_paper")
