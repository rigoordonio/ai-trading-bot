"""Order and fill objects.

An ``Order`` can only reach a broker if it carries a risk approval
(``risk_decision_id``). Brokers refuse anything else, so neither a strategy,
the ML model nor the LLM layer can place an order around the risk engine.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"     # filled at the next bar's open (+ costs)
    LIMIT = "LIMIT"       # filled at next open if marketable, else at limit if touched


class OrderStatus(str, Enum):
    PROPOSED = "PROPOSED"                # created by the risk engine, not yet validated
    VALIDATED = "VALIDATED"              # passed the order validator
    SUBMITTING = "SUBMITTING"            # intent recorded, request in flight
    SUBMITTED = "SUBMITTED"              # broker received it (backtest: queued for next open)
    ACKNOWLEDGED = "ACKNOWLEDGED"        # broker accepted it
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"                  # response lost: must be RECONCILED, never assumed failed


TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
OPEN_STATES = {OrderStatus.SUBMITTING, OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED,
               OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN}

_S = OrderStatus
ALLOWED_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    _S.PROPOSED: {_S.VALIDATED, _S.REJECTED, _S.CANCELLED},
    _S.VALIDATED: {_S.SUBMITTING, _S.SUBMITTED, _S.REJECTED, _S.CANCELLED},
    _S.SUBMITTING: {_S.SUBMITTED, _S.ACKNOWLEDGED, _S.REJECTED, _S.UNKNOWN},
    _S.SUBMITTED: {_S.ACKNOWLEDGED, _S.PARTIALLY_FILLED, _S.FILLED, _S.CANCELLED, _S.REJECTED, _S.UNKNOWN},
    _S.ACKNOWLEDGED: {_S.PARTIALLY_FILLED, _S.FILLED, _S.CANCELLED, _S.REJECTED, _S.UNKNOWN},
    _S.PARTIALLY_FILLED: {_S.PARTIALLY_FILLED, _S.FILLED, _S.CANCELLED, _S.UNKNOWN},
    # Only reconciliation with the broker may move an order out of UNKNOWN.
    _S.UNKNOWN: {_S.SUBMITTED, _S.ACKNOWLEDGED, _S.PARTIALLY_FILLED, _S.FILLED, _S.CANCELLED, _S.REJECTED},
    _S.FILLED: set(),
    _S.CANCELLED: set(),
    _S.REJECTED: set(),
}


class RiskNotApprovedError(PermissionError):
    """Raised when an order without a valid risk approval reaches a broker."""


class InvalidOrderTransition(ValueError):
    pass


def idempotency_key(strategy: str, symbol: str, signal_ts, side: "Side | str", account: str = "paper") -> str:
    """Deterministic key for one order INTENTION. The same decision on the
    same bar always yields the same key, so a retry cannot create a twin."""
    import hashlib

    side_v = side.value if isinstance(side, Side) else str(side)
    raw = f"{account}|{strategy}|{symbol}|{pd.Timestamp(signal_ts):%Y-%m-%dT%H:%M}|{side_v}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


_seq = itertools.count(1)


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: int
    created_at: pd.Timestamp
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reason: str = ""                       # why: "ensemble BUY", "stop_loss", ...
    strategy: str = ""
    risk_decision_id: str | None = None    # set ONLY by RiskEngine
    reference_price: float | None = None   # close at decision time (for audit / slippage stats)
    action: str = ""                       # BUY / REDUCE / EXIT / STOP
    account: str = "paper"
    idempotency_key: str = ""
    broker_order_id: str | None = None
    order_id: str = field(default_factory=lambda: f"O{next(_seq):07d}-{uuid.uuid4().hex[:6]}")
    status: OrderStatus = OrderStatus.PROPOSED
    filled_quantity: int = 0
    avg_fill_price: float | None = None
    reject_reason: str | None = None
    execute_on: pd.Timestamp | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        import math
        import numbers

        q = self.quantity
        whole = isinstance(q, numbers.Integral) or (isinstance(q, numbers.Real) and float(q).is_integer())
        if isinstance(q, bool) or not whole:   # rejects NaN, inf, 1.5, "10", None
            raise ValueError(f"order quantity must be a whole number, got {q!r}")
        if not math.isfinite(float(self.quantity)) or self.quantity <= 0:
            raise ValueError(f"order quantity must be finite and positive, got {self.quantity!r}")
        self.quantity = int(self.quantity)
        if not isinstance(self.side, Side):
            self.side = Side(self.side)
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or not math.isfinite(self.limit_price)
                                                   or self.limit_price <= 0):
            raise ValueError("limit orders need a finite positive limit_price")
        if not self.idempotency_key:
            self.idempotency_key = idempotency_key(self.strategy or "manual", self.symbol, self.created_at,
                                                   self.side, self.account)
        self.history.append((self.status.value, str(self.created_at)))

    def transition(self, new: OrderStatus, note: str = "") -> None:
        if new not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidOrderTransition(f"{self.order_id}: {self.status.value} -> {new.value} not allowed")
        self.status = new
        self.history.append((new.value, note))

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATES

    @property
    def risk_approved(self) -> bool:
        return bool(self.risk_decision_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"], d["order_type"], d["status"] = self.side.value, self.order_type.value, self.status.value
        d.pop("history", None)
        d.pop("meta", None)
        return d


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float              # all-in execution price incl. spread / slippage / impact
    reference_price: float    # bar open the price was derived from
    commission: float
    timestamp: pd.Timestamp
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost + self.impact_cost

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["total_cost"] = self.total_cost
        return d
