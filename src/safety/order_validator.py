"""Last line of defence before an order reaches a broker.

Every check raises ``OrderValidationError`` with the rule that fired. The
validator never "fixes" an order; resizing is the risk engine's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import pandas as pd

from src.errors import OrderValidationError
from src.execution.orders import Order, OrderStatus, OrderType, Side
from src.risk.engine import RiskEngine

SUPPORTED_TYPES = {OrderType.MARKET, OrderType.LIMIT}


@dataclass
class OrderContext:
    as_of: pd.Timestamp                   # "now" (decision or submission time)
    cash: float
    equity: float
    gross_value: float                    # current invested value
    held_quantity: int
    known_symbols: set[str]


class OrderValidator:
    def __init__(self, max_order_notional: float, max_signal_age_days: int, max_gross_exposure: float,
                 cost_buffer: float = 0.01):
        self.max_notional = max_order_notional
        self.max_age = max_signal_age_days
        self.max_gross = max_gross_exposure
        self.cost_buffer = cost_buffer

    @classmethod
    def from_config(cls, risk_cfg) -> "OrderValidator":
        return cls(risk_cfg.safety.max_order_notional, risk_cfg.safety.max_signal_age_days,
                   risk_cfg.limits.max_gross_exposure)

    def validate(self, order: Order, ctx: OrderContext) -> None:
        def fail(rule: str, msg: str) -> NoReturn:
            raise OrderValidationError(f"{order.symbol}: {msg}", context={"rule": rule, "order_id": order.order_id})

        if order.status != OrderStatus.PROPOSED:
            fail("state", f"order must be PROPOSED to validate, is {order.status.value}")
        if not order.symbol or order.symbol not in ctx.known_symbols:
            fail("symbol", f"unknown symbol {order.symbol!r}")
        if not isinstance(order.side, Side):
            fail("side", f"invalid side {order.side!r}")
        q = order.quantity
        if not isinstance(q, int) or isinstance(q, bool) or not math.isfinite(q) or q <= 0:
            fail("quantity", f"invalid quantity {q!r}")
        if order.order_type not in SUPPORTED_TYPES:
            fail("order_type", f"unsupported order type {order.order_type}")
        px = order.reference_price
        if px is None or not math.isfinite(px) or px <= 0:
            fail("price", f"invalid reference price {px!r}")
        notional = q * px
        if not math.isfinite(notional) or notional > self.max_notional:
            fail("max_order_notional", f"notional {notional:,.0f} exceeds max {self.max_notional:,.0f}")
        if not order.risk_decision_id or not RiskEngine.verify(order):
            fail("risk_approval", "missing or invalid risk approval")
        age = (pd.Timestamp(ctx.as_of).normalize() - pd.Timestamp(order.created_at).normalize()).days
        if age < 0:
            fail("signal_time", "signal timestamp is in the future")
        if age > self.max_age:
            fail("stale_signal", f"stale signal: {age} days old (max {self.max_age})")
        if order.side == Side.BUY:
            if notional * (1 + self.cost_buffer) > ctx.cash:
                fail("cash", f"insufficient cash {ctx.cash:,.0f} for {notional:,.0f}")
            if ctx.equity <= 0 or (ctx.gross_value + notional) / ctx.equity > self.max_gross + 1e-9:
                fail("exposure", "order would exceed max gross exposure")
        else:
            if q > ctx.held_quantity:
                fail("position", f"sell {q} exceeds held {ctx.held_quantity} (shorting disabled)")
        order.transition(OrderStatus.VALIDATED, "order validator passed")
