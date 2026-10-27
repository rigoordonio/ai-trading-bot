"""OrderManager: the ONLY path from a validated order to a broker.

    gate closed?  -> TradingDisabledError (no new orders, ever)
    circuit open? -> CircuitOpenError
    duplicate idempotency key in the ledger (not terminal-failed)? -> refuse
    1. ledger.record_intent(order, SUBMITTING)    DB fails -> do NOT submit
    2. broker.submit_order(order)                 NOT retried blindly
         timeout / lost response -> UNKNOWN -> reconcile_order():
             broker has it (by client id) -> adopt broker state
             broker does not have it      -> CANCELLED("never reached broker");
                                             a NEW decision may resubmit later
             broker unreachable           -> stays UNKNOWN, trading halted
         rejection      -> REJECTED
    3. ledger.update(order)                       DB fails AFTER broker success
                                                  -> order is live; we halt trading
                                                     and require reconciliation,
                                                     never resubmit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.errors import (
    BrokerError,
    BrokerTimeoutError,
    CircuitOpenError,
    DatabaseError,
    DuplicateOrderError,
    OrderRejectedError,
    ReconciliationError,
    Severity,
    TradingDisabledError,
)
from src.execution.broker import BrokerAdapter, BrokerOrder
from src.execution.orders import ALLOWED_TRANSITIONS, TERMINAL, Order, OrderStatus
from src.logging_setup import get_logger
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthStatus, TradingGate

log = get_logger(__name__)

BROKER_TO_INTERNAL = {"ACKNOWLEDGED": OrderStatus.ACKNOWLEDGED, "SUBMITTED": OrderStatus.SUBMITTED,
                      "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED, "FILLED": OrderStatus.FILLED,
                      "CANCELLED": OrderStatus.CANCELLED, "REJECTED": OrderStatus.REJECTED}


class OrderLedger(Protocol):
    def record_intent(self, order: Order) -> None: ...
    def update(self, order: Order) -> None: ...
    def get_by_key(self, key: str) -> Order | None: ...
    def open_orders(self) -> list[Order]: ...
    def all(self) -> list[Order]: ...


@dataclass
class InMemoryLedger:
    orders: dict[str, Order] = field(default_factory=dict)          # by idempotency key

    def record_intent(self, order: Order) -> None:
        self.orders[order.idempotency_key] = order

    def update(self, order: Order) -> None:
        self.orders[order.idempotency_key] = order

    def get_by_key(self, key: str) -> Order | None:
        return self.orders.get(key)

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.status not in TERMINAL]

    def all(self) -> list[Order]:
        return list(self.orders.values())


class FileOrderLedger(InMemoryLedger):
    """InMemoryLedger persisted to JSON after every change (atomic replace).
    Used by paper trading when no database is configured, so order intents,
    UNKNOWN states and bot attribution survive restarts."""

    def __init__(self, path):
        import json
        from pathlib import Path

        import pandas as pd

        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                o = Order(symbol=d["symbol"], side=d["side"], quantity=int(d["quantity"]),
                          created_at=pd.Timestamp(d["created_at"]), reason=d.get("reason", ""),
                          strategy=d.get("strategy", ""), risk_decision_id=d.get("risk_decision_id"),
                          reference_price=d.get("reference_price"), action=d.get("action", ""),
                          idempotency_key=d["idempotency_key"], order_id=d["order_id"])
                o.status = OrderStatus(d["status"])
                o.broker_order_id = d.get("broker_order_id")
                o.filled_quantity = int(d.get("filled_quantity", 0))
                o.meta["bot_id"] = d.get("bot_id") or o.strategy
                self.orders[o.idempotency_key] = o

    def _flush(self) -> None:
        import json
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        rows = [{**o.to_dict(), "created_at": str(o.created_at), "bot_id": o.meta.get("bot_id", o.strategy)}
                for o in self.orders.values()]
        tmp.write_text(json.dumps(rows, indent=1, default=str))
        os.replace(tmp, self.path)

    def record_intent(self, order: Order) -> None:
        super().record_intent(order)
        self._flush()

    def update(self, order: Order) -> None:
        super().update(order)
        self._flush()


@dataclass
class SubmitResult:
    order: Order
    submitted: bool
    broker_order: BrokerOrder | None = None
    note: str = ""


class OrderManager:
    def __init__(self, broker: BrokerAdapter, ledger: OrderLedger, gate: TradingGate,
                 breaker: CircuitBreaker | None = None, allow_risk_reducing_when_halted: bool = False):
        self.broker = broker
        self.ledger = ledger
        self.gate = gate
        self.breaker = breaker or CircuitBreaker("broker")
        if self.breaker not in gate.breakers:
            gate.breakers.append(self.breaker)
        self.allow_reducing = allow_risk_reducing_when_halted
        self.events: list[dict] = []

    # --------------------------------------------------------------- submit
    def submit(self, order: Order) -> SubmitResult:
        if not self.gate.trading_enabled and not (self.allow_reducing and order.side.value == "SELL"):
            raise TradingDisabledError("; ".join(self.gate.blocking_reasons()) or "trading disabled")
        if order.status != OrderStatus.VALIDATED:
            raise BrokerError(f"order {order.order_id} must be VALIDATED before submission "
                              f"(is {order.status.value})", severity=Severity.ERROR)
        try:
            existing = self.ledger.get_by_key(order.idempotency_key)
        except Exception as exc:  # noqa: BLE001 - cannot prove it is not a duplicate -> halt, do not send
            self.gate.handle_error(DatabaseError(f"ledger lookup failed: {exc}"), "database")
            raise DatabaseError(f"cannot check for duplicate orders, order NOT submitted: {exc}") from exc
        if existing is not None and existing is not order and existing.status not in (
                OrderStatus.REJECTED, OrderStatus.CANCELLED):
            raise DuplicateOrderError(f"order intention {order.idempotency_key} already exists "
                                      f"({existing.order_id}, {existing.status.value})")
        if not self.breaker.allow():
            raise CircuitOpenError(f"broker circuit {self.breaker.state.value}: order not sent")

        order.transition(OrderStatus.SUBMITTING, "intent recorded")
        try:
            self.ledger.record_intent(order)
        except Exception as exc:  # noqa: BLE001 - ANY persistence failure means: do not send
            # Nothing was sent: safe to abandon. The DB is unhealthy -> halt.
            order.transition(OrderStatus.REJECTED, "intent could not be persisted")
            self.gate.handle_error(DatabaseError(f"could not persist order intent: {exc}"), "database")
            raise DatabaseError(f"order intent not persisted, order NOT submitted: {exc}") from exc

        try:
            bo = self.broker.submit_order(order)
        except (BrokerTimeoutError, TimeoutError, ConnectionError) as exc:
            self.breaker.record_failure(str(exc))
            order.transition(OrderStatus.UNKNOWN, f"response lost: {exc}")
            self._safe_ledger_update(order)
            log.error("order %s state UNKNOWN after %s: reconciling, NOT resubmitting", order.order_id, exc)
            return SubmitResult(order, False, None, self.reconcile_order(order))
        except (OrderRejectedError, DuplicateOrderError, TradingDisabledError) as exc:
            self.breaker.record_success()           # the broker answered: it is reachable
            order.transition(OrderStatus.REJECTED, str(exc))
            order.reject_reason = str(exc)
            self._safe_ledger_update(order)
            raise
        except Exception as exc:  # noqa: BLE001 - unknown broker failure => state UNKNOWN, reconcile
            self.breaker.record_failure(str(exc))
            # We cannot know whether the broker accepted it -> UNKNOWN, never "failed".
            order.transition(OrderStatus.UNKNOWN, f"unexpected broker error: {type(exc).__name__}: {exc}")
            self._safe_ledger_update(order)
            return SubmitResult(order, False, None, self.reconcile_order(order))

        self.breaker.record_success()
        order.broker_order_id = bo.broker_order_id
        order.transition(BROKER_TO_INTERNAL.get(bo.status, OrderStatus.SUBMITTED), "broker accepted")
        self.gate.health.mark("last_order", f"{order.symbol} {order.side.value} {order.quantity} "
                                            f"({order.status.value})")
        try:
            self.ledger.update(order)
        except Exception as exc:  # noqa: BLE001 - any DB failure after broker success => halt
            # Broker HAS the order. Never resubmit; halt and force reconciliation.
            self.gate.handle_error(DatabaseError(
                f"order {order.order_id} accepted by broker ({bo.broker_order_id}) but DB update failed: {exc}. "
                "Reconcile before trading resumes."), "database")
            return SubmitResult(order, True, bo, "broker accepted; database update failed; trading halted")
        return SubmitResult(order, True, bo, "submitted")

    def _safe_ledger_update(self, order: Order) -> None:
        try:
            self.ledger.update(order)
        except Exception as exc:  # noqa: BLE001 - recorded as CRITICAL, trading halts
            self.gate.handle_error(DatabaseError(f"ledger update failed for {order.order_id}: {exc}"), "database")

    # ------------------------------------------------------------ reconcile
    def reconcile_order(self, order: Order) -> str:
        """Resolve an UNKNOWN order by asking the broker (idempotency key)."""
        try:
            # A successful READ does not reset the breaker: repeated submit
            # timeouts must still be able to open it.
            bo = self.broker.get_order_by_client_id(order.idempotency_key)
        except Exception as exc:  # noqa: BLE001 - cannot query broker => stays UNKNOWN, halt
            self.breaker.record_failure(str(exc))
            self.gate.handle_error(ReconciliationError(
                f"order {order.order_id} is UNKNOWN and the broker cannot be queried: {exc}"), "broker")
            return "UNKNOWN: broker unreachable; trading halted until reconciled"
        if bo is None:
            order.transition(OrderStatus.CANCELLED, "reconciled: broker never received the order")
            self._safe_ledger_update(order)
            return "reconciled: not at broker (safe to make a NEW decision later; not auto-resubmitted)"
        order.broker_order_id = bo.broker_order_id
        order.filled_quantity = bo.filled_quantity
        order.transition(BROKER_TO_INTERNAL.get(bo.status, OrderStatus.SUBMITTED),
                         f"reconciled: broker state {bo.status}")
        self._safe_ledger_update(order)
        return f"reconciled: broker has the order ({bo.status}); NOT resubmitted"

    def reconcile_unknown(self) -> list[str]:
        return [self.reconcile_order(o) for o in self.ledger.open_orders() if o.status == OrderStatus.UNKNOWN]

    def refresh_open_orders(self) -> None:
        """Pull broker state for every non-terminal order in the ledger."""
        for o in self.ledger.open_orders():
            try:
                bo = self.broker.get_order_by_client_id(o.idempotency_key)
            except Exception as exc:  # noqa: BLE001 - broker unhealthy is recorded, refresh stops
                self.breaker.record_failure(str(exc))
                self.gate.health.set("broker", HealthStatus.UNHEALTHY, f"order refresh failed: {exc}")
                return
            if bo is None:
                continue
            new = BROKER_TO_INTERNAL.get(bo.status)
            if new and new != o.status and new in ALLOWED_TRANSITIONS[o.status]:
                o.filled_quantity = bo.filled_quantity
                o.avg_fill_price = bo.avg_fill_price
                o.transition(new, "broker update")
                self._safe_ledger_update(o)
