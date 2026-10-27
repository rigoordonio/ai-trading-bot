"""Internal books vs broker (authoritative). Any discrepancy disables trading."""
import pandas as pd

from src.execution.broker import BrokerOrder
from src.execution.order_manager import InMemoryLedger
from src.execution.orders import OrderStatus
from src.execution.paper_broker import PaperBroker
from src.execution.reconciliation import reconcile
from src.safety.health import HealthMonitor, TradingGate
from tests.fakes import FaultyBroker
from tests.helpers import approved_order, bar


def broker_with_position(sym="AAPL", qty=10):
    b = PaperBroker(100_000)
    o = approved_order(sym, qty=qty)
    b.submit_order(o)
    b.process_open(pd.Timestamp("2024-01-03"), {sym: bar(o=100, h=100, low=100, c=100)})
    return b, o


def gate():
    return TradingGate(HealthMonitor(), operator_enabled=True)


def types(rep):
    return {d["type"] for d in rep.discrepancies}


def test_matching_books_pass():
    b, o = broker_with_position()
    led = InMemoryLedger()
    led.record_intent(o)
    g = gate()
    rep = reconcile({"AAPL": 10}, b.cash, led, b, g)
    assert rep.ok and g.trading_enabled


def test_missing_unexpected_and_quantity_mismatch():
    b, o = broker_with_position()
    led = InMemoryLedger()
    led.record_intent(o)
    g = gate()
    rep = reconcile({"AAPL": 12, "MSFT": 5}, b.cash, led, b, g)
    assert {"quantity_mismatch", "missing_position"} <= types(rep)
    assert not g.trading_enabled
    rep2 = reconcile({}, b.cash, led, b, gate())
    assert "unexpected_position" in types(rep2)


def test_unknown_open_order_and_duplicate_order_and_cash():
    b = PaperBroker(100_000)
    o = approved_order()
    b.submit_order(o)                                    # open at broker, not in our ledger
    b.orders["dup"] = BrokerOrder("dup", o.idempotency_key, "AAPL", "BUY", 1, "ACKNOWLEDGED")
    rep = reconcile({}, 99_000, InMemoryLedger(), b, gate())
    assert {"unknown_open_order", "duplicate_order", "cash_mismatch"} <= types(rep)


def test_unresolved_unknown_order_fails_reconciliation():
    b = PaperBroker(100_000)
    led = InMemoryLedger()
    o = approved_order()
    o.transition(OrderStatus.VALIDATED)
    o.transition(OrderStatus.SUBMITTING)
    o.transition(OrderStatus.UNKNOWN)
    led.record_intent(o)
    assert "unresolved_unknown_order" in types(reconcile({}, 100_000, led, b, gate()))


def test_position_reconciliation_failure_when_broker_unreachable():
    b = FaultyBroker(100_000, mode="query_down")
    g = gate()
    rep = reconcile({}, 100_000, InMemoryLedger(), b, g)
    assert not rep.ok and "broker_unreachable" in types(rep) and not g.trading_enabled
