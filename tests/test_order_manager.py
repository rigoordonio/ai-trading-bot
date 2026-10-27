"""OrderManager: idempotency, UNKNOWN state, reconcile-before-retry, DB failure modes."""
import pytest

from src.errors import (
    BrokerError,
    CircuitOpenError,
    DatabaseError,
    DuplicateOrderError,
    OrderRejectedError,
    TradingDisabledError,
)
from src.execution.order_manager import FileOrderLedger, InMemoryLedger, OrderManager
from src.execution.orders import OrderStatus, Side
from src.execution.reconciliation import reconcile
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthMonitor, TradingGate
from tests.fakes import FaultyBroker, FaultyLedger
from tests.helpers import D0, approved_order


def setup(mode=None, ledger=None, enabled=True):
    gate = TradingGate(HealthMonitor(), operator_enabled=enabled)
    broker = FaultyBroker(100_000, mode=mode, trading_enabled=lambda: gate.trading_enabled)
    om = OrderManager(broker, ledger if ledger is not None else InMemoryLedger(), gate, CircuitBreaker("broker", 3))
    return om, broker, gate


def validated(**kw):
    o = approved_order(**kw)
    o.transition(OrderStatus.VALIDATED)
    return o


def test_successful_submission_records_ledger():
    om, broker, gate = setup()
    r = om.submit(validated())
    assert r.submitted and r.order.status == OrderStatus.ACKNOWLEDGED
    assert om.ledger.get_by_key(r.order.idempotency_key) is r.order
    assert r.order.broker_order_id


def test_broker_timeout_after_accept_reconciles_and_never_resubmits():
    om, broker, gate = setup("timeout_after_accept")
    o = validated()
    r = om.submit(o)
    assert broker.submit_calls == 1
    assert o.status == OrderStatus.ACKNOWLEDGED            # adopted from the broker, not "failed"
    assert "NOT resubmitted" in r.note
    with pytest.raises(DuplicateOrderError):                # a retry of the same intention is refused
        om.submit(validated())
    assert broker.submit_calls == 1 and len(broker.get_orders()) == 1


def test_broker_timeout_before_accept_marks_cancelled_after_reconcile():
    om, broker, gate = setup("timeout_before_accept")
    o = validated()
    r = om.submit(o)
    assert o.status == OrderStatus.CANCELLED and "not at broker" in r.note
    assert broker.get_orders() == []
    # A new decision with the same key is now allowed (previous attempt provably never reached the broker)
    broker.mode = None
    assert om.submit(validated()).submitted


def test_unknown_stays_unknown_when_broker_unreachable_and_trading_halts():
    om, broker, gate = setup("timeout_after_accept")
    o = validated()
    broker.mode = "timeout_after_accept"
    # make lookups fail too: we cannot learn the truth
    orig = broker.get_order_by_client_id
    broker.get_order_by_client_id = lambda k: (_ for _ in ()).throw(ConnectionError("down"))
    r = om.submit(o)
    assert o.status == OrderStatus.UNKNOWN and "UNKNOWN" in r.note
    assert not gate.trading_enabled
    broker.get_order_by_client_id = orig
    assert om.reconcile_unknown()[0].startswith("reconciled: broker has the order")
    assert o.status == OrderStatus.ACKNOWLEDGED


def test_unexpected_error_after_accept_is_unknown_not_failed():
    om, broker, gate = setup("error_after_accept")
    o = validated()
    om.submit(o)
    assert o.status == OrderStatus.ACKNOWLEDGED and broker.submit_calls == 1


def test_broker_rejection():
    om, broker, gate = setup("reject")
    o = validated()
    with pytest.raises(OrderRejectedError):
        om.submit(o)
    assert o.status == OrderStatus.REJECTED and gate.trading_enabled


def test_db_failure_before_submit_means_no_order():
    om, broker, gate = setup(ledger=FaultyLedger(fail_intent=True))
    o = validated()
    with pytest.raises(DatabaseError):
        om.submit(o)
    assert broker.submit_calls == 0 and o.status == OrderStatus.REJECTED
    assert not gate.trading_enabled


def test_broker_succeeded_database_failed_halts_and_does_not_resubmit():
    ledger = FaultyLedger()
    om, broker, gate = setup(ledger=ledger)
    o = validated()
    ledger.fail_update = True
    r = om.submit(o)
    assert r.submitted and "database update failed" in r.note
    assert not gate.trading_enabled                         # must reconcile before trading again
    assert len(broker.get_orders()) == 1
    with pytest.raises(TradingDisabledError):
        om.submit(validated(symbol="MSFT"))


def test_kill_switch_prevents_new_orders_but_state_is_readable():
    om, broker, gate = setup(enabled=False)
    with pytest.raises(TradingDisabledError):
        om.submit(validated())
    assert broker.submit_calls == 0
    assert broker.get_account().cash == 100_000          # reads still work
    assert reconcile({}, 100_000, om.ledger, broker, gate).ok


def test_circuit_breaker_blocks_orders():
    om, broker, gate = setup("timeout_before_accept")
    for i in range(3):
        om.submit(validated(symbol="AAPL", date=D0 + __import__("pandas").Timedelta(days=i)))
    assert om.breaker.is_open and not gate.trading_enabled
    with pytest.raises(TradingDisabledError):
        om.submit(validated(symbol="MSFT"))


def test_circuit_open_refuses_even_if_gate_check_bypassed():
    om, broker, gate = setup()
    om.breaker.record_failure()
    om.breaker.record_failure()
    om.breaker.record_failure()
    gate.breakers.clear()                                   # simulate a wiring mistake
    with pytest.raises(CircuitOpenError):
        om.submit(validated())
    assert broker.submit_calls == 0


def test_only_validated_orders_can_be_submitted():
    om, broker, gate = setup()
    o = approved_order()                                    # PROPOSED, not validated
    with pytest.raises(BrokerError):
        om.submit(o)
    assert broker.submit_calls == 0


def test_file_ledger_survives_restart(tmp_path):
    led = FileOrderLedger(tmp_path / "orders.json")
    om, broker, gate = setup(ledger=led)
    o = validated(bot_id="SWING_V1", strategy="SWING_V1")
    om.submit(o)
    again = FileOrderLedger(tmp_path / "orders.json")
    got = again.get_by_key(o.idempotency_key)
    assert got.status == OrderStatus.ACKNOWLEDGED and got.meta["bot_id"] == "SWING_V1"
    assert got.side == Side.BUY


def test_ledger_read_failure_in_duplicate_check_halts():
    class BrokenLookup(InMemoryLedger):
        def get_by_key(self, key):
            raise OSError("database unreachable")

    om, broker, gate = setup(ledger=BrokenLookup())
    with pytest.raises(DatabaseError, match="duplicate"):
        om.submit(validated())
    assert broker.submit_calls == 0 and not gate.trading_enabled
