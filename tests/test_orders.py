"""Order sanity: quantities, approvals, staleness, notional, state machine, duplicates."""
import pandas as pd
import pytest

from src.errors import DuplicateOrderError, OrderRejectedError, OrderValidationError, TradingDisabledError
from src.execution.orders import InvalidOrderTransition, Order, OrderStatus, OrderType, Side, idempotency_key
from src.execution.paper_broker import PaperBroker
from src.risk.engine import RiskEngine
from src.safety.order_validator import OrderContext, OrderValidator
from tests.helpers import D0, approved_order

V = OrderValidator(max_order_notional=25_000, max_signal_age_days=4, max_gross_exposure=0.95)


def ctx(cash=100_000.0, equity=100_000.0, gross=0.0, held=0, as_of=D0):
    return OrderContext(as_of, cash, equity, gross, held, {"AAPL", "MSFT"})


@pytest.mark.parametrize("q", [float("nan"), float("inf"), -float("inf"), 1.5, "10", None, True])
def test_invalid_quantity(q):
    with pytest.raises(ValueError):
        Order("AAPL", Side.BUY, q, D0)


def test_negative_quantity():
    with pytest.raises(ValueError):
        Order("AAPL", Side.BUY, -5, D0)


def test_zero_quantity():
    with pytest.raises(ValueError):
        Order("AAPL", Side.BUY, 0, D0)


def test_numpy_integer_quantity_accepted():
    import numpy as np
    assert Order("AAPL", Side.BUY, np.int64(7), D0).quantity == 7


def test_order_without_risk_approval():
    o = Order("AAPL", Side.BUY, 10, D0, reference_price=100.0)
    with pytest.raises(OrderValidationError, match="risk approval"):
        V.validate(o, ctx())
    forged = Order("AAPL", Side.BUY, 10, D0, reference_price=100.0, risk_decision_id="R-forged")
    assert not RiskEngine.verify(forged)
    with pytest.raises(OrderValidationError, match="risk approval"):
        V.validate(forged, ctx())
    broker = PaperBroker(100_000)
    with pytest.raises(OrderRejectedError):
        broker.submit_order(forged)


def test_quantity_above_approval_is_not_verified():
    o = approved_order(qty=10)
    o.quantity = 11
    assert not RiskEngine.verify(o)


def test_stale_signal():
    o = approved_order(date=D0)
    with pytest.raises(OrderValidationError, match="stale"):
        V.validate(o, ctx(as_of=D0 + pd.Timedelta(days=10)))
    o2 = approved_order(date=D0 + pd.Timedelta(days=3))
    with pytest.raises(OrderValidationError, match="future"):
        V.validate(o2, ctx(as_of=D0))


def test_max_order_notional():
    o = approved_order(qty=300, price=100.0)            # $30,000 > $25,000
    with pytest.raises(OrderValidationError, match="max_order_notional|exceeds"):
        V.validate(o, ctx())


def test_unknown_symbol_cash_exposure_and_oversell():
    with pytest.raises(OrderValidationError, match="unknown symbol"):
        V.validate(approved_order("ZZZZ"), ctx())
    with pytest.raises(OrderValidationError, match="cash"):
        V.validate(approved_order(qty=100), ctx(cash=5_000))
    with pytest.raises(OrderValidationError, match="exposure"):
        V.validate(approved_order(qty=100), ctx(gross=90_000))
    with pytest.raises(OrderValidationError, match="shorting"):
        V.validate(approved_order(side=Side.SELL, qty=10), ctx(held=5))


def test_valid_order_passes_and_is_validated():
    o = approved_order(qty=10)
    V.validate(o, ctx())
    assert o.status == OrderStatus.VALIDATED


def test_state_machine_transitions():
    o = approved_order()
    with pytest.raises(InvalidOrderTransition):
        o.transition(OrderStatus.FILLED)                 # PROPOSED -> FILLED is illegal
    for s in (OrderStatus.VALIDATED, OrderStatus.SUBMITTING, OrderStatus.UNKNOWN, OrderStatus.ACKNOWLEDGED,
              OrderStatus.FILLED):
        o.transition(s)
    with pytest.raises(InvalidOrderTransition):
        o.transition(OrderStatus.CANCELLED)              # terminal
    assert [h[0] for h in o.history][-1] == "FILLED"


def test_unknown_is_never_marked_failed_without_reconciliation():
    from src.execution.orders import ALLOWED_TRANSITIONS
    # UNKNOWN can only leave via a broker-backed state; there is no generic "FAILED" state
    assert "FAILED" not in OrderStatus.__members__
    assert OrderStatus.UNKNOWN not in ALLOWED_TRANSITIONS[OrderStatus.FILLED]


def test_idempotency_key_deterministic():
    k1 = idempotency_key("SWING_V1", "AAPL", D0, Side.BUY)
    assert k1 == idempotency_key("SWING_V1", "AAPL", D0, "BUY")
    assert k1 != idempotency_key("TREND_V1", "AAPL", D0, Side.BUY)
    assert k1 != idempotency_key("SWING_V1", "AAPL", D0 + pd.Timedelta(days=1), Side.BUY)
    assert approved_order(strategy="X").idempotency_key == approved_order(strategy="X").idempotency_key


def test_duplicate_order():
    broker = PaperBroker(100_000)
    o1 = approved_order(strategy="BOT")
    broker.submit_order(o1)
    o2 = approved_order(strategy="BOT")                  # same intention -> same key
    with pytest.raises(DuplicateOrderError):
        broker.submit_order(o2)
    assert len(broker.get_orders()) == 1


def test_kill_switch_prevents_new_orders():
    broker = PaperBroker(100_000, trading_enabled=lambda: False)
    with pytest.raises(TradingDisabledError):
        broker.submit_order(approved_order())
    assert broker.get_orders() == []


def test_limit_order_requires_valid_price():
    with pytest.raises(ValueError):
        Order("AAPL", Side.BUY, 1, D0, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        Order("AAPL", Side.BUY, 1, D0, order_type=OrderType.LIMIT, limit_price=float("nan"))
