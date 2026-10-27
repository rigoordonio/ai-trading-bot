"""Property-based invariants.

Uses Hypothesis when installed; otherwise the same properties run on 300
seeded random examples (so the invariants are still exercised offline)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.errors import TradingSystemError
from src.execution.orders import Order, OrderStatus, Side
from src.risk import position_sizing as ps
from src.risk.engine import BLOCKED, OrderProposal, RiskEngine
from src.safety.order_validator import OrderContext, OrderValidator
from src.safety.validation import validate_bar_values
from tests.helpers import D0, cfg_with, portfolio_with, snapshot

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False

N = 300
WEIRD = [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, 1e-12, 1e12]


def floats(rng, lo, hi, weird=0.1):
    return float(rng.choice(WEIRD)) if rng.random() < weird else float(rng.uniform(lo, hi))


def run_examples(fn, gen):
    rng = np.random.default_rng(1234)
    for _ in range(N):
        fn(*gen(rng))


# ---------------------------------------------------------------- invariants
def prop_position_size_non_negative_finite(equity, price, vol, atr, frac):
    for q in (ps.fixed_fractional(equity, price, frac), ps.volatility_target(equity, price, vol, 0.15),
              ps.atr_based(equity, price, atr, 0.01, 2.0)):
        assert isinstance(q, int) and q >= 0 and math.isfinite(q)


ENGINE = RiskEngine(cfg_with(limits={"min_avg_dollar_volume": 0, "max_portfolio_volatility": 10.0}).risk)
LIM = ENGINE.limits


def prop_accepted_buys_respect_exposure(cash, held_val, price, qty):
    eq = cash + held_val
    p = portfolio_with(cash, {"MSFT": (1, held_val)} if held_val > 0 else {})
    prop = OrderProposal("AAPL", Side.BUY, qty, "p", "T", "BUY", 0.9, {})
    orders, decs, _ = ENGINE.review([prop], p, snapshot({"AAPL": price, "MSFT": held_val}))
    d = decs[0]
    if d.status == BLOCKED:
        assert d.approved_quantity == 0 and orders == []          # rejection never creates an order
        return
    o = orders[0]
    assert o.quantity == d.approved_quantity > 0 and math.isfinite(o.quantity)
    notional = o.quantity * price
    assert (p.positions_value + notional) / eq <= LIM.max_gross_exposure + 1e-9
    assert notional / eq <= LIM.max_position_weight + 1e-9
    assert cash - notional >= LIM.min_cash_reserve * eq - 1e-6
    assert RiskEngine.verify(o)


V = OrderValidator(25_000, 4, 0.95)


def prop_nan_inf_never_reach_an_order(o, h, low, c, v, qty_choice):
    """Invalid market data or quantities can never become an executable order."""
    bar = pd.Series({"open": o, "high": h, "low": low, "close": c, "volume": v})
    bar_ok = True
    try:
        validate_bar_values("AAPL", bar, D0, D0)
    except TradingSystemError:
        bar_ok = False
    vals = [o, h, low, c, v]
    if any(not math.isfinite(x) for x in vals) or min(o, h, low, c) <= 0 or h < max(o, c, low) or low > min(o, c, h):
        assert not bar_ok
    try:
        order = Order("AAPL", Side.BUY, qty_choice, D0, reference_price=c if math.isfinite(c) else None)
    except ValueError:
        return                                                   # invalid quantity cannot even be constructed
    assert math.isfinite(order.quantity) and order.quantity > 0
    try:
        V.validate(order, OrderContext(D0, 1e9, 1e9, 0, 0, {"AAPL"}))
    except TradingSystemError:
        assert order.status == OrderStatus.PROPOSED              # stays non-executable
        return
    pytest.fail("an unapproved order passed validation")        # no risk approval -> must not validate


# -------------------------------------------------------------- generators
def gen_sizing(rng):
    return (floats(rng, 1e3, 1e7), floats(rng, 0.5, 2000), floats(rng, 0.01, 1.5), floats(rng, 0.01, 50),
            floats(rng, 0.0, 0.5))


def gen_exposure(rng):
    return (float(rng.uniform(1_000, 1e6)), float(rng.choice([0.0, rng.uniform(0, 1e6)])),
            float(rng.uniform(1, 2_000)), int(rng.integers(1, 100_000)))


def gen_bar(rng):
    c = floats(rng, 1, 500, 0.15)
    o = floats(rng, 1, 500, 0.15)
    h = floats(rng, 1, 600, 0.15)
    low = floats(rng, 0.5, 500, 0.15)
    v = floats(rng, -10, 1e7, 0.1)
    q = rng.choice([1, 10, 0, -3, float("nan"), float("inf"), 2.5, 100])
    return o, h, low, c, v, q.item() if hasattr(q, "item") else q


if HAVE_HYPOTHESIS:
    any_f = st.floats(allow_nan=True, allow_infinity=True, width=64)

    @settings(max_examples=N, deadline=None)
    @given(any_f, any_f, any_f, any_f, any_f)
    def test_position_size_non_negative(e, p, v, a, f):
        prop_position_size_non_negative_finite(e, p, v, a, f)

    @settings(max_examples=N, deadline=None)
    @given(st.floats(1_000, 1e6), st.one_of(st.just(0.0), st.floats(0, 1e6)), st.floats(1, 2_000),
           st.integers(1, 100_000))
    def test_accepted_orders_respect_limits(c, h, p, q):
        prop_accepted_buys_respect_exposure(c, h, p, q)

    @settings(max_examples=N, deadline=None)
    @given(any_f, any_f, any_f, any_f, any_f, st.one_of(st.integers(-5, 1000), any_f))
    def test_invalid_inputs_never_become_orders(o, h, low, c, v, q):
        prop_nan_inf_never_reach_an_order(o, h, low, c, v, q)
else:
    def test_position_size_non_negative():
        run_examples(prop_position_size_non_negative_finite, gen_sizing)

    def test_accepted_orders_respect_limits():
        run_examples(prop_accepted_buys_respect_exposure, gen_exposure)

    def test_invalid_inputs_never_become_orders():
        run_examples(prop_nan_inf_never_reach_an_order, gen_bar)
