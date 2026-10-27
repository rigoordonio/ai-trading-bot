"""Risk engine, sizing, stops, drawdown/kill switch, portfolio-level limits."""
import math

import pandas as pd
import pytest

from src.execution.orders import OrderStatus, Side
from src.portfolio.portfolio import Portfolio
from src.portfolio.risk_manager import AccountState, PortfolioRiskManager
from src.risk import position_sizing as ps
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import BLOCKED, RESIZED, OrderProposal, RiskEngine
from src.risk.portfolio_risk import portfolio_volatility, risk_contributions
from src.risk.stop_loss import StopLossManager
from tests.helpers import D0, approved_order, cfg_with, portfolio_with, snapshot


def buy(sym="AAPL", qty=100, score=0.9):
    return OrderProposal(sym, Side.BUY, qty, "test", "T", "BUY", score, {})


def engine(**limits):
    cfg = cfg_with(limits={"min_avg_dollar_volume": 0, "max_portfolio_volatility": 10.0, **limits})
    return RiskEngine(cfg.risk, {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "XOM": "Energy"})


# ------------------------------------------------------------------ sizing
def test_fixed_fractional_volatility_and_atr_sizing():
    assert ps.fixed_fractional(100_000, 50.0, 0.10) == 200
    assert ps.volatility_target(100_000, 100.0, asset_vol=0.30, target_vol=0.15) == 500    # weight 0.5
    assert ps.volatility_target(100_000, 100.0, asset_vol=0.05, target_vol=0.15) == 1000   # capped at weight 1
    assert ps.atr_based(100_000, 100.0, atr=2.0, risk_per_trade=0.01, stop_multiple=2.0) == 250


def test_sizing_with_denormal_price_does_not_overflow():
    # found by Hypothesis on a real machine: 1.0 / 2.2e-309 overflows to inf
    assert ps.fixed_fractional(1.0, 2.225073858507203e-309, 1.0) == 0
    assert ps.volatility_target(1.0, 2.2e-309, 0.2, 0.15) == 0
    assert ps.atr_based(1e308, 100.0, 1e-308, 1.0, 1.0) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, None])
def test_sizing_never_negative_or_from_bad_inputs(bad):
    assert ps.fixed_fractional(100_000, bad, 0.1) == 0
    assert ps.volatility_target(100_000, 100.0, bad, 0.15) == 0
    assert ps.atr_based(100_000, 100.0, bad, 0.01, 2.0) == 0
    assert ps.fixed_fractional(bad, 100.0, 0.1) == 0


# ------------------------------------------------------------- engine rules
def test_max_position():
    eng = engine(max_position_weight=0.10)
    orders, decs, _ = eng.review([buy(qty=500)], portfolio_with(100_000), snapshot({"AAPL": 100.0}))
    assert decs[0].status == RESIZED and decs[0].approved_quantity == 100
    assert decs[0].rule_triggered == "max_position_weight"
    assert orders[0].quantity == 100


def test_insufficient_cash():
    eng = engine(max_position_weight=1.0, max_gross_exposure=0.95, min_cash_reserve=0.05)
    p = portfolio_with(1_000, {"MSFT": (90, 1000.0)})            # equity 91,000, cash 1,000
    _, decs, _ = eng.review([buy(qty=100)], p, snapshot({"AAPL": 100.0, "MSFT": 1000.0}))
    assert decs[0].status == BLOCKED and not decs[0].approved


def test_max_exposure():
    eng = engine(max_position_weight=1.0, max_gross_exposure=0.50, min_cash_reserve=0.0)
    p = portfolio_with(60_000, {"XOM": (400, 100.0)})              # 40% invested
    _, decs, _ = eng.review([buy("SPY", 500)], p, snapshot({"SPY": 100.0, "XOM": 100.0}))
    assert decs[0].approved_quantity == 100 and decs[0].rule_triggered == "max_gross_exposure"


def test_sector_concentration_and_max_positions():
    eng = engine(max_position_weight=1.0, max_sector_weight=0.30, max_concurrent_positions=2)
    p = portfolio_with(75_000, {"MSFT": (250, 100.0)})            # Tech 25%
    _, decs, _ = eng.review([buy("NVDA", 200)], p, snapshot({"NVDA": 100.0, "MSFT": 100.0}))
    assert decs[0].approved_quantity == 50 and decs[0].rule_triggered == "max_sector_weight"
    p2 = portfolio_with(50_000, {"MSFT": (100, 100.0), "XOM": (100, 100.0)})
    _, decs2, _ = eng.review([buy("AAPL", 10)], p2, snapshot({"AAPL": 100.0, "MSFT": 100.0, "XOM": 100.0}))
    assert decs2[0].status == BLOCKED and decs2[0].rule_triggered == "max_concurrent_positions"


def test_min_price_liquidity_and_bad_price_blocked():
    cfg = cfg_with(limits={"min_avg_dollar_volume": 1e6})
    eng = RiskEngine(cfg.risk)
    feats = pd.DataFrame({"dollar_volume_20": {"AAPL": 10.0, "PENNY": 1e9}})
    _, d, _ = eng.review([buy("AAPL", 10)], portfolio_with(100_000), snapshot({"AAPL": 100.0}, features=feats))
    assert d[0].rule_triggered == "liquidity"
    _, d, _ = eng.review([buy("PENNY", 10)], portfolio_with(100_000), snapshot({"PENNY": 1.0}, features=feats))
    assert d[0].rule_triggered == "min_price"
    for bad in (float("nan"), float("inf"), -5.0):
        _, d, _ = engine().review([buy("AAPL", 10)], portfolio_with(100_000), snapshot({"AAPL": bad}))
        assert d[0].status == BLOCKED


def test_sells_always_allowed_even_with_kill_switch():
    eng = engine()
    eng.drawdown.state.kill_switch = True
    p = portfolio_with(10_000, {"AAPL": (50, 100.0)})
    orders, decs, _ = eng.review([OrderProposal("AAPL", Side.SELL, 80, "exit", "T", "EXIT")], p,
                                 snapshot({"AAPL": 100.0}))
    assert decs[0].approved_quantity == 50 and orders[0].side == Side.SELL       # capped at held
    _, d, _ = eng.review([buy()], p, snapshot({"AAPL": 100.0}))
    assert d[0].status == BLOCKED and d[0].rule_triggered == "kill_switch"


def test_risk_decision_fields_and_rejections_recorded():
    eng = engine(max_concurrent_positions=0)
    _, decs, events = eng.review([buy()], portfolio_with(100_000), snapshot({"AAPL": 100.0}))
    d = decs[0]
    assert d.approved is False and d.reason and d.rule_triggered == "max_concurrent_positions"
    assert d.timestamp == D0
    assert events and events[0]["event"] == "ORDER_BLOCKED"


def test_vol_limit_scales_buy():
    cfg = cfg_with(limits={"min_avg_dollar_volume": 0, "max_portfolio_volatility": 0.10, "max_position_weight": 1.0,
                           "min_cash_reserve": 0.0, "max_gross_exposure": 1.0},
                   safety={"max_order_notional": 1e9})
    eng = RiskEngine(cfg.risk)
    import numpy as np
    rets = pd.DataFrame({"AAPL": np.random.default_rng(1).normal(0, 0.02, 80)},
                        index=pd.bdate_range("2023-09-01", periods=80))         # ~32% annual vol
    snap = snapshot({"AAPL": 100.0})
    snap.returns = rets
    _, d, _ = eng.review([buy("AAPL", 900)], portfolio_with(100_000), snap)
    assert d[0].status == RESIZED and d[0].rule_triggered == "max_portfolio_volatility"
    w = d[0].approved_quantity * 100 / 100_000
    assert w < 0.9


def test_portfolio_vol_and_contributions():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = {"A": 0.5, "B": 0.5}
    assert portfolio_volatility(w, cov) == pytest.approx(math.sqrt(0.25 * 0.04 + 0.25 * 0.09))
    rc = risk_contributions(w, cov)
    assert sum(rc.values()) == pytest.approx(1.0)
    assert rc["B"] > rc["A"]


# ------------------------------------------------------------ stops / drawdown
def test_stop_loss_rules():
    from src.portfolio.portfolio import Position
    pos = Position("AAPL", 10, 100.0, pd.Timestamp("2024-01-02"), 100.0, 120.0)
    sl = StopLossManager(True, 0.10, 3.0, 30)
    assert sl.check(pos, 89.0, None, pd.Timestamp("2024-01-10")).rule == "fixed_stop"
    assert sl.check(pos, 112.0, 2.5, pd.Timestamp("2024-01-10")).rule == "trailing_atr_stop"   # <= 120 - 7.5
    assert sl.check(pos, 115.0, 2.5, pd.Timestamp("2024-02-05")).rule == "time_stop"
    assert sl.check(pos, 115.0, 2.5, pd.Timestamp("2024-01-10")) is None
    assert StopLossManager(False).check(pos, 1.0, 1.0, pd.Timestamp("2024-01-10")) is None
    assert sl.check(pos, float("nan"), 2.5, pd.Timestamp("2024-01-10")) is None


def test_drawdown_limit_and_kill_switch_latches():
    m = DrawdownMonitor(max_drawdown=0.20, max_daily_loss=None)
    days = pd.bdate_range("2024-01-01", periods=6)
    for d, e in zip(days, [100, 110, 95, 88, 120, 130], strict=True):
        m.update(d, e)
    assert m.state.kill_switch is True                 # 88 / 110 - 1 = -20%
    assert m.state.kill_switch_since == days[3]
    assert m.buys_blocked                              # still on after recovery: latched
    ev = m.manual_reset(days[5], 130)
    assert ev[0]["event"] == "KILL_SWITCH_RESET" and not m.state.kill_switch and m.state.peak == 130


def test_kill_switch_auto_reset_and_daily_loss():
    m = DrawdownMonitor(0.10, max_daily_loss=0.05, reset={"days": 2})
    d = pd.bdate_range("2024-01-01", periods=6)
    m.update(d[0], 100)
    ev = m.update(d[1], 94)
    assert {e["event"] for e in ev} == {"DAILY_LOSS_LIMIT"} and m.buys_blocked
    m.update(d[2], 89)
    assert m.state.kill_switch
    m.update(d[3], 90)
    m.update(d[4], 91)
    assert not m.state.kill_switch                     # auto reset after 2 bars


def test_drawdown_state_round_trip():
    m = DrawdownMonitor(0.2)
    m.update(D0, 100)
    m.update(D0 + pd.Timedelta(days=1), 70)
    m2 = DrawdownMonitor(0.2).load_dict(m.to_dict())
    assert m2.state.kill_switch and m2.state.peak == 100


# ----------------------------------------------------- account-level manager
def test_portfolio_manager_caps_symbol_across_bots_and_global_kill_switch():
    cfg = cfg_with(portfolio={"max_symbol_weight": 0.20, "max_gross_exposure": 0.95, "max_total_positions": 10})
    prm = PortfolioRiskManager.from_config(cfg.risk)
    a = portfolio_with(40_000, {"AAPL": (100, 100.0)})       # bot A holds $10k AAPL
    b = Portfolio(cash=50_000)
    acct = AccountState({"A": a, "B": b})                   # equity 100k
    o = approved_order("AAPL", Side.BUY, 200, 100.0, bot_id="B")
    approved, decs, _ = prm.review([o], acct, {"AAPL": 100.0}, D0)
    assert approved[0].quantity == 100                       # 20% cap: 10k held + 10k more
    assert decs[0].rule_triggered == "max_symbol_weight"
    prm.drawdown.state.kill_switch = True
    o2 = approved_order("MSFT", Side.BUY, 10, 100.0, bot_id="B")
    approved, decs, ev = prm.review([o2], acct, {"MSFT": 100.0}, D0)
    assert approved == [] and o2.status == OrderStatus.REJECTED and decs[0].rule_triggered == "global_kill_switch"
    s = approved_order("AAPL", Side.SELL, 100, 100.0, bot_id="A")
    approved, _, _ = prm.review([s], acct, {"AAPL": 100.0}, D0)
    assert approved == [s]                                   # sells always pass
