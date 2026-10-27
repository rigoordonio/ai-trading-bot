"""Edge paths of the trading cycle, simulator limit orders, paper broker state."""
import math

import pandas as pd
import pytest

from src.backtesting.engine import StrategyBot
from src.config import load_config
from src.execution.orders import OrderStatus, OrderType, Side
from src.execution.paper_broker import PaperBroker
from src.execution.simulator import CostModel, ExecutionSimulator
from src.portfolio.risk_manager import PortfolioRiskManager
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import RiskEngine
from src.risk.exposure import gross_exposure, largest_position, net_exposure, sector_weights
from src.risk.stop_loss import StopLossManager
from src.safety.health import HealthMonitor, TradingGate
from src.safety.order_validator import OrderValidator
from src.trading_cycle import BotRuntime, MarketState, TradingCycle, is_rebalance_day
from tests.helpers import D0, approved_order, bar, portfolio_with, scripted_signal

SYMS = ["AAPL", "MSFT"]


def make_cycle(sleeve, scores=None, freq="daily", active=lambda s, d: True, risk_patch=None, prev=None):
    cfg = load_config(overrides={"risk": {"limits": {"min_avg_dollar_volume": 0, "max_portfolio_volatility": 10},
                                          **(risk_patch or {})}})
    sig = scripted_signal([D0], SYMS, scores or {}, default=0.5)
    feats = pd.DataFrame({"atr_14": {s: 2.0 for s in SYMS}, "realized_vol_20": {s: 0.2 for s in SYMS},
                          "close": {s: 100.0 for s in SYMS}, "dollar_volume_20": {s: 1e9 for s in SYMS}})
    bot = StrategyBot("BOT", SYMS, cfg.risk, ["close"], freq)
    gate = TradingGate(HealthMonitor(), operator_enabled=True)
    rt = BotRuntime(bot, sig, lambda d: feats, sleeve,
                    RiskEngine(cfg.risk, {}, DrawdownMonitor.from_config(cfg.risk.drawdown)),
                    StopLossManager.from_config(cfg.risk.stops), active)
    cyc = TradingCycle({"BOT": rt}, PortfolioRiskManager.from_config(cfg.risk), OrderValidator.from_config(cfg.risk),
                       gate, cfg.risk.safety, set(SYMS))
    cyc._prev_date = prev
    return cyc, gate, rt


def market(closes=None, date=D0):
    closes = closes or {s: 100.0 for s in SYMS}
    bars = {s: bar(o=c, h=c * 1.01, low=c * 0.99, c=c) for s, c in closes.items()}
    return MarketState(date, bars, {s: date for s in closes}, {s: 100.0 for s in closes})


def test_corrupt_portfolio_state_fails_closed():
    p = portfolio_with(float("nan"))
    cyc, gate, _ = make_cycle(p, {(D0, "AAPL"): 1.0})
    res = cyc.run(market())
    assert res.orders == [] and not gate.trading_enabled


def test_portfolio_risk_manager_failure_rejects_everything(monkeypatch):
    cyc, gate, _ = make_cycle(portfolio_with(100_000), {(D0, "AAPL"): 1.0})
    monkeypatch.setattr(PortfolioRiskManager, "review", lambda *a, **k: 1 / 0)
    res = cyc.run(market())
    assert res.orders == [] and res.rejected and not gate.trading_enabled


def test_held_position_with_invalid_bar_is_not_traded():
    p = portfolio_with(50_000, {"AAPL": (100, 100.0)})
    cyc, _, _ = make_cycle(p, {(D0, "AAPL"): 0.0})
    m = market()
    m.bars["AAPL"]["close"] = float("nan")
    res = cyc.run(m)
    assert all(o.symbol != "AAPL" for o in res.orders)
    assert any("held position not evaluated" in (a.get("reason") or "") for a in res.audit)


def test_stop_loss_exit_generated_by_cycle():
    p = portfolio_with(50_000, {"AAPL": (100, 120.0)})            # bought at 120, now 100: -16.7%
    cyc, _, _ = make_cycle(p, {(D0, "AAPL"): 1.0})                 # signal still bullish
    res = cyc.run(market())
    sells = [o for o in res.orders if o.symbol == "AAPL"]
    assert sells and sells[0].side == Side.SELL and "fixed_stop" in sells[0].reason


def test_kill_switch_liquidation_when_configured():
    p = portfolio_with(50_000, {"AAPL": (100, 100.0)})
    cyc, gate, rt = make_cycle(p, risk_patch={"drawdown": {"liquidate_on_kill_switch": True},
                                              "safety": {"allow_risk_reducing_orders_when_halted": True}})
    rt.risk.drawdown.state.kill_switch = True
    res = cyc.run(market())
    assert [o.side for o in res.orders if o.symbol == "AAPL"] == [Side.SELL]
    assert all(o.side == Side.SELL for o in res.orders)


def test_leaving_point_in_time_universe_forces_exit():
    p = portfolio_with(50_000, {"MSFT": (100, 100.0)})
    cyc, _, _ = make_cycle(p, {(D0, "MSFT"): 1.0}, active=lambda s, d: s != "MSFT")
    res = cyc.run(market())
    msft = [o for o in res.orders if o.symbol == "MSFT"]
    assert msft and msft[0].side == Side.SELL and "universe" in msft[0].reason


def test_weekly_bot_only_trades_on_rebalance_days():
    thursday, friday, monday = pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")
    assert not is_rebalance_day("weekly", friday, thursday)
    assert is_rebalance_day("weekly", monday, friday)
    assert is_rebalance_day("monthly", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-01-31"))
    with pytest.raises(ValueError):
        is_rebalance_day("hourly", monday, friday)


def test_gate_closed_rejects_validated_orders():
    cyc, gate, _ = make_cycle(portfolio_with(100_000), {(D0, "AAPL"): 1.0})
    gate.operator_enabled = False
    res = cyc.run(market())
    assert res.orders == [] and res.rejected and all(o.status == OrderStatus.REJECTED for o in res.rejected)


# ------------------------------------------------------------ simulator limits
def sub(o):
    o.transition(OrderStatus.VALIDATED)
    o.transition(OrderStatus.SUBMITTED)
    return o


ZERO = CostModel(0, 0, 0, 0, 0, 0, 1.0)


def test_limit_orders():
    sim = ExecutionSimulator(ZERO)
    o = sub(approved_order(qty=10, order_type=OrderType.LIMIT))
    o.limit_price = 101.0
    f = sim.execute(o, bar(o=100, h=102, low=99, c=101), 1e6, 0, D0)
    assert f.price == pytest.approx(100.0)                       # marketable at the open
    o = sub(approved_order(qty=10, order_type=OrderType.LIMIT))
    o.limit_price = 99.5
    f = sim.execute(o, bar(o=100, h=102, low=99, c=101), 1e6, 0, D0)
    assert f.price == pytest.approx(99.5)                        # touched intraday
    o = sub(approved_order(qty=10, order_type=OrderType.LIMIT))
    o.limit_price = 95.0
    assert sim.execute(o, bar(o=100, h=102, low=99, c=101), 1e6, 0, D0) is None
    s = sub(approved_order(side=Side.SELL, qty=10, order_type=OrderType.LIMIT))
    s.limit_price = 101.5
    f = sim.execute(s, bar(o=100, h=102, low=99, c=101), 0, 10, D0)
    assert f.price == pytest.approx(101.5)
    s2 = sub(approved_order(side=Side.SELL, qty=10, order_type=OrderType.LIMIT))
    s2.limit_price = 110.0
    assert sim.execute(s2, bar(o=100, h=102, low=99, c=101), 0, 10, D0) is None


def test_execute_requires_submitted_state():
    with pytest.raises(ValueError):
        ExecutionSimulator().execute(approved_order(), bar(), 1e6, 0, D0)


def test_cost_model_scaling():
    c = CostModel().scaled(2.0)
    assert c.slippage_bps == pytest.approx(10.0) and c.max_participation == CostModel().max_participation
    assert CostModel().commission(0, 100) == 0.0


# --------------------------------------------------------------- paper broker
def test_paper_broker_cancel_persist_and_sell_first(tmp_path):
    b = PaperBroker(10_000, CostModel(0, 0, 0, 0, 0, 0, 1.0), state_path=tmp_path / "s.json")
    o = approved_order(qty=50, price=100.0)
    bo = b.submit_order(o)
    assert b.cancel_order(bo.broker_order_id).status == "CANCELLED"
    o2 = approved_order(qty=50, price=100.0, date=D0 + pd.Timedelta(days=1))
    b.submit_order(o2)
    b2 = PaperBroker(cost_model=CostModel(0, 0, 0, 0, 0, 0, 1.0), state_path=tmp_path / "s.json")  # restart
    fills = b2.process_open(D0 + pd.Timedelta(days=2), {"AAPL": bar(o=100, h=100, low=100, c=100)})
    assert len(fills) == 1 and b2.positions["AAPL"].quantity == 50 and b2.cash == pytest.approx(5_000)
    from src.errors import OrderRejectedError
    with pytest.raises(OrderRejectedError):
        b2.cancel_order("nope")
    acct = b2.get_account()
    assert math.isclose(acct.equity, 10_000) and acct.buying_power == pytest.approx(5_000)


def test_exposure_helpers():
    w = {"A": 0.3, "B": 0.2, "SPY": 0.4}
    assert gross_exposure(w) == pytest.approx(0.9) and net_exposure(w) == pytest.approx(0.9)
    assert largest_position(w) == ("SPY", 0.4) and largest_position({}) == (None, 0.0)
    assert sector_weights(w, {"A": "Tech", "B": "Tech", "SPY": "ETF"}) == {"Tech": pytest.approx(0.5)}


@pytest.mark.slow
def test_research_backtest_workflow(cfg, prices):
    from src.workflows import research_backtest
    res = research_backtest(cfg, prices, "trend_following", "2018-01-01", "2019-12-31")
    assert res.metrics["start"] == "2018-01-01" and "sharpe" in res.metrics
    res2 = research_backtest(cfg, prices, "trend_following", "2018-01-01", "2019-12-31", cost_multiplier=3.0)
    if len(res.fills) and len(res2.fills):
        def bps(f):
            return float((f["slippage_cost"] / (f["quantity"] * f["reference_price"])).mean() * 1e4)
        assert bps(res.fills) == pytest.approx(5.0) and bps(res2.fills) == pytest.approx(15.0)
