"""Financial correctness of the ENGINE itself (exact numbers, deterministic data).

These tests check arithmetic, not profitability."""
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine, BotInput, StrategyBot
from src.config import load_config
from src.execution.orders import OrderStatus, Side
from src.execution.simulator import CostModel, ExecutionSimulator
from src.portfolio.portfolio import Portfolio
from tests.helpers import approved_order, bar, flat_prices, scripted_signal

ZERO = CostModel(0, 0, 0, 0, 0, 0, 1.0)


def _fill(order, b, cash, held, costs=ZERO):
    order.transition(OrderStatus.VALIDATED)
    order.transition(OrderStatus.SUBMITTED)
    return ExecutionSimulator(costs).execute(order, b, cash, held, pd.Timestamp("2024-01-03"))


# --------------------------------------------------- the spec's worked example
def test_buy_10_at_100_sell_10_at_110_no_costs_exact():
    p = Portfolio(cash=10_000.0)
    buy = _fill(approved_order("XYZ", Side.BUY, 10, 100.0), bar(o=100, h=100, low=100, c=100), p.cash, 0)
    p.apply_fill(buy)
    assert p.cash == pytest.approx(9_000.0)
    assert p.quantity("XYZ") == 10
    p.mark_to_market({"XYZ": 105.0})
    assert p.positions["XYZ"].unrealized_pnl == pytest.approx(50.0)
    assert p.equity == pytest.approx(10_050.0)
    sell = _fill(approved_order("XYZ", Side.SELL, 10, 110.0), bar(o=110, h=110, low=110, c=110), p.cash, 10)
    trade = p.apply_fill(sell, reason="exit")
    assert trade.pnl == pytest.approx(100.0)
    assert p.cash == pytest.approx(10_100.0)
    assert p.quantity("XYZ") == 0 and "XYZ" not in p.positions
    assert p.realized_pnl == pytest.approx(100.0)


def test_same_round_trip_with_commission_and_slippage_exact():
    costs = CostModel(commission_per_share=0.0, commission_min=1.0, commission_pct=0.0, half_spread_bps=0.0,
                      slippage_bps=10.0, impact_coefficient=0.0, max_participation=1.0)
    p = Portfolio(cash=10_000.0)
    buy = _fill(approved_order("XYZ", Side.BUY, 10, 100.0), bar(o=100, h=100, low=100, c=100), p.cash, 0, costs)
    assert buy.price == pytest.approx(100.10)          # +10 bps
    assert buy.commission == pytest.approx(1.0)
    p.apply_fill(buy)
    assert p.cash == pytest.approx(10_000 - 1001.0 - 1.0)
    sell = _fill(approved_order("XYZ", Side.SELL, 10, 110.0), bar(o=110, h=110, low=110, c=110), p.cash, 10, costs)
    assert sell.price == pytest.approx(109.89)         # -10 bps
    trade = p.apply_fill(sell)
    # gross (109.89 - 100.10) * 10 = 97.90, minus 2 x $1 commission = 95.90
    assert trade.pnl == pytest.approx(95.90)
    assert p.cash == pytest.approx(10_095.90)
    assert p.total_commission == pytest.approx(2.0)


def test_spread_slippage_impact_components():
    c = CostModel(0.005, 1.0, 0.0, 2.0, 5.0, 10.0, 1.0)
    px, parts = c.execution_price(Side.BUY, 100.0, 1_000, 100_000)   # 1% participation -> impact 10 bps
    assert parts["spread"] == pytest.approx(100 * 2e-4 * 1000)
    assert parts["slippage"] == pytest.approx(100 * 5e-4 * 1000)
    assert parts["impact"] == pytest.approx(100 * 10e-4 * 1000)
    assert px == pytest.approx(100 * (1 + 17e-4))
    assert c.commission(1_000, px) == pytest.approx(5.0)
    assert c.commission(10, px) == pytest.approx(1.0)     # minimum


def test_participation_cap_partial_fill_and_cash_shrink():
    costs = CostModel(0, 0, 0, 0, 0, 0, max_participation=0.01)
    o = approved_order("XYZ", Side.BUY, 500, 100.0)
    f = _fill(o, bar(o=100, h=100, low=100, c=100, v=10_000), 1e9, 0, costs)
    assert f.quantity == 100 and o.status == OrderStatus.PARTIALLY_FILLED
    o2 = approved_order("XYZ", Side.BUY, 50, 100.0)
    f2 = _fill(o2, bar(o=110, h=110, low=110, c=110), 2_200.0, 0)     # gap up: only 20 affordable
    assert f2.quantity == 20 and f2.notional <= 2_200.0


def test_rejections_no_bar_no_cash_no_position():
    o = approved_order("XYZ", Side.BUY, 10, 100.0)
    assert _fill(o, None, 1e6, 0) is None and o.status == OrderStatus.REJECTED
    o = approved_order("XYZ", Side.BUY, 10, 100.0)
    assert _fill(o, bar(), 50.0, 0) is None and "cash" in o.reject_reason
    o = approved_order("XYZ", Side.SELL, 10, 100.0)
    assert _fill(o, bar(), 0.0, 0) is None and "shorting" in o.reject_reason
    o = approved_order("XYZ", Side.BUY, 10, 100.0)
    assert _fill(o, bar(v=0), 1e6, 0) is None


def test_average_cost_and_partial_sell():
    p = Portfolio(cash=100_000.0)
    p.apply_fill(_fill(approved_order("XYZ", Side.BUY, 10, 100.0), bar(o=100, h=100, low=100, c=100), p.cash, 0))
    p.apply_fill(_fill(approved_order("XYZ", Side.BUY, 10, 120.0, date=pd.Timestamp("2024-01-04")),
                       bar(o=120, h=120, low=120, c=120), p.cash, 10))
    assert p.positions["XYZ"].avg_cost == pytest.approx(110.0)
    t = p.apply_fill(_fill(approved_order("XYZ", Side.SELL, 5, 130.0), bar(o=130, h=130, low=130, c=130), p.cash, 20))
    assert t is None and p.quantity("XYZ") == 15
    assert p.realized_pnl == pytest.approx(5 * 20.0)
    from src.execution.orders import Fill

    with pytest.raises(ValueError):   # selling more than held must never be booked
        p.apply_fill(Fill("x", "XYZ", Side.SELL, 99, 130.0, 130.0, 0.0, pd.Timestamp("2024-01-05")))


# ------------------------------------------------------------ engine end-to-end
def _engine(closes, scores, delay=1, risk_patch=None):
    from src.config import deep_merge

    overrides = {
        "settings": {"backtest": {"execution_delay_bars": delay, "initial_capital": 10_000}},
        "risk": {"limits": {"max_position_weight": 1.0, "max_gross_exposure": 1.0, "min_cash_reserve": 0.0,
                            "min_price": 0.0, "min_avg_dollar_volume": 0, "max_portfolio_volatility": 10.0,
                            "min_trade_value": 0},
                 "position_sizing": {"method": "fixed_fractional", "fixed_fraction": 0.5},
                 "stops": {"enabled": False}, "drawdown": {"max_daily_loss": None},
                 "portfolio": {"max_symbol_weight": 1.0, "max_gross_exposure": 1.0, "max_daily_loss": None},
                 "safety": {"max_order_notional": 1e9}}}
    if risk_patch:
        overrides["risk"] = deep_merge(overrides["risk"], risk_patch)
    cfg = load_config(overrides=overrides)
    prices = flat_prices(["XYZ"], {"XYZ": closes})
    dates = prices["XYZ"].index
    idx = pd.MultiIndex.from_product([dates, ["XYZ"]], names=["date", "symbol"])
    panel = pd.DataFrame({"close": closes, "dollar_volume_20": 1e9, "atr_14": 1.0, "realized_vol_20": 0.2},
                         index=idx)
    sig = scripted_signal(dates, ["XYZ"], {(dates[i], "XYZ"): v for i, v in scores.items()}, (0.6, 0.6, 0.6),
                          default=0.0)
    bot = StrategyBot("TEST", ["XYZ"], cfg.risk, [], "daily")
    return BacktestEngine(cfg, prices, [BotInput(bot, panel, sig)], start=dates[0], end=dates[-1],
                          cost_model=ZERO, benchmark="NONE").run()


def test_engine_executes_at_next_open_never_same_bar():
    closes = [100, 100, 120, 130, 130, 130]
    res = _engine(closes, {1: 1.0, 2: 1.0, 3: 0.0})     # BUY decided day1 close, EXIT decided day3 close
    fills = res.fills
    assert list(fills["side"]) == ["BUY", "SELL"]
    # decided on day 1 (close 100) -> executed at day 2 OPEN (= 120 here), not at 100
    assert pd.Timestamp(fills.iloc[0]["timestamp"]) == res.equity.index[2]
    assert fills.iloc[0]["price"] == pytest.approx(120.0)
    assert pd.Timestamp(fills.iloc[1]["timestamp"]) == res.equity.index[4]
    # 50% of $10,000 at decision close 100 -> 50 shares proposed; filled at 120: 50*120 = 6000 <= cash
    assert fills.iloc[0]["quantity"] == 50
    tr = res.trades.iloc[0]
    assert tr["pnl"] == pytest.approx(50 * (130 - 120))
    assert res.equity["equity"].iloc[-1] == pytest.approx(10_000 + 500)


def test_engine_delay_two_bars():
    res = _engine([100, 100, 100, 110, 120, 120, 120], {1: 1.0}, delay=2)
    assert pd.Timestamp(res.fills.iloc[0]["timestamp"]) == res.equity.index[3]
    assert res.fills.iloc[0]["price"] == pytest.approx(110.0)


def test_account_level_symbol_cap_resizes_and_is_audited():
    """Same scenario with the default account-wide 25% symbol cap: resized, and the audit says why."""
    from tests import test_backtest_engine as m
    cfg_patch = {"portfolio": {"max_symbol_weight": 0.25}}
    res = _engine([100, 100, 120, 130, 130, 130], {1: 1.0, 2: 1.0, 3: 0.0}, risk_patch=cfg_patch)
    assert res.fills.iloc[0]["quantity"] == 25
    row = res.audit[res.audit["final_action"] == "BUY"].iloc[0]
    assert row["quantity"] == 25 and row["rule_triggered"] == "max_symbol_weight"
    assert m is not None


def test_engine_no_signal_no_trades_and_cash_constant():
    res = _engine([100] * 8, {})
    assert res.fills.empty and res.trades.empty
    assert (res.equity["equity"] == 10_000).all()
    assert (res.audit["final_action"] == "NO_TRADE").all() if len(res.audit) else True
