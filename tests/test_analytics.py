"""Robustness simulations, attribution, result store and dashboard data helpers."""
import numpy as np
import pandas as pd
import pytest

from src.analytics.attribution import cost_breakdown, exit_reason_breakdown, pnl_by
from src.analytics.performance import save_result
from src.analytics.store import ResultsStore
from src.backtesting.robustness import block_bootstrap, summarise, sweep, trade_shuffle


def test_bootstrap_and_shuffle_are_deterministic_and_bounded():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 750))
    a = block_bootstrap(r, 200, 20, seed=1)
    b = block_bootstrap(r, 200, 20, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["max_drawdown"] <= 0).all() and (a["max_drawdown"] >= -1).all()
    s = summarise(a)
    assert {"p05", "median", "p95"} <= set(s.columns) and 0 <= s.loc["cagr", "prob_negative"] <= 1
    t = trade_shuffle([100, -50, 30, -20, 80, -10], 10_000, 100, seed=2)
    assert np.allclose(t["final_equity"], 10_000 + 130)      # order changes the path, not the sum
    with pytest.raises(ValueError):
        trade_shuffle([1, 2], 100)


def test_sweep_collects_rows():
    df = sweep(lambda m: {"total_return": m * 0.1, "sharpe": m}, [0, 1, 2], "cost_multiplier")
    assert list(df["cost_multiplier"]) == [0, 1, 2] and df["sharpe"].tolist() == [0, 1, 2]


def test_attribution():
    trades = pd.DataFrame({"symbol": ["A", "A", "B"], "pnl": [10.0, -5.0, 3.0],
                           "exit_reason": ["fixed_stop: x", "BOT EXIT (score 0.1)", "fixed_stop: y"],
                           "exit_date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-02-03"])})
    by = pnl_by(trades, "symbol")
    assert by.loc["A", "pnl"] == 5 and by.loc["A", "win_rate"] == 0.5
    ex = exit_reason_breakdown(trades)
    assert ex.loc["fixed_stop", "trades"] == 2
    fills = pd.DataFrame({"commission": [1.0], "spread_cost": [2.0], "slippage_cost": [3.0], "impact_cost": [4.0],
                          "quantity": [100], "price": [100.0]})
    c = cost_breakdown(fills)
    assert c["total"] == 10 and c["cost_bps_of_notional"] == pytest.approx(10.0)


def test_store_round_trip(tmp_path):
    from src.backtesting.engine import BacktestResult
    idx = pd.bdate_range("2024-01-01", periods=3, name="date")
    eq = pd.DataFrame({"equity": [100.0, 101, 102], "cash": 100.0, "positions_value": 0.0, "gross_exposure": 0.0,
                       "n_positions": 0}, index=idx)
    res = BacktestResult("RUN1", eq, {"B": eq}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                         pd.DataFrame(), None, {"total_return": 0.02, "sharpe": 1.0, "max_drawdown": 0.0},
                         {"B": {}}, {"bots": ["B"], "start": "2024-01-01", "end": "2024-01-03"})
    save_result(res, tmp_path / "reports")
    st = ResultsStore(tmp_path / "reports", tmp_path / "paper")
    assert st.runs()[0]["run_id"] == "RUN1"
    assert st.run_frame("RUN1", "equity")["equity"].iloc[-1] == 102
    assert st.system_status()["blocking_reasons"] == ["no paper run yet"]


def test_dashboard_market_table(cfg, prices):
    from src.bots.bot import build_bots
    from src.dashboard.data import market_table, overview_kpis
    bot = build_bots(cfg, ["TREND_V1"])["TREND_V1"]
    panel = bot.compute_features(prices)
    t = market_table(bot, panel)                            # no model -> NO SIGNAL (fail closed)
    assert set(t["Signal"]) == {"NO SIGNAL"}
    assert len(t) == len([s for s in bot.symbols if s in prices])
    idx = pd.bdate_range("2024-01-01", periods=3)
    k = overview_kpis(pd.DataFrame({"equity": [100.0, 110, 99], "cash": 50.0, "gross_exposure": 0.5}, index=idx))
    assert k["Max drawdown"] == pytest.approx(99 / 110 - 1) and k["Total P&L"] == pytest.approx(-1)
