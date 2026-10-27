"""Metric MATH against hand-computed values. No profitability assertions."""
import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting import metrics as M


def eq(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_total_return_and_drawdown_exact():
    e = eq([100, 110, 99, 120, 90, 95])
    assert M.total_return(e) == pytest.approx(-0.05)
    assert M.max_drawdown(e) == pytest.approx(90 / 120 - 1)      # -25%
    dd = M.drawdown_series(e)
    assert dd.iloc[2] == pytest.approx(99 / 110 - 1)


def test_sharpe_and_sortino_exact():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    mean, sd = r.mean(), r.std(ddof=1)
    assert M.sharpe_ratio(r) == pytest.approx(mean / sd * math.sqrt(252))
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert M.sortino_ratio(r) == pytest.approx(mean / downside * math.sqrt(252))
    rf = 0.0252
    assert M.sharpe_ratio(r, rf) == pytest.approx((r - rf / 252).mean() / sd * math.sqrt(252))


def test_cagr_calmar_and_vol():
    idx = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])          # 731 days
    e = pd.Series([100.0, 121.0], index=idx)
    yrs = 731 / 365.25
    assert M.cagr(e) == pytest.approx(1.21 ** (1 / yrs) - 1)
    e2 = eq([100, 120, 90, 150])
    assert M.calmar_ratio(e2) == pytest.approx(M.cagr(e2) / 0.25)
    r = pd.Series([0.01, -0.01] * 10)
    assert M.annualised_volatility(r) == pytest.approx(r.std(ddof=1) * math.sqrt(252))


def test_trade_stats_exact():
    s = M.trade_stats([100, -50, 200, -25, 0])
    assert s["num_trades"] == 5
    assert s["win_rate"] == pytest.approx(2 / 5)
    assert s["avg_win"] == pytest.approx(150)
    assert s["avg_loss"] == pytest.approx(-25)                      # (-50 - 25 + 0) / 3
    assert s["profit_factor"] == pytest.approx(300 / 75)
    assert s["expectancy"] == pytest.approx(45)
    assert M.trade_stats([])["num_trades"] == 0
    assert M.trade_stats([10, 20])["profit_factor"] == math.inf


def test_turnover_and_benchmark_relative():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
    e = pd.Series([100.0, 100.0], index=idx)
    assert M.turnover(366.0, e) == pytest.approx(366 / 100 / (366 / 365.25))
    b = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    r = 2 * b                                                       # beta exactly 2, zero alpha
    st = M.benchmark_stats(r, b)
    assert st["beta"] == pytest.approx(2.0)
    assert st["alpha_annual"] == pytest.approx(0.0, abs=1e-12)
    assert st["correlation"] == pytest.approx(1.0)


def test_degenerate_inputs_do_not_produce_nan_or_crash():
    flat = eq([100] * 10)
    m = M.compute_metrics(flat)
    for k in ("sharpe", "sortino", "max_drawdown", "calmar", "annualised_volatility"):
        assert m[k] == 0.0
    assert M.compute_metrics(eq([100]))["total_return"] == 0.0


def test_monthly_returns_table():
    e = pd.Series([100.0, 110.0, 99.0], index=pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"]))
    t = M.monthly_returns(e)
    assert t.loc[2020, 2] == pytest.approx(0.10)
    assert t.loc[2020, 3] == pytest.approx(-0.10)
