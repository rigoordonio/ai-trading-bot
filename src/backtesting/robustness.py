"""Robustness analysis: report DISTRIBUTIONS, not one lucky backtest.

1. Sensitivity sweeps (re-run the backtest): transaction-cost multipliers,
   strategy parameters, start dates.
2. Return bootstrap (stationary block bootstrap of daily returns).
   Assumes the return-generating process is stationary and that serial
   dependence fits within the block length. It cannot create regimes that
   never occurred in the sample.
3. Trade-sequence Monte Carlo (shuffle closed-trade P&Ls). Assumes trades
   are independent and position sizes comparable; it only tests sequencing
   risk (drawdown depth), not whether the edge is real.

Neither simulation proves a strategy works; a wide distribution or a large
share of losing paths is a warning, a narrow positive one is not a guarantee.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.backtesting.metrics import TRADING_DAYS


def sweep(run: Callable[[object], dict], values: Iterable, label: str = "value") -> pd.DataFrame:
    """Call ``run(v)`` -> metrics dict for each value; returns one row per value."""
    rows = []
    for v in values:
        m = run(v)
        rows.append({label: v if not isinstance(v, dict) else str(v), **{k: m.get(k) for k in (
            "total_return", "annualised_return", "sharpe", "max_drawdown", "num_trades", "turnover")}})
    return pd.DataFrame(rows)


def _max_dd(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1).min())


def block_bootstrap(returns: pd.Series, n_sims: int = 1000, block: int = 20, seed: int = 0) -> pd.DataFrame:
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        raise ValueError("not enough returns for a block bootstrap")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sims):
        idx: list[int] = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend(((start + np.arange(rng.geometric(1 / block))) % n).tolist())
        sim = r[np.array(idx[:n])]
        path = np.cumprod(1 + sim)
        yrs = n / TRADING_DAYS
        sd = sim.std(ddof=1)
        out.append({"cagr": path[-1] ** (1 / yrs) - 1, "max_drawdown": _max_dd(path),
                    "sharpe": sim.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0, "final_multiple": path[-1]})
    return pd.DataFrame(out)


def trade_shuffle(pnls: Iterable[float], start_equity: float, n_sims: int = 1000, seed: int = 0) -> pd.DataFrame:
    p = np.asarray(list(pnls), dtype=float)
    if len(p) < 5:
        raise ValueError("need at least 5 trades")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sims):
        path = start_equity + np.cumsum(rng.permutation(p))
        path = np.concatenate([[start_equity], path])
        out.append({"final_equity": path[-1], "max_drawdown": _max_dd(np.maximum(path, 1e-9))})
    return pd.DataFrame(out)


def summarise(dist: pd.DataFrame) -> pd.DataFrame:
    q = dist.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    q.columns = ["p05", "p25", "median", "p75", "p95"]
    q["mean"] = dist.mean()
    if "cagr" in dist:
        q.loc["cagr", "prob_negative"] = float((dist["cagr"] < 0).mean())
    return q
