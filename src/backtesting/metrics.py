"""Performance metrics. Pure functions, deterministic, unit-tested against
hand-computed values (tests/test_metrics.py). No profitability assumptions.

Conventions
    returns         simple daily returns of the equity curve
    annualisation   252 trading days
    Sharpe          mean(excess daily) / std(daily, ddof=1) * sqrt(252)
    Sortino         mean(excess daily) / downside deviation * sqrt(252),
                    downside deviation = sqrt(mean(min(r - rf, 0)^2))
    max drawdown    min over t of equity_t / max(equity_<=t) - 1   (negative)
    Calmar          CAGR / |max drawdown|
    turnover        sum(|traded notional|) / mean(equity) / years   (annual)
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0


def years_between(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    return (index[-1] - index[0]).days / 365.25


def cagr(equity: pd.Series) -> float:
    yrs = years_between(equity.index)
    if yrs <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    ex = returns - risk_free_annual / TRADING_DAYS
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    ex = returns - risk_free_annual / TRADING_DAYS
    dd = math.sqrt(float((np.minimum(ex, 0.0) ** 2).mean()))
    return float(ex.mean() / dd * math.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def calmar_ratio(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity) / abs(mdd)) if mdd < 0 else 0.0


def trade_stats(pnls: Iterable[float]) -> dict:
    p = np.asarray(list(pnls), dtype=float)
    n = len(p)
    wins, losses = p[p > 0], p[p <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "num_trades": n,
        "win_rate": float(len(wins) / n) if n else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
        "expectancy": float(p.mean()) if n else 0.0,
    }


def turnover(traded_notional: float, equity: pd.Series) -> float:
    yrs = years_between(equity.index)
    if yrs <= 0 or equity.mean() <= 0:
        return 0.0
    return float(traded_notional / equity.mean() / yrs)


def benchmark_stats(returns: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(df) < 2:
        return {}
    r, b = df.iloc[:, 0], df.iloc[:, 1]
    var_b = b.var(ddof=1)
    beta = float(r.cov(b) / var_b) if var_b > 0 else 0.0
    alpha = float((r.mean() - beta * b.mean()) * TRADING_DAYS)
    active = r - b
    te = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS))
    ir = float(active.mean() * TRADING_DAYS / te) if te > 0 else 0.0
    return {"beta": beta, "alpha_annual": alpha, "tracking_error": te, "information_ratio": ir,
            "correlation": float(r.corr(b)),
            "excess_return_annual": float(active.mean() * TRADING_DAYS)}


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    return t.pivot(index="year", columns="month", values="ret")


def compute_metrics(equity: pd.Series, trade_pnls: Iterable[float] = (), traded_notional: float = 0.0,
                    benchmark_equity: pd.Series | None = None, risk_free_annual: float = 0.0) -> dict:
    equity = equity.dropna()
    r = returns_from_equity(equity)
    out = {
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "total_return": total_return(equity),
        "annualised_return": cagr(equity),
        "annualised_volatility": annualised_volatility(r),
        "sharpe": sharpe_ratio(r, risk_free_annual),
        "sortino": sortino_ratio(r, risk_free_annual),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "turnover": turnover(traded_notional, equity),
        **trade_stats(trade_pnls),
    }
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        b = benchmark_equity.reindex(equity.index).ffill().dropna()
        out["benchmark_total_return"] = total_return(b)
        out["benchmark_annualised_return"] = cagr(b)
        out["benchmark_sharpe"] = sharpe_ratio(returns_from_equity(b), risk_free_annual)
        out["benchmark_max_drawdown"] = max_drawdown(b)
        out.update(benchmark_stats(r, returns_from_equity(b)))
    return out


def multi_bot_metrics(account: pd.DataFrame, bot_equity: dict[str, pd.DataFrame],
                      allocations: dict[str, float]) -> dict:
    """Portfolio-level exposure, capital utilisation and MEASURED
    diversification between bots (never assumed).

    diversification_ratio = sum_b(w_b * vol_b) / vol(sum of bot equity);
    > 1 means the bots' combination was less volatile than its parts."""
    out: dict = {}
    if account.empty:
        return out
    eq = account["equity"].where(account["equity"] > 0)
    expo = (account["positions_value"] / eq).dropna()
    out["avg_gross_exposure"] = float(expo.mean()) if len(expo) else 0.0
    out["max_gross_exposure"] = float(expo.max()) if len(expo) else 0.0
    alloc = sum(allocations.values())
    out["capital_utilisation"] = float((expo / alloc).mean()) if alloc > 0 and len(expo) else 0.0
    rets = pd.DataFrame({b: returns_from_equity(df["equity"]) for b, df in bot_equity.items()}).dropna(how="all")
    if rets.shape[1] >= 2 and len(rets) > 20:
        corr = rets.corr()
        out["bot_return_correlation"] = {a: {b: float(corr.at[a, b]) for b in corr.columns} for a in corr.index}
        pairs = [corr.iat[i, j] for i in range(len(corr)) for j in range(i + 1, len(corr))]
        out["avg_pairwise_bot_correlation"] = float(np.nanmean(pairs)) if pairs else float("nan")
        total = sum(df["equity"] for df in bot_equity.values())
        w = {b: float((df["equity"] / total).mean()) for b, df in bot_equity.items()}
        vols = {b: annualised_volatility(rets[b].dropna()) for b in rets}
        combo = annualised_volatility(returns_from_equity(total))
        out["diversification_ratio"] = (float(sum(w[b] * vols[b] for b in vols) / combo)
                                        if combo > 0 else float("nan"))
    return out
