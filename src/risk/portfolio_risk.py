"""Portfolio-level risk statistics from trailing returns (past data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def covariance(returns: pd.DataFrame, symbols: list[str], min_obs: int = 20) -> pd.DataFrame | None:
    r = returns.reindex(columns=symbols).dropna(how="all")
    if len(r) < min_obs:
        return None
    return r.cov(min_periods=min_obs).fillna(0.0) * 252


def portfolio_volatility(weights: dict[str, float], cov: pd.DataFrame | None) -> float:
    if not weights or cov is None:
        return 0.0
    syms = [s for s in weights if s in cov.index]
    w = np.array([weights[s] for s in syms])
    c = cov.loc[syms, syms].to_numpy()
    var = float(w @ c @ w)
    return float(np.sqrt(max(var, 0.0)))


def risk_contributions(weights: dict[str, float], cov: pd.DataFrame | None) -> dict[str, float]:
    """Share of total portfolio variance from each position (sums to 1)."""
    if not weights or cov is None:
        return {}
    syms = [s for s in weights if s in cov.index]
    w = np.array([weights[s] for s in syms])
    c = cov.loc[syms, syms].to_numpy()
    var = float(w @ c @ w)
    if var <= 0:
        return {s: 0.0 for s in syms}
    contrib = w * (c @ w) / var
    return dict(zip(syms, contrib.tolist(), strict=True))
