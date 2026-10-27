"""Position sizing. Each method returns a whole number of shares (never < 0).

fixed_fractional  shares = equity * fraction / price
volatility        weight = target_vol / asset_vol (capped at 1), shares = equity * weight / price
                  -> a 60%-vol stock gets a quarter of the weight of a 15%-vol stock
atr               shares = (equity * risk_per_trade) / (atr * stop_multiple)
                  -> if price moves `stop_multiple` ATRs against us we lose
                     `risk_per_trade` of equity

All results are later capped by the risk engine's hard limits.
"""
from __future__ import annotations

import math


def _valid(*xs: float | None) -> bool:
    return all(x is not None and x == x and math.isfinite(x) for x in xs)


def _shares(x: float) -> int:
    """Whole shares from a float; NaN/inf/negative -> 0 (e.g. a denormal price
    makes equity / price overflow to infinity)."""
    return int(x) if math.isfinite(x) and x > 0 else 0


def fixed_fractional(equity: float, price: float, fraction: float) -> int:
    if not _valid(equity, price, fraction) or price <= 0 or equity <= 0 or fraction <= 0:
        return 0
    return _shares(equity * fraction // price)


def volatility_target(equity: float, price: float, asset_vol: float | None, target_vol: float,
                      max_weight: float = 1.0) -> int:
    if asset_vol is None or not _valid(equity, price, asset_vol, target_vol) or price <= 0 or asset_vol <= 0 \
            or equity <= 0:
        return 0
    weight = min(target_vol / asset_vol, max_weight)
    return _shares(equity * weight // price)


def atr_based(equity: float, price: float, atr: float | None, risk_per_trade: float, stop_multiple: float) -> int:
    if atr is None or not _valid(equity, price, atr, risk_per_trade, stop_multiple) or atr <= 0 or price <= 0 \
            or equity <= 0:
        return 0
    return _shares((equity * risk_per_trade) / (atr * stop_multiple))


def size_position(method: str, equity: float, price: float, *, fraction: float = 0.1,
                  asset_vol: float | None = None, target_vol: float = 0.15, atr: float | None = None,
                  risk_per_trade: float = 0.01, stop_multiple: float = 2.0) -> int:
    if method == "fixed_fractional":
        return fixed_fractional(equity, price, fraction)
    if method == "volatility":
        return volatility_target(equity, price, asset_vol, target_vol)
    if method == "atr":
        return atr_based(equity, price, atr, risk_per_trade, stop_multiple)
    raise ValueError(f"unknown sizing method '{method}'")
