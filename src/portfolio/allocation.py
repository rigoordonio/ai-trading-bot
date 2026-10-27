"""Turn actions into order PROPOSALS (quantities) for one bot's sleeve.

    BUY    -> size with the bot's sizing method (fixed_fractional / volatility / atr)
    REDUCE -> sell (1 - reduce_fraction) of the current position, once
    EXIT   -> sell everything

Proposals are requests; the risk engines decide what is actually allowed.
"""
from __future__ import annotations

import math

import pandas as pd

from src.execution.orders import Side
from src.portfolio.portfolio import Portfolio
from src.risk.engine import OrderProposal
from src.risk.position_sizing import size_position
from src.strategies.base import BUY, EXIT, REDUCE


def propose(action: str, symbol: str, portfolio: Portfolio, price: float, features: pd.Series | None,
            sizing_cfg, bot_id: str, score: float | None, components: dict, reduce_fraction: float = 0.5,
            reason: str = "") -> OrderProposal | None:
    if action == EXIT:
        qty = portfolio.quantity(symbol)
        if qty <= 0:
            return None
        return OrderProposal(symbol, Side.SELL, qty, reason or f"{bot_id} EXIT (score {score:.2f})", bot_id, EXIT,
                             score, components)
    if action == REDUCE:
        held = portfolio.quantity(symbol)
        qty = int(math.floor(held * (1 - reduce_fraction)))
        if qty <= 0:
            return None
        return OrderProposal(symbol, Side.SELL, qty, reason or f"{bot_id} REDUCE (score {score:.2f})", bot_id,
                             REDUCE, score, components)
    if action == BUY:
        f = features if features is not None else pd.Series(dtype=float)
        qty = size_position(
            sizing_cfg.method, portfolio.equity, price,
            fraction=sizing_cfg.fixed_fraction,
            asset_vol=float(f.get(sizing_cfg.vol_lookback_feature, float("nan"))),
            target_vol=sizing_cfg.target_position_vol,
            atr=float(f.get("atr_14", float("nan"))),
            risk_per_trade=sizing_cfg.atr_risk_per_trade,
            stop_multiple=sizing_cfg.atr_stop_multiple)
        if qty <= 0:
            return None
        return OrderProposal(symbol, Side.BUY, int(qty), reason or f"{bot_id} BUY (score {score:.2f})", bot_id, BUY,
                             score, components)
    return None
