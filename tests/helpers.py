"""Deterministic builders shared by tests (no network, no randomness unless seeded)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.execution.orders import Order, OrderType, Side
from src.portfolio.portfolio import Portfolio
from src.risk.engine import MarketSnapshot, OrderProposal, RiskDecision, RiskEngine
from src.strategies.base import SignalFrame

D0 = pd.Timestamp("2024-01-02")


def cfg_with(**risk_overrides):
    return load_config(overrides={"risk": risk_overrides} if risk_overrides else None)


def bar(o=100.0, h=101.0, low=99.0, c=100.0, v=1_000_000.0) -> pd.Series:
    return pd.Series({"open": o, "high": h, "low": low, "close": c, "volume": v})


def approved_order(symbol="AAPL", side=Side.BUY, qty=10, price=100.0, date=D0, strategy="TEST", risk_cfg=None,
                   order_type=OrderType.MARKET, bot_id=None) -> Order:
    """A genuinely risk-approved order (registered in the RiskEngine approval registry)."""
    rc = risk_cfg or load_config().risk
    eng = RiskEngine(rc)
    prop = OrderProposal(symbol, side, qty, "test", strategy, "BUY" if side == Side.BUY else "EXIT", 0.9,
                         {"bot_id": bot_id or strategy})
    dec = RiskDecision(prop, "PASS", qty)
    o = eng._make_order(prop, qty, MarketSnapshot(date, {symbol: price}, pd.DataFrame()), dec)
    o.order_type = order_type
    return o


def scripted_signal(dates, symbols, scores: dict[tuple, float], thresholds=(0.6, 0.4, 0.25), name="scripted",
                    default=0.5) -> SignalFrame:
    """SignalFrame with explicit scores; anything not given gets ``default``."""
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), symbols], names=["date", "symbol"])
    s = pd.Series(default, index=idx, dtype=float)
    for (d, sym), v in scores.items():
        s.loc[(pd.Timestamp(d), sym)] = v
    b, h, e = thresholds
    return SignalFrame(name, pd.DataFrame({"score": s, name: s}), b, h, e)


def flat_prices(symbols, closes: dict[str, list[float]], start="2024-01-01", volume=10_000_000.0,
                open_equals_close=True) -> dict[str, pd.DataFrame]:
    """Deterministic bars: open == close (so fills at next open are easy to compute)."""
    out = {}
    for s in symbols:
        c = np.asarray(closes[s], dtype=float)
        idx = pd.bdate_range(start, periods=len(c), name="date")
        o = c.copy() if open_equals_close else np.concatenate([[c[0]], c[:-1]])
        out[s] = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                               "adj_close": c, "volume": volume, "dividends": 0.0, "splits": 0.0}, index=idx)
    return out


def portfolio_with(cash: float, positions: dict[str, tuple[int, float]] | None = None, date=D0) -> Portfolio:
    from src.portfolio.portfolio import Position

    p = Portfolio(cash=cash)
    for s, (q, px) in (positions or {}).items():
        p.positions[s] = Position(s, q, px, date, px, px)
    return p


def snapshot(prices: dict[str, float], date=D0, features: pd.DataFrame | None = None) -> MarketSnapshot:
    if features is None:
        features = pd.DataFrame({"dollar_volume_20": {s: 1e9 for s in prices}, "atr_14": {s: 2.0 for s in prices},
                                 "realized_vol_20": {s: 0.2 for s in prices}})
    return MarketSnapshot(date, prices, features, None)
