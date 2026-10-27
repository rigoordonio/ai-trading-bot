"""Portfolio accounting: cash, positions, realised/unrealised P&L, round trips.

Average-cost accounting. Commissions are charged to cash immediately and
attributed to the round-trip trade they belong to, so win rate / profit
factor are measured net of costs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.execution.orders import Fill, Side


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: pd.Timestamp
    last_price: float
    highest_close: float
    entry_atr: float | None = None
    reduced: bool = False
    realized_pnl: float = 0.0          # booked so far within this round trip
    costs: float = 0.0                 # commissions within this round trip
    entry_reason: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.quantity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["market_value"], d["unrealized_pnl"] = self.market_value, self.unrealized_pnl
        return d


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int            # peak quantity held
    pnl: float               # net of commissions
    return_pct: float        # pnl / capital deployed at entry
    holding_days: int
    entry_reason: str
    exit_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    _peak_qty: dict[str, int] = field(default_factory=dict)
    _cost_basis: dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------------- values
    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.positions_value

    def weight(self, symbol: str) -> float:
        eq = self.equity
        p = self.positions.get(symbol)
        return p.market_value / eq if p and eq > 0 else 0.0

    def weights(self) -> dict[str, float]:
        eq = self.equity
        return {s: p.market_value / eq for s, p in self.positions.items()} if eq > 0 else {}

    def quantity(self, symbol: str) -> int:
        p = self.positions.get(symbol)
        return p.quantity if p else 0

    # ---------------------------------------------------------------- marks
    def mark_to_market(self, closes: dict[str, float]) -> None:
        for s, p in self.positions.items():
            px = closes.get(s)
            if px is not None and px == px and px > 0:   # skip NaN / missing -> keep last price
                p.last_price = float(px)
                p.highest_close = max(p.highest_close, float(px))

    # ---------------------------------------------------------------- fills
    def apply_fill(self, fill: Fill, reason: str = "", atr: float | None = None) -> ClosedTrade | None:
        self.total_commission += fill.commission
        if fill.side == Side.BUY:
            cost = fill.quantity * fill.price + fill.commission
            if cost > self.cash + 1e-6:
                raise ValueError(f"fill for {fill.symbol} exceeds cash ({cost:.2f} > {self.cash:.2f})")
            self.cash -= cost
            p = self.positions.get(fill.symbol)
            if p is None:
                self.positions[fill.symbol] = Position(
                    fill.symbol, fill.quantity, fill.price, fill.timestamp, fill.price, fill.price,
                    entry_atr=atr, costs=fill.commission, entry_reason=reason)
                self._peak_qty[fill.symbol] = fill.quantity
                self._cost_basis[fill.symbol] = fill.quantity * fill.price
            else:
                new_qty = p.quantity + fill.quantity
                p.avg_cost = (p.avg_cost * p.quantity + fill.price * fill.quantity) / new_qty
                p.quantity = new_qty
                p.costs += fill.commission
                p.reduced = False
                self._peak_qty[fill.symbol] = max(self._peak_qty.get(fill.symbol, 0), new_qty)
                self._cost_basis[fill.symbol] = self._cost_basis.get(fill.symbol, 0) + fill.quantity * fill.price
            return None

        # SELL
        p = self.positions.get(fill.symbol)
        if p is None or fill.quantity > p.quantity:
            raise ValueError(f"cannot sell {fill.quantity} {fill.symbol}: position is {p.quantity if p else 0}")
        self.cash += fill.quantity * fill.price - fill.commission
        pnl = (fill.price - p.avg_cost) * fill.quantity
        p.realized_pnl += pnl
        p.costs += fill.commission
        self.realized_pnl += pnl - fill.commission
        p.quantity -= fill.quantity
        if p.quantity == 0:
            net = p.realized_pnl - p.costs
            basis = self._cost_basis.pop(fill.symbol, p.avg_cost * self._peak_qty.get(fill.symbol, 1))
            trade = ClosedTrade(
                symbol=fill.symbol, entry_date=p.entry_date, exit_date=fill.timestamp, entry_price=p.avg_cost,
                exit_price=fill.price, quantity=self._peak_qty.pop(fill.symbol, fill.quantity), pnl=net,
                return_pct=net / basis if basis else 0.0,
                holding_days=int((pd.Timestamp(fill.timestamp) - pd.Timestamp(p.entry_date)).days),
                entry_reason=p.entry_reason, exit_reason=reason)
            self.closed_trades.append(trade)
            del self.positions[fill.symbol]
            return trade
        return None

    def snapshot(self, date: pd.Timestamp) -> dict:
        eq = self.equity
        gross = self.positions_value
        return {"date": date, "equity": eq, "cash": self.cash, "positions_value": gross,
                "gross_exposure": gross / eq if eq > 0 else 0.0, "n_positions": len(self.positions),
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values())}
