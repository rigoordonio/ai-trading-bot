"""PaperBroker: a simulated account that behaves like a broker's paper API.

* the broker's own cash/positions/orders are AUTHORITATIVE for reconciliation
* orders are acknowledged on submission and filled at the next session's
  open (``process_open``) with the same cost model as the backtester
* rejects: trading disabled, missing/invalid risk approval, duplicate
  client_order_id (idempotency), unknown symbol, insufficient cash/position
* state persists to JSON with an atomic write (tmp file + os.replace)
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.errors import DuplicateOrderError, OrderRejectedError, TradingDisabledError
from src.execution.broker import AccountInfo, BrokerAdapter, BrokerOrder, BrokerPosition
from src.execution.orders import Order, OrderStatus, Side
from src.execution.simulator import CostModel, ExecutionSimulator
from src.logging_setup import get_logger
from src.risk.engine import RiskEngine

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperBroker(BrokerAdapter):
    name = "paper"
    is_paper = True

    def __init__(self, initial_cash: float = 100_000.0, cost_model: CostModel | None = None,
                 state_path: str | Path | None = None, market_data: Callable | None = None,
                 trading_enabled: Callable[[], bool] = lambda: True, account_id: str = "PAPER-001",
                 known_symbols: set[str] | None = None):
        self.account_id = account_id
        self.cash = float(initial_cash)
        self.positions: dict[str, BrokerPosition] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self._by_client: dict[str, str] = {}
        self._pending: dict[str, Order] = {}           # broker_order_id -> order copy awaiting a fill
        self.sim = ExecutionSimulator(cost_model or CostModel())
        self.state_path = Path(state_path) if state_path else None
        self._market_data = market_data
        self._trading_enabled = trading_enabled
        self.known_symbols = known_symbols
        self.last_prices: dict[str, float] = {}
        if self.state_path and self.state_path.exists():
            self.load()

    # --------------------------------------------------------------- reads
    def get_account(self) -> AccountInfo:
        mv = sum(p.quantity * self.last_prices.get(s, p.avg_price) for s, p in self.positions.items())
        reserved = sum(o.quantity * (o.reference_price or 0) for o in self._pending.values() if o.side == Side.BUY)
        return AccountInfo(self.account_id, self.cash, self.cash + mv, max(0.0, self.cash - reserved))

    def get_positions(self) -> dict[str, BrokerPosition]:
        return {s: copy.copy(p) for s, p in self.positions.items()}

    def get_market_data(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        if self._market_data is None:
            raise NotImplementedError("PaperBroker has no market data source configured")
        return self._market_data(symbol, start, end)

    def get_orders(self, status: str | None = None) -> list[BrokerOrder]:
        out = list(self.orders.values())
        return [o for o in out if status is None or o.status == status]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        bid = self._by_client.get(client_order_id)
        return copy.deepcopy(self.orders[bid]) if bid else None

    # -------------------------------------------------------------- writes
    def submit_order(self, order: Order) -> BrokerOrder:
        if not self._trading_enabled():
            raise TradingDisabledError("paper broker: trading is disabled; order refused")
        if not order.risk_decision_id or not RiskEngine.verify(order):
            raise OrderRejectedError(f"{order.symbol}: order lacks a valid risk approval")
        if order.idempotency_key in self._by_client:
            raise DuplicateOrderError(f"duplicate client_order_id {order.idempotency_key} "
                                      f"(existing {self._by_client[order.idempotency_key]})")
        if self.known_symbols is not None and order.symbol not in self.known_symbols:
            raise OrderRejectedError(f"unknown symbol {order.symbol}")
        if order.side == Side.SELL:
            held = self.positions.get(order.symbol)
            if held is None or held.quantity < order.quantity:
                raise OrderRejectedError(f"{order.symbol}: sell {order.quantity} > position "
                                         f"{held.quantity if held else 0}")
        bid = f"PB-{uuid.uuid4().hex[:12]}"
        bo = BrokerOrder(bid, order.idempotency_key, order.symbol, order.side.value, order.quantity, "ACKNOWLEDGED",
                         submitted_at=_now(), internal_order_id=order.order_id)
        self.orders[bid] = bo
        self._by_client[order.idempotency_key] = bid
        pending = copy.deepcopy(order)
        pending.status = OrderStatus.ACKNOWLEDGED
        self._pending[bid] = pending
        self.save()
        return copy.deepcopy(bo)

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        bo = self.orders.get(broker_order_id)
        if bo is None:
            raise OrderRejectedError(f"unknown broker order {broker_order_id}")
        if bo.status in ("ACKNOWLEDGED", "PARTIALLY_FILLED"):
            bo.status = "CANCELLED"
            self._pending.pop(broker_order_id, None)
            self.save()
        return copy.deepcopy(bo)

    def process_open(self, date: pd.Timestamp, bars: dict[str, pd.Series]) -> list[dict]:
        """Fill acknowledged orders at this session's open. Sells first."""
        fills = []
        for bid, o in sorted(self._pending.items(), key=lambda kv: 0 if kv[1].side == Side.SELL else 1):
            bo = self.orders[bid]
            pos = self.positions.get(o.symbol)
            held = pos.quantity if pos is not None else 0
            fill = self.sim.execute(o, bars.get(o.symbol), self.cash, held, date)
            if fill is None:
                bo.status, bo.reject_reason = "REJECTED", o.reject_reason
                continue
            if fill.side == Side.BUY:
                self.cash -= fill.notional + fill.commission
                p = self.positions.get(o.symbol)
                if p is None:
                    self.positions[o.symbol] = BrokerPosition(o.symbol, fill.quantity, fill.price)
                else:
                    tot = p.quantity + fill.quantity
                    p.avg_price = (p.avg_price * p.quantity + fill.price * fill.quantity) / tot
                    p.quantity = tot
            else:
                self.cash += fill.notional - fill.commission
                p = self.positions[o.symbol]
                p.quantity -= fill.quantity
                if p.quantity == 0:
                    del self.positions[o.symbol]
            bo.filled_quantity = fill.quantity
            bo.avg_fill_price = fill.price
            bo.status = "FILLED" if fill.quantity == o.quantity else "PARTIALLY_FILLED"
            fd = {**fill.to_dict(), "broker_order_id": bid, "client_order_id": bo.client_order_id,
                  "timestamp": str(pd.Timestamp(date).date())}
            bo.fills.append(fd)
            fills.append(fd)
        # day orders: anything not filled today is done (filled, partial->closed, or rejected)
        self._pending.clear()
        for bo in self.orders.values():
            if bo.status == "PARTIALLY_FILLED":
                bo.status = "CANCELLED"   # remainder of a day order expires
        for s, b in bars.items():
            if b is not None and "close" in b:
                self.last_prices[s] = float(b["close"])
        self.save()
        return fills

    def sync(self, as_of: pd.Timestamp, bars: dict[str, pd.Series]) -> None:
        self.process_open(as_of, bars)

    def mark(self, closes: dict[str, float]) -> None:
        self.last_prices.update({k: float(v) for k, v in closes.items() if v == v})

    # ---------------------------------------------------------- persistence
    def to_state(self) -> dict:
        return {"account_id": self.account_id, "cash": self.cash,
                "positions": {s: p.__dict__ for s, p in self.positions.items()},
                "orders": {k: v.__dict__ for k, v in self.orders.items()},
                "pending": {k: {**o.to_dict(), "created_at": str(o.created_at)} for k, o in self._pending.items()},
                "last_prices": self.last_prices, "saved_at": _now()}

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_state(), indent=2, default=str))
        os.replace(tmp, self.state_path)          # atomic on POSIX

    def load(self) -> None:
        if self.state_path is None:
            return
        st = json.loads(self.state_path.read_text())
        self.account_id = st["account_id"]
        self.cash = float(st["cash"])
        self.positions = {s: BrokerPosition(**p) for s, p in st["positions"].items()}
        self.orders = {k: BrokerOrder(**v) for k, v in st["orders"].items()}
        self._by_client = {v.client_order_id: k for k, v in self.orders.items()}
        self.last_prices = st.get("last_prices", {})
        self._pending = {}
        for k, d in st.get("pending", {}).items():
            o = Order(symbol=d["symbol"], side=Side(d["side"]), quantity=int(d["quantity"]),
                      created_at=pd.Timestamp(d["created_at"]), reason=d.get("reason", ""),
                      strategy=d.get("strategy", ""), risk_decision_id=d.get("risk_decision_id"),
                      reference_price=d.get("reference_price"), action=d.get("action", ""),
                      idempotency_key=d["idempotency_key"], order_id=d["order_id"])
            o.status = OrderStatus.ACKNOWLEDGED
            self._pending[k] = o
