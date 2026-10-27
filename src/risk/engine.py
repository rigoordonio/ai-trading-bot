"""The deterministic risk engine: the ONLY component that creates orders.

    proposals (from strategies / ensemble / stops)
          |
          v
    RiskEngine.review()  -- PASS / RESIZED / BLOCKED, with reasons
          |
          v
    approved Orders (carry a risk_decision_id registered here)
          |
          v
    Broker.submit_order()  -- verifies the approval with RiskEngine.verify()

Order of checks for BUYS (sells are always allowed, they reduce risk):
    1 kill switch / daily-loss block      6 sector concentration
    2 minimum price                       7 gross exposure cap
    3 minimum liquidity                   8 cash reserve
    4 max concurrent positions            9 forecast portfolio volatility
    5 max position weight                10 minimum trade value
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.execution.orders import Order, OrderType, Side
from src.logging_setup import get_logger
from src.portfolio.portfolio import Portfolio
from src.risk.drawdown import DrawdownMonitor
from src.risk.exposure import NON_SECTOR, gross_exposure, sector_weights
from src.risk.portfolio_risk import covariance, portfolio_volatility

log = get_logger(__name__)

PASS, RESIZED, BLOCKED = "PASS", "RESIZED", "BLOCKED"

# Registry of approvals issued in this process: {decision_id: (symbol, side, max_qty)}
_APPROVALS: dict[str, tuple[str, str, int]] = {}


@dataclass
class OrderProposal:
    symbol: str
    side: Side
    quantity: int
    reason: str
    strategy: str = ""
    action: str = ""                     # BUY / REDUCE / EXIT
    score: float | None = None
    components: dict[str, Any] = field(default_factory=dict)
    is_risk_exit: bool = False           # stop-loss / kill-switch liquidation


@dataclass
class MarketSnapshot:
    """Everything known at the decision time (close of ``date``)."""
    date: pd.Timestamp
    closes: dict[str, float]
    features: pd.DataFrame               # index = symbol, row = features at `date`
    returns: pd.DataFrame | None = None  # trailing daily returns up to `date` (dates x symbols)


@dataclass
class RiskDecision:
    proposal: OrderProposal
    status: str
    approved_quantity: int
    reasons: list[str] = field(default_factory=list)
    rule_triggered: str = ""
    timestamp: pd.Timestamp | None = None
    decision_id: str = field(default_factory=lambda: f"R{uuid.uuid4().hex[:12]}")

    @property
    def approved(self) -> bool:
        return self.status != BLOCKED and self.approved_quantity > 0

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)

    def to_dict(self) -> dict:
        p = self.proposal
        return {"decision_id": self.decision_id, "timestamp": self.timestamp, "symbol": p.symbol,
                "side": p.side.value, "action": p.action, "requested_qty": p.quantity,
                "approved_qty": self.approved_quantity, "status": self.status, "approved": self.approved,
                "rule_triggered": self.rule_triggered, "reasons": self.reason, "strategy": p.strategy,
                "score": p.score, "proposal_reason": p.reason}


def _blocked(prop: "OrderProposal", rule: str, reason: str) -> RiskDecision:
    return RiskDecision(prop, BLOCKED, 0, [reason], rule_triggered=rule)


class RiskEngine:
    def __init__(self, risk_cfg, sectors: dict[str, str] | None = None, drawdown: DrawdownMonitor | None = None,
                 name: str = "risk"):
        self.cfg = risk_cfg
        self.name = name
        self.limits = risk_cfg.limits
        self.max_order_notional = risk_cfg.safety.max_order_notional
        self.sectors = sectors or {}
        self.drawdown = drawdown or DrawdownMonitor.from_config(risk_cfg.drawdown)

    # ------------------------------------------------------------ verification
    @staticmethod
    def verify(order: Order) -> bool:
        """True only for orders whose approval was issued by a RiskEngine and
        whose quantity does not exceed the approved quantity."""
        rec = _APPROVALS.get(order.risk_decision_id or "")
        if rec is None:
            return False
        return rec[0] == order.symbol and rec[1] == order.side.value and order.quantity <= rec[2]

    # ------------------------------------------------------------------ review
    def review(self, proposals: list[OrderProposal], portfolio: Portfolio,
               snapshot: MarketSnapshot) -> tuple[list[Order], list[RiskDecision], list[dict]]:
        equity = portfolio.equity
        orders: list[Order] = []
        decisions: list[RiskDecision] = []
        events: list[dict] = []
        values = {s: p.market_value for s, p in portfolio.positions.items()}
        cash = portfolio.cash
        n_positions = len(portfolio.positions)

        sells = [p for p in proposals if p.side == Side.SELL]
        buys = sorted([p for p in proposals if p.side == Side.BUY], key=lambda p: -(p.score or 0))

        for prop in sells:
            held = portfolio.quantity(prop.symbol)
            qty = min(prop.quantity, held)
            if qty <= 0:
                decisions.append(_blocked(prop, "position", "nothing to sell"))
                continue
            dec = RiskDecision(prop, PASS if qty == prop.quantity else RESIZED, qty,
                               ["risk-reducing trade allowed"], rule_triggered="" if qty == prop.quantity
                               else "held_quantity")
            decisions.append(dec)
            px = snapshot.closes.get(prop.symbol, 0.0)
            values[prop.symbol] = max(0.0, values.get(prop.symbol, 0.0) - qty * px)
            cash += qty * px * 0.99          # same haircut the order validator applies
            if values[prop.symbol] <= 1e-9:
                n_positions -= 1
                values.pop(prop.symbol, None)
            orders.append(self._make_order(prop, qty, snapshot, dec))

        cov = None
        if buys and snapshot.returns is not None and self.limits.max_portfolio_volatility:
            cov = covariance(snapshot.returns.tail(60), list(set(values) | {b.symbol for b in buys}))

        for prop in buys:
            dec = self._review_buy(prop, equity, values, cash, n_positions, snapshot, cov)
            decisions.append(dec)
            if dec.status != BLOCKED:
                px = snapshot.closes[prop.symbol]
                if prop.symbol not in values:
                    n_positions += 1
                values[prop.symbol] = values.get(prop.symbol, 0.0) + dec.approved_quantity * px
                cash -= dec.approved_quantity * px
                orders.append(self._make_order(prop, dec.approved_quantity, snapshot, dec))
        for dec in decisions:
            dec.timestamp = snapshot.date
            if dec.status in (BLOCKED, RESIZED):
                events.append({"date": snapshot.date, "event": f"ORDER_{dec.status}", "severity": "WARNING"
                               if dec.status == BLOCKED else "INFO", "symbol": dec.proposal.symbol,
                               "rule": dec.rule_triggered, "component": self.name,
                               "value": dec.approved_quantity, "limit": dec.proposal.quantity,
                               "detail": dec.reason})
        return orders, decisions, events

    def _review_buy(self, prop: OrderProposal, equity: float, values: dict[str, float], cash: float,
                    n_positions: int, snap: MarketSnapshot, cov) -> RiskDecision:
        L = self.limits
        reasons: list[str] = []
        sym = prop.symbol
        px = snap.closes.get(sym)

        rule = ""
        if self.drawdown.state.kill_switch:
            return _blocked(prop, "kill_switch", "KILL SWITCH active: max portfolio drawdown breached")
        if self.drawdown.state.daily_loss_breached:
            return _blocked(prop, "daily_loss_limit", "daily loss limit breached")
        if px is None or not np.isfinite(px) or px <= 0:
            return _blocked(prop, "price", "no valid price")
        if not np.isfinite(equity) or equity <= 0:
            return _blocked(prop, "equity", "portfolio equity is not positive/finite")
        if not isinstance(prop.quantity, (int, np.integer)) or prop.quantity <= 0:
            return _blocked(prop, "quantity", f"invalid proposed quantity {prop.quantity!r}")
        if px < L.min_price:
            return _blocked(prop, "min_price", f"price {px:.2f} < min_price {L.min_price}")
        feats = snap.features.loc[sym] if sym in snap.features.index else None
        if L.min_avg_dollar_volume:
            dv = feats.get("dollar_volume_20", np.nan) if feats is not None else np.nan
            if not np.isfinite(dv) or dv < L.min_avg_dollar_volume:
                return _blocked(prop, "liquidity", "insufficient liquidity (20d dollar volume)")
        is_new = sym not in values
        if is_new and n_positions >= L.max_concurrent_positions:
            return _blocked(prop, "max_concurrent_positions",
                            f"max concurrent positions ({L.max_concurrent_positions})")

        qty = int(prop.quantity)

        def cap(limit_value: float, label: str, rule_name: str) -> None:
            nonlocal qty, rule
            allowed = int(max(0.0, limit_value) / px + 1e-9)   # epsilon: 0.30-0.25 must not lose a share
            if allowed < qty:
                qty = allowed
                reasons.append(label)
                rule = rule_name

        cap(L.max_position_weight * equity - values.get(sym, 0.0),
            f"max position weight {L.max_position_weight:.0%}", "max_position_weight")
        sector = self.sectors.get(sym)
        if sector is not None and sector not in NON_SECTOR:
            sw = sector_weights({s: v / equity for s, v in values.items()}, self.sectors).get(sector, 0.0)
            cap((L.max_sector_weight - sw) * equity, f"max sector weight {L.max_sector_weight:.0%} ({sector})",
                "max_sector_weight")
        cap(L.max_gross_exposure * equity - sum(values.values()),
            f"max gross exposure {L.max_gross_exposure:.0%}", "max_gross_exposure")
        # keep the cash reserve, with a small buffer for costs and opening gaps
        cap((cash - L.min_cash_reserve * equity) * 0.985, f"min cash reserve {L.min_cash_reserve:.0%}",
            "min_cash_reserve")
        cap(self.max_order_notional, f"max order notional {self.max_order_notional:,.0f}", "max_order_notional")

        if qty > 0 and cov is not None and sym in cov.index:
            def vol_with(q: int) -> float:
                w = {s: v / equity for s, v in values.items()}
                w[sym] = w.get(sym, 0.0) + q * px / equity
                return portfolio_volatility(w, cov)

            if vol_with(qty) > L.max_portfolio_volatility:
                lo, hi = 0, qty
                while lo < hi:                    # largest q with vol <= limit
                    mid = (lo + hi + 1) // 2
                    if vol_with(mid) <= L.max_portfolio_volatility:
                        lo = mid
                    else:
                        hi = mid - 1
                qty = lo
                reasons.append(f"forecast portfolio vol limit {L.max_portfolio_volatility:.0%}")
                rule = "max_portfolio_volatility"

        if qty <= 0 or qty * px < L.min_trade_value:
            reasons.append(f"trade value below minimum {L.min_trade_value}")
            return RiskDecision(prop, BLOCKED, 0, reasons, rule_triggered=rule or "min_trade_value")
        status = PASS if qty == prop.quantity else RESIZED
        return RiskDecision(prop, status, qty, reasons or ["all checks passed"], rule_triggered=rule)

    def _make_order(self, prop: OrderProposal, qty: int, snap: MarketSnapshot, dec: RiskDecision) -> Order:
        _APPROVALS[dec.decision_id] = (prop.symbol, prop.side.value, qty)
        return Order(symbol=prop.symbol, side=prop.side, quantity=int(qty), created_at=snap.date,
                     order_type=OrderType.MARKET, reason=prop.reason, strategy=prop.strategy,
                     risk_decision_id=dec.decision_id, reference_price=snap.closes.get(prop.symbol),
                     action=prop.action, account=prop.components.get("account", "paper"),
                     meta={"bot_id": prop.components.get("bot_id", prop.strategy), "score": prop.score})

    # --------------------------------------------------------------- reporting
    def status(self, portfolio: Portfolio, returns: pd.DataFrame | None = None) -> dict:
        w = portfolio.weights()
        cov = covariance(returns.tail(60), list(w)) if returns is not None and w else None
        st = self.drawdown.state
        return {"gross_exposure": gross_exposure(w), "net_exposure": sum(w.values()),
                "largest_position": max(w.values(), default=0.0), "portfolio_vol": portfolio_volatility(w, cov),
                "drawdown": st.drawdown, "kill_switch": st.kill_switch, "cash_weight": portfolio.cash / portfolio.equity
                if portfolio.equity > 0 else 0.0, "positions": len(w)}
