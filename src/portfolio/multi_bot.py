"""MultiBotPortfolioManager: LEVEL 2 (account-wide) risk above every bot.

Three bots decide independently; this layer decides whether those decisions
may become orders. Each bot's RiskEngine (level 1) enforces limits inside
its own capital sleeve; this layer enforces limits on the WHOLE account,
where bots can collide:
    - global drawdown kill switch and daily-loss limit (account equity)
    - combined weight of one symbol held by several bots
    - combined weight of one bot (strategy concentration)
    - total gross exposure across bots
    - total number of open symbols
    - conflicting signals (reported; each bot only trades its own shares)

Capital: every bot trades from its own sleeve cash, so one bot can never
spend another bot's capital (the OrderValidator checks sleeve cash).

It can only shrink or block BUY orders. Sells always pass (risk-reducing).
Any malfunction here is CRITICAL in the TradingCycle: ALL new orders stop.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from src.bots.signal import SignalType, TradingSignal
from src.execution.orders import Order, OrderStatus, Side
from src.portfolio.portfolio import Portfolio
from src.risk.drawdown import DrawdownMonitor


@dataclass
class PortfolioDecision:
    order_id: str
    bot_id: str
    symbol: str
    approved: bool
    approved_quantity: int
    rule_triggered: str = ""
    reason: str = ""
    timestamp: pd.Timestamp | None = None


@dataclass
class AccountState:
    sleeves: dict[str, Portfolio]
    unallocated_cash: float = 0.0
    events: list[dict] = field(default_factory=list)

    @property
    def equity(self) -> float:
        return self.unallocated_cash + sum(p.equity for p in self.sleeves.values())

    def symbol_values(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for p in self.sleeves.values():
            for s, pos in p.positions.items():
                out[s] += pos.market_value
        return dict(out)

    def aggregate_positions(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for p in self.sleeves.values():
            for s, pos in p.positions.items():
                out[s] += pos.quantity
        return dict(out)

    @property
    def cash(self) -> float:
        return self.unallocated_cash + sum(p.cash for p in self.sleeves.values())

    def bot_values(self) -> dict[str, float]:
        return {b: p.positions_value for b, p in self.sleeves.items()}


class MultiBotPortfolioManager:
    def __init__(self, limits, drawdown: DrawdownMonitor | None = None):
        self.limits = limits
        self.drawdown = drawdown or DrawdownMonitor(limits.max_portfolio_drawdown, limits.max_daily_loss)

    @classmethod
    def from_config(cls, risk_cfg) -> "MultiBotPortfolioManager":
        return cls(risk_cfg.portfolio)

    def update(self, date: pd.Timestamp, account: AccountState) -> list[dict]:
        events = self.drawdown.update(date, account.equity)
        for e in events:
            e["component"] = "portfolio_risk"
        return events

    def review_signals(self, signals: list[TradingSignal], account: AccountState,
                       date: pd.Timestamp) -> list[dict]:
        """Aggregate the bots' TradingSignals per symbol and report conflicts.

        Resolution policy (deterministic, never a vote): each bot acts only on
        its OWN virtual position. A SWING EXIT sells SWING's shares only; an
        INVESTOR BUY/HOLD keeps INVESTOR's shares. Several BUYs for one symbol
        are capped together by max_symbol_weight in review()."""
        by_sym: dict[str, list[TradingSignal]] = defaultdict(list)
        for sig in signals:
            by_sym[sig.symbol].append(sig)
        events = []
        held = {s: {b: p.positions[s].quantity for b, p in account.sleeves.items() if s in p.positions}
                for s in by_sym}
        for sym, sigs in by_sym.items():
            types = {sig.bot_id: sig.signal_type for sig in sigs}
            buying = sorted(b for b, t in types.items() if t == SignalType.BUY)
            selling = sorted(b for b, t in types.items() if t in (SignalType.EXIT, SignalType.REDUCE))
            if buying and selling:
                events.append({"date": date, "event": "SIGNAL_CONFLICT", "severity": "INFO", "symbol": sym,
                               "component": "portfolio_risk", "rule": "per_bot_attribution",
                               "detail": f"buy={buying} sell={selling} holdings={held.get(sym, {})}: each bot "
                                         "trades only its own shares"})
            elif len(buying) > 1:
                events.append({"date": date, "event": "SIGNAL_OVERLAP", "severity": "INFO", "symbol": sym,
                               "component": "portfolio_risk", "rule": "max_symbol_weight",
                               "detail": f"{buying} all want {sym}: combined exposure is capped"})
        return events

    def review(self, orders: list[Order], account: AccountState, closes: dict[str, float],
               date: pd.Timestamp) -> tuple[list[Order], list[PortfolioDecision], list[dict]]:
        L = self.limits
        equity = account.equity
        sym_val = account.symbol_values()
        bot_val = account.bot_values()
        gross = sum(sym_val.values())
        max_bot = getattr(L, "max_bot_weight", 1.0)
        n_pos = len(sym_val)
        approved, decisions, events = [], [], []

        for o in sorted(orders, key=lambda o: 0 if o.side == Side.SELL else 1):
            bot = o.meta.get("bot_id", o.strategy)
            if o.side == Side.SELL:
                approved.append(o)
                px = closes.get(o.symbol, 0.0)
                sym_val[o.symbol] = max(0.0, sym_val.get(o.symbol, 0.0) - o.quantity * px)
                bot_val[bot] = max(0.0, bot_val.get(bot, 0.0) - o.quantity * px)
                gross = max(0.0, gross - o.quantity * px)
                decisions.append(PortfolioDecision(o.order_id, bot, o.symbol, True, o.quantity, "", "sell", date))
                continue
            px_opt = closes.get(o.symbol)
            rule, reason, qty = "", "", o.quantity

            def block(r: str, why: str, o: Order = o, bot: str = bot) -> None:
                o.transition(OrderStatus.REJECTED, f"portfolio risk: {why}")
                o.reject_reason = f"portfolio risk: {why}"
                decisions.append(PortfolioDecision(o.order_id, bot, o.symbol, False, 0, r, why, date))
                events.append({"date": date, "event": "PORTFOLIO_BLOCK", "severity": "WARNING", "symbol": o.symbol,
                               "bot_id": bot, "rule": r, "detail": why, "component": "portfolio_risk"})

            if self.drawdown.state.kill_switch:
                block("global_kill_switch", "account drawdown limit breached")
                continue
            if self.drawdown.state.daily_loss_breached:
                block("global_daily_loss", "account daily loss limit breached")
                continue
            if px_opt is None or not px_opt > 0 or equity <= 0:
                block("price", "no valid price / equity")
                continue
            px = float(px_opt)
            is_new = sym_val.get(o.symbol, 0.0) <= 0
            if is_new and n_pos >= L.max_total_positions:
                block("max_total_positions", f"account already holds {n_pos} symbols")
                continue
            room = {"max_symbol_weight": L.max_symbol_weight * equity - sym_val.get(o.symbol, 0.0),
                    "max_account_gross_exposure": L.max_gross_exposure * equity - gross,
                    "max_bot_weight": max_bot * equity - bot_val.get(bot, 0.0)}
            binding = min(room, key=lambda k: room[k])
            allowed = int(max(0.0, room[binding]) // px)
            if allowed < qty:
                rule = binding
                reason = f"resized {qty} -> {allowed} by {rule}"
                qty = allowed
            if qty <= 0:
                block(rule or "capacity", reason or "no capacity")
                continue
            if qty < o.quantity:
                o.quantity = qty            # shrinking stays within the sleeve approval
                events.append({"date": date, "event": "PORTFOLIO_RESIZE", "severity": "INFO", "symbol": o.symbol,
                               "bot_id": bot, "rule": rule, "detail": reason, "component": "portfolio_risk"})
            approved.append(o)
            decisions.append(PortfolioDecision(o.order_id, bot, o.symbol, True, qty, rule, reason or "pass", date))
            sym_val[o.symbol] = sym_val.get(o.symbol, 0.0) + qty * px
            bot_val[bot] = bot_val.get(bot, 0.0) + qty * px
            gross += qty * px
            if is_new:
                n_pos += 1
        return approved, decisions, events


# Backward-compatible name.
PortfolioRiskManager = MultiBotPortfolioManager
