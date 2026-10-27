"""The fail-closed decision cycle, shared by backtesting and paper trading.

Runs once per bar close:

    portfolio state check               corrupt -> NO_TRADE (all), CRITICAL halt
    for each bot:
        sleeve drawdown monitor
        stop-loss exits                         (risk rules; failure -> CRITICAL halt)
        on rebalance days, if the bot's signal generation is enabled,
        for each symbol:
            [VALIDATE current bar]     invalid -> NO_TRADE (that symbol only)
            [VALIDATE features]        invalid -> NO_TRADE (symbol, new buys)
            signal score -> action     NaN (e.g. model failed) -> HOLD
            -> TradingSignal, size -> proposal
        failure in this bot's signal path -> BOT DISABLED for this cycle
                                       (no orders from it; other bots continue)
        sleeve RiskEngine.review       exception -> NO_TRADE (all), CRITICAL halt
    MultiBotPortfolioManager           exception -> NO_TRADE (all), CRITICAL halt
        review_signals (conflicts), review (account limits)
    OrderValidator                     invalid -> order REJECTED
    TradingGate                        closed  -> every order REJECTED

Failure scope:
    one bot's features / model / signal code  -> that bot only (fail closed)
    risk engines, portfolio state, gate         -> ALL new trading stops
A disabled bot's sleeve still counts in account exposure and its
stop-losses still run. Exceptions are never swallowed: every failure is
logged with traceback and recorded as an event.

Every evaluated (bot, symbol) gets an AUDIT record answering "why did the
bot buy?" and "why did it refuse?".
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from src.bots.signal import TradingSignal
from src.errors import (
    FeatureSchemaError,
    OrderValidationError,
    PortfolioStateError,
    RiskEngineError,
    TradingSystemError,
)
from src.execution.orders import Order, OrderStatus, Side
from src.logging_setup import get_logger
from src.portfolio.allocation import propose
from src.portfolio.portfolio import Portfolio
from src.portfolio.risk_manager import AccountState, PortfolioRiskManager
from src.risk.engine import MarketSnapshot, OrderProposal, RiskEngine
from src.risk.stop_loss import StopLossManager
from src.safety.health import HealthStatus, TradingGate
from src.safety.order_validator import OrderContext, OrderValidator
from src.safety.validation import validate_bar_values, validate_feature_row
from src.strategies.base import BUY, EXIT, HOLD, SignalFrame

log = get_logger(__name__)

NO_TRADE = "NO_TRADE"


class BotLike(Protocol):
    bot_id: str
    symbols: list[str]
    risk_cfg: Any

    @property
    def signal_features(self) -> list[str]: ...

    @property
    def rebalance_frequency(self) -> str: ...

    @property
    def versions(self) -> Any: ...


@dataclass
class MarketState:
    date: pd.Timestamp
    bars: dict[str, pd.Series]             # latest bar per symbol on/before `date`
    bar_dates: dict[str, pd.Timestamp]
    prev_close: dict[str, float | None]
    returns: pd.DataFrame | None = None    # trailing daily returns up to `date`

    @property
    def closes(self) -> dict[str, float]:
        return {s: float(b["close"]) for s, b in self.bars.items() if b is not None and "close" in b}


@dataclass
class BotRuntime:
    bot: BotLike
    signal: SignalFrame
    features_on: Callable[[pd.Timestamp], pd.DataFrame]   # date -> features (index = symbol)
    sleeve: Portfolio
    risk: RiskEngine
    stops: StopLossManager
    active: Callable[[str, pd.Timestamp], bool] = lambda s, d: True
    signals_enabled: bool = True           # False: bot disabled (e.g. its model failed); risk exits still run
    disabled_reason: str = ""
    buy_size_multiplier: float = 1.0       # set by the Reviewer: 1 active, <1 probation, 0 paused (risk-reducing only)
    reviewer_status: str = "ACTIVE"

    @classmethod
    def disabled(cls, bot: BotLike, sleeve: Portfolio, risk: RiskEngine, stops: StopLossManager, reason: str,
                 features_on: Callable[[pd.Timestamp], pd.DataFrame] | None = None) -> "BotRuntime":
        """Runtime for a bot whose signal path failed: it proposes nothing new,
        but its existing positions keep their stop-losses and still count in
        account-level risk."""
        empty = pd.DataFrame({"score": pd.Series(dtype=float)},
                             index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
        sf = SignalFrame(bot.bot_id, empty, 1.0, 0.0, 0.0)
        return cls(bot, sf, features_on or (lambda d: pd.DataFrame()), sleeve, risk, stops,
                   signals_enabled=False, disabled_reason=reason)


@dataclass
class CycleResult:
    orders: list[Order] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    rejected: list[Order] = field(default_factory=list)
    signals: list[TradingSignal] = field(default_factory=list)
    disabled_bots: dict[str, str] = field(default_factory=dict)


def is_rebalance_day(freq: str, date: pd.Timestamp, prev: pd.Timestamp | None) -> bool:
    if freq == "daily" or prev is None:
        return True
    if freq == "weekly":
        return date.isocalendar()[:2] != prev.isocalendar()[:2]
    if freq == "monthly":
        return (date.year, date.month) != (prev.year, prev.month)
    raise ValueError(f"unknown rebalance frequency {freq}")


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


class TradingCycle:
    def __init__(self, runtimes: dict[str, BotRuntime], portfolio_risk: PortfolioRiskManager,
                 validator: OrderValidator, gate: TradingGate, safety_cfg, known_symbols: set[str],
                 unallocated_cash: Callable[[], float] = lambda: 0.0, environment: str = "backtest",
                 run_id: str = "", sleeves: dict[str, Portfolio] | None = None):
        """``sleeves``: every bot's sleeve, including bots that have no runtime
        this cycle, so account-level risk always sees the whole account."""
        self.runtimes = runtimes
        self.sleeves = sleeves or {}
        self.portfolio_risk = portfolio_risk
        self.validator = validator
        self.gate = gate
        self.safety = safety_cfg
        self.known_symbols = known_symbols
        self.unallocated_cash = unallocated_cash
        self.environment = environment
        self.run_id = run_id
        self._prev_date: pd.Timestamp | None = None

    def account(self) -> AccountState:
        return AccountState({**self.sleeves, **{b: rt.sleeve for b, rt in self.runtimes.items()}},
                            self.unallocated_cash())

    # ------------------------------------------------------------------ run
    def run(self, market: MarketState) -> CycleResult:
        res = CycleResult()
        date = market.date
        account = self.account()
        try:
            self._check_portfolio_state(account)
            res.events += self.portfolio_risk.update(date, account)
        except Exception as exc:  # noqa: BLE001 - fail closed, recorded below
            self._fail(res, exc, "portfolio", date)
            self._prev_date = date
            return res

        for b in account.sleeves:
            if b not in self.runtimes:
                res.disabled_bots[b] = "no runtime this cycle"
        proposed: list[Order] = []
        for bot_id, rt in self.runtimes.items():
            try:
                proposed += self._run_bot(rt, market, res)
            except (RiskEngineError, PortfolioStateError) as exc:
                self._fail(res, exc, getattr(exc, "component", "risk_engine"), date, bot_id)
            except Exception as exc:  # noqa: BLE001 - bot-scoped: this bot proposes nothing, others continue
                self._disable_bot(res, bot_id, exc, date)
                proposed = [o for o in proposed if o.meta.get("bot_id") != bot_id]

        try:
            if hasattr(self.portfolio_risk, "review_signals"):
                res.events += self.portfolio_risk.review_signals(res.signals, account, date)
        except Exception as exc:  # noqa: BLE001 - portfolio layer broken -> nothing may trade
            self._fail(res, exc, "portfolio_risk", date)
            for o in proposed:
                self._reject(o, "portfolio risk manager failed", res)
            proposed = []

        if proposed:
            try:
                approved, pdecisions, pevents = self.portfolio_risk.review(proposed, account, market.closes, date)
                res.events += pevents
                pmap = {d.order_id: d for d in pdecisions}
                for rec in res.audit:
                    d = pmap.get(rec.get("order_id") or "")
                    if d is None:
                        continue
                    if not d.approved:
                        rec.update(final_action=NO_TRADE, portfolio_status="BLOCKED", quantity=0,
                                   rule_triggered=d.rule_triggered, reason=f"portfolio risk blocked: {d.reason}")
                    elif d.rule_triggered:
                        rec.update(portfolio_status="RESIZED", quantity=d.approved_quantity,
                                   rule_triggered=d.rule_triggered, portfolio_reason=d.reason)
                    else:
                        rec.update(portfolio_status="PASS")
            except Exception as exc:  # noqa: BLE001
                self._fail(res, exc, "portfolio_risk", date)
                for o in proposed:
                    self._reject(o, "portfolio risk manager failed", res)
                approved = []
            proceeds: dict[str, float] = {}
            for o in approved:      # sells come first: their proceeds fund same-cycle buys (sold at next open)
                self._finalise(o, account, market, res, proceeds)
        self._prev_date = date
        return res

    # ------------------------------------------------------------ internals
    def _check_portfolio_state(self, account: AccountState) -> None:
        for bot_id, p in account.sleeves.items():
            if not np.isfinite(p.cash) or not np.isfinite(p.equity):
                raise PortfolioStateError(f"{bot_id}: non-finite cash/equity")
            for s, pos in p.positions.items():
                if pos.quantity <= 0 or not np.isfinite(pos.avg_cost) or pos.avg_cost <= 0:
                    raise PortfolioStateError(f"{bot_id}: corrupt position {s}: qty={pos.quantity}, "
                                              f"avg_cost={pos.avg_cost}")

    def _run_bot(self, rt: BotRuntime, market: MarketState, res: CycleResult) -> list[Order]:
        bot, date = rt.bot, market.date
        res.events += [{**e, "bot_id": bot.bot_id, "component": f"{bot.bot_id}.drawdown"}
                       for e in rt.risk.drawdown.update(date, rt.sleeve.equity)]
        feats = rt.features_on(date)
        proposals: list[OrderProposal] = []
        audit_by_symbol: dict[str, dict] = {}
        versions = bot.versions.as_dict() if hasattr(bot.versions, "as_dict") else dict(bot.versions)

        def audit(symbol: str, **kw) -> dict:
            rec = {"signal_id": TradingSignal.make_id(bot.bot_id, symbol, date), "timestamp": date,
                   "data_timestamp": market.bar_dates.get(symbol, date), "run_id": self.run_id,
                   "environment": self.environment, "bot_id": bot.bot_id, "symbol": symbol,
                   "timeframe": getattr(bot, "timeframe", "daily"),
                   "strategy": rt.signal.name, **versions, "final_action": NO_TRADE, "risk_status": None,
                   "rule_triggered": None, "order_id": None, "order_status": None, "quantity": 0}
            rec.update(kw)
            audit_by_symbol[symbol] = rec
            res.audit.append(rec)
            return rec

        # 1) forced exits: kill-switch liquidation and stop-losses (held symbols)
        forced = set()
        for sym, pos in list(rt.sleeve.positions.items()):
            bar = market.bars.get(sym)
            try:
                validate_bar_values(sym, bar if bar is not None else {}, market.bar_dates.get(sym, date), date,
                                    market.prev_close.get(sym), self.safety.max_data_staleness_days,
                                    self.safety.max_bar_abs_return)
            except TradingSystemError as exc:
                audit(sym, action=HOLD, reason=f"held position not evaluated: {exc}", data_status="INVALID")
                res.events.append(self._event(exc, date, bot.bot_id, sym))
                continue
            assert bar is not None  # validated above
            close = float(bar["close"])
            if rt.risk.drawdown.state.kill_switch and rt.risk.drawdown.liquidate:
                proposals.append(OrderProposal(sym, Side.SELL, pos.quantity, "kill-switch liquidation", bot.bot_id,
                                               EXIT, None, {"bot_id": bot.bot_id}, is_risk_exit=True))
                forced.add(sym)
                audit(sym, action=EXIT, reason="kill-switch liquidation")
                continue
            atr = _f(feats.loc[sym].get("atr_14")) if sym in feats.index else None
            try:
                stop = rt.stops.check(pos, close, atr, date)
            except Exception as exc:  # noqa: BLE001 - re-raised as CRITICAL risk failure
                raise RiskEngineError(f"{bot.bot_id}: stop-loss check failed for {sym}: "
                                      f"{type(exc).__name__}: {exc}") from exc
            if stop is not None:
                proposals.append(OrderProposal(sym, Side.SELL, pos.quantity, f"{stop.rule}: {stop.detail}",
                                               bot.bot_id, EXIT, None, {"bot_id": bot.bot_id}, is_risk_exit=True))
                forced.add(sym)
                audit(sym, action=EXIT, reason=f"{stop.rule}: {stop.detail}")

        # 2) signal-driven decisions on rebalance days
        rebalance = is_rebalance_day(bot.rebalance_frequency, date, self._prev_date)
        if rebalance and not rt.signals_enabled:
            res.disabled_bots[bot.bot_id] = rt.disabled_reason or "signal generation disabled"
            res.events.append({"date": date, "event": "BOT_SIGNALS_DISABLED", "severity": "ERROR",
                               "component": f"bot:{bot.bot_id}", "bot_id": bot.bot_id, "symbol": None,
                               "detail": rt.disabled_reason or "signal generation disabled", "rule": None})
        elif rebalance:
            signals = rt.signal.on(date)
            for sym in bot.symbols:
                if sym in forced:
                    continue
                held = rt.sleeve.quantity(sym) > 0
                srow = signals.loc[sym] if sym in signals.index else None
                score = _f(srow["score"]) if srow is not None else None
                comps = {k: _f(v) for k, v in (srow.drop(labels=["score"]).items() if srow is not None else [])}
                comp_scores: dict[str, float | None] = {k: v for k, v in comps.items() if k != "ml_probability"}
                base: dict[str, Any] = {"score": score, "ml_probability": comps.get("ml_probability"),
                                        "components": comp_scores}
                bar = market.bars.get(sym)
                try:
                    validate_bar_values(sym, bar if bar is not None else {}, market.bar_dates.get(sym, date), date,
                                        market.prev_close.get(sym), self.safety.max_data_staleness_days,
                                        self.safety.max_bar_abs_return)
                except TradingSystemError as exc:
                    audit(sym, action=HOLD, reason=f"market data invalid: {exc}", data_status="INVALID", **base)
                    res.events.append(self._event(exc, date, bot.bot_id, sym))
                    continue
                assert bar is not None  # validated above
                price = float(bar["close"])
                if held and not rt.active(sym, date):
                    action, why = EXIT, "symbol left the point-in-time universe"
                else:
                    cur = rt.sleeve.positions.get(sym)
                    action = rt.signal.decide(score if score is not None else float("nan"), held,
                                              cur.reduced if cur else False)
                    why = "" if score is not None else "no valid signal score (missing data or model output)"
                    if action == BUY and not rt.active(sym, date):
                        action, why = HOLD, "symbol not in the point-in-time universe"
                if action == BUY:
                    try:
                        row = feats.loc[sym] if sym in feats.index else pd.Series(dtype=float)
                        validate_feature_row(sym, row, bot.signal_features)
                    except FeatureSchemaError as exc:
                        audit(sym, action=HOLD, reason=f"features invalid: {exc}", **base)
                        res.events.append(self._event(exc, date, bot.bot_id, sym))
                        continue
                if action in (HOLD,):
                    if held or score is not None:
                        audit(sym, action=action, reason=why or "no action", **base)
                    continue
                prop = propose(action, sym, rt.sleeve, price, feats.loc[sym] if sym in feats.index else None,
                               bot.risk_cfg.position_sizing, bot.bot_id, score,
                               {**comp_scores, "bot_id": bot.bot_id, "account": "paper"},
                               rt.signal.reduce_fraction, reason=f"{bot.bot_id} {action}: {why}" if why else "")
                if prop is None:
                    audit(sym, action=action, reason="position size rounds to zero", **base)
                    continue
                if action == BUY and rt.buy_size_multiplier < 1.0:
                    mult = max(0.0, rt.buy_size_multiplier)             # can only shrink, never enlarge
                    qty = int(prop.quantity * mult)
                    if qty <= 0:
                        audit(sym, action=action, reason=f"reviewer: {rt.reviewer_status}, no new buys", **base,
                              rule_triggered=f"reviewer_{rt.reviewer_status.lower()}")
                        continue
                    prop.quantity = qty
                    prop.reason += f" (reviewer {rt.reviewer_status}: size x{mult:g})"
                proposals.append(prop)
                audit(sym, action=action, reason=prop.reason, **base)

        self._emit_signals(rt, audit_by_symbol, date, res)
        if not proposals:
            return []

        # 3) sleeve risk engine (deterministic; any malfunction blocks everything)
        snap = MarketSnapshot(date, market.closes, feats, market.returns)
        try:
            orders, decisions, events = rt.risk.review(proposals, rt.sleeve, snap)
        except Exception as exc:
            raise RiskEngineError(f"{bot.bot_id}: risk engine failed: {type(exc).__name__}: {exc}") from exc
        res.events += [{**e, "bot_id": bot.bot_id} for e in events]
        by_dec = {o.risk_decision_id: o for o in orders}
        for d in decisions:
            rec = audit_by_symbol.get(d.proposal.symbol)
            if rec is None:
                continue
            o = by_dec.get(d.decision_id)
            rec.update(risk_status=d.status, rule_triggered=d.rule_triggered or None, risk_reason=d.reason,
                       risk_decision_id=d.decision_id, quantity=d.approved_quantity,
                       order_id=o.order_id if o else None)
            if not d.approved:
                rec.update(final_action=NO_TRADE, reason=f"risk blocked: {d.reason}")
        for o in orders:
            o.meta["audit"] = audit_by_symbol.get(o.symbol)
            o.meta["signal_id"] = (audit_by_symbol.get(o.symbol) or {}).get("signal_id")
            o.meta["data_timestamp"] = str(pd.Timestamp(date).date())
        return orders

    def _emit_signals(self, rt: BotRuntime, audit_by_symbol: dict[str, dict], date: pd.Timestamp,
                      res: CycleResult) -> None:
        """One TradingSignal per evaluated symbol (the bot's own decision,
        before any risk layer). Research bots without make_signal skip this."""
        make = getattr(rt.bot, "make_signal", None)
        if make is None:
            return
        for sym, rec in audit_by_symbol.items():
            res.signals.append(make(sym, rec.get("data_timestamp", date), rec.get("action") or NO_TRADE,
                                    rec.get("score"), rec.get("ml_probability"), rec.get("reason", "")))

    def _disable_bot(self, res: CycleResult, bot_id: str, exc: Exception, date) -> None:
        """Bot-scoped failure: this bot proposes nothing this cycle; other bots
        are unaffected. Logged with traceback; never silently ignored."""
        tb = traceback.format_exc()
        log.error("BOT %s DISABLED for %s: %s: %s\n%s", bot_id, getattr(date, "date", lambda: date)(),
                  type(exc).__name__, exc, tb)
        res.disabled_bots[bot_id] = f"{type(exc).__name__}: {exc}"
        self.gate.health.set(f"bot:{bot_id}", HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}")
        res.events.append({"date": date, "event": "BOT_DISABLED", "severity": "ERROR", "component": f"bot:{bot_id}",
                           "bot_id": bot_id, "symbol": None, "detail": f"{type(exc).__name__}: {exc}",
                           "rule": None, "traceback": tb[-2000:]})

    def _finalise(self, o: Order, account: AccountState, market: MarketState, res: CycleResult,
                  proceeds: dict[str, float]) -> None:
        bot = o.meta.get("bot_id", o.strategy)
        sleeve = account.sleeves[bot]
        rec = o.meta.get("audit")
        try:
            freed = proceeds.get(bot, 0.0)
            ctx = OrderContext(as_of=market.date, cash=sleeve.cash + freed, equity=sleeve.equity,
                               gross_value=max(0.0, sleeve.positions_value - freed),
                               held_quantity=sleeve.quantity(o.symbol), known_symbols=self.known_symbols)
            self.validator.validate(o, ctx)
        except OrderValidationError as exc:
            self._reject(o, f"order validation: {exc}", res, rule=exc.context.get("rule"))
            res.events.append(self._event(exc, market.date, bot, o.symbol))
            return
        if not self.gate.trading_enabled:
            reducing = o.side == Side.SELL and self.safety.allow_risk_reducing_orders_when_halted
            if not reducing:
                self._reject(o, "trading disabled: " + "; ".join(self.gate.blocking_reasons()), res,
                             rule="trading_gate")
                return
        res.orders.append(o)
        if o.side == Side.SELL:
            # conservative: proceeds net of a 1% haircut for costs / adverse open
            proceeds[bot] = proceeds.get(bot, 0.0) + o.quantity * (o.reference_price or 0.0) * 0.99
        if rec is not None:
            rec.update(final_action=o.side.value, order_status=o.status.value, quantity=o.quantity)

    def _reject(self, o: Order, why: str, res: CycleResult, rule: str | None = None) -> None:
        if o.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
            o.transition(OrderStatus.REJECTED, why)
        o.reject_reason = why
        res.rejected.append(o)
        rec = o.meta.get("audit")
        if rec is not None:
            rec.update(final_action=NO_TRADE, order_status=o.status.value, reason=why,
                       rule_triggered=rule or rec.get("rule_triggered"))

    def _event(self, exc: Exception, date, bot_id=None, symbol=None) -> dict:
        sev = getattr(exc, "severity", None)
        return {"date": date, "event": type(exc).__name__, "severity": getattr(sev, "value", "CRITICAL"),
                "component": getattr(exc, "component", "system"), "bot_id": bot_id, "symbol": symbol,
                "detail": str(exc), "rule": getattr(exc, "context", {}).get("rule")}

    def _fail(self, res: CycleResult, exc: Exception, component: str, date, bot_id: str | None = None) -> None:
        """Fail closed: record, log with traceback, halt if CRITICAL/unknown."""
        tb = traceback.format_exc()
        log.error("cycle failure [%s %s]: %s\n%s", component, bot_id or "", exc, tb)
        evt = self.gate.handle_error(exc, component)
        res.events.append({**evt, "date": date, "bot_id": bot_id, "traceback": tb[-2000:]})
        if not isinstance(exc, TradingSystemError):
            log.critical("UNKNOWN exception in trading path treated as CRITICAL (%s)", type(exc).__name__)
