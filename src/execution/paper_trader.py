"""Daily paper-trading run (after the US close). Fails CLOSED at every step.

    0  banner: TRADING MODE: PAPER / REAL MONEY: DISABLED
    1  fill yesterday's acknowledged orders at today's open (paper broker)
    2  apply broker fills to the bots' sleeves
    3  resolve UNKNOWN orders, reconcile sleeves vs broker  (fail -> halt)
    4  download + validate market data                      (fail -> no trade for that symbol / halt)
    5  per bot: features -> model inference (validated)    (fail -> THAT bot makes no new
                                                             trades; other bots continue;
                                                             its stop-losses still run)
    6  TradingCycle: signals -> sleeve risk -> portfolio risk -> order validator -> gate
    7  OrderManager.submit (idempotent, UNKNOWN-aware)
    8  persist sleeves, audit trail, events, health status

The same FeaturePipeline / bots / TradingCycle as the backtester are used.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtesting.engine import _lookup
from src.bots.bot import TradingBot, build_bots
from src.config import AppConfig, resolve_path
from src.data.downloader import build_downloader
from src.data.provider import MarketDataProvider, get_provider
from src.errors import Severity, StaleMarketDataError, TradingSystemError
from src.execution.broker import BrokerAdapter, create_broker, startup_banner
from src.execution.order_manager import FileOrderLedger, OrderManager
from src.execution.orders import Side
from src.execution.paper_broker import PaperBroker
from src.execution.reconciliation import reconcile
from src.execution.simulator import CostModel
from src.logging_setup import get_logger
from src.models.prediction import Predictor, load_model
from src.portfolio.attribution import virtual_positions
from src.portfolio.portfolio import Portfolio, Position
from src.portfolio.risk_manager import PortfolioRiskManager
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import RiskEngine
from src.risk.stop_loss import StopLossManager
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthMonitor, HealthStatus, TradingGate
from src.safety.order_validator import OrderValidator
from src.trading_cycle import BotRuntime, MarketState, TradingCycle

log = get_logger(__name__)


NY = ZoneInfo("America/New_York")


def last_completed_session(now: datetime) -> pd.Timestamp:
    """Date of the most recent New York session that has CLOSED (16:30 ET buffer
    for the daily bar to settle). Weekends roll back to Friday. Exchange
    holidays are handled later: the data simply has no bar for them."""
    ny = now.astimezone(NY)
    d = pd.Timestamp(ny.date())
    if ny.time() < time(16, 30):
        d -= pd.Timedelta(days=1)
    while d.dayofweek >= 5:
        d -= pd.Timedelta(days=1)
    return d


# ------------------------------------------------------------ sleeve state
def sleeve_to_dict(p: Portfolio) -> dict:
    return {"cash": p.cash, "realized_pnl": p.realized_pnl,
            "positions": {s: {**pos.to_dict(), "entry_date": str(pos.entry_date)} for s, pos in p.positions.items()},
            "closed_trades": [{**t.to_dict(), "entry_date": str(t.entry_date), "exit_date": str(t.exit_date)}
                              for t in p.closed_trades[-500:]]}


def sleeve_from_dict(d: dict) -> Portfolio:
    p = Portfolio(cash=float(d["cash"]), realized_pnl=float(d.get("realized_pnl", 0.0)))
    for s, x in d.get("positions", {}).items():
        p.positions[s] = Position(s, int(x["quantity"]), float(x["avg_cost"]), pd.Timestamp(x["entry_date"]),
                                  float(x["last_price"]), float(x["highest_close"]), x.get("entry_atr"),
                                  bool(x.get("reduced", False)), float(x.get("realized_pnl", 0.0)),
                                  float(x.get("costs", 0.0)), x.get("entry_reason", ""))
    return p


@dataclass
class PaperRunReport:
    run_id: str
    as_of: str
    trading_enabled: bool
    blocking_reasons: list[str]
    submitted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    reconciliation: dict | None = None
    events: list[dict] = field(default_factory=list)


class PaperTrader:
    def __init__(self, cfg: AppConfig, state_dir: str | Path = "data/paper", provider: MarketDataProvider | None = None,
                 ledger=None, repo=None, gate: TradingGate | None = None, bots: dict[str, TradingBot] | None = None,
                 initial_capital: float | None = None, broker: BrokerAdapter | None = None,
                 broker_name: str | None = None):
        self.cfg = cfg
        self.state_dir = resolve_path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider or get_provider(cfg.settings.data.provider)
        self.gate = gate or TradingGate(HealthMonitor())
        self.repo = repo
        self.ledger = ledger if ledger is not None else FileOrderLedger(self.state_dir / "orders.json")
        self.bots = bots if bots is not None else build_bots(cfg)
        if not any(b.ml_cfg for b in self.bots.values()):
            self.gate.health.disable_component("model")
        if repo is None:
            self.gate.health.set("database", HealthStatus.DEGRADED, "no database configured (in-memory ledger)")
        known = set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark}
        name = (broker_name or os.getenv("BROKER") or "simulated").strip().lower()
        if broker is not None:
            self.broker = broker
        elif name == "alpaca_paper":
            self.broker = create_broker("PAPER", "alpaca_paper", trading_enabled=lambda: self.gate.trading_enabled,
                                        known_symbols=known)
        else:
            cap0 = float(initial_capital or cfg.settings.backtest.initial_capital)
            self.broker = PaperBroker(cap0, CostModel.from_config(cfg.settings.backtest.costs),
                                      self.state_dir / "broker_state.json",
                                      trading_enabled=lambda: self.gate.trading_enabled, known_symbols=known)
        if initial_capital is not None:
            capital = float(initial_capital)
        elif not self._sleeves_path().exists() and not isinstance(self.broker, PaperBroker):
            # first run against an external paper account: start from ITS cash, so books match
            capital = float(self.broker.get_account().cash)
        else:
            capital = float(cfg.settings.backtest.initial_capital)
        self.data_breaker = CircuitBreaker.from_config("market_data", cfg.risk.safety.circuit_breaker)
        self.broker_breaker = CircuitBreaker.from_config("broker", cfg.risk.safety.circuit_breaker)
        self.gate.breakers += [self.data_breaker]
        self.orders = OrderManager(self.broker, self.ledger, self.gate, self.broker_breaker,
                                   cfg.risk.safety.allow_risk_reducing_orders_when_halted)
        self.sleeves = self._load_sleeves(capital)
        status_path = self.state_dir / "system_status.json"
        if status_path.exists():   # carry "last market data / signal / order / reconciliation" across runs
            try:
                prev = json.loads(status_path.read_text())
                self.gate.health.last_seen.update(prev.get("health", {}).get("last_seen", {}))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("previous status file unreadable (%s); starting fresh", exc)
        self.applied_fills_path = self.state_dir / "applied_fills.json"
        self.booked_fills: list[dict] = []          # fills booked to sleeves during this run

    # ------------------------------------------------------------ persistence
    def _sleeves_path(self) -> Path:
        return self.state_dir / "sleeves.json"

    def _load_sleeves(self, capital: float) -> dict[str, Portfolio]:
        path = self._sleeves_path()
        self.saved_drawdown: dict = {}
        self.last_run_date: pd.Timestamp | None = None
        if path.exists():
            raw = json.loads(path.read_text())
            out = {b: sleeve_from_dict(d) for b, d in raw.get("sleeves", {}).items()}
            self.unallocated = float(raw.get("unallocated_cash", 0.0))
            self.saved_drawdown = raw.get("drawdown", {})
            self.last_run_date = pd.Timestamp(raw["last_run_date"]) if raw.get("last_run_date") else None
        else:
            out = {b: Portfolio(cash=capital * bot.capital_allocation) for b, bot in self.bots.items()}
            self.unallocated = capital * (1 - sum(b.capital_allocation for b in self.bots.values()))
        for b in self.bots:
            out.setdefault(b, Portfolio(cash=0.0))
        return out

    def _save_sleeves(self, drawdown: dict | None = None, last_run_date=None) -> None:
        tmp = self._sleeves_path().with_suffix(".tmp")
        tmp.write_text(json.dumps({"unallocated_cash": self.unallocated,
                                   "last_run_date": str(last_run_date or self.last_run_date or ""),
                                   "drawdown": drawdown if drawdown is not None else self.saved_drawdown,
                                   "sleeves": {b: sleeve_to_dict(p) for b, p in self.sleeves.items()}},
                                  indent=2, default=str))
        os.replace(tmp, self._sleeves_path())

    # --------------------------------------------------------------- helpers
    def internal_positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.sleeves.values():
            for s, pos in p.positions.items():
                out[s] = out.get(s, 0) + pos.quantity
        return out

    def internal_cash(self) -> float:
        return self.unallocated + sum(p.cash for p in self.sleeves.values())

    def _apply_broker_fills(self) -> None:
        """Book broker fills into the bots' sleeves exactly once.

        Simulated broker: itemised fills (with commissions).
        External paper API (Alpaca): per-order cumulative filled quantity and
        average price; only the increment since the last run is booked."""
        from src.execution.orders import Fill

        raw = json.loads(self.applied_fills_path.read_text()) if self.applied_fills_path.exists() else {}
        if isinstance(raw, list):                     # older format: list of itemised fill ids
            raw = {"fills": raw, "qty": {}, "notional": {}}
        applied = set(raw.get("fills", []))
        applied_qty: dict[str, int] = raw.get("qty", {})
        applied_notional: dict[str, float] = raw.get("notional", {})

        for bo in self.broker.get_orders():
            if not bo.fills and bo.filled_quantity > applied_qty.get(bo.broker_order_id, 0):
                prev_q = applied_qty.get(bo.broker_order_id, 0)
                prev_n = applied_notional.get(bo.broker_order_id, 0.0)
                dq = bo.filled_quantity - prev_q
                if bo.avg_fill_price is None or not dq > 0:
                    raise TradingSystemError(f"broker order {bo.broker_order_id}: filled qty without a price")
                # price of the increment = total notional now minus notional already booked
                total_n = bo.avg_fill_price * bo.filled_quantity
                px = (total_n - prev_n) / dq
                o = self.ledger.get_by_key(bo.client_order_id)
                bot = (o.meta.get("bot_id") if o is not None else None) or (o.strategy if o is not None else None)
                if bot not in self.sleeves:
                    raise TradingSystemError(f"fill of {bo.broker_order_id} cannot be attributed to a bot "
                                             "(order not in our ledger)")
                ts = pd.Timestamp(bo.filled_at) if bo.filled_at else pd.Timestamp.today()
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
                fill = Fill(bo.broker_order_id, bo.symbol, Side(bo.side), dq, px, px, 0.0, ts.normalize())
                self.sleeves[bot].apply_fill(fill, reason=o.reason if o else "")
                self.booked_fills.append({**fill.to_dict(), "bot_id": bot, "broker_order_id": bo.broker_order_id})
                if o is not None and o.action == "REDUCE" and bo.symbol in self.sleeves[bot].positions:
                    self.sleeves[bot].positions[bo.symbol].reduced = True
                applied_qty[bo.broker_order_id] = bo.filled_quantity
                applied_notional[bo.broker_order_id] = total_n
                continue
            for f in bo.fills:
                fid = f"{bo.broker_order_id}:{f['timestamp']}:{f['quantity']}"
                if fid in applied:
                    continue
                o = self.ledger.get_by_key(bo.client_order_id)
                bot = (o.meta.get("bot_id") if o is not None else None) or (o.strategy if o is not None else None)
                if bot not in self.sleeves:
                    raise TradingSystemError(f"fill {fid} cannot be attributed to a bot")
                fill = Fill(bo.broker_order_id, f["symbol"], Side(f["side"]), int(f["quantity"]), float(f["price"]),
                            float(f["reference_price"]), float(f["commission"]), pd.Timestamp(f["timestamp"]),
                            f.get("spread_cost", 0.0), f.get("slippage_cost", 0.0), f.get("impact_cost", 0.0))
                self.sleeves[bot].apply_fill(fill, reason=o.reason if o else "")
                self.booked_fills.append({**fill.to_dict(), "bot_id": bot, "broker_order_id": bo.broker_order_id})
                if o is not None and o.action == "REDUCE" and f["symbol"] in self.sleeves[bot].positions:
                    self.sleeves[bot].positions[f["symbol"]].reduced = True
                applied.add(fid)
        # Persist the books BEFORE marking fills as applied: a run that stops
        # later (e.g. a refused re-run of the same session) must not lose them.
        # A crash between the two writes double-books at worst, which the next
        # reconciliation catches (fail closed); losing fills silently is worse.
        self._save_sleeves()
        self.applied_fills_path.write_text(json.dumps({"fills": sorted(applied), "qty": applied_qty,
                                                       "notional": applied_notional}))
        self.orders.refresh_open_orders()

    # ------------------------------------------------------------------ run
    def run(self, as_of: pd.Timestamp | None = None, now: datetime | None = None) -> PaperRunReport:
        """``as_of`` replays a specific date (tests / catch-up). Without it the
        run uses the latest COMPLETED New York session, whatever the local
        (e.g. Manila) date is, and never a bar from a session still trading."""
        startup_banner("PAPER")
        live = as_of is None
        if live:
            as_of = last_completed_session(now or datetime.now(tz=NY))
        as_of = pd.Timestamp(as_of).normalize()
        run_id = f"PAPER-{as_of:%Y%m%d}-{uuid.uuid4().hex[:6]}"
        events: list[dict] = []
        cfg = self.cfg
        symbols = sorted(set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark})

        # 4) data (needed for fills too)
        dl = build_downloader(cfg, self.provider)
        start = as_of - pd.Timedelta(days=cfg.settings.data.paper_history_days)
        try:
            etl = self.data_breaker.call(dl.run, symbols, start, as_of)
            if etl.failed:
                events.append({"event": "DATA_FAILURES", "severity": "ERROR", "component": "market_data",
                               "detail": json.dumps(etl.failed)})
            if not etl.data:
                raise TradingSystemError("no market data at all")
            self.gate.health.set("market_data", HealthStatus.HEALTHY if not etl.failed else HealthStatus.DEGRADED,
                                 f"{len(etl.data)} symbols")
            self.gate.health.mark("last_market_data", str(max(df.index.max() for df in etl.data.values()).date()))
        except Exception as exc:  # noqa: BLE001
            self.gate.handle_error(exc if isinstance(exc, TradingSystemError) else
                                   TradingSystemError(f"market data failed: {exc}"), "market_data")
            why = (f"market data unavailable ({type(exc).__name__}: {str(exc)[:160]}). Nothing was decided or "
                   "sent; check the internet connection (the scheduled run retries by itself).")
            log.warning(why)
            report = self._finish(run_id, as_of, events)
            report.blocking_reasons = [*report.blocking_reasons, why]
            return report
        prices = {s: df.loc[:as_of] for s, df in etl.data.items() if not df.loc[:as_of].empty}

        # Decide on the latest session actually present in the data (US time),
        # and refuse if that session is too old (stale feed / long outage).
        ref = prices.get(cfg.settings.universe.benchmark)
        decision_date = (ref.index.max() if ref is not None and not ref.empty
                         else max(df.index.max() for df in prices.values()))
        lag = (as_of - decision_date).days
        if lag > cfg.risk.safety.max_data_staleness_days:
            self.gate.handle_error(StaleMarketDataError(
                f"latest market data is {decision_date.date()}, {lag} days before {as_of.date()}"), "market_data")
            return self._finish(run_id, as_of, events)
        expected_session = as_of
        as_of = decision_date
        run_id = f"PAPER-{as_of:%Y%m%d}-{uuid.uuid4().hex[:6]}"

        # 1-2) yesterday's orders fill at today's open, then apply to sleeves
        try:
            open_bars = {s: df.loc[as_of] for s, df in prices.items() if as_of in df.index}
            self.broker.sync(as_of, open_bars)
            self._apply_broker_fills()
            self.gate.health.set("broker", HealthStatus.HEALTHY, f"{self.broker.name} reachable")
        except Exception as exc:  # noqa: BLE001
            self.gate.handle_error(exc, "broker")

        # 3) reconcile (UNKNOWN orders first)
        self.orders.reconcile_unknown()
        rep = reconcile(self.internal_positions(), self.internal_cash(), self.ledger, self.broker, self.gate,
                        cfg.risk.safety.reconciliation_cash_tolerance, attribution=virtual_positions(self.sleeves))

        # Refuse to DECIDE when the broker cannot accept orders now (e.g. outside
        # Alpaca's OPG window): deciding anyway would reject every order AND use
        # up this session, so a later run in the window would be refused.
        window_ok, why = self.broker.submission_window_open()
        if not window_ok:
            events.append({"event": "SUBMISSION_WINDOW_CLOSED", "severity": "WARNING", "component": "broker",
                           "detail": why})
            log.warning("no decision made: %s", why)
            report = self._finish(run_id, as_of, events, rep)
            report.blocking_reasons = [*report.blocking_reasons, why]
            return report

        # 5) features + model per bot. A failure is BOT-SCOPED: that bot's
        #    signal generation is disabled (no prediction is ever replaced by a
        #    default/bullish/previous value), the other bots continue, and its
        #    existing positions keep their stop-losses and count in account risk.
        runtimes: dict[str, BotRuntime] = {}
        model_failures: dict[str, str] = {}
        from src.reviewer.review import multipliers as reviewer_multipliers
        review_mult, review_err = reviewer_multipliers(self.state_dir, list(self.bots))
        if review_err:
            events.append({"event": "REVIEWER_STATE_ERROR", "severity": "ERROR", "component": "reviewer",
                           "detail": review_err})
            log.error(review_err)
        run_predictions: dict[str, tuple] = {}
        for bot_id, bot in self.bots.items():
            rc = bot.risk_cfg
            dd = DrawdownMonitor.from_config(rc.drawdown).load_dict(self.saved_drawdown.get(bot_id))
            risk = RiskEngine(rc, cfg.settings.universe.sectors, dd, f"{bot_id}.risk")
            stops = StopLossManager.from_config(rc.stops)
            try:
                panel = bot.compute_features(prices)
            except Exception as exc:  # noqa: BLE001 - bot-scoped, recorded below
                log.exception("%s: feature calculation failed; bot disabled for this run", bot_id)
                self._bot_failed(bot_id, "features", exc, events)
                runtimes[bot_id] = BotRuntime.disabled(bot, self.sleeves[bot_id], risk, stops,
                                                       f"features failed: {type(exc).__name__}: {exc}")
                continue
            fb = {d: g.droplevel("date") for d, g in panel.groupby(level="date")}
            preds = None
            if bot.ml_cfg:
                try:
                    lm = load_model(resolve_path(cfg.settings.ml.artifacts_dir), bot_id, bot.ml_cfg.model_version,
                                    bot.feature_version)
                    last = panel.xs(panel.index.get_level_values("date").max(), level="date", drop_level=False)
                    preds = Predictor.from_loaded(lm).predict(last, bot.feature_version)
                    run_predictions[bot_id] = (bot.ml_cfg.model_version, bot.feature_version, preds)
                except Exception as exc:  # noqa: BLE001 - MODEL FAILURE -> this bot makes NO new trades
                    model_failures[bot_id] = f"{type(exc).__name__}: {exc}"
                    self._bot_failed(bot_id, "model", exc, events)
                    runtimes[bot_id] = BotRuntime.disabled(bot, self.sleeves[bot_id], risk, stops,
                                                           f"model failed: {type(exc).__name__}: {exc}",
                                                           features_on=_lookup(fb))
                    continue
            try:
                sf = bot.signal_frame(panel, preds)
            except Exception as exc:  # noqa: BLE001
                self._bot_failed(bot_id, "signal", exc, events)
                runtimes[bot_id] = BotRuntime.disabled(bot, self.sleeves[bot_id], risk, stops,
                                                       f"signal failed: {type(exc).__name__}: {exc}",
                                                       features_on=_lookup(fb))
                continue
            mult, rstatus = review_mult.get(bot_id, (1.0, "ACTIVE"))
            runtimes[bot_id] = BotRuntime(bot, sf, _lookup(fb), self.sleeves[bot_id], risk, stops,
                                          buy_size_multiplier=mult, reviewer_status=rstatus)
            last_scores = sf.on(panel.index.get_level_values("date").max())["score"]
            if len(last_scores) and last_scores.isna().all():
                # Not an error (fail closed = HOLD), but a bot that can never score must be visible.
                missing = sorted({c for c in (bot.ml_features + bot.signal_features) if c in panel.columns
                                  and panel.xs(panel.index.get_level_values("date").max(), level="date")[c]
                                  .isna().all()})
                detail = f"no valid score for any symbol (all HOLD); features missing: {missing or 'model output'}"
                self.gate.health.set(f"bot:{bot_id}", HealthStatus.DEGRADED, detail)
                events.append({"event": "BOT_NO_SCORES", "severity": "WARNING", "component": f"bot:{bot_id}",
                               "bot_id": bot_id, "detail": detail})
                log.warning("%s: %s", bot_id, detail)
            else:
                self.gate.health.set(f"bot:{bot_id}", HealthStatus.HEALTHY, "signals enabled")
        if any(b.ml_cfg for b in self.bots.values()):
            if model_failures:
                self.gate.health.set("model", HealthStatus.DEGRADED,
                                     "disabled bots: " + "; ".join(f"{b} ({e})" for b, e in model_failures.items()))
            else:
                self.gate.health.set("model", HealthStatus.HEALTHY, "all bot models loaded")
        if not runtimes:
            return self._finish(run_id, as_of, events, rep)

        # 6) decide
        closes = {s: float(df["close"].loc[:as_of].iloc[-1]) for s, df in prices.items() if not df.loc[:as_of].empty}
        for p in self.sleeves.values():
            p.mark_to_market(closes)
        self.broker.mark(closes)
        bars: dict[str, pd.Series] = {}
        bdates: dict[str, pd.Timestamp] = {}
        prev: dict[str, float | None] = {}
        for s, df in prices.items():
            d = df.loc[:as_of]
            if d.empty:
                continue
            bars[s], bdates[s] = d.iloc[-1], d.index[-1]
            prev[s] = float(d["close"].iloc[-2]) if len(d) > 1 else None
        rets = pd.DataFrame({s: df["close"] for s, df in prices.items()}).pct_change().loc[:as_of].tail(80)
        prm = PortfolioRiskManager.from_config(cfg.risk)
        prm.drawdown.load_dict(self.saved_drawdown.get("ACCOUNT"))
        cycle = TradingCycle(runtimes, prm, OrderValidator.from_config(cfg.risk),
                             self.gate, cfg.risk.safety, set(symbols), lambda: self.unallocated, "paper", run_id,
                             sleeves=self.sleeves)
        if self.last_run_date is not None and self.last_run_date < as_of:
            cycle._prev_date = self.last_run_date          # weekly/monthly bots rebalance on schedule
        if self.last_run_date is not None and self.last_run_date >= as_of and expected_session > as_of:
            # The provider has not delivered the latest session yet (e.g. a blank row
            # right after the close). Not an error and nothing is used up: a later
            # run the same day decides normally once the data is complete.
            why = (f"prices for the {expected_session.date()} session are not available yet from the data "
                   f"provider (latest complete session: {as_of.date()}, already decided). Nothing was decided "
                   "or sent; a later run today will pick it up.")
            events.append({"event": "DATA_NOT_READY", "severity": "WARNING", "component": "market_data",
                           "detail": why})
            log.warning(why)
            report = self._finish(run_id, as_of, events, rep)
            report.blocking_reasons = [*report.blocking_reasons, why]
            return report
        if self.last_run_date is not None and self.last_run_date >= as_of:
            self.gate.handle_error(TradingSystemError(
                f"already ran for {as_of.date()} (last run {self.last_run_date.date()}): refusing to decide twice",
                severity=Severity.ERROR), "execution_engine")
            return self._finish(run_id, as_of, events, rep)
        res = cycle.run(MarketState(as_of, bars, bdates, prev, rets))
        events += res.events
        self.gate.health.mark("last_signal", f"{as_of.date()} ({len(res.audit)} evaluations)")

        # 7) submit through the OrderManager (idempotent, UNKNOWN-aware)
        submitted, rejected = [], [o.to_dict() for o in res.rejected]
        for o in res.orders:
            try:
                r = self.orders.submit(o)
                submitted.append({**o.to_dict(), "note": r.note})
                if o.meta.get("audit"):
                    o.meta["audit"].update(order_status=o.status.value, broker_order_id=o.broker_order_id)
            except TradingSystemError as exc:
                rejected.append({**o.to_dict(), "error": str(exc)})
                events.append({"event": type(exc).__name__, "severity": exc.severity.value, "component": "broker",
                               "symbol": o.symbol, "detail": str(exc)})
                if o.meta.get("audit"):
                    o.meta["audit"].update(final_action="NO_TRADE", order_status=o.status.value, reason=str(exc))

        # 8) persist (sleeves, drawdown/kill-switch state, last run date)
        dd_state = {b: rt.risk.drawdown.to_dict() for b, rt in runtimes.items()}
        dd_state["ACCOUNT"] = prm.drawdown.to_dict()
        self.saved_drawdown, self.last_run_date = dd_state, as_of
        self._save_sleeves(dd_state, as_of)
        self._save_history(as_of, res.signals, closes.get(cfg.settings.universe.benchmark), res.audit)
        if self.repo is not None:
            try:
                self.repo.save_audit(res.audit)
                self.repo.save_events(events + self.gate.events, run_id)
                self.repo.save_paper_state(run_id, as_of, self.sleeves, self.booked_fills, self.unallocated)
                for b, (mv, fv, p) in run_predictions.items():
                    self.repo.save_predictions(b, mv, fv, p, out_of_sample=True)
                self.gate.health.set("database", HealthStatus.HEALTHY, "writes ok")
            except Exception as exc:  # noqa: BLE001
                self.gate.handle_error(exc, "database")
        return self._finish(run_id, as_of, events, rep, submitted, rejected)

    def _save_history(self, as_of: pd.Timestamp, signals: list, benchmark_close: float | None = None,
                      audit: list | None = None) -> None:
        """Per-bot daily snapshots (for the multi-bot dashboard: daily P&L,
        return, volatility, Sharpe, drawdown) and the latest TradingSignals.
        Reporting only: a failure here is logged and never affects trading."""
        try:
            rows = [{"date": str(as_of.date()), "bot_id": b, "equity": p.equity, "cash": p.cash,
                     "positions_value": p.positions_value, "n_positions": len(p.positions),
                     "n_trades": len(p.closed_trades), "realized_pnl": p.realized_pnl}
                    for b, p in self.sleeves.items()]
            rows.append({"date": str(as_of.date()), "bot_id": "ACCOUNT",
                         "equity": self.unallocated + sum(p.equity for p in self.sleeves.values()),
                         "cash": self.internal_cash(),
                         "positions_value": sum(p.positions_value for p in self.sleeves.values()),
                         "n_positions": len(self.internal_positions()),
                         "n_trades": sum(len(p.closed_trades) for p in self.sleeves.values()),
                         "realized_pnl": sum(p.realized_pnl for p in self.sleeves.values())})
            for r in rows:
                r["benchmark_close"] = benchmark_close
            path = self.state_dir / "snapshots.csv"
            df = pd.DataFrame(rows)
            if path.exists():                     # keep columns aligned with older files
                old_cols = list(pd.read_csv(path, nrows=0).columns)
                if "benchmark_close" not in old_cols:
                    old = pd.read_csv(path)
                    old["benchmark_close"] = None
                    old.to_csv(path, index=False)
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            if audit:                             # every decision: signal, reason, veto, order (step "log it all")
                ddir = self.state_dir / "decisions"
                ddir.mkdir(parents=True, exist_ok=True)
                cols = ["timestamp", "bot_id", "symbol", "action", "score", "ml_probability", "risk_status",
                        "portfolio_status", "rule_triggered", "reason", "final_action", "quantity", "order_id",
                        "order_status", "signal_id"]
                pd.DataFrame(audit).reindex(columns=cols).to_csv(ddir / f"decisions-{as_of.date()}.csv",
                                                                 index=False)
            (self.state_dir / "last_signals.json").write_text(
                json.dumps([sig.to_dict() for sig in signals], indent=1, default=str))
        except Exception as exc:  # noqa: BLE001 - reporting only
            log.warning("could not write paper history: %s", exc)

    def _bot_failed(self, bot_id: str, stage: str, exc: Exception, events: list[dict]) -> None:
        detail = f"{stage} failed: {type(exc).__name__}: {exc}"
        self.gate.health.set(f"bot:{bot_id}", HealthStatus.UNHEALTHY, detail)
        events.append({"event": "BOT_DISABLED", "severity": "ERROR", "component": f"bot:{bot_id}",
                       "bot_id": bot_id, "detail": detail})
        log.error("%s DISABLED for this run (%s); other bots continue", bot_id, detail)

    def _account_summary(self) -> dict:
        try:
            a = self.broker.get_account()
            pos = self.broker.get_positions()
            return {"broker": self.broker.name, "broker_cash": a.cash, "broker_equity": a.equity,
                    "positions": {s: p.quantity for s, p in pos.items()}}
        except Exception as exc:  # noqa: BLE001 - status reporting only; trading decisions never use this
            return {"broker": self.broker.name, "error": f"unavailable: {type(exc).__name__}"}

    def _finish(self, run_id, as_of, events, rep=None, submitted=(), rejected=()) -> PaperRunReport:
        report = PaperRunReport(run_id, str(as_of.date()), self.gate.trading_enabled, self.gate.blocking_reasons(),
                                list(submitted), list(rejected),
                                {"ok": rep.ok, "discrepancies": rep.discrepancies, "at": rep.at} if rep else None,
                                events + self.gate.events)
        status = {**self.gate.status(), "run_id": run_id, "as_of": report.as_of,
                  "reconciliation": report.reconciliation, "submitted": len(report.submitted),
                  "rejected": len(report.rejected),
                  "account": self._account_summary(),
                  "recent_events": [{k: str(v) for k, v in e.items()} for e in report.events[-50:]]}
        (self.state_dir / "system_status.json").write_text(json.dumps(status, indent=2, default=str))
        return report
