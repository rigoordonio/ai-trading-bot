"""Event-driven daily backtester (framework-independent).

Day t loop:
    1. OPEN of t   : fill orders decided at the close of t-delay (sells first)
    2. CLOSE of t  : mark sleeves to market
    3. CLOSE of t  : TradingCycle (same code as paper trading) decides
    4. queue approved orders for the open of t+delay
    5. snapshot sleeves and account

No order is ever filled at a price that was known when it was decided: the
reference price is the NEXT open, plus spread, slippage and impact.

Works for a single strategy (StrategyBot adapter) or several bots sharing one
account, each with its own capital sleeve.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_metrics, multi_bot_metrics
from src.bots.bot import BotVersions
from src.config import AppConfig, RiskConfig
from src.data.universe import Universe
from src.execution.orders import Order, OrderStatus, Side
from src.execution.simulator import CostModel, ExecutionSimulator
from src.logging_setup import get_logger
from src.portfolio.portfolio import Portfolio
from src.portfolio.risk_manager import PortfolioRiskManager
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import RiskEngine
from src.risk.stop_loss import StopLossManager
from src.safety.health import HealthMonitor, TradingGate
from src.safety.order_validator import OrderValidator
from src.strategies.base import SignalFrame
from src.trading_cycle import BotRuntime, MarketState, TradingCycle

log = get_logger(__name__)


class StrategyBot:
    """Adapter so a single strategy (or the classic ensemble) runs as a 'bot'
    with 100% of capital. Used for independent strategy backtests."""

    def __init__(self, name: str, symbols: list[str], risk_cfg: RiskConfig, signal_features: list[str],
                 rebalance_frequency: str = "daily", feature_version: str = "", strategy_version: str = "research"):
        self.bot_id = name
        self.symbols = symbols
        self.risk_cfg = risk_cfg
        self._signal_features = signal_features
        self._freq = rebalance_frequency
        self._versions = BotVersions(name, strategy_version, feature_version, None)
        self.capital_allocation = 1.0

    @property
    def signal_features(self):
        return self._signal_features

    @property
    def rebalance_frequency(self):
        return self._freq

    @property
    def versions(self):
        return self._versions


@dataclass
class BotInput:
    bot: Any                    # TradingBot or StrategyBot
    panel: pd.DataFrame         # this bot's features, (date, symbol)
    signal: SignalFrame


@dataclass
class BacktestResult:
    run_id: str
    equity: pd.DataFrame                  # account-level daily snapshot
    bot_equity: dict[str, pd.DataFrame]
    trades: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    audit: pd.DataFrame
    risk_events: pd.DataFrame
    benchmark: pd.Series | None
    metrics: dict
    bot_metrics: dict[str, dict]
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        keys = ["total_return", "annualised_return", "annualised_volatility", "sharpe", "sortino", "max_drawdown",
                "calmar", "num_trades", "win_rate", "profit_factor", "turnover", "benchmark_total_return",
                "benchmark_sharpe", "beta", "information_ratio"]
        lines = [f"Backtest {self.run_id}  {m.get('start')} -> {m.get('end')}"]
        for k in keys:
            if k in m:
                v = m[k]
                lines.append(f"  {k:<26}{v:>12.4f}" if isinstance(v, float) else f"  {k:<26}{v!s:>12}")
        for k in ("avg_gross_exposure", "capital_utilisation", "avg_holding_days", "avg_pairwise_bot_correlation",
                  "diversification_ratio"):
            if k in m and isinstance(m[k], float):
                lines.append(f"  {k:<26}{m[k]:>12.4f}")
        for b, bm in self.bot_metrics.items():
            lines.append(f"  [{b}] return {bm['total_return']:+.2%}  ann {bm['annualised_return']:+.2%}  "
                         f"vol {bm['annualised_volatility']:.2%}  sharpe {bm['sharpe']:.2f}  "
                         f"sortino {bm['sortino']:.2f}  maxDD {bm['max_drawdown']:.2%}  trades {bm['num_trades']}  "
                         f"win {bm['win_rate']:.0%}  PF {bm['profit_factor']:.2f}  "
                         f"hold {bm.get('avg_holding_days', float('nan')):.0f}d  "
                         f"turnover {bm['turnover']:.2f}  costs ${bm.get('transaction_costs', 0.0):,.0f}")
        return "\n".join(lines)


def _lookup(by_date: dict) -> Callable[[pd.Timestamp], pd.DataFrame]:
    def get(d: pd.Timestamp) -> pd.DataFrame:
        return by_date.get(d, pd.DataFrame())
    return get


def _activity(universe: Universe | None) -> Callable[[str, pd.Timestamp], bool]:
    """Point-in-time membership check (survivorship-bias control)."""
    if universe is None or universe.membership is None:
        return lambda s, d: True
    return universe.is_active


def _wide(prices: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return pd.DataFrame({s: df[col] for s, df in prices.items()}).sort_index()


class BacktestEngine:
    def __init__(self, cfg: AppConfig, prices: dict[str, pd.DataFrame], inputs: list[BotInput],
                 start=None, end=None, initial_capital: float | None = None, cost_model: CostModel | None = None,
                 universe: Universe | None = None, benchmark: str | None = None):
        self.cfg = cfg
        self.prices = prices
        self.inputs = inputs
        bt = cfg.settings.backtest
        self.start = pd.Timestamp(start or bt.start)
        self.end = pd.Timestamp(end) if (end or bt.end) else None
        self.capital = float(initial_capital or bt.initial_capital)
        self.delay = bt.execution_delay_bars
        self.costs = cost_model or CostModel.from_config(bt.costs)
        self.universe = universe
        self.benchmark = benchmark or cfg.settings.universe.benchmark
        self.run_id = f"BT-{pd.Timestamp.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    def run(self) -> BacktestResult:
        cfg = self.cfg
        sim = ExecutionSimulator(self.costs)
        OPN, HIG, LOW, CLS, VOL = (_wide(self.prices, c) for c in ("open", "high", "low", "close", "volume"))
        rets = CLS.pct_change()
        dates = CLS.index[(CLS.index >= self.start) & ((CLS.index <= self.end) if self.end is not None else True)]
        if len(dates) < 2:
            raise ValueError("backtest window has fewer than 2 trading days")

        total_alloc = sum(getattr(i.bot, "capital_allocation", 1.0) for i in self.inputs)
        if total_alloc > 1 + 1e-9:
            raise ValueError("bot capital allocations exceed 100%")
        unallocated = self.capital * (1 - total_alloc)

        gate = TradingGate(HealthMonitor(), operator_enabled=True)
        runtimes: dict[str, BotRuntime] = {}
        feats_by_bot: dict[str, dict] = {}
        for inp in self.inputs:
            b = inp.bot
            rc = b.risk_cfg
            feats_by_bot[b.bot_id] = {d: g.droplevel("date") for d, g in inp.panel.groupby(level="date")}
            fb = feats_by_bot[b.bot_id]
            runtimes[b.bot_id] = BotRuntime(
                bot=b, signal=inp.signal,
                features_on=_lookup(fb),
                sleeve=Portfolio(cash=self.capital * getattr(b, "capital_allocation", 1.0)),
                risk=RiskEngine(rc, cfg.settings.universe.sectors, DrawdownMonitor.from_config(rc.drawdown),
                                name=f"{b.bot_id}.risk"),
                stops=StopLossManager.from_config(rc.stops),
                active=_activity(self.universe))
        cycle = TradingCycle(runtimes, PortfolioRiskManager.from_config(cfg.risk),
                             OrderValidator.from_config(cfg.risk), gate, cfg.risk.safety, set(CLS.columns),
                             unallocated_cash=lambda: unallocated, environment="backtest", run_id=self.run_id)

        pending: dict[int, list[Order]] = {}
        all_orders: list[Order] = []
        fills, audit, events, account_rows = [], [], [], []
        bot_rows: dict[str, list] = {b: [] for b in runtimes}
        traded_notional = {b: 0.0 for b in runtimes}

        # previous VALID close for every bar (used for the discontinuity check)
        prev_close_frame = CLS.apply(lambda col: col.dropna().shift(1).reindex(col.index))
        bar_date_frame = pd.DataFrame(np.where(CLS.notna(), CLS.index.values[:, None], np.datetime64("NaT")),
                                      index=CLS.index, columns=CLS.columns).ffill()

        for i, date in enumerate(dates):
            # 1) fills at the open
            todays = pending.pop(i, [])
            for o in sorted(todays, key=lambda o: 0 if o.side == Side.SELL else 1):
                bot_id = o.meta.get("bot_id", o.strategy)
                sleeve = runtimes[bot_id].sleeve
                sym = o.symbol
                bar = None
                if sym in OPN.columns and pd.notna(OPN.at[date, sym]):
                    bar = pd.Series({"open": OPN.at[date, sym], "high": HIG.at[date, sym], "low": LOW.at[date, sym],
                                     "close": CLS.at[date, sym], "volume": VOL.at[date, sym]})
                fill = sim.execute(o, bar, sleeve.cash, sleeve.quantity(sym), date)
                rec = o.meta.get("audit")
                if fill is None:
                    if rec is not None:
                        rec.update(order_status=o.status.value, execution=f"rejected: {o.reject_reason}")
                    events.append({"date": date, "event": "ORDER_REJECTED_AT_EXECUTION", "severity": "WARNING",
                                   "symbol": sym, "bot_id": bot_id, "detail": o.reject_reason,
                                   "component": "execution"})
                    continue
                atr = None
                prev_feats = feats_by_bot[bot_id].get(dates[i - 1]) if i > 0 else None
                if prev_feats is not None and sym in prev_feats.index:
                    atr = prev_feats.loc[sym].get("atr_14")
                sleeve.apply_fill(fill, reason=o.reason, atr=atr)
                if o.action == "REDUCE" and sym in sleeve.positions:
                    sleeve.positions[sym].reduced = True
                traded_notional[bot_id] += fill.notional
                fd = fill.to_dict()
                fd.update(bot_id=bot_id, action=o.action, reason=o.reason)
                fills.append(fd)
                if rec is not None:
                    rec.update(order_status=o.status.value, fill_price=fill.price, fill_qty=fill.quantity,
                               execution=f"filled {fill.quantity} @ {fill.price:.4f}")

            # 2) mark to market at the close
            closes_today = CLS.loc[date].to_dict()
            for rt in runtimes.values():
                rt.sleeve.mark_to_market(closes_today)

            # 3) decide at the close
            bars: dict[str, pd.Series] = {}
            bdates: dict[str, pd.Timestamp] = {}
            prev: dict[str, float | None] = {}
            for sym in CLS.columns:
                bd = bar_date_frame.at[date, sym]
                if pd.isna(bd):
                    continue
                bd = pd.Timestamp(bd)
                bars[sym] = pd.Series({"open": OPN.at[bd, sym], "high": HIG.at[bd, sym], "low": LOW.at[bd, sym],
                                       "close": CLS.at[bd, sym], "volume": VOL.at[bd, sym]})
                bdates[sym] = bd
                pc = prev_close_frame.at[bd, sym]
                prev[sym] = None if pd.isna(pc) else float(pc)
            ms = MarketState(date, bars, bdates, prev, rets.loc[:date].tail(80))
            res = cycle.run(ms)
            audit += res.audit
            events += [{**e, "date": e.get("date", date)} for e in res.events]
            all_orders += res.orders + res.rejected
            if i + self.delay < len(dates):
                for o in res.orders:
                    o.execute_on = dates[i + self.delay]
                    o.transition(OrderStatus.SUBMITTED, "queued for next open")
                    pending.setdefault(i + self.delay, []).append(o)
            else:
                for o in res.orders:
                    o.transition(OrderStatus.CANCELLED, "backtest ended before execution")

            # 4) snapshots
            acct = cycle.account()
            for b, rt in runtimes.items():
                bot_rows[b].append(rt.sleeve.snapshot(date))
            snap = {"date": date, "equity": acct.equity, "cash": acct.cash,
                    "positions_value": sum(p.positions_value for p in acct.sleeves.values()),
                    "n_positions": len(acct.symbol_values()), "kill_switch": cycle.portfolio_risk.drawdown.state
                    .kill_switch, "trading_enabled": gate.trading_enabled}
            snap["gross_exposure"] = snap["positions_value"] / snap["equity"] if snap["equity"] > 0 else 0.0
            account_rows.append(snap)

        for lst in pending.values():
            for o in lst:
                o.transition(OrderStatus.CANCELLED, "backtest ended")

        equity = pd.DataFrame(account_rows).set_index("date")
        bot_equity = {b: pd.DataFrame(r).set_index("date") for b, r in bot_rows.items()}
        trades = pd.DataFrame([{**t.to_dict(), "bot_id": b} for b, rt in runtimes.items()
                               for t in rt.sleeve.closed_trades])
        bench = None
        if self.benchmark in CLS.columns:
            bc = CLS[self.benchmark].reindex(equity.index).ffill()
            bench = self.capital * bc / bc.dropna().iloc[0]
        metrics = compute_metrics(equity["equity"], trades["pnl"] if len(trades) else [],
                                  sum(traded_notional.values()), bench)
        bot_metrics = {}
        for b, be in bot_equity.items():
            bt = trades[trades["bot_id"] == b]["pnl"] if len(trades) else []
            bench_b = bench * (be["equity"].iloc[0] / self.capital) if bench is not None else None
            bot_metrics[b] = compute_metrics(be["equity"], bt, traded_notional[b], bench_b)
            bt_df = trades[trades["bot_id"] == b] if len(trades) else pd.DataFrame()
            bot_metrics[b]["avg_holding_days"] = (float(bt_df["holding_days"].mean())
                                                  if len(bt_df) else float("nan"))
            bf = [f for f in fills if f.get("bot_id") == b]
            cost = float(sum(f.get("commission", 0.0) + f.get("spread_cost", 0.0) + f.get("slippage_cost", 0.0)
                             + f.get("impact_cost", 0.0) for f in bf))
            bot_metrics[b]["transaction_costs"] = cost
            bot_metrics[b]["cost_bps_of_notional"] = (cost / traded_notional[b] * 1e4
                                                      if traded_notional[b] > 0 else 0.0)
        alloc = {i.bot.bot_id: float(getattr(i.bot, "capital_allocation", 1.0)) for i in self.inputs}
        metrics.update(multi_bot_metrics(equity, bot_equity, alloc))
        metrics["avg_holding_days"] = float(trades["holding_days"].mean()) if len(trades) else float("nan")
        log.info("backtest %s done: %d fills, %d trades, %d events", self.run_id, len(fills), len(trades), len(events))
        return BacktestResult(
            run_id=self.run_id, equity=equity, bot_equity=bot_equity, trades=trades, fills=pd.DataFrame(fills),
            orders=pd.DataFrame([o.to_dict() | {"bot_id": o.meta.get("bot_id")} for o in all_orders]),
            audit=pd.DataFrame(audit), risk_events=pd.DataFrame(events), benchmark=bench, metrics=metrics,
            bot_metrics=bot_metrics,
            config={"start": str(dates[0].date()), "end": str(dates[-1].date()), "capital": self.capital,
                    "costs": self.costs.__dict__, "bots": [i.bot.bot_id for i in self.inputs],
                    "delay_bars": self.delay})
