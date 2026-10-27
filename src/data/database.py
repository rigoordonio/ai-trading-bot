"""Database models and repository (SQLAlchemy 2.0; SQLite locally, PostgreSQL in docker).

Tables: market_prices, features, signals (decision audit), orders, executions,
positions, portfolio_snapshots, model_predictions, backtest_runs,
backtest_bot_results, risk_events.

"Show everything SWING_V1 did with NVDA": Repository.bot_activity("SWING_V1", "NVDA")
or `python scripts/bot_activity.py --bot SWING_V1 --symbol NVDA --db`.

Schema changes: Repository.init() creates missing tables and ADDS missing
nullable columns to existing tables (additive migration only; nothing is
dropped or rewritten).

Every table that relates to decisions carries ``bot_id`` so each bot's
signal history and performance are separable. ``signals`` is the audit
trail: one row per evaluated (bot, symbol, bar) with the scores, model
probability, versions, risk decision, rule, order id/status and execution.

All writes go through ``Repository``; any SQLAlchemy failure is re-raised as
``DatabaseError`` (CRITICAL) after a rollback, never swallowed.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.errors import DatabaseError
from src.execution.orders import TERMINAL, Order, OrderStatus, Side


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MarketPrice(Base):
    __tablename__ = "market_prices"
    __table_args__ = (UniqueConstraint("symbol", "date", "source", name="uq_price"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    source: Mapped[str] = mapped_column(String(32), default="processed")
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    raw_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    adj_factor: Mapped[float | None] = mapped_column(Float, nullable=True)


class FeatureRow(Base):
    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("bot_id", "symbol", "date", "feature_version", name="uq_feature"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    feature_version: Mapped[str] = mapped_column(String(64))
    values: Mapped[dict] = mapped_column(JSON)


class SignalRecord(Base):
    """Decision audit trail: WHY did the bot buy / refuse?"""
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    environment: Mapped[str] = mapped_column(String(16), default="backtest")
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(8), nullable=True)
    data_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)   # bar the decision used
    feature_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    components: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    final_action: Mapped[str] = mapped_column(String(16))
    risk_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rule_triggered: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    order_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    execution: Mapped[str | None] = mapped_column(Text, nullable=True)


class OrderRecord(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String(8))
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class Execution(Base):
    __tablename__ = "executions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    reference_price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    spread_cost: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
    impact_cost: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)


class PositionRecord(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("bot_id", "symbol", "environment", name="uq_position"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    environment: Mapped[str] = mapped_column(String(16), default="paper")
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[int] = mapped_column(Integer)
    avg_cost: Mapped[float] = mapped_column(Float)
    last_price: Mapped[float] = mapped_column(Float)
    entry_date: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)   # NULL = paper account
    bot_id: Mapped[str] = mapped_column(String(32), index=True)                         # "ACCOUNT" = total
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    gross_exposure: Mapped[float] = mapped_column(Float)
    n_positions: Mapped[int] = mapped_column(Integer)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(128))
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    probability: Mapped[float] = mapped_column(Float)
    out_of_sample: Mapped[bool] = mapped_column(Boolean, default=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    bots: Mapped[list] = mapped_column(JSON)
    start: Mapped[str] = mapped_column(String(10))
    end: Mapped[str] = mapped_column(String(10))
    config: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    bot_metrics: Mapped[dict] = mapped_column(JSON)


class BacktestBotResult(Base):
    """One row per (backtest run, bot): bot-level metrics are queryable by bot_id."""
    __tablename__ = "backtest_bot_results"
    __table_args__ = (UniqueConstraint("run_id", "bot_id", name="uq_bt_bot"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    bot_id: Mapped[str] = mapped_column(String(32), index=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON)


class RiskEventRecord(Base):
    __tablename__ = "risk_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    bot_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    component: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)


# ------------------------------------------------------------------ helpers
def _clean(v):
    """JSON/DB-safe scalar: NaN/inf -> None, numpy -> python, Timestamp -> datetime."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def _ts(v) -> datetime | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return pd.Timestamp(v).to_pydatetime()


class Repository:
    def __init__(self, url: str, echo: bool = False):
        self.url = url
        try:
            kwargs: dict[str, Any] = {"pool_pre_ping": True}
            if url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False}
            self.engine = create_engine(url, echo=echo, **kwargs)
            self._session = sessionmaker(self.engine, expire_on_commit=False)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"cannot create database engine: {exc}") from exc

    def init(self) -> None:
        try:
            Base.metadata.create_all(self.engine)
            self.migrate()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"schema creation failed: {exc}") from exc

    def migrate(self) -> list[str]:
        """Additive migration: add columns that exist in the models but not in
        the database (e.g. signals.timeframe on a database created by an
        older version). Only NULLABLE columns are added; returns their names."""
        from sqlalchemy import inspect

        added = []
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    continue
                have = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in have:
                        continue
                    if not col.nullable:
                        raise DatabaseError(f"cannot auto-migrate NOT NULL column {table.name}.{col.name}")
                    ddl = col.type.compile(dialect=self.engine.dialect)
                    conn.exec_driver_sql(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}')
                    added.append(f"{table.name}.{col.name}")
        return added

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session()
        try:
            yield s
            s.commit()
        except SQLAlchemyError as exc:
            s.rollback()
            raise DatabaseError(f"database operation failed and was rolled back: {exc}") from exc
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.exec_driver_sql("SELECT 1")
            return True
        except SQLAlchemyError:
            return False

    # ---------------------------------------------------------------- writes
    def save_prices(self, symbol: str, df: pd.DataFrame, source: str = "processed") -> int:
        with self.session() as s:
            existing = {d for (d,) in s.execute(select(MarketPrice.date).where(
                MarketPrice.symbol == symbol, MarketPrice.source == source))}
            rows = [MarketPrice(symbol=symbol, date=_ts(d), source=source, open=r["open"], high=r["high"], low=r["low"],
                                close=r["close"], adj_close=r["adj_close"], volume=r["volume"],
                                raw_close=_clean(r.get("raw_close")), adj_factor=_clean(r.get("adj_factor")))
                    for d, r in df.iterrows() if _ts(d) not in existing]
            s.add_all(rows)
            return len(rows)

    def save_audit(self, records: list[dict]) -> int:
        cols = {c.name for c in SignalRecord.__table__.columns} - {"id"}
        with self.session() as s:
            for r in records:
                d = {k: _clean(v) for k, v in r.items() if k in cols}
                d["timestamp"] = _ts(r.get("timestamp"))
                d["data_timestamp"] = _ts(r.get("data_timestamp"))
                d.setdefault("final_action", "NO_TRADE")
                d.setdefault("strategy", r.get("bot_id", ""))
                s.add(SignalRecord(**d))
            return len(records)

    def save_events(self, events: list[dict], run_id: str | None = None) -> int:
        with self.session() as s:
            for e in events:
                s.add(RiskEventRecord(run_id=run_id, date=_ts(e.get("date") or e.get("timestamp")),
                                      bot_id=e.get("bot_id"), symbol=e.get("symbol"), event=str(e.get("event")),
                                      severity=str(e.get("severity", "INFO")), component=e.get("component"),
                                      rule=e.get("rule"), detail=str(e.get("detail", ""))[:4000]))
            return len(events)

    def upsert_order(self, o: Order, run_id: str | None = None) -> None:
        with self.session() as s:
            q = select(OrderRecord).where(OrderRecord.idempotency_key == o.idempotency_key)
            rec = s.execute(q).scalar_one_or_none()
            data = dict(order_id=o.order_id, idempotency_key=o.idempotency_key, run_id=run_id,
                        bot_id=o.meta.get("bot_id", o.strategy), symbol=o.symbol, side=o.side.value,
                        quantity=o.quantity, order_type=o.order_type.value, limit_price=o.limit_price,
                        reference_price=_clean(o.reference_price), status=o.status.value, action=o.action,
                        reason=o.reason, risk_decision_id=o.risk_decision_id, broker_order_id=o.broker_order_id,
                        filled_quantity=o.filled_quantity, avg_fill_price=_clean(o.avg_fill_price),
                        reject_reason=o.reject_reason, created_at=_ts(o.created_at))
            if rec is None:
                s.add(OrderRecord(**data))
            else:
                for k, v in data.items():
                    setattr(rec, k, v)

    def save_backtest(self, result) -> None:
        with self.session() as s:
            s.add(BacktestRun(run_id=result.run_id, bots=result.config.get("bots", []), start=result.config["start"],
                              end=result.config["end"], config=_clean(result.config),
                              metrics=json.loads(json.dumps(result.metrics, default=str)),
                              bot_metrics=json.loads(json.dumps(result.bot_metrics, default=str))))
            for bot_id, bm in result.bot_metrics.items():
                s.add(BacktestBotResult(run_id=result.run_id, bot_id=bot_id,
                                        total_return=_clean(bm.get("total_return")), sharpe=_clean(bm.get("sharpe")),
                                        max_drawdown=_clean(bm.get("max_drawdown")),
                                        num_trades=_clean(bm.get("num_trades")),
                                        metrics=json.loads(json.dumps(bm, default=str))))
            for bot_id, eq in [("ACCOUNT", result.equity), *result.bot_equity.items()]:
                for d, r in eq.iterrows():
                    s.add(PortfolioSnapshot(run_id=result.run_id, bot_id=bot_id, date=_ts(d), equity=r["equity"],
                                            cash=r["cash"], positions_value=r["positions_value"],
                                            gross_exposure=r.get("gross_exposure", 0.0),
                                            n_positions=int(r["n_positions"])))
            for f in result.fills.to_dict("records") if len(result.fills) else []:
                s.add(Execution(order_id=f["order_id"], run_id=result.run_id, bot_id=f.get("bot_id", ""),
                                symbol=f["symbol"], side=f["side"], quantity=int(f["quantity"]), price=f["price"],
                                reference_price=f["reference_price"], commission=f["commission"],
                                spread_cost=f["spread_cost"], slippage_cost=f["slippage_cost"],
                                impact_cost=f["impact_cost"], timestamp=_ts(f["timestamp"])))
        self.save_audit([{**r, "run_id": result.run_id} for r in result.audit.to_dict("records")])
        self.save_events(result.risk_events.to_dict("records") if len(result.risk_events) else [], result.run_id)

    def save_paper_state(self, run_id: str, date, sleeves: dict, fills: list[dict], unallocated_cash: float = 0.0
                         ) -> None:
        """Paper run: executions booked this run, CURRENT per-bot positions
        (replaced, environment='paper') and per-bot + ACCOUNT snapshots."""
        with self.session() as s:
            for f in fills:
                s.add(Execution(order_id=f.get("order_id") or f.get("broker_order_id", ""), run_id=run_id,
                                bot_id=f["bot_id"], symbol=f["symbol"], side=f["side"], quantity=int(f["quantity"]),
                                price=float(f["price"]), reference_price=float(f.get("reference_price") or f["price"]),
                                commission=float(f.get("commission", 0.0)),
                                spread_cost=float(f.get("spread_cost", 0.0) or 0.0),
                                slippage_cost=float(f.get("slippage_cost", 0.0) or 0.0),
                                impact_cost=float(f.get("impact_cost", 0.0) or 0.0), timestamp=_ts(f["timestamp"])))
            s.query(PositionRecord).filter(PositionRecord.environment == "paper").delete()
            for bot_id, p in sleeves.items():
                for sym, pos in p.positions.items():
                    s.add(PositionRecord(environment="paper", bot_id=bot_id, symbol=sym, quantity=pos.quantity,
                                         avg_cost=float(pos.avg_cost), last_price=float(pos.last_price),
                                         entry_date=_ts(pos.entry_date)))
            total = {"equity": unallocated_cash, "cash": unallocated_cash, "positions_value": 0.0, "n": 0}
            for bot_id, p in sleeves.items():
                snap = p.snapshot(pd.Timestamp(date))
                s.add(PortfolioSnapshot(run_id=None, bot_id=bot_id, date=_ts(date), equity=snap["equity"],
                                        cash=snap["cash"], positions_value=snap["positions_value"],
                                        gross_exposure=snap["gross_exposure"], n_positions=snap["n_positions"]))
                total["equity"] += snap["equity"]
                total["cash"] += snap["cash"]
                total["positions_value"] += snap["positions_value"]
                total["n"] += snap["n_positions"]
            eq = total["equity"]
            s.add(PortfolioSnapshot(run_id=None, bot_id="ACCOUNT", date=_ts(date), equity=eq, cash=total["cash"],
                                    positions_value=total["positions_value"],
                                    gross_exposure=total["positions_value"] / eq if eq > 0 else 0.0,
                                    n_positions=int(total["n"])))

    def save_predictions(self, bot_id: str, model_version: str, feature_version: str, preds: pd.Series,
                         out_of_sample: bool = True) -> int:
        with self.session() as s:
            n = 0
            for (d, sym), p in preds.dropna().items():
                s.add(ModelPrediction(bot_id=bot_id, model_version=model_version, feature_version=feature_version,
                                      symbol=sym, date=_ts(d), probability=float(p), out_of_sample=out_of_sample))
                n += 1
            return n

    # ----------------------------------------------------------------- reads
    BOT_TABLES = ("signals", "orders", "executions", "positions", "model_predictions", "risk_events",
                  "portfolio_snapshots", "backtest_bot_results")

    def bot_activity(self, bot_id: str, symbol: str | None = None, run_id: str | None = None
                     ) -> dict[str, pd.DataFrame]:
        """Everything one bot did (optionally with one symbol / in one run):
        signals, orders, executions, positions, predictions, risk events,
        snapshots and backtest results. Parameterised queries only."""
        from sqlalchemy import inspect

        insp = inspect(self.engine)
        out: dict[str, pd.DataFrame] = {}
        for table in self.BOT_TABLES:
            if not insp.has_table(table):
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            where, params = ["bot_id = :bot"], {"bot": bot_id}
            if symbol and "symbol" in cols:
                where.append("symbol = :sym")
                params["sym"] = symbol.upper()
            elif symbol and table not in ("portfolio_snapshots", "backtest_bot_results"):
                continue
            if run_id and "run_id" in cols:
                where.append("run_id = :run")
                params["run"] = run_id
            out[table] = self.read_df(table, " AND ".join(where), params)
        return out
    def read_df(self, table: str, where: str | None = None, params: dict | None = None) -> pd.DataFrame:
        from sqlalchemy import text

        q = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        try:
            with self.engine.connect() as c:
                return pd.read_sql(text(q), c, params=params or {})
        except (SQLAlchemyError, pd.errors.DatabaseError) as exc:   # pandas wraps driver errors
            raise DatabaseError(f"read failed on {table}: {exc}") from exc


class SqlOrderLedger:
    """OrderLedger backed by the ``orders`` table. Keeps live Order objects in
    memory and mirrors every state change to the DB (raising on failure)."""

    def __init__(self, repo: Repository, run_id: str | None = None):
        self.repo = repo
        self.run_id = run_id
        self._orders: dict[str, Order] = {}

    def record_intent(self, order: Order) -> None:
        self.repo.upsert_order(order, self.run_id)
        self._orders[order.idempotency_key] = order

    def update(self, order: Order) -> None:
        self._orders[order.idempotency_key] = order
        self.repo.upsert_order(order, self.run_id)

    def get_by_key(self, key: str) -> Order | None:
        if key in self._orders:
            return self._orders[key]
        df = self.repo.read_df("orders", "idempotency_key = :k", {"k": key})
        if df.empty:
            return None
        r = df.iloc[0]
        o = Order(symbol=r["symbol"], side=Side(r["side"]), quantity=int(r["quantity"]),
                  created_at=pd.Timestamp(r["created_at"]), reason=r["reason"] or "", strategy=r["bot_id"],
                  risk_decision_id=r["risk_decision_id"], reference_price=r["reference_price"],
                  action=r["action"] or "",
                  idempotency_key=r["idempotency_key"], order_id=r["order_id"])
        o.status = OrderStatus(r["status"])
        o.broker_order_id = r["broker_order_id"]
        self._orders[key] = o
        return o

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status not in TERMINAL]

    def all(self) -> list[Order]:
        return list(self._orders.values())
