"""Read-only FastAPI service.

There is deliberately NO endpoint that creates, modifies or cancels orders:
orders exist only through the TradingCycle -> risk engines -> OrderManager
path. The explain endpoint returns text.

    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import math

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.ai.explanation import DecisionFacts, explain
from src.analytics.store import ResultsStore
from src.execution.broker import trading_mode

app = FastAPI(title="AI Trading Platform (paper/backtest only)", version="0.1.0")
store = ResultsStore()


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if limit:
        df = df.tail(limit)
    out = df.reset_index().to_dict("records") if df.index.name else df.to_dict("records")
    return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()} for r in out]


@app.get("/health")
def health() -> dict:
    st = store.system_status()
    return {"mode": trading_mode(), "real_money": "DISABLED", "trading_enabled": st.get("trading_enabled"),
            "system_health": st.get("health", {}).get("overall"), "blocking_reasons": st.get("blocking_reasons", []),
            "circuit_breakers": st.get("circuit_breakers", {}), "last_seen": st.get("health", {}).get("last_seen", {})}


@app.get("/backtests")
def backtests() -> list[dict]:
    return store.runs()


@app.get("/backtests/{run_id}")
def backtest(run_id: str) -> dict:
    try:
        return store.run_metrics(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"unknown run {run_id}") from exc


@app.get("/backtests/{run_id}/equity")
def backtest_equity(run_id: str) -> list[dict]:
    return _records(store.run_frame(run_id, "equity"))


@app.get("/backtests/{run_id}/trades")
def backtest_trades(run_id: str) -> list[dict]:
    return _records(store.run_frame(run_id, "trades"))


@app.get("/backtests/{run_id}/signals")
def backtest_signals(run_id: str, symbol: str | None = None, bot_id: str | None = None, limit: int = 500):
    df = store.run_frame(run_id, "audit")
    if symbol and not df.empty:
        df = df[df["symbol"] == symbol]
    if bot_id and not df.empty:
        df = df[df["bot_id"] == bot_id]
    return _records(df, limit)


@app.get("/paper/portfolio")
def paper_portfolio() -> dict:
    return {"positions": _records(store.paper_positions()), "broker": store.broker_state().get("positions", {}),
            "cash": store.broker_state().get("cash")}


@app.get("/paper/orders")
def paper_orders(limit: int = 200) -> list[dict]:
    return _records(store.paper_orders(), limit)


@app.get("/risk/events")
def risk_events() -> list[dict]:
    return store.system_status().get("recent_events", [])


@app.post("/explain")
def explain_decision(record: dict) -> dict:
    """Body: one audit record. Returns an explanation string only."""
    return {"explanation": explain(DecisionFacts.from_audit(record))}
