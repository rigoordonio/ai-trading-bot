"""KPI data for the local KPI page (read-only: never touches the broker or orders).

Sources (all written by the paper trader / Reviewer in data/paper/):
    snapshots.csv          per-bot + ACCOUNT equity per run (+ SPY close)
    sleeves.json           positions and closed trades per bot
    orders.json            order ledger
    review_latest.json     Reviewer grades, statuses and alerts
    system_status.json     last run: trading enabled, reconciliation, health
    decisions/*.csv        every decision of the latest run
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOT_ORDER = ["SWING_V1", "TREND_V1", "INVESTOR_V1"]


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _series_stats(eq: pd.Series) -> dict:
    eq = eq.dropna().astype(float)
    if eq.empty:
        return {}
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if len(r) >= 5 and r.std() > 0 else None
    return {"equity": float(eq.iloc[-1]), "start": float(eq.iloc[0]),
            "total_pnl": float(eq.iloc[-1] - eq.iloc[0]),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1) if eq.iloc[0] > 0 else None,
            "daily_pnl": float(eq.iloc[-1] - eq.iloc[-2]) if len(eq) > 1 else None,
            "drawdown": float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1),
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "volatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else None,
            "sharpe": sharpe, "days": int(len(eq))}


def build_kpis(state_dir: str | Path, allocations: dict[str, float] | None = None) -> dict:
    d = Path(state_dir)
    snaps = pd.read_csv(d / "snapshots.csv") if (d / "snapshots.csv").exists() else pd.DataFrame()
    sleeves = (_json(d / "sleeves.json") or {}).get("sleeves", {})
    orders = _json(d / "orders.json") or []
    review = _json(d / "review_latest.json") or {}
    status = _json(d / "system_status.json") or {}
    if len(snaps):
        snaps["date"] = pd.to_datetime(snaps["date"])
        snaps = snaps.drop_duplicates(["date", "bot_id"], keep="last").sort_values("date")
    bots = [b for b in BOT_ORDER if b in sleeves or (len(snaps) and b in set(snaps["bot_id"]))]
    bots += sorted(b for b in sleeves if b not in bots)
    rgrades = {g["bot_id"]: g for g in review.get("bots", [])}

    # ----- equity history (indexed to 100 on one axis: bots have different capital)
    history: dict[str, list] = {}
    dates: list[str] = []
    if len(snaps):
        piv = snaps.pivot(index="date", columns="bot_id", values="equity").sort_index()
        dates = [str(x.date()) for x in piv.index]
        for col in [*bots, "ACCOUNT"]:
            if col in piv:
                s = piv[col].astype(float)
                base = s.dropna().iloc[0] if s.notna().any() else np.nan
                history[col] = [None if pd.isna(v) else round(float(v / base * 100), 3) for v in s]
        if "benchmark_close" in snaps:
            b = snaps[snaps["bot_id"] == "ACCOUNT"].set_index("date")["benchmark_close"].reindex(piv.index)
            if b.notna().any():
                base = b.dropna().iloc[0]
                history["SPY"] = [None if pd.isna(v) else round(float(v / base * 100), 3) for v in b]

    # ----- per bot
    open_orders = [o for o in orders if o.get("status") in ("ACKNOWLEDGED", "SUBMITTED", "PARTIALLY_FILLED",
                                                              "SUBMITTING", "UNKNOWN")]
    bot_rows = []
    for b in bots:
        s = snaps[snaps["bot_id"] == b].set_index("date")["equity"] if len(snaps) else pd.Series(dtype=float)
        st = _series_stats(s)
        sl = sleeves.get(b, {})
        trades = sl.get("closed_trades", [])
        pnls = [float(t.get("pnl", 0)) for t in trades]
        wins, losses = [p for p in pnls if p > 0], [-p for p in pnls if p < 0]
        positions = sl.get("positions", {})
        pos_value = sum(float(p.get("market_value", 0) or 0) for p in positions.values())
        eq = st.get("equity") or (float(sl.get("cash", 0)) + pos_value)
        g = rgrades.get(b, {})
        bot_rows.append({
            "bot_id": b, **st, "allocation": (allocations or {}).get(b),
            "cash": _f(sl.get("cash")), "exposure": pos_value / eq if eq else None,
            "positions": len(positions), "open_orders": len([o for o in open_orders if o.get("bot_id") == b]),
            "trades": len(trades), "win_rate": len(wins) / len(trades) if trades else None,
            "profit_factor": (sum(wins) / sum(losses)) if losses else (None if not wins else float("inf")),
            "avg_holding_days": float(np.mean([t.get("holding_days", 0) for t in trades])) if trades else None,
            "realized_pnl": _f(sl.get("realized_pnl")),
            "unrealized_pnl": sum(float(p.get("unrealized_pnl", 0) or 0) for p in positions.values()),
            "status": g.get("status", "ACTIVE"), "grade": g.get("grade", "—"),
            "size_multiplier": g.get("size_multiplier", 1.0), "review": "; ".join(g.get("reasons", [])),
        })

    # ----- account
    acc_s = snaps[snaps["bot_id"] == "ACCOUNT"].set_index("date")["equity"] if len(snaps) else pd.Series(dtype=float)
    account = _series_stats(acc_s)
    if "SPY" in history and len([v for v in history["SPY"] if v is not None]) >= 2:
        spy = [v for v in history["SPY"] if v is not None]
        account["benchmark_return"] = spy[-1] / spy[0] - 1
    account["positions_value"] = sum(r["exposure"] * r["equity"] for r in bot_rows
                                     if r.get("exposure") is not None and r.get("equity"))
    account["exposure"] = (account["positions_value"] / account["equity"]) if account.get("equity") else None
    account["open_orders"] = len(open_orders)

    # ----- positions with per-bot attribution
    positions = []
    for b in bots:
        for sym, p in sleeves.get(b, {}).get("positions", {}).items():
            positions.append({"bot_id": b, "symbol": sym, "quantity": p.get("quantity"),
                              "avg_cost": _f(p.get("avg_cost")), "last_price": _f(p.get("last_price")),
                              "market_value": _f(p.get("market_value")), "unrealized_pnl": _f(p.get("unrealized_pnl")),
                              "entry_date": str(p.get("entry_date", ""))[:10]})

    # ----- latest decisions
    decisions: list[dict] = []
    files = sorted((d / "decisions").glob("decisions-*.csv")) if (d / "decisions").exists() else []
    if files:
        df = pd.read_csv(files[-1])
        keep = df[(df["action"].isin(["BUY", "EXIT", "REDUCE"])) | (df["final_action"] != "NO_TRADE")]
        decisions = json.loads(keep.head(60).to_json(orient="records"))

    recent_orders = sorted(orders, key=lambda o: o.get("order_id", ""), reverse=True)[:20]
    alerts = review.get("alerts", [])
    return {"as_of": str(snaps["date"].max().date()) if len(snaps) else None,
            "reviewed_at": review.get("reviewed_at"),
            "trading_enabled": status.get("trading_enabled"),
            "blocking_reasons": status.get("blocking_reasons", []),
            "reconciliation_ok": (status.get("reconciliation") or {}).get("ok"),
            "account": account, "bots": bot_rows, "history": {"dates": dates, "series": history},
            "positions": positions, "decisions": decisions,
            "orders": [{k: o.get(k) for k in ("order_id", "bot_id", "symbol", "side", "quantity", "status",
                                              "filled_quantity", "avg_fill_price", "created_at", "reject_reason")}
                       for o in recent_orders],
            "alerts": alerts}


def live_view(snapshot: dict | None, kpis: dict, error: str | None = None, at: str | None = None) -> dict:
    """Merge Alpaca's live snapshot with the bots' books: who owns each position.
    Positions filled after the last bot run are not booked to a bot yet; they
    are attributed from the order ledger and marked as pending."""
    if snapshot is None:
        return {"available": False, "error": error, "at": at}
    owners: dict[str, list[str]] = {}
    for p in kpis.get("positions", []):
        owners.setdefault(p["symbol"], []).append(f"{p['bot_id']} {p['quantity']}")
    pending: dict[str, set] = {}
    for o in kpis.get("orders", []):
        if o.get("side") == "BUY" and o.get("status") in ("ACKNOWLEDGED", "SUBMITTED", "PARTIALLY_FILLED", "FILLED"):
            pending.setdefault(o["symbol"], set()).add(o.get("bot_id") or "?")
    rows = []
    for p in snapshot.get("positions", []):
        sym = p["symbol"]
        if sym in owners:
            who = ", ".join(owners[sym])
        elif sym in pending:
            who = ", ".join(sorted(pending[sym])) + " (filled after last run; booked at next run)"
        else:
            who = "not from the bots"
        rows.append({**p, "owner": who})
    return {"available": True, "error": None, "at": at, "account": snapshot.get("account", {}), "positions": rows}
