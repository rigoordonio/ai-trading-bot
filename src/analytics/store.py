"""Read-only access to results for the API and dashboard.

Sources (no database required):
    reports/<run_id>/            backtest outputs written by save_result()
    data/paper/                  paper broker, sleeves, ledger and system status
A database (DATABASE_URL) is optional and read through Repository when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import resolve_path


class ResultsStore:
    def __init__(self, reports_dir: str | Path = "reports", paper_dir: str | Path = "data/paper"):
        self.reports = resolve_path(reports_dir)
        self.paper = resolve_path(paper_dir)

    # -------------------------------------------------------------- backtests
    def runs(self) -> list[dict]:
        out: list[dict] = []
        if not self.reports.exists():
            return out
        for d in sorted(self.reports.iterdir(), reverse=True):
            m = d / "metrics.json"
            if d.is_dir() and m.exists():
                j = json.loads(m.read_text())
                out.append({"run_id": j["run_id"], **j.get("config", {}),
                            "total_return": j["metrics"].get("total_return"), "sharpe": j["metrics"].get("sharpe"),
                            "max_drawdown": j["metrics"].get("max_drawdown")})
        return out

    def run_metrics(self, run_id: str) -> dict:
        return json.loads((self.reports / run_id / "metrics.json").read_text())

    def run_frame(self, run_id: str, name: str) -> pd.DataFrame:
        p = self.reports / run_id / f"{name}.csv"
        if not p.exists():
            return pd.DataFrame()
        idx = "date" if name.startswith("equity") or name == "benchmark" else None
        df = pd.read_csv(p, parse_dates=[idx] if idx else None, index_col=idx)
        return df

    # ------------------------------------------------------------------ paper
    def _json(self, name: str):
        p = self.paper / name
        return json.loads(p.read_text()) if p.exists() else None

    def system_status(self) -> dict:
        return self._json("system_status.json") or {"trading_enabled": None, "health": {"overall": "UNKNOWN"},
                                                     "blocking_reasons": ["no paper run yet"]}

    def broker_state(self) -> dict:
        return self._json("broker_state.json") or {}

    def sleeves(self) -> dict:
        return self._json("sleeves.json") or {}

    def paper_orders(self) -> pd.DataFrame:
        rows = self._json("orders.json") or []
        return pd.DataFrame(rows)

    def paper_positions(self) -> pd.DataFrame:
        sl = self.sleeves().get("sleeves", {})
        rows = []
        for bot, s in sl.items():
            for sym, p in s.get("positions", {}).items():
                rows.append({"bot_id": bot, "symbol": sym, "quantity": p["quantity"], "avg_cost": p["avg_cost"],
                             "last_price": p["last_price"], "market_value": p["quantity"] * p["last_price"],
                             "unrealized_pnl": (p["last_price"] - p["avg_cost"]) * p["quantity"],
                             "entry_date": p["entry_date"]})
        return pd.DataFrame(rows)
