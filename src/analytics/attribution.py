"""P&L attribution: by bot, symbol, month, exit reason, and cost drag."""
from __future__ import annotations

import pandas as pd


def pnl_by(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(key)["pnl"]
    return pd.DataFrame({"pnl": g.sum(), "trades": g.size(), "win_rate": g.apply(lambda x: (x > 0).mean())}) \
        .sort_values("pnl", ascending=False)


def pnl_by_month(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_date"]).dt.to_period("M")
    return t.groupby("month")["pnl"].sum()


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["exit_type"] = t["exit_reason"].str.split(":").str[0].str.replace(r"\(.*\)", "", regex=True).str.strip()
    return pnl_by(t, "exit_type")


def cost_breakdown(fills: pd.DataFrame) -> dict:
    if fills.empty:
        return {}
    cols = ["commission", "spread_cost", "slippage_cost", "impact_cost"]
    out = {c: float(fills[c].sum()) for c in cols if c in fills}
    out["total"] = sum(out.values())
    out["traded_notional"] = float((fills["quantity"] * fills["price"]).sum())
    out["cost_bps_of_notional"] = out["total"] / out["traded_notional"] * 1e4 if out["traded_notional"] else 0.0
    return out
