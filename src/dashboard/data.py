"""Data helpers for the dashboard (no Streamlit imports: unit-testable)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.store import ResultsStore
from src.backtesting.metrics import drawdown_series, monthly_returns, sharpe_ratio
from src.bots.bot import TradingBot
from src.config import AppConfig, resolve_path


def artifacts(cfg: AppConfig) -> list[dict]:
    root = resolve_path(cfg.settings.ml.artifacts_dir)
    out = []
    for meta in sorted(root.glob("*/*/metadata.json")):
        try:
            out.append(json.loads(meta.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def market_table(bot: TradingBot, panel: pd.DataFrame, preds: pd.Series | None = None) -> pd.DataFrame:
    """Latest row per symbol: price, daily %, trend, momentum, vol, ML probability, final signal."""
    last_date = panel.index.get_level_values("date").max()
    last = panel.xs(last_date, level="date")
    sf = bot.signal_frame(panel, preds)
    sig = sf.on(last_date)
    rows = []
    for sym in last.index:
        r = last.loc[sym]
        score = sig.loc[sym, "score"] if sym in sig.index else np.nan
        trend_up = r.get("close", np.nan) > r.get("sma_50", np.nan) > r.get("sma_200", np.nan)
        rows.append({
            "Ticker": sym, "Price": r.get("close"), "Daily %": r.get("ret_1d"),
            "Trend": "Up" if trend_up else "Down/flat",
            "Momentum 20d": r.get("ret_20d", r.get("w_ret_4d")),
            "Volatility 20d": r.get("realized_vol_20"),
            "ML probability": (sig.loc[sym, "ml_probability"] if sym in sig.index and "ml_probability" in sig
                               else np.nan),
            "Score": score,
            "Signal": sf.decide(score, held=False) if np.isfinite(score) else "NO SIGNAL",
        })
    return pd.DataFrame(rows).set_index("Ticker")


def overview_kpis(equity: pd.DataFrame) -> dict:
    if equity is None or equity.empty:
        return {}
    e = equity["equity"]
    r = e.pct_change().dropna()
    return {"Portfolio value": float(e.iloc[-1]), "Daily P&L": float(e.iloc[-1] - e.iloc[-2]) if len(e) > 1 else 0.0,
            "Total P&L": float(e.iloc[-1] - e.iloc[0]), "Cash": float(equity["cash"].iloc[-1]),
            "Market exposure": float(equity.get("gross_exposure", pd.Series([0.0])).iloc[-1]),
            "Max drawdown": float(drawdown_series(e).min()), "Sharpe": sharpe_ratio(r)}


def drawdown(equity: pd.Series) -> pd.Series:
    return drawdown_series(equity)


def monthly(equity: pd.Series) -> pd.DataFrame:
    return monthly_returns(equity)


def latest_run(store: ResultsStore) -> str | None:
    runs = store.runs()
    return runs[0]["run_id"] if runs else None


def file_age(path: Path) -> str:
    if not path.exists():
        return "never"
    secs = pd.Timestamp.now().timestamp() - path.stat().st_mtime
    return f"{secs / 3600:.1f} h ago"


# ------------------------------------------------------------ multi-bot view
OVERVIEW_ROWS = ["Capital allocation", "Portfolio value", "Cash", "Positions", "Daily P&L", "Total P&L", "Return",
                 "Volatility", "Sharpe", "Max drawdown", "Trade count", "Active signals"]


def snapshots_from_backtest(bot_equity: dict[str, pd.DataFrame], account: pd.DataFrame,
                            trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """Backtest frames -> the long format used by multi_bot_overview."""
    frames = []
    for b, df in {**bot_equity, "ACCOUNT": account}.items():
        if df is None or df.empty:
            continue
        f = df.reset_index()[["date", "equity", "cash", "positions_value", "n_positions"]].copy()
        f["bot_id"] = b
        if trades is not None and len(trades):
            t = trades if b == "ACCOUNT" else trades[trades["bot_id"] == b]
            exits = pd.to_datetime(t["exit_date"]).sort_values()
            f["n_trades"] = exits.searchsorted(pd.to_datetime(f["date"]), side="right")
        else:
            f["n_trades"] = 0
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def multi_bot_overview(snapshots: pd.DataFrame, allocations: dict[str, float],
                       active_signals: dict[str, int] | None = None) -> pd.DataFrame:
    """Rows = OVERVIEW_ROWS, columns = bots + TOTAL PORTFOLIO.

    ``snapshots``: long format (date, bot_id, equity, cash, positions_value,
    n_positions, n_trades); bot_id "ACCOUNT" is the whole account. Model
    accuracy is deliberately not shown: it is not a measure of trading quality."""
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(index=OVERVIEW_ROWS)
    snaps = snapshots.copy()
    snaps["date"] = pd.to_datetime(snaps["date"])
    active_signals = active_signals or {}
    cols: dict[str, dict] = {}
    for b in [*[x for x in allocations if x in set(snaps["bot_id"])], "ACCOUNT"]:
        g = snaps[snaps["bot_id"] == b].sort_values("date").drop_duplicates("date", keep="last")
        if g.empty:
            continue
        e = g.set_index("date")["equity"].astype(float)
        r = e.pct_change().dropna()
        name = "TOTAL PORTFOLIO" if b == "ACCOUNT" else b
        cols[name] = {
            "Capital allocation": sum(allocations.values()) if b == "ACCOUNT" else allocations.get(b, np.nan),
            "Portfolio value": float(e.iloc[-1]), "Cash": float(g["cash"].iloc[-1]),
            "Positions": int(g["n_positions"].iloc[-1]),
            "Daily P&L": float(e.iloc[-1] - e.iloc[-2]) if len(e) > 1 else 0.0,
            "Total P&L": float(e.iloc[-1] - e.iloc[0]),
            "Return": float(e.iloc[-1] / e.iloc[0] - 1) if e.iloc[0] > 0 else np.nan,
            "Volatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan,
            "Sharpe": sharpe_ratio(r) if len(r) > 1 else np.nan,
            "Max drawdown": float(drawdown_series(e).min()),
            "Trade count": int(g["n_trades"].iloc[-1]) if "n_trades" in g else 0,
            "Active signals": (sum(active_signals.values()) if b == "ACCOUNT" else active_signals.get(b, 0)),
        }
    return pd.DataFrame(cols).reindex(OVERVIEW_ROWS)


def active_signal_counts(signals: list[dict] | pd.DataFrame) -> dict[str, int]:
    """Actionable signals (BUY / REDUCE / EXIT) per bot in the latest decision."""
    df = pd.DataFrame(signals)
    if df.empty:
        return {}
    col = "signal_type" if "signal_type" in df else "action"
    if "timestamp" in df and col == "action":
        df = df[pd.to_datetime(df["timestamp"]) == pd.to_datetime(df["timestamp"]).max()]
    act = df[df[col].isin(["BUY", "REDUCE", "EXIT"])]
    return act.groupby("bot_id").size().astype(int).to_dict()


def paper_history(state_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    snaps = pd.read_csv(state_dir / "snapshots.csv") if (state_dir / "snapshots.csv").exists() else pd.DataFrame()
    try:
        sigs = json.loads((state_dir / "last_signals.json").read_text())
    except (OSError, json.JSONDecodeError):
        sigs = []
    return snaps, sigs
