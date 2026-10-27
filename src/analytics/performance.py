"""Performance reporting: tables + equity/drawdown charts (PNG via matplotlib)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting.metrics import drawdown_series, monthly_returns


def save_result(result, out_dir: str | Path) -> Path:
    """Write a backtest result as CSV/JSON files the dashboard can read
    without a database."""
    d = Path(out_dir) / result.run_id
    d.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(d / "equity.csv")
    for b, e in result.bot_equity.items():
        e.to_csv(d / f"equity_{b}.csv")
    if result.benchmark is not None:
        result.benchmark.rename("benchmark").to_csv(d / "benchmark.csv")
    for name in ("trades", "fills", "orders", "audit", "risk_events"):
        df = getattr(result, name)
        df.to_csv(d / f"{name}.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({"run_id": result.run_id, "config": result.config,
                                                "metrics": result.metrics, "bot_metrics": result.bot_metrics},
                                               indent=2, default=str))
    return d


def plot_equity(result, path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = result.equity["equity"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(eq.index, eq.values, label="Account", lw=1.6)
    for b, e in result.bot_equity.items():
        if len(result.bot_equity) > 1:
            a1.plot(e.index, e["equity"].values, label=b, lw=0.9, alpha=0.8)
    if result.benchmark is not None:
        a1.plot(result.benchmark.index, result.benchmark.values, label="Benchmark (buy & hold)", lw=1, ls="--")
    a1.set_ylabel("Equity")
    a1.legend(loc="upper left", frameon=False)
    a1.grid(alpha=0.2)
    dd = drawdown_series(eq)
    a2.fill_between(dd.index, dd.values, 0, alpha=0.4)
    a2.set_ylabel("Drawdown")
    a2.grid(alpha=0.2)
    fig.suptitle(f"{result.run_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def monthly_table(result) -> pd.DataFrame:
    return monthly_returns(result.equity["equity"])
