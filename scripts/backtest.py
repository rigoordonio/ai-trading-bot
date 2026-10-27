"""Backtests.

    python scripts/backtest.py --strategy trend_following      # one rule strategy, 100% capital
    python scripts/backtest.py --strategy all                   # each rule strategy independently
    python scripts/backtest.py --bots                           # all bots on one account (walk-forward ML)
    python scripts/backtest.py --bots --bot SWING_V1            # one bot independently
Options: --provider synthetic --start 2016-01-01 --end 2024-12-31 --cost-mult 2 --db
"""
import argparse

import _bootstrap  # noqa: F401

from src.analytics.attribution import cost_breakdown, pnl_by
from src.analytics.performance import plot_equity, save_result
from src.config import load_config, resolve_path
from src.logging_setup import setup_logging
from src.workflows import bots_backtest, load_market_data, research_backtest


def report(res, db: bool, cfg) -> None:
    print(res.summary())
    if len(res.trades):
        print(pnl_by(res.trades, "symbol").round(2).head(10).to_string())
    print("costs:", {k: round(v, 2) for k, v in cost_breakdown(res.fills).items()})
    out = save_result(res, resolve_path("reports"))
    plot_equity(res, out / "equity.png")
    print("saved:", out)
    if db:
        from src.data.database import Repository
        repo = Repository(cfg.settings.database_url)
        repo.init()
        repo.save_backtest(res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--bots", action="store_true")
    ap.add_argument("--bot", action="append")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--db", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    prices = load_market_data(cfg, args.provider).data
    if args.bots or args.bot:
        res, wf = bots_backtest(cfg, prices, args.bot, args.start, args.end, args.cost_mult)
        for b, w in wf.items():
            if w is not None:
                print(f"[{b}] walk-forward OOS from {w.first_oos_date}: "
                      f"auc={w.oos_metrics.get('roc_auc', float('nan')):.3f} "
                      f"ic={w.oos_metrics.get('ic_spearman', float('nan')):.3f}")
        report(res, args.db, cfg)
        return 0
    names = list(cfg.strategies.strategies) if args.strategy == "all" else [args.strategy or "trend_following"]
    for name in names:
        if name == "ml":
            continue
        report(research_backtest(cfg, prices, name, args.start, args.end, args.cost_mult), args.db, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
