"""Robustness: cost / parameter / start-date sensitivity + Monte Carlo.

    python scripts/robustness.py --strategy trend_following --provider synthetic
"""
import argparse

import _bootstrap  # noqa: F401

from src.backtesting.metrics import returns_from_equity
from src.backtesting.robustness import block_bootstrap, summarise, sweep, trade_shuffle
from src.config import load_config
from src.logging_setup import setup_logging
from src.workflows import load_market_data, research_backtest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="trend_following")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--sims", type=int, default=500)
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config()
    prices = load_market_data(cfg, args.provider, save=False).data
    s = args.strategy
    print("== transaction-cost sensitivity")
    print(sweep(lambda m: research_backtest(cfg, prices, s, cost_multiplier=m).metrics, [0, 1, 2, 4],
                "cost_multiplier").round(4).to_string(index=False))
    print("== start-date sensitivity")
    print(sweep(lambda d: research_backtest(cfg, prices, s, start=d).metrics,
                ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"], "start").round(4).to_string(index=False))
    if s == "trend_following":
        print("== parameter sensitivity (fast/slow SMA)")
        grid = [{"fast_sma": 20, "slow_sma": 200}, {"fast_sma": 50, "slow_sma": 200}]
        print(sweep(lambda p: research_backtest(cfg, prices, s, params=p).metrics, grid, "params")
              .round(4).to_string(index=False))
    base = research_backtest(cfg, prices, s)
    print("== block bootstrap of daily returns (assumes stationarity; 20-day blocks)")
    print(summarise(block_bootstrap(returns_from_equity(base.equity["equity"]), args.sims)).round(4).to_string())
    if len(base.trades) >= 5:
        print("== trade-sequence shuffle (assumes independent trades)")
        print(summarise(trade_shuffle(base.trades["pnl"], cfg.settings.backtest.initial_capital, args.sims))
              .round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
