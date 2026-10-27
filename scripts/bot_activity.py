"""Show everything one bot did, optionally with one symbol.

    python scripts/bot_activity.py --bot SWING_V1 --symbol NVDA
    python scripts/bot_activity.py --bot INVESTOR_V1 --run BT-20261018-224252-a8101b

Reads the database configured in settings.yaml (database_url). Backtests
are written there with `scripts/backtest.py --bots --db` and paper runs
with `scripts/paper_trade.py --db`.
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    from src.data.database import Repository

    repo = Repository(cfg.settings.database_url)
    repo.init()
    tables = repo.bot_activity(args.bot.upper(), args.symbol, args.run)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 14)
    scope = f"{args.bot.upper()}" + (f" / {args.symbol.upper()}" if args.symbol else "")
    print(f"=== activity for {scope} ===")
    for name, df in tables.items():
        print(f"\n--- {name}: {len(df)} rows ---")
        if len(df):
            print(df.tail(args.rows).to_string(index=False, max_colwidth=40))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
