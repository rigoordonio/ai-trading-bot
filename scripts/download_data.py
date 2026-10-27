"""Download, validate, adjust and store market data.

    python scripts/download_data.py                 # provider from settings.yaml
    python scripts/download_data.py --provider synthetic
    python scripts/download_data.py --db            # also load into the database
"""
import argparse
import json

import _bootstrap  # noqa: F401

from src.config import load_config
from src.logging_setup import setup_logging
from src.workflows import load_market_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--db", action="store_true", help="also write prices to the database")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    res = load_market_data(cfg, args.provider)
    for rep in res.reports.values():
        print(rep.summary())
    if res.failed:
        print("FAILED:", json.dumps(res.failed, indent=2))
    if args.db:
        from src.data.database import Repository
        repo = Repository(cfg.settings.database_url)
        repo.init()
        for sym, df in res.data.items():
            print(sym, "rows inserted:", repo.save_prices(sym, df))
    return 1 if res.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
