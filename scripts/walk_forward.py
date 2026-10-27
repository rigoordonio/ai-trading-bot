"""Walk-forward ML evaluation per bot (aggregated out-of-sample statistics).

    python scripts/walk_forward.py --provider synthetic
"""
import argparse
import json

import _bootstrap  # noqa: F401

from src.bots.bot import build_bots
from src.config import load_config
from src.logging_setup import setup_logging
from src.workflows import bot_walk_forward, load_market_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", action="append")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    prices = load_market_data(cfg, args.provider).data
    for bot_id, bot in build_bots(cfg, args.bot).items():
        wf = bot_walk_forward(cfg, bot, bot.compute_features(prices))
        if wf is None:
            continue
        print(f"\n=== {bot_id}")
        print(wf.windows.to_string(index=False))
        keep = ("roc_auc", "accuracy", "ic_spearman", "quintile_spread", "top_quintile_hit", "n")
        print("aggregated OOS:", json.dumps({k: wf.oos_metrics.get(k) for k in keep}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
