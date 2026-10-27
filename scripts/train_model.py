"""Train each bot's model on the chronological split in settings.yaml
(train -> validation for model choice -> one final test evaluation).

    python scripts/train_model.py                    # all ML-enabled bots
    python scripts/train_model.py --bot SWING_V1 --provider synthetic
"""
import argparse
import json

import _bootstrap  # noqa: F401

from src.bots.bot import build_bots
from src.config import load_config
from src.logging_setup import setup_logging
from src.models.evaluation import feature_importance
from src.workflows import load_market_data, train_bot_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", action="append")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    prices = load_market_data(cfg, args.provider).data
    for bot_id, bot in build_bots(cfg, args.bot).items():
        if not bot.ml_cfg:
            continue
        tr = train_bot_model(cfg, bot, prices)
        print(f"\n=== {bot_id}  best={tr.best_name}  n_train={tr.n_train}")
        print("validation:", json.dumps({k: round(v.get("roc_auc", float("nan")), 4)
                                         for k, v in tr.validation_scores.items()}))
        if tr.test_metrics:
            keep = ("roc_auc", "accuracy", "precision", "recall", "ic_spearman", "quintile_spread", "top_quintile_hit")
            print("test (touched once):", json.dumps({k: round(tr.test_metrics[k], 4) for k in keep
                                                      if k in tr.test_metrics}))
        print("top features:", feature_importance(tr.model, tr.features).head(8).round(4).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
