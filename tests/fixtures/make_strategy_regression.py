"""Regenerate the strategy regression snapshot. ONLY run this when a strategy
change is intentional, and review the diff of strategy_regression.json.

    python tests/fixtures/make_strategy_regression.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.pipeline import FeaturePipeline  # noqa: E402
from src.strategies.ensemble import RULE_STRATEGIES  # noqa: E402
from tests.fixtures.regression_data import regression_prices  # noqa: E402

DEFAULTS = {"trend_following": {}, "mean_reversion": {"require_long_term_uptrend": False},
            "momentum": {"top_n": 1}, "breakout": {"lookback": 55, "exit_lookback": 20, "volume_ratio_min": 1.2}}


def compute():
    panel = FeaturePipeline().compute(regression_prices())
    out = {}
    for name, params in DEFAULTS.items():
        s = RULE_STRATEGIES[name](dict(params)).score(panel)
        sub = s.dropna()
        sub = sub[sub.index.get_level_values("date").isin(sorted(set(sub.index.get_level_values("date")))[::7])]
        out[name] = {f"{d:%Y-%m-%d}|{sym}": round(float(v), 6) for (d, sym), v in sub.items()}
    return out


if __name__ == "__main__":
    path = Path(__file__).with_name("strategy_regression.json")
    path.write_text(json.dumps(compute(), indent=0, sort_keys=True))
    print("wrote", path)
