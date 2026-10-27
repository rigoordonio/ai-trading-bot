"""The Reviewer: grade the bots and manage their risk (down only).

    python scripts/reviewer.py                   # review now (also runs after every paper_trade.py)
    python scripts/reviewer.py --report          # print the latest report
    python scripts/reviewer.py --resume SWING_V1 # lift a PAUSE / probation for one bot (your decision)

Reports: data/paper/reviews/review-YYYY-MM-DD.md
"""
import argparse
import json

import _bootstrap  # noqa: F401

from src.config import resolve_path
from src.logging_setup import setup_logging
from src.reviewer.review import Reviewer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="data/paper")
    ap.add_argument("--resume", default=None, help="bot id to set back to ACTIVE")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    setup_logging()
    rv = Reviewer(resolve_path(args.state_dir))
    if args.resume:
        print(json.dumps({args.resume.upper(): rv.resume(args.resume.upper())}, indent=2))
        return 0
    if args.report:
        p = resolve_path(args.state_dir) / "review_latest.json"
        print(Reviewer.markdown(json.loads(p.read_text())) if p.exists() else "no review yet")
        return 0
    print(Reviewer.summary(rv.run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
