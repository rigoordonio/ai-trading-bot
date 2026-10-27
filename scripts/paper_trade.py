"""Run one paper-trading cycle (schedule it once per day after the US close).

    TRADING_MODE=PAPER python scripts/paper_trade.py
    python scripts/paper_trade.py --provider synthetic --as-of 2024-03-01
    python scripts/paper_trade.py --status            # show trading gate / health only
    python scripts/paper_trade.py --broker alpaca_paper --check-broker   # read-only connection test

With BROKER=alpaca_paper, run while the US market is CLOSED (08:15 Manila works
all year): orders are queued and fill at the next open. After the run, the
Reviewer grades the bots (scripts/reviewer.py).

If the latest prices are not published yet, or the internet / Alpaca cannot be
reached, the run waits --retry-minutes (30) and tries again, up to
--max-retries (10) times. It never retries after an order was sent, and never
for problems that need you (e.g. books that do not match Alpaca).

Real-money trading is not implemented and is refused by configuration.
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.provider import get_provider
from src.execution.broker import assert_safe_mode
from src.execution.paper_trader import PaperTrader
from src.logging_setup import setup_logging


def retryable(rep) -> bool:
    """Retry only when waiting can fix it: the latest prices are not published
    yet, or market data / the broker could not be reached. Never after any
    order was sent, and never for problems that need a human (e.g. books that
    do not match the broker)."""
    if rep.submitted or rep.rejected:
        return False
    reasons = rep.blocking_reasons or []
    if any("not available yet" in r or "market data unavailable" in r for r in reasons):
        return True
    if rep.trading_enabled or not reasons:
        return False
    connectivity = ("HALT[market_data]", "UNHEALTHY: market_data", "breaker OPEN: market_data",
                    "UNHEALTHY: broker", "breaker OPEN: broker", "unreachable", "timed out", "Timeout",
                    "ConnectionError", "Max retries", "Name or service not known", "nodename nor servname")
    needs_human = ("reconciliation failed", "do not match", "UNKNOWN", "database", "kill switch", "drawdown")
    if any(any(h in r for h in needs_human) for r in reasons):
        return False
    return all(any(c in r for c in connectivity) for r in reasons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--state-dir", default="data/paper")
    ap.add_argument("--db", action="store_true", help="persist ledger/audit to DATABASE_URL")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--broker", default=None, help="simulated | alpaca_paper (default: BROKER env var)")
    ap.add_argument("--check-broker", action="store_true", help="read-only: show account, positions, orders")
    ap.add_argument("--retry-minutes", type=float, default=30.0,
                    help="if prices are not ready yet or the internet is down, wait this long and retry")
    ap.add_argument("--max-retries", type=int, default=10, help="0 = single attempt")
    args = ap.parse_args()
    setup_logging()
    assert_safe_mode("PAPER")
    cfg = load_config()
    if args.check_broker:
        from src.execution.broker import create_broker
        b = create_broker("PAPER", args.broker)
        acct = b.get_account()
        print(json.dumps({"broker": b.name, "account": acct.account_id, "cash": acct.cash, "equity": acct.equity,
                          "positions": {s: p.quantity for s, p in b.get_positions().items()},
                          "open_orders": len([o for o in b.get_orders() if o.status in ("ACKNOWLEDGED",
                                                                                         "PARTIALLY_FILLED")])},
                         indent=2))
        return 0
    repo = ledger = None
    if args.db:
        from src.data.database import Repository, SqlOrderLedger
        repo = Repository(cfg.settings.database_url)
        repo.init()
        ledger = SqlOrderLedger(repo)
    def make_trader():
        return PaperTrader(cfg, args.state_dir, get_provider(args.provider or cfg.settings.data.provider),
                           ledger=ledger, repo=repo, broker_name=args.broker)

    trader = make_trader()
    if args.status:
        print(json.dumps(trader.gate.status(), indent=2, default=str))
        return 0
    attempts = 1 + (0 if args.as_of else max(0, args.max_retries))   # replays (--as-of) never retry
    for attempt in range(1, attempts + 1):
        rep = trader.run(args.as_of)
        print(json.dumps({"run_id": rep.run_id, "as_of": rep.as_of, "trading_enabled": rep.trading_enabled,
                          "blocking_reasons": rep.blocking_reasons, "submitted": len(rep.submitted),
                          "rejected": len(rep.rejected), "reconciliation": rep.reconciliation,
                          "attempt": f"{attempt}/{attempts}"}, indent=2, default=str))
        if attempt == attempts or not retryable(rep):
            break
        print(f"waiting {args.retry_minutes:g} min, then retrying (prices not ready or connection problem)",
              flush=True)
        time.sleep(args.retry_minutes * 60)
        trader = make_trader()                # fresh state: books, gate and health re-read from disk
    try:                                     # the Reviewer grades the run; it can never affect this run's orders
        from src.reviewer.review import Reviewer, ReviewerConfig
        rcfg = ReviewerConfig.load()
        if rcfg.enabled:
            print(Reviewer.summary(Reviewer(trader.state_dir, rcfg).run()))
    except Exception as exc:  # noqa: BLE001 - reporting must never break trading
        print(f"reviewer failed: {type(exc).__name__}: {exc}")
    return 0 if rep.trading_enabled else 2


if __name__ == "__main__":
    raise SystemExit(main())
