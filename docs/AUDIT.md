# Safety audit: fail-closed review

Scope: every module that can influence whether an order is created, sized, sent or recorded.
Principle: **no trade is better than a bad trade caused by a system error.**

## 1. Trading-critical modules

| Stage | Module | Fails closed by |
|---|---|---|
| Market data | `src/data/provider.py`, `downloader.py` | retries + backoff, schema check, `market_data` circuit breaker |
| Historical cleaning | `src/data/validator.py` | flags issues; removes only rows that are provably unusable (documented rules) |
| Current bar | `src/safety/validation.py::validate_bar_values` | NaN/inf, ≤0 prices, OHLC, stale, timezone, future bar, discontinuity → **no trade for that symbol** |
| Features | `src/features/*`, `validate_feature_row` | NaN/inf in a required feature → no BUY; version hash on every schema |
| Model | `src/models/prediction.py` | missing / corrupted (sha256) / wrong version / wrong schema / NaN / out-of-range → `ModelInferenceError` → ML component NaN → **HOLD**, trading halted |
| Strategy / ensemble | `src/strategies/*`, `bots/bot.py` | a required ML component that is NaN makes the score NaN. Never "trade on the remaining rules". |
| Sleeve risk | `src/risk/engine.py` | deterministic. Any exception → `RiskEngineError` (CRITICAL) → all orders blocked |
| Account risk | `src/portfolio/risk_manager.py` | global kill switch, symbol cap across bots; exception → all orders rejected |
| Order sanity | `src/safety/order_validator.py` | qty, price, notional, approval, staleness, cash, exposure |
| Gate | `src/safety/health.py::TradingGate` | `TRADING_ENABLED` = operator switch AND no halt AND no UNHEALTHY component AND no OPEN breaker |
| Submission | `src/execution/order_manager.py` | idempotency key, intent recorded first, UNKNOWN → reconcile, never blind retry |
| Broker | `src/execution/paper_broker.py` | refuses orders when gate closed, unapproved, duplicate, unknown symbol, oversell |
| Reconciliation | `src/execution/reconciliation.py` | any discrepancy → CRITICAL halt |
| Persistence | `src/data/database.py`, `FileOrderLedger` | rollback + `DatabaseError` (CRITICAL); atomic JSON writes |

## 2. Can an order be generated when…

| Condition | Result | Test |
|---|---|---|
| data is stale | `StaleMarketDataError` → NO_TRADE for symbol | `test_stale_data` |
| market data missing / NaN / inf | `DataValidationError` → NO_TRADE | `test_missing_prices`, `test_nan_prices`, `test_infinite_values`, `test_bad_data_blocks_trade` |
| features contain NaN | `FeatureSchemaError` → no BUY | `test_feature_schema_and_nan_handling` |
| model prediction fails | ML NaN → HOLD + CRITICAL halt | `test_model_failure_blocks_trade`, `test_model_failure_never_becomes_a_buy` |
| risk calculation fails | `RiskEngineError` → no orders, halt | `test_risk_failure_blocks_trade` |
| portfolio state unknown (NaN cash, corrupt position) | `PortfolioStateError` → no orders, halt | `test_corrupt_portfolio_state_fails_closed` |
| broker state unknown | reconciliation fails → halt | `test_position_reconciliation_failure*` |
| database write fails before submit | order not sent, halt | `test_db_failure_before_submit_means_no_order` |
| database fails after broker accepted | order kept, halt, no resubmit | `test_broker_succeeded_database_failed_halts_and_does_not_resubmit` |
| an order may already have been submitted | idempotency key + broker lookup | `test_broker_timeout_after_accept_reconciles_and_never_resubmits`, `test_duplicate_order_prevention` |
| operator sets `TRADING_ENABLED=false` | every new order refused, reads still work | `test_kill_switch_prevents_new_orders*` |

## 3. Exception handling inventory

No `except: pass` and no bare `except:` anywhere in `src/`. Every broad `except Exception` is listed
here with its justification; each records an event and/or re-raises a typed error.

| Location | Why broad | What happens |
|---|---|---|
| `trading_cycle.py` (3×) | unknown error in a bot / portfolio stage must stop order generation | logged with traceback, `gate.handle_error` (unknown ⇒ CRITICAL) |
| `trading_cycle.py::_run_bot` risk call | wrap any risk failure | re-raised as `RiskEngineError` |
| `order_manager.py` (6×) | broker/DB libraries raise anything; the safe reaction is known | UNKNOWN + reconcile, or halt; never "assume failed" |
| `reconciliation.py` | broker unreachable | report not OK + halt |
| `paper_trader.py` (4×) | per-stage fail-closed boundaries | `gate.handle_error`, run returns without orders |
| `prediction.py` (2×) | unpickling / native inference errors | re-raised as `ModelInferenceError` |
| `downloader.py` (2×) | vendor libraries; one bad symbol must not kill the batch | recorded in `ETLResult.failed`; symbol has no data → no trade |
| `circuit_breaker.py`, `api_errors.py` | count/classify, then **re-raise** | nothing is swallowed |
| `config.py` | pydantic error → `ConfigurationError` | re-raised, original chained |
| `training.py` | permutation importance (diagnostics only) | warning; never affects trading |
| `logging_setup.py` | a formatting bug must not break logging | message left unredacted-safe |
| `dashboard/app.py` (3×) | non-trading UI | error shown to the user |

## 4. Issues found and fixed during this work

These came from running the system and the tests, not from inspection alone.

1. **Object-dtype feature (`breadth_50`) halted paper trading.** `bool.where()` produced object dtype. The
   schema check correctly refused it (fail closed). Fixed with an explicit float cast, plus a test that every feature is numeric.
2. **Circuit breaker could never open on repeated submit timeouts.** A successful reconciliation *read*
   reset the failure counter. Reads no longer reset it (`test_circuit_breaker_blocks_orders`).
3. **Audit trail missed account-level resizes.** The portfolio risk manager shrank an order but the audit
   still showed the sleeve-approved quantity. The audit now records `RESIZED`, the rule and the final quantity.
4. **Forced exits were logged with a generic reason.** "Left the universe" now appears in the order reason.
5. **Cash projection mismatch between risk engine and validator.** Sell proceeds that fund same-cycle
   buys were counted by one and not the other, so valid orders were rejected. Both now use the same 1% haircut.
6. **Floating-point boundaries.** `88/110-1 = -0.19999…` did not trip a 20% kill switch, and `0.30-0.25`
   lost a share in sector caps. Fixed with explicit tolerances, and tested at the boundary.
7. **Paper trading lost risk state between runs.** Drawdown peak / kill switch, last run date (weekly
   rebalancing) and the order ledger are now persisted. A second run on the same day is refused.
8. **Explainer crashed on audit rows read back from CSV** (NaN components). Fixed and tested.
9. **mypy found 34 unchecked `None` cases**, e.g. a LIMIT order without a price reaching the simulator, a
   missing reference price in notional checks, and an unguarded sector lookup. All fixed. `mypy src` is clean.
10. **yfinance call had no explicit timeout.** Now `timeout=20, raise_errors=True`, so failures reach the
    retry/breaker layer instead of returning silently empty frames.

### Multi-bot revision (2026-10-19)

11. **One bot's model failure halted the whole account.** `ModelInferenceError` is CRITICAL, so a corrupt
    SWING model closed the trading gate for TREND and INVESTOR too. Failures are now scoped. A bot's
    features, model or strategy code failing disables that bot only: no new trades, stop-losses still run,
    and its positions still count in account risk. Sleeve risk engine, portfolio manager, portfolio
    state, reconciliation and DB/broker integrity failures still stop ALL new trading
    (`tests/test_multi_bot.py`, `test_integration.py::test_paper_trader_end_to_end_and_model_corruption`).
12. **A disabled bot could have vanished from account-level risk.** Account state was built only from
    bots with a runtime. `TradingCycle(sleeves=...)` now always includes every sleeve
    (`test_disabled_bot_still_counts_in_account_risk_and_keeps_stops`).
13. **Fills booked during a refused re-run were lost.** A manual second run on the same session booked
    overnight Alpaca fills, then refused to decide twice without saving the sleeves. The next run's
    reconciliation would then fail and halt trading. Sleeves are now persisted immediately after fills
    are booked (`test_fills_booked_during_a_refused_rerun_are_not_lost`, confirmed failing before the fix).
14. **Model artifacts lacked provenance.** Artifacts now record model_id, training_start/end,
    validation_period, target_definition, algorithm and hyperparameters, and loading refuses artifacts
    without them (retrain once after upgrading).

## 5. NaN / inf / division hazards

| Calculation | Guard |
|---|---|
| position sizing (÷ price, vol, ATR) | `_valid()` rejects None/NaN/inf/≤0 → size 0 |
| risk caps (÷ price) | price validated first, epsilon-safe floor |
| RSI (÷ avg loss) | loss = 0 ⇒ RSI 100, no division |
| risk-adjusted momentum (÷ vol) | vol ≤ 0 ⇒ NaN ⇒ no signal |
| volume z-score (÷ std) | std ≤ 0 ⇒ NaN |
| metrics (÷ std, ÷ max DD) | return 0.0 for degenerate series (tested) |
| order quantity | `Order.__post_init__` rejects NaN, inf, fractions, bool, str, ≤0 |

## 6. Known gaps (not yet addressed)

* No exchange calendar: gaps of ≤ 3 business days are treated as holidays.
* yfinance has no delisted symbols, so a large-universe backtest on it is survivorship-biased even with the
  membership file. A vendor with delisted history is needed.
* The paper broker fills at the next open, all-or-nothing up to the participation cap. There is no queue
  position or intraday modelling.
* Approval registry is in-process memory. A broker process restarted between approval and submission
  will refuse the order (fail closed), and the next run re-decides it.
* `allow_risk_reducing_orders_when_halted=false` (strict default) means a CRITICAL error also blocks
  stop-loss exits until recovery. This is a deliberate trade-off; set it to `true` to allow exits only.
* Correlated exposure across bots is limited only indirectly (symbol cap, per-bot weight, sleeve
  portfolio-volatility forecast). There is no account-level correlation limit yet.
* The investor's 3-12 month horizon is an expectation, not a forced exit (no `max_holding_days`).
* Reconciliation cash tolerance is $1. If the Alpaca paper account ever credits dividends or fees
  that the bot's books do not model, reconciliation will halt trading (fail closed) until reviewed.
