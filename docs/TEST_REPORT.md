# Test report

## Multi-bot revision, 2026-10-19 (latest)

Environment: cloud sandbox, Python 3.13.16. PyPI was blocked, so these packages were installed from their
official GitHub source tags (pure Python, no C extensions) and the tests ran under the **real** tools:
pytest 8.3.4, Hypothesis 6.112.0, SQLAlchemy 2.0.36, FastAPI 0.115.6 (+ starlette, httpx), coverage.py 7.6.9 and
mypy 1.13.0. ruff could not be installed (it is a compiled Rust binary); pyflakes 3.2.0 and a 120-column
line check ran on the changed files instead. Run `make ci` on your machine for the authoritative result.

| Check | Result |
|---|---|
| `pytest` (whole suite, nothing skipped) | **308 passed, 0 failed, 0 skipped (also with a hostile .env present)** |
| Coverage, trading-critical set (CI gate 85%) | 94% |
| Coverage, new multi-bot code (bots/, multi_bot, attribution, trading_cycle, reconciliation, resampling) | 95% |
| Coverage, all of src | 90% |
| mypy src | 4 errors, identical to the baseline before this revision (yaml stubs not installed; pydantic/sklearn stub mismatches in config.py:136, training.py:167, alpaca_paper_broker.py:153). No new errors |
| pyflakes | only `_bootstrap imported but unused` in scripts (intentional, `# noqa: F401`) |

New tests: `tests/test_multi_bot.py`. It has all 15 named tests from the specification §19, the §24 final
acceptance scenario, and contract, signal, conflict, concentration, isolation, overview and
diversification tests. Also new: DB tests for `bot_activity`, additive migration and paper state, plus a
paper-run regression test.

---

## Previous report (2026-10-18)

## Environment caveat
The build sandbox could not reach PyPI. pytest, pytest-cov, Hypothesis, SQLAlchemy, FastAPI, Streamlit and
plotly were NOT installable. Therefore:

* Tests ran under a small pytest-compatible runner (fixtures, parametrize, raises, approx, tmp_path,
  monkeypatch, capsys). Run `pytest` on your machine for the authoritative result.
* Coverage was measured with Python's built-in `sys.monitoring` line tracing, not pytest-cov.
  Expect small differences in the numbers.
* Property tests ran on their seeded-random fallback (300 examples each), not Hypothesis.
* `tests/test_database.py` (5 tests) and `tests/test_api.py` (4 tests) were SKIPPED: SQLAlchemy and
  FastAPI are missing. They have **not been executed**.
* The Streamlit dashboard was smoke-run page by page with Streamlit stubbed out (logic paths only, plus
  one in-page bot backtest). Real browser rendering was not verified.

ruff 0.16.8 and mypy 2.3.1 WERE available and ran for real.

## Results
| Check | Result |
|---|---|
| ruff check src tests scripts | All checks passed |
| mypy src (79 files) | Success: no issues found (pandas-stubs not installed: pandas is typed as Any) |
| Tests executed | **261 passed, 0 failed** (incl. 17 Alpaca adapter tests on a fake API) |
| Tests not executed | 9 (2 modules skipped: SQLAlchemy / FastAPI missing) |
| Line coverage, all of src | 77% |
| Line coverage, trading-critical set (CI gate) | 92.9% (gate: 85%) |

## Trading-critical coverage
| File | Cover | Uncovered lines |
|---|---|---|
| src/data/validator.py | 84% | 35-36,62-64,67-68,88-89,94-95,97-98,100,151 |
| src/execution/order_manager.py | 95% | 205-206,239-242,244 |
| src/execution/orders.py | 97% | 120,135,139 |
| src/execution/paper_broker.py | 88% | 69-71,91,93-96,127-128,139-143,155,183 |
| src/execution/reconciliation.py | 100% |  |
| src/execution/simulator.py | 93% | 49,84,95,110-111,113 |
| src/portfolio/portfolio.py | 94% | 82-84,87-88,108 |
| src/risk/drawdown.py | 100% |  |
| src/risk/engine.py | 90% | 86-91,136-137,185,189,191,262-268 |
| src/risk/exposure.py | 100% |  |
| src/risk/portfolio_risk.py | 88% | 17,28,34 |
| src/risk/position_sizing.py | 96% | 52 |
| src/risk/stop_loss.py | 100% |  |
| src/safety/api_errors.py | 92% | 51,64-67 |
| src/safety/circuit_breaker.py | 92% | 36,66,102-104 |
| src/safety/health.py | 98% | 66-67 |
| src/safety/order_validator.py | 93% | 49,53,56,58 |
| src/safety/validation.py | 93% | 23-24,34,40,102,114 |
| src/trading_cycle.py | 96% | 117-118,171-172,196-197,278,283-286,315 |

The uncovered critical lines are mainly reporting helpers (`RiskDecision.to_dict`, `RiskEngine.status`) and
secondary exception branches, e.g. a ledger write failing *during* a broker-order refresh.

## Not covered at all (needs the missing packages)
`src/data/database.py`, `src/api/main.py`, `src/dashboard/app.py` (0%); `src/workflows.py` (36%, exercised
end to end by the scripts rather than unit tests).

## Bugs found by running the system (all fixed and covered by tests)
See docs/AUDIT.md section 4.
