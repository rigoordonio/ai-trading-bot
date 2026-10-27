# AI-Assisted Quantitative Trading Platform

> 🚧 **In progress.** Backtesting and paper trading work today; more strategies and reporting are on the way.


**Backtesting, simulation and paper trading only. Real-money trading is not implemented and is refused by configuration.**

One platform, three independent strategy bots, one deterministic risk layer that nothing can bypass.
Decisions come from defined rules and trained models; an optional LLM only *explains* them.

```
                      MARKET DATA (provider abstraction: yfinance / CSV / synthetic)
                           │
                     [VALIDATE] ── invalid ──> STOP (no trade for that symbol)
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
     SWING_V1          TREND_V1         INVESTOR_V1          each bot: own features, components,
     daily, 5-20d      daily, 20-60d    weekly, 3-12m         ML target + model version, risk
         │                 │                 │                parameters and capital sleeve
   features→model→TradingSignal (fail closed: a bot's model failure = NO new trades
         │                 │                 │       for THAT bot; other bots continue)
         │                 │                 │
   sleeve RiskEngine  sleeve RiskEngine  sleeve RiskEngine    sizing, limits, stops, drawdown
         └─────────────────┼─────────────────┘
                           ▼
          MULTI-BOT PORTFOLIO MANAGER          account-wide: kill switch, symbol cap across bots,
                           │                   strategy concentration, conflicting signals
                           ▼
                    ORDER VALIDATOR            qty / price / notional / approval / staleness
                           ▼
                  TRADING GATE (TRADING_ENABLED)   operator switch · health · circuit breakers
                           ▼
                     ORDER MANAGER             idempotency key, UNKNOWN → reconcile, no blind retry
                           ▼
                     PAPER BROKER  ──>  RECONCILIATION  ──>  AUDIT TRAIL / DATABASE  ──>  DASHBOARD
```

The **same** feature pipeline, bots and `TradingCycle` run in the backtester and in paper trading.

---

## 1. Quick start (fully offline, synthetic data)

```bash
cd ai-trading-bot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python scripts/train_model.py   --provider synthetic        # train + validate + test each bot's model
python scripts/backtest.py --bots --provider synthetic      # 3 bots on one account, walk-forward ML
python scripts/backtest.py --strategy all --provider synthetic   # each rule strategy independently
python scripts/robustness.py --provider synthetic           # cost/param/start-date sweeps + Monte Carlo
python scripts/paper_trade.py --provider synthetic --as-of 2026-09-01   # one paper-trading day
streamlit run src/dashboard/app.py                          # dashboard (dark theme)
uvicorn src.api.main:app --reload                           # read-only API
```

Synthetic data is fake. It exists so everything runs and tests without a network. **Never draw
investment conclusions from it.** For real research, use `--provider yfinance` (prototyping only), or
drop vendor CSVs into `data/raw/csv/` and use `--provider csv`.

Docker: `docker compose up --build` (PostgreSQL + API + dashboard); `docker compose run --rm paper` runs one paper cycle.

---

## 2. Repository layout

```
config/       settings.yaml (universe, data, costs, ML split) · strategies.yaml (research library)
              risk.yaml (limits, stops, kill switch, safety, account limits)
              bots/swing.yaml · bots/trend.yaml · bots/investor.yaml (one file per bot)
src/
  data/       provider · downloader (ETL + retries) · validator · corporate_actions · universe · database
  features/   technical · momentum · volatility · volume · market_regime · weekly · pipeline (+ versioning)
  strategies/ trend_following · mean_reversion · momentum · breakout · volume_confirmation · long_term · ml · ensemble
  bots/       base (BaseTradingBot contract) · signal (TradingSignal) · registry
              swing/ trend/ investor/  each: bot · features · strategy · model
  data/resampling.py  daily -> weekly candles (one validation path)
  models/     training (chronological, purged) · prediction (registry, checksums, validation) · evaluation
  risk/       engine · position_sizing · stop_loss · exposure · drawdown (kill switch) · portfolio_risk
  portfolio/  portfolio (accounting) · allocation (proposals) · multi_bot (MultiBotPortfolioManager)
              attribution (per-bot virtual positions vs broker)
  execution/  orders (state machine) · simulator (costs) · broker (interface) · paper_broker
              order_manager (idempotency, UNKNOWN) · reconciliation · paper_trader
  safety/     validation · order_validator · health (TRADING_ENABLED) · circuit_breaker · api_errors
  backtesting/ engine · metrics · walk_forward · robustness
  analytics/  performance · attribution · store      ai/  explanation (text only) · sentiment
  api/        read-only FastAPI                       dashboard/  Streamlit app
  trading_cycle.py   the fail-closed decision cycle shared by backtest and paper trading
  errors.py          error taxonomy + severity model
scripts/      download_data · train_model · backtest · walk_forward · robustness · paper_trade · bot_activity
tests/        308 tests (unit, integration, property-based, regression, multi-bot acceptance)
docs/AUDIT.md safety audit: every failure path and how it is closed
```

---

## 3. The three bots (`config/bots/*.yaml`)

| | SWING_V1 | TREND_V1 | INVESTOR_V1 |
|---|---|---|---|
| Code | `src/bots/swing/` | `src/bots/trend/` | `src/bots/investor/` |
| Candles | daily | daily | **weekly** (built from validated daily bars) |
| Horizon | 5-20 trading days | 20-60 trading days | 3-12 months |
| Components | short-term momentum, 20d breakout, mean reversion, volume confirmation | 50/200 trend, 20/100 trend, relative momentum | 52-4 week momentum, 10/40-week trend, low volatility, drawdown filter |
| Own features | (shared pipeline set, pinned) | `trend_strength` | `w_drawdown_52`, `w_trend_strength` (weekly space) |
| ML model | `swing_model_v1`: fwd 5d return > **+1%** | `trend_model_v1`: fwd 40d > +2% | `investor_model_v1`: fwd 63d > +3%, weekly features only |
| Sizing | ATR (0.75% risk/trade) | volatility target | fixed fractional |
| Stops | 6% fixed, 2×ATR trailing, 30-day time | 12% fixed, 3.5×ATR trailing | 20% fixed |
| Capital | 30% | 40% | 30% |
| Evaluated | each completed day | completed daily data, once a week | each completed week |

**Contracts.** Each bot class checks its configuration when built: its timeframe, the strategy component
types that belong to its logic, the allowed ML horizon range (swing 3-10, trend 20-60, investor 55-130
trading days) and, for the investor, that model and rules use weekly (`w_`) features only. A config edit
that breaks the design fails at start-up instead of quietly turning one bot into a copy of another.

**Signals.** Every evaluated symbol produces a `TradingSignal` (BUY / HOLD / REDUCE / EXIT / NO_TRADE) with
`signal_id`, bot, versions, score, model probability, expected horizon and the **data timestamp** (the bar
it used). The order timestamp is separate and set by the OrderManager. Signal ids are deterministic, so a
re-run never invents new signals.

**Attribution.** Each bot trades its own capital sleeve and owns its shares. The broker sees
`AAPL 225 = SWING 50 + TREND 75 + INVESTOR 100`; a SWING exit sells only SWING's 50. Reconciliation
compares the sum with the broker and reports the per-bot breakdown on any mismatch.

**Two risk levels.** Level 1, the bot's sleeve RiskEngine: size, limits, stops, sleeve drawdown. Level 2, the
MultiBotPortfolioManager: global kill switch, combined symbol weight, one bot's weight of the account
(`max_bot_weight`), total exposure and positions. It also reports conflicting signals. Either level can veto; nothing
(ML or LLM) can override them.

**Failure scope.**

| Failure | Effect |
|---|---|
| One bot's features / model / strategy code | That bot makes **no new trades** this run; its stop-losses still run; other bots continue |
| Market data for one symbol | That decision is blocked |
| Portfolio risk manager, sleeve risk engine, portfolio state | **All** new trading stops |
| Broker reconciliation, unknown DB/broker state | **All** new trading stops |

**Weekly look-ahead rule:** a week's bar becomes usable on the first trading day *after* that week ends.
Tests prove weekly features and signals are identical whether the data ends mid-week or not.

**Fundamentals:** `investor.yaml` has a `fundamentals` block, disabled. No provider here supplies reliable
fundamentals, so enabling it stops the bot at start-up rather than inventing numbers.

**Thresholds are hypotheses.** Targets are not "return > 0" (costs exist). All thresholds must be validated
with `scripts/walk_forward.py` and never tuned on the test period.

**"Show everything SWING_V1 did with NVDA":** `python scripts/bot_activity.py --bot SWING_V1 --symbol NVDA`
(reads the database written by `backtest.py --db` / `paper_trade.py --db`).

---

## 4. How a decision becomes (or does not become) an order

1. **Bar validation:** finite, positive, OHLC-consistent, not stale, tz-naive, no >40% jump. Otherwise NO_TRADE for the symbol.
2. **Feature validation:** every feature the bot needs to open a position is present and finite.
3. **Model:** the artifact must exist, pass its sha256 checksum, match the bot's model version and feature
   version, and return probabilities in [0, 1]. Any failure means no prediction, which means a NaN score
   and therefore HOLD. A failure is **never** a default buy and never re-uses an old prediction.
4. **Signal:** weighted ensemble of components (weights in YAML). NaN means HOLD. BUY / HOLD / REDUCE / EXIT, depending on the position.
5. **Sleeve risk engine:** kill switch, daily loss, min price, liquidity, max positions, max position
   weight, sector cap, gross exposure, cash reserve, max order notional, forecast portfolio volatility.
   Returns `RiskDecision(approved, reason, rule_triggered, timestamp)`.
6. **Multi-bot portfolio manager:** global kill switch, combined symbol weight, per-bot weight, account exposure, total positions.
7. **Order validator:** sanity checks plus *verification* that the risk approval is genuine (forged ids fail).
8. **Trading gate:** `TRADING_ENABLED=false`, a CRITICAL error, an UNHEALTHY component or an OPEN circuit breaker refuses every new order.
9. **Order manager:** writes the intent first (DB down means nothing is sent), then submits. A timeout or lost
   response leaves the order UNKNOWN; the manager queries the broker by idempotency key and adopts its answer. It never resubmits blindly.
10. **Reconciliation** every run: positions, cash, open orders, duplicates, unresolved UNKNOWNs. Any mismatch halts trading.

Every evaluated (bot, symbol, bar) is written to the **audit trail**: scores, ML probability, versions,
action, risk status, rule, quantity, order id/status and execution. That answers both *"why did it buy?"*
and *"why did it refuse?"*. The dashboard's Signals page can turn a row into plain English.

---

## 5. Backtesting methodology

* Signals use data up to the close of day *t*; orders fill at the **open of t+1** (configurable delay ≥ 1).
* Costs: commission (per share, minimum, %), half-spread, slippage, √participation impact, a volume
  participation cap (partial fills), cash checks at the fill (gap risk), and rejections for missing bars.
* Prices are split- **and** dividend-adjusted (total return); raw prices are kept for audit.
* Point-in-time universe membership file (`universe.membership_file`) prevents trading symbols outside
  their membership dates (survivorship-bias control). Note that yfinance has no delisted tickers.
* Benchmark: buy-and-hold of `universe.benchmark` with beta, alpha, tracking error and information ratio.
* ML: chronological train → validation (model choice) → test (touched once), with labels **purged** at
  every boundary. The scaler is fit inside the pipeline on training rows only. Walk-forward retrains per window,
  and only out-of-sample predictions ever reach a backtest.
* Robustness: cost multipliers, parameter grids, start dates, block bootstrap, trade-sequence shuffle.
  Each reports distributions and states its assumptions.

---

## 6. Testing and CI

```bash
make ci            # ruff -> mypy -> unit -> integration -> coverage gate (85% on trading-critical code)
pytest -m "not integration and not slow"
pytest -m integration
```

Test groups: data validation, decision-time validation, indicators, **look-ahead / leakage** (truncation
and future-shock tests, static `shift(-n)` guard, scaler-on-train check, purge and walk-forward boundaries),
strategies (+ a regression snapshot on formula-based data), ML safety (missing / corrupted / wrong version /
wrong schema / NaN / out-of-range / inference failure), risk, orders, the order state machine, idempotency,
broker timeout/UNKNOWN, DB failure before/after submit, reconciliation, circuit breaker, HTTP failure
classes, retries with backoff and jitter, the kill switch, security (secrets, redaction, live mode refused),
property-based invariants (Hypothesis, with a seeded fallback), and exact-number engine tests (the
$10,000 → buy 10 @ $100 → sell 10 @ $110 → $10,100 example, with and without costs).

Tests never assert profitability. They assert arithmetic and safety.

---

## 7. Security

* Secrets only in `.env` (git-ignored). `.env.example` has empty values.
* Log handlers redact any env value whose name contains KEY/SECRET/TOKEN/PASSWORD, plus credentials in DB URLs.
* `TRADING_MODE` accepts `BACKTEST` / `PAPER`; `LIVE` raises `LiveTradingNotAllowedError`. There is no live broker adapter.
* The API has no endpoint that can create, modify or cancel orders; the AI package cannot import execution or risk code (tested).

---

## 8. Alpaca paper account

1. Paper Trading dashboard → **API Keys** → generate. Put them in `.env` (never in code or chat):
   ```
   BROKER=alpaca_paper
   ALPACA_API_KEY=...
   ALPACA_API_SECRET=...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
2. Test the connection (read-only, places nothing):
   `python scripts/paper_trade.py --check-broker`
3. Daily run **after 19:00 New York time** (08:15 Manila is safe all year). Orders go in as
   opening-auction (OPG) orders and fill at the next open, like the backtest:
   ```
   15 8 * * 2-6 cd ~/Projects/ai-trading-bot && .venv/bin/python scripts/paper_trade.py >> data/paper/cron.log 2>&1
   ```
   (Tuesday to Saturday in Manila = Monday to Friday in New York.)

Safety: the adapter refuses any URL other than `paper-api.alpaca.markets`; every order carries our
idempotency key as Alpaca's `client_order_id` (a lost response is looked up, never resent); POSTs are never
retried. On the first run the bots' sleeves start from the Alpaca account's cash; after that, Alpaca's
positions and cash are reconciled daily and any mismatch halts trading. The account's margin
"buying power" is never used: the bot is long-only and capped at 95% of equity.
Do not trade manually in the same paper account. Reconciliation will (correctly) halt the bot.

## 9. Adding another broker's paper API

Implement `BrokerAdapter` (`get_account`, `get_positions`, `get_market_data`, `submit_order`,
`cancel_order`, `get_orders`, `get_order_by_client_id`). Send our `idempotency_key` as the broker's
client order id, map its HTTP errors through `src/safety/api_errors.py`, and wrap calls in the broker
circuit breaker. Do not retry `submit_order`; let `OrderManager` reconcile. Strategy and risk code do not change.

See `docs/AUDIT.md` for known limitations.
