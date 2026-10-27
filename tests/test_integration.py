"""DATA -> FEATURES -> MODEL -> STRATEGY -> RISK -> ORDER -> PAPER BROKER -> LEDGER/DB.

Each test injects one failure and asserts the system FAILS CLOSED."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import StrategyBot
from src.config import EnsembleConfig, load_config
from src.errors import DuplicateOrderError, ModelInferenceError, OrderRejectedError, TradingSystemError
from src.execution.order_manager import InMemoryLedger, OrderManager
from src.execution.orders import OrderStatus, Side
from src.execution.reconciliation import reconcile
from src.features.pipeline import FeaturePipeline
from src.models.prediction import Predictor
from src.portfolio.portfolio import Portfolio
from src.portfolio.risk_manager import PortfolioRiskManager
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import RiskEngine
from src.risk.stop_loss import StopLossManager
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthMonitor, TradingGate
from src.safety.order_validator import OrderValidator
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.trading_cycle import BotRuntime, MarketState, TradingCycle
from tests.fakes import FaultyBroker, FaultyLedger, MockModel
from tests.helpers import flat_prices

pytestmark = pytest.mark.integration
SYMS = ["AAPL", "MSFT", "SPY"]
FEATS = ["ret_5d", "ret_20d", "rsi_14", "atr_pct", "dist_sma_50"]


def make_prices():
    n = 320
    closes = {s: list(np.linspace(100 + 10 * i, 180 + 10 * i, n)) for i, s in enumerate(SYMS)}
    p = flat_prices(SYMS, closes, start="2023-01-02", open_equals_close=False)
    for s in SYMS:                    # realistic high/low around the bar
        p[s]["high"] = p[s][["open", "close"]].max(axis=1) * 1.005
        p[s]["low"] = p[s][["open", "close"]].min(axis=1) * 0.995
    return p


class World:
    """One account, one bot (trend + required ML), paper broker, ledger."""

    def __init__(self, model=None, broker_mode=None, ledger=None, prices=None):
        self.cfg = load_config(overrides={"risk": {"limits": {"min_avg_dollar_volume": 0}}})
        self.prices = prices or make_prices()
        self.gate = TradingGate(HealthMonitor(), operator_enabled=True)
        self.broker = FaultyBroker(100_000, mode=broker_mode, trading_enabled=lambda: self.gate.trading_enabled,
                                   known_symbols=set(SYMS))
        self.ledger = ledger if ledger is not None else InMemoryLedger()
        self.om = OrderManager(self.broker, self.ledger, self.gate, CircuitBreaker("broker", 3))
        self.sleeve = Portfolio(cash=100_000)
        self.model = model or MockModel(output=0.9)
        self.dd = DrawdownMonitor.from_config(self.cfg.risk.drawdown)
        self.prm = PortfolioRiskManager.from_config(self.cfg.risk)
        self.cycle_prev = None

    def decide_and_submit(self, as_of: pd.Timestamp, corrupt: dict | None = None):
        prices = {s: df.loc[:as_of].copy() for s, df in self.prices.items()}
        for sym, patch in (corrupt or {}).items():
            for k, v in patch.items():
                prices[sym].loc[prices[sym].index[-1], k] = v
        panel = FeaturePipeline().compute(prices)                                   # FEATURES
        last = panel.xs(as_of, level="date", drop_level=False)
        preds = None
        try:
            preds = Predictor(self.model, FEATS, "fv").predict(last, "fv")         # MODEL
        except ModelInferenceError as exc:
            self.gate.handle_error(exc, "model")                                  # -> NO TRADE
        trend = TrendFollowingStrategy()
        from src.strategies.ml_strategy import MLStrategy
        ml = MLStrategy(preds if preds is not None else pd.Series(np.nan, index=panel.index), entry_threshold=0.5)
        ens = EnsembleStrategy({"trend_following": trend},
                               EnsembleConfig(weights={"trend_following": 0.6, "ml": 0.4}, buy_threshold=0.6,
                                              hold_threshold=0.4, exit_threshold=0.25),
                               ml=ml, required_components=("ml",), min_components=1)
        sf = ens.signal_frame(panel)                                                # STRATEGY
        bot = StrategyBot("BOT", SYMS, self.cfg.risk, list(trend.required_features))
        feats_today = panel.xs(as_of, level="date")
        rt = BotRuntime(bot, sf, lambda d: feats_today, self.sleeve,
                        RiskEngine(self.cfg.risk, {}, self.dd), StopLossManager.from_config(self.cfg.risk.stops))
        cycle = TradingCycle({"BOT": rt}, self.prm, OrderValidator.from_config(self.cfg.risk), self.gate,
                             self.cfg.risk.safety, set(SYMS))
        cycle._prev_date = self.cycle_prev
        bars = {s: df.iloc[-1] for s, df in prices.items()}
        res = cycle.run(MarketState(as_of, bars, {s: df.index[-1] for s, df in prices.items()},
                                    {s: float(df["close"].iloc[-2]) for s, df in prices.items()},
                                    pd.DataFrame({s: df["close"] for s, df in prices.items()}).pct_change()))
        self.cycle_prev = as_of
        submitted, errors = [], []
        for o in res.orders:                                                       # ORDER -> BROKER
            try:
                submitted.append(self.om.submit(o))
            except TradingSystemError as exc:
                errors.append(exc)
        return res, submitted, errors

    def next_open(self, date):
        bars = {s: df.loc[date] for s, df in self.prices.items()}
        fills = self.broker.process_open(date, bars)
        from src.execution.orders import Fill
        for f in fills:
            self.sleeve.apply_fill(Fill(f["order_id"], f["symbol"], Side(f["side"]), f["quantity"], f["price"],
                                        f["reference_price"], f["commission"], pd.Timestamp(f["timestamp"])))
        return fills

    def reconcile(self):
        return reconcile({s: p.quantity for s, p in self.sleeve.positions.items()}, self.sleeve.cash, self.ledger,
                         self.broker, self.gate)


def dates(w):
    idx = w.prices["AAPL"].index
    return idx[260], idx[261], idx[262]


def test_successful_trade_pipeline():
    w = World()
    d0, d1, _ = dates(w)
    res, sub, err = w.decide_and_submit(d0)
    assert not err and len(sub) >= 1 and all(s.submitted for s in sub)
    assert all(r.order.status == OrderStatus.ACKNOWLEDGED for r in sub)
    buys = [a for a in res.audit if a["final_action"] == "BUY"]
    assert buys and buys[0]["risk_status"] in ("PASS", "RESIZED") and buys[0]["ml_probability"] == pytest.approx(0.9)
    fills = w.next_open(d1)
    assert len(fills) == len(sub)
    assert w.reconcile().ok and w.gate.trading_enabled


def test_bad_data_blocks_trade():
    w = World()
    d0, _, _ = dates(w)
    res, sub, _ = w.decide_and_submit(d0, corrupt={"AAPL": {"high": 1.0}})      # high < low
    assert "AAPL" not in {s.order.symbol for s in sub}
    rec = [a for a in res.audit if a["symbol"] == "AAPL"][0]
    assert rec["final_action"] == "NO_TRADE" and rec.get("data_status") == "INVALID"
    res2, sub2, _ = World().decide_and_submit(d0, corrupt={"MSFT": {"close": np.nan}})
    assert "MSFT" not in {s.order.symbol for s in sub2}


def test_model_failure_blocks_trade():
    w = World(model=MockModel(raise_exc=RuntimeError("segfault in native lib")))
    d0, _, _ = dates(w)
    res, sub, err = w.decide_and_submit(d0)
    assert sub == [] and w.broker.get_orders() == []
    assert not w.gate.trading_enabled
    assert all(a["final_action"] == "NO_TRADE" for a in res.audit)
    w2 = World(model=MockModel(output=float("nan")))
    _, sub2, _ = w2.decide_and_submit(d0)
    assert sub2 == [] and w2.broker.get_orders() == []


def test_risk_failure_blocks_trade(monkeypatch):
    w = World()
    d0, _, _ = dates(w)

    def broken(*a, **k):
        raise ZeroDivisionError("bug in risk code")

    monkeypatch.setattr(RiskEngine, "review", broken)
    res, sub, _ = w.decide_and_submit(d0)
    assert sub == [] and w.broker.get_orders() == []
    assert not w.gate.trading_enabled
    assert any(e.get("event") == "TRADING_HALTED" and e.get("severity") == "CRITICAL" for e in res.events)


def test_database_failure_handling():
    w = World(ledger=FaultyLedger(fail_intent=True))
    d0, _, _ = dates(w)
    _, sub, err = w.decide_and_submit(d0)
    assert sub == [] and err and w.broker.submit_calls == 0         # intent not recorded -> not sent
    assert not w.gate.trading_enabled
    led = FaultyLedger()
    w2 = World(ledger=led)
    led.fail_update = True                                          # broker succeeds, DB fails afterwards
    _, sub2, _ = w2.decide_and_submit(d0)
    assert w2.broker.submit_calls == 1 and not w2.gate.trading_enabled   # halted after the first, no repeats


def test_broker_rejection():
    w = World(broker_mode="reject")
    d0, _, _ = dates(w)
    _, sub, err = w.decide_and_submit(d0)
    assert sub == [] and err and all(isinstance(e, OrderRejectedError) for e in err)
    assert all(o.status == OrderStatus.REJECTED for o in w.ledger.all())


def test_broker_timeout():
    w = World(broker_mode="timeout_after_accept")
    d0, _, _ = dates(w)
    _, sub, _ = w.decide_and_submit(d0)
    n = len(w.broker.get_orders())
    assert n >= 1 and all(r.order.status == OrderStatus.ACKNOWLEDGED for r in sub)   # reconciled, not resubmitted
    assert w.broker.submit_calls == n


def test_duplicate_order_prevention():
    w = World()
    d0, _, _ = dates(w)
    _, sub1, _ = w.decide_and_submit(d0)
    w.cycle_prev = None
    _, sub2, err2 = w.decide_and_submit(d0)                          # same decision replayed
    assert sub1 and not sub2 and err2 and all(isinstance(e, DuplicateOrderError) for e in err2)
    keys = [o.client_order_id for o in w.broker.get_orders()]
    assert len(keys) == len(set(keys)) == len(sub1)


def test_position_reconciliation_failure():
    w = World()
    d0, d1, d2 = dates(w)
    w.decide_and_submit(d0)
    w.next_open(d1)
    sym = next(iter(w.sleeve.positions))
    w.sleeve.positions[sym].quantity += 7                             # internal books drift
    rep = w.reconcile()
    assert not rep.ok and not w.gate.trading_enabled
    _, sub, _ = w.decide_and_submit(d1)
    assert sub == []                                                  # TRADING_ENABLED = False


def test_circuit_breaker_blocks_orders():
    w = World()
    d0, _, _ = dates(w)
    cb = CircuitBreaker("market_data", 1)
    w.gate.breakers.append(cb)
    cb.record_failure("vendor 503")
    res, sub, _ = w.decide_and_submit(d0)
    assert sub == [] and w.broker.get_orders() == []
    assert any("circuit breaker OPEN" in (a.get("reason") or "") for a in res.audit)


def test_operator_kill_switch_prevents_new_orders(monkeypatch):
    monkeypatch.setenv("TRADING_ENABLED", "false")
    w = World()
    w.gate = TradingGate(HealthMonitor())
    w.om = OrderManager(w.broker, w.ledger, w.gate, CircuitBreaker("broker", 3))
    w.broker._trading_enabled = lambda: w.gate.trading_enabled
    d0, _, _ = dates(w)
    _, sub, _ = w.decide_and_submit(d0)
    assert sub == [] and w.broker.get_orders() == []
    assert w.broker.get_account().cash == 100_000                     # state still readable


# ----------------------------------------------------- full PaperTrader run
@pytest.mark.slow
def test_paper_trader_end_to_end_and_model_corruption(tmp_path, monkeypatch):
    from src.bots.bot import build_bots
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for b in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=b.bot_id, model_version=b.ml_cfg.model_version,
                   feature_set_version=b.feature_version, features=b.ml_features, target={}, periods={},
                   metrics={}, model_name="mock")
    days = pd.bdate_range("2026-06-01", periods=3)
    reports = []
    for d in days:
        pt = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots)
        reports.append(pt.run(d))
    assert all(r.trading_enabled for r in reports), [r.blocking_reasons for r in reports]
    assert all(r.reconciliation["ok"] for r in reports)
    assert sum(len(r.submitted) for r in reports) > 0
    # corrupt ONE bot's model -> that bot fails closed (no new SWING orders),
    # the other bots are not crashed and the account is not halted (spec §18)
    f = tmp_path / "models" / "SWING_V1" / bots["SWING_V1"].ml_cfg.model_version / "model.joblib"
    f.write_bytes(b"corrupted")
    pt = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots)
    r = pt.run(days[-1] + pd.offsets.BDay(1))
    assert r.trading_enabled, r.blocking_reasons
    assert not [o for o in r.submitted if o.get("strategy") == "SWING_V1" and o["side"] == "BUY"]
    disabled = [e for e in r.events if e.get("event") == "BOT_DISABLED"]
    assert disabled and disabled[0]["bot_id"] == "SWING_V1" and "checksum" in disabled[0]["detail"]
    assert pt.gate.health.components["model"].status.value == "DEGRADED"
