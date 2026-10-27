"""Alpaca PAPER adapter against an in-memory fake API (never touches a real account)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.errors import (
    ConfigurationError,
    DuplicateOrderError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    Severity,
    TradingDisabledError,
)
from src.execution.alpaca_paper_broker import PAPER_URL, AlpacaPaperBroker, opg_window_open
from src.execution.broker import create_broker
from src.execution.order_manager import InMemoryLedger, OrderManager
from src.execution.orders import OrderStatus
from src.execution.reconciliation import reconcile
from src.safety.api_errors import APIError, MalformedResponseError
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthMonitor, TradingGate
from tests.fake_alpaca import FakeAlpaca
from tests.helpers import D0, approved_order

ET = ZoneInfo("America/New_York")
EVENING = datetime(2026, 6, 1, 20, 0, tzinfo=ET)        # 20:00 ET: OPG window open
MIDDAY = datetime(2026, 6, 1, 12, 0, tzinfo=ET)         # 12:00 ET: OPG window closed


def broker(fake=None, clock=EVENING, enabled=lambda: True, tif="opg"):
    fake = fake or FakeAlpaca()
    b = AlpacaPaperBroker("PKTESTKEY123", "secretsecretsecret", PAPER_URL, time_in_force=tif, session=fake,
                          clock=lambda: clock, trading_enabled=enabled, known_symbols={"AAPL", "MSFT"},
                          sleep=lambda s: None)
    return b, fake


def validated(**kw):
    o = approved_order(**kw)
    o.transition(OrderStatus.VALIDATED)
    return o


# ------------------------------------------------------------- safety locks
@pytest.mark.parametrize("url", ["https://api.alpaca.markets", "https://evil.example.com",
                                 "http://paper-api.alpaca.markets"])
def test_refuses_anything_but_the_paper_endpoint(url):
    with pytest.raises(LiveTradingNotAllowedError):
        AlpacaPaperBroker("k", "s", url, session=FakeAlpaca())


def test_requires_keys_and_never_exposes_them(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(ConfigurationError):
        AlpacaPaperBroker(base_url=PAPER_URL, session=FakeAlpaca())
    b, _ = broker()
    assert "secretsecretsecret" not in repr(b) and "PKTESTKEY123" not in repr(b)
    assert AlpacaPaperBroker("k", "s", PAPER_URL + "/v2", session=FakeAlpaca()).base_url == PAPER_URL


def test_factory_selects_alpaca_only_in_paper_mode():
    b = create_broker("PAPER", "alpaca_paper", api_key="k", api_secret="s", session=FakeAlpaca())
    assert isinstance(b, AlpacaPaperBroker)
    with pytest.raises(LiveTradingNotAllowedError):
        create_broker("BACKTEST", "alpaca_paper", api_key="k", api_secret="s", session=FakeAlpaca())
    with pytest.raises(LiveTradingNotAllowedError):
        create_broker("PAPER", "alpaca_live")


def test_opg_window():
    assert opg_window_open(datetime(2026, 6, 1, 19, 0, tzinfo=ET))
    assert opg_window_open(datetime(2026, 6, 2, 9, 27, tzinfo=ET))
    assert not opg_window_open(datetime(2026, 6, 2, 9, 28, tzinfo=ET))
    assert not opg_window_open(MIDDAY)
    manila_815 = datetime(2026, 11, 3, 8, 15, tzinfo=ZoneInfo("Asia/Manila"))   # after US DST ends
    assert opg_window_open(manila_815)
    manila_815_summer = datetime(2026, 7, 3, 8, 15, tzinfo=ZoneInfo("Asia/Manila"))
    assert opg_window_open(manila_815_summer)


# ------------------------------------------------------------------- orders
def test_submit_sends_idempotency_key_and_opg():
    b, fake = broker()
    o = approved_order(qty=7)
    bo = b.submit_order(o)
    sent = next(iter(fake.orders.values()))
    assert sent["client_order_id"] == o.idempotency_key and sent["time_in_force"] == "opg"
    assert sent["qty"] == "7" and sent["side"] == "buy" and sent["type"] == "market"
    assert bo.status == "ACKNOWLEDGED" and bo.client_order_id == o.idempotency_key


def test_outside_opg_window_refused_locally():
    b, fake = broker(clock=MIDDAY)
    with pytest.raises(OrderRejectedError, match="OPG"):
        b.submit_order(approved_order())
    assert ("POST", "/v2/orders") not in fake.calls
    b2, fake2 = broker(clock=MIDDAY, tif="day")                    # market open: refused locally too
    with pytest.raises(OrderRejectedError, match="market open"):
        b2.submit_order(approved_order())
    assert ("POST", "/v2/orders") not in fake2.calls
    b3, fake3 = broker(clock=EVENING, tif="day")                   # after the close: queued for the open
    assert b3.submit_order(approved_order()).status == "ACKNOWLEDGED"
    assert next(iter(fake3.orders.values()))["time_in_force"] == "day"


def test_day_orders_are_the_default_and_window_rules():
    from src.execution.alpaca_paper_broker import day_window_open
    b = AlpacaPaperBroker("k", "s", PAPER_URL, session=FakeAlpaca())
    assert b.time_in_force == "day"
    assert day_window_open(datetime(2026, 6, 1, 16, 5, tzinfo=ET))          # Mon after close
    assert day_window_open(datetime(2026, 6, 2, 9, 27, tzinfo=ET))          # Tue pre-open
    assert not day_window_open(datetime(2026, 6, 2, 9, 28, tzinfo=ET))      # about to open
    assert not day_window_open(MIDDAY)
    assert day_window_open(datetime(2026, 6, 6, 12, 0, tzinfo=ET))          # Saturday
    assert day_window_open(datetime(2026, 10, 20, 8, 15, tzinfo=ZoneInfo("Asia/Manila")))
    assert day_window_open(datetime(2026, 11, 3, 8, 15, tzinfo=ZoneInfo("Asia/Manila")))


def test_unapproved_disabled_and_unknown_symbol_never_sent():
    from src.execution.orders import Order, Side
    b, fake = broker()
    with pytest.raises(OrderRejectedError):
        b.submit_order(Order("AAPL", Side.BUY, 1, D0, reference_price=100.0, risk_decision_id="R-forged"))
    with pytest.raises(OrderRejectedError):
        b.submit_order(approved_order("TSLA"))
    b2, fake2 = broker(enabled=lambda: False)
    with pytest.raises(TradingDisabledError):
        b2.submit_order(approved_order())
    assert not [c for c in fake.calls + fake2.calls if c[0] == "POST"]


def test_rejections_and_duplicates_are_classified():
    b, fake = broker()
    o = approved_order()
    b.submit_order(o)
    with pytest.raises(DuplicateOrderError):
        b.submit_order(approved_order())                       # same intention -> same client_order_id
    fake.faults.append(("POST", "/v2/orders", "http_403"))
    with pytest.raises(OrderRejectedError):                    # e.g. insufficient buying power
        b.submit_order(approved_order("MSFT"))
    fake.faults.append(("GET", "/v2/account", "http_401"))
    with pytest.raises(APIError) as e:
        b.get_account()
    assert e.value.severity == Severity.CRITICAL                # bad credentials halt trading


def test_post_is_never_retried_but_get_is():
    b, fake = broker()
    fake.faults.append(("POST", "/v2/orders", "http_503"))
    with pytest.raises(APIError):
        b.submit_order(approved_order())
    assert fake.calls.count(("POST", "/v2/orders")) == 1
    fake.faults += [("GET", "/v2/account", "http_503"), ("GET", "/v2/account", "http_429")]
    assert b.get_account().cash == 100_000
    assert fake.calls.count(("GET", "/v2/account")) == 3


def test_malformed_responses():
    b, fake = broker()
    fake.faults.append(("GET", "/v2/positions", "malformed"))
    fake.faults.append(("GET", "/v2/positions", "malformed"))
    fake.faults.append(("GET", "/v2/positions", "malformed"))
    with pytest.raises(MalformedResponseError):
        b.get_positions()


# ---------------------------------------------- order manager + reconciliation
def manager(b):
    gate = TradingGate(HealthMonitor(), operator_enabled=True)
    return OrderManager(b, InMemoryLedger(), gate, CircuitBreaker("broker", 3)), gate


def test_timeout_after_accept_is_reconciled_not_resubmitted():
    b, fake = broker()
    om, gate = manager(b)
    fake.faults.append(("POST", "/v2/orders", "timeout_after"))
    o = validated()
    r = om.submit(o)
    assert o.status == OrderStatus.ACKNOWLEDGED and "NOT resubmitted" in r.note
    assert len(fake.orders) == 1 and fake.calls.count(("POST", "/v2/orders")) == 1


def test_timeout_before_accept_and_5xx_are_resolved_by_lookup():
    b, fake = broker()
    om, gate = manager(b)
    fake.faults.append(("POST", "/v2/orders", "timeout_before"))
    o = validated()
    om.submit(o)
    assert o.status == OrderStatus.CANCELLED and fake.orders == {}
    fake.faults.append(("POST", "/v2/orders", "http_502"))
    o2 = validated(symbol="MSFT")
    om.submit(o2)
    assert o2.status == OrderStatus.CANCELLED and fake.orders == {}


def test_status_mapping_and_reconciliation_against_alpaca():
    b, fake = broker()
    om, gate = manager(b)
    o = validated(qty=10)
    om.submit(o)
    oid = next(iter(fake.orders))
    fake.fill(oid, 4, 100.0)
    assert b.get_order_by_client_id(o.idempotency_key).status == "PARTIALLY_FILLED"
    fake.fill(oid, 6, 102.0)
    bo = b.get_order_by_client_id(o.idempotency_key)
    assert bo.status == "FILLED" and bo.filled_quantity == 10 and bo.avg_fill_price == pytest.approx(101.2)
    assert b.get_order_by_client_id("nope") is None
    rep = reconcile({"AAPL": 10}, fake.cash, om.ledger, b, gate)
    assert rep.ok, rep.discrepancies
    rep2 = reconcile({"AAPL": 9}, fake.cash, om.ledger, b, gate)
    assert not rep2.ok and not gate.trading_enabled


def test_cancel():
    b, fake = broker()
    bo = b.submit_order(approved_order())
    assert b.cancel_order(bo.broker_order_id).status == "CANCELLED"


# ------------------------------------------------------ full paper-trader runs
@pytest.mark.integration
@pytest.mark.slow
def test_paper_trader_with_alpaca_books_partial_fills_exactly_once(tmp_path):
    from src.bots.bot import build_bots
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model
    from tests.fakes import MockModel

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for bt in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=bt.bot_id,
                   model_version=bt.ml_cfg.model_version, feature_set_version=bt.feature_version,
                   features=bt.ml_features, target={}, periods={}, metrics={}, model_name="mock")
    fake = FakeAlpaca(cash=100_000)
    known = set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark}

    def trader():
        b = AlpacaPaperBroker("k", "s", PAPER_URL, session=fake, clock=lambda: EVENING, known_symbols=known,
                              sleep=lambda s: None)
        t = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots, broker=b)
        b._trading_enabled = lambda: t.gate.trading_enabled
        return t

    days = pd.bdate_range("2026-06-01", periods=4)
    r1 = trader().run(days[0])
    assert r1.trading_enabled and r1.submitted, r1.blocking_reasons
    assert len(fake.orders) == len(r1.submitted)
    fake.fill_all(price=100.0, fraction=0.5)            # opening auction: partial fills
    r2 = trader().run(days[1])
    assert r2.reconciliation["ok"], r2.reconciliation
    fake.fill_all(price=101.0)                          # remainder fills later
    r3 = trader().run(days[2])
    assert r3.reconciliation["ok"], r3.reconciliation
    r4 = trader().run(days[3])                          # nothing new: nothing booked twice
    assert r4.reconciliation["ok"] and r4.trading_enabled


@pytest.mark.integration
@pytest.mark.slow
def test_fills_booked_during_a_refused_rerun_are_not_lost(tmp_path):
    """Regression: a manual re-run on the same session books overnight fills,
    then refuses to decide twice. The booked fills must be persisted, or the
    next run's reconciliation would fail (books short of the broker)."""
    from src.bots.bot import build_bots
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model
    from tests.fakes import MockModel

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for bt in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=bt.bot_id,
                   model_version=bt.ml_cfg.model_version, feature_set_version=bt.feature_version,
                   features=bt.ml_features, target={}, periods={}, metrics={}, model_name="mock")
    fake = FakeAlpaca(cash=100_000)
    known = set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark}

    def trader():
        b = AlpacaPaperBroker("k", "s", PAPER_URL, session=fake, clock=lambda: EVENING, known_symbols=known,
                              sleep=lambda s: None)
        t = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots, broker=b)
        b._trading_enabled = lambda: t.gate.trading_enabled
        return t

    days = pd.bdate_range("2026-06-01", periods=2)
    r1 = trader().run(days[0])
    assert r1.submitted
    fake.fill_all(price=100.0)                          # orders fill at the next open
    r_again = trader().run(days[0])                     # user re-runs: refused, but fills get booked
    assert r_again.submitted == []
    r2 = trader().run(days[1])
    assert r2.reconciliation["ok"], r2.reconciliation
    assert r2.trading_enabled, r2.blocking_reasons


@pytest.mark.integration
@pytest.mark.slow
def test_run_outside_opg_window_decides_nothing_and_does_not_use_up_the_session(tmp_path):
    """Regression: a run at 14:49 ET decided, had every order refused (OPG window)
    and recorded the session as done, so the proper evening run was refused."""
    from src.bots.bot import build_bots
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model
    from tests.fakes import MockModel

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for bt in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=bt.bot_id,
                   model_version=bt.ml_cfg.model_version, feature_set_version=bt.feature_version,
                   features=bt.ml_features, target={}, periods={}, metrics={}, model_name="mock")
    fake = FakeAlpaca(cash=100_000)
    known = set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark}

    def trader(clock):
        b = AlpacaPaperBroker("k", "s", PAPER_URL, session=fake, clock=lambda: clock, known_symbols=known,
                              sleep=lambda s: None)
        t = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots, broker=b)
        b._trading_enabled = lambda: t.gate.trading_enabled
        return t

    day = pd.Timestamp("2026-06-01")
    early = trader(MIDDAY).run(day)
    assert early.submitted == [] and early.rejected == []
    assert any("Nothing was decided" in r for r in early.blocking_reasons)
    assert not [c for c in fake.calls if c[0] == "POST"]
    later = trader(EVENING).run(day)                    # same session, inside the window: decides normally
    assert later.submitted, later.blocking_reasons
