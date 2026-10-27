"""Database layer on a temporary SQLite file. Requires SQLAlchemy."""
import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from src.data.database import Repository, SqlOrderLedger  # noqa: E402
from src.errors import DatabaseError  # noqa: E402
from src.execution.orders import OrderStatus  # noqa: E402
from tests.helpers import approved_order  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    r = Repository(f"sqlite:///{tmp_path / 't.db'}")
    r.init()
    return r


def test_schema_and_order_round_trip(repo):
    led = SqlOrderLedger(repo)
    o = approved_order(bot_id="SWING_V1")
    led.record_intent(o)
    o.transition(OrderStatus.VALIDATED)
    led.update(o)
    df = repo.read_df("orders")
    assert len(df) == 1 and df.iloc[0]["status"] == "VALIDATED" and df.iloc[0]["bot_id"] == "SWING_V1"
    fresh = SqlOrderLedger(repo)
    assert fresh.get_by_key(o.idempotency_key).status == OrderStatus.VALIDATED
    tables = set(sqlalchemy.inspect(repo.engine).get_table_names())
    assert {"market_prices", "features", "signals", "orders", "executions", "positions", "portfolio_snapshots",
            "model_predictions", "backtest_runs", "risk_events"} <= tables


def test_audit_and_events(repo):
    rec = {"signal_id": "B-AAPL-20240102", "timestamp": pd.Timestamp("2024-01-02"), "bot_id": "B", "symbol": "AAPL",
           "strategy": "B", "score": float("nan"), "final_action": "NO_TRADE", "risk_status": "BLOCKED",
           "rule_triggered": "kill_switch", "reason": "KILL SWITCH active", "components": {"a": float("nan")}}
    assert repo.save_audit([rec]) == 1
    row = repo.read_df("signals").iloc[0]
    assert row["rule_triggered"] == "kill_switch" and pd.isna(row["score"])
    repo.save_events([{"date": "2024-01-02", "event": "KILL_SWITCH_ON", "severity": "CRITICAL"}], "RUN1")
    assert repo.read_df("risk_events").iloc[0]["severity"] == "CRITICAL"


def test_constraint_violation_rolls_back(repo):
    from src.data.database import OrderRecord
    o = approved_order()
    repo.upsert_order(o)
    with pytest.raises(DatabaseError):
        with repo.session() as s:
            s.add(OrderRecord(order_id="other", idempotency_key=o.idempotency_key, bot_id="X", symbol="AAPL",
                              side="BUY", quantity=1, order_type="MARKET", status="PROPOSED",
                              created_at=pd.Timestamp("2024-01-02").to_pydatetime()))
    assert len(repo.read_df("orders")) == 1


def test_connection_and_read_failures(tmp_path):
    bad = Repository(f"sqlite:///{tmp_path / 'missing_dir' / 'x.db'}")
    with pytest.raises(DatabaseError):
        bad.init()
    assert not bad.ping()
    ok = Repository(f"sqlite:///{tmp_path / 'y.db'}")
    ok.init()
    with pytest.raises(DatabaseError):
        ok.read_df("no_such_table")


def test_write_failure_in_order_manager_halts(tmp_path):
    from src.execution.order_manager import OrderManager
    from src.execution.paper_broker import PaperBroker
    from src.safety.health import HealthMonitor, TradingGate
    r = Repository(f"sqlite:///{tmp_path / 'z.db'}")
    r.init()
    gate = TradingGate(HealthMonitor(), operator_enabled=True)
    om = OrderManager(PaperBroker(100_000), SqlOrderLedger(r), gate)
    r.engine.dispose()
    (tmp_path / "z.db").unlink()
    (tmp_path / "z.db").mkdir()                     # path is now a directory: every write fails
    o = approved_order()
    o.transition(OrderStatus.VALIDATED)
    with pytest.raises(DatabaseError):
        om.submit(o)
    assert om.broker.get_orders() == [] and not gate.trading_enabled


def test_bot_activity_everything_a_bot_did_with_a_symbol(repo):
    from src.execution.orders import Side
    led = SqlOrderLedger(repo)
    for bot, sym in (("SWING_V1", "NVDA"), ("SWING_V1", "AAPL"), ("TREND_V1", "NVDA")):
        o = approved_order(sym, bot_id=bot, strategy=bot)
        led.record_intent(o)
    recs = [{"signal_id": f"{b}-{s}", "timestamp": pd.Timestamp("2024-01-02"), "data_timestamp": pd.Timestamp(
        "2024-01-02"), "bot_id": b, "symbol": s, "strategy": b, "timeframe": "daily", "action": "BUY",
        "final_action": "BUY"} for b, s in (("SWING_V1", "NVDA"), ("TREND_V1", "NVDA"), ("SWING_V1", "MSFT"))]
    repo.save_audit(recs)
    repo.save_events([{"date": pd.Timestamp("2024-01-02"), "bot_id": "SWING_V1", "symbol": "NVDA",
                       "event": "ORDER_RESIZED", "severity": "INFO"}])
    out = repo.bot_activity("SWING_V1", "NVDA")
    assert len(out["signals"]) == 1 and out["signals"].iloc[0]["timeframe"] == "daily"
    assert out["signals"].iloc[0]["data_timestamp"] is not None
    assert len(out["orders"]) == 1 and out["orders"].iloc[0]["side"] == Side.BUY.value
    assert len(out["risk_events"]) == 1
    assert set(repo.bot_activity("TREND_V1")["signals"]["symbol"]) == {"NVDA"}
    # parameterised: an injection attempt is just a bot id that does not exist
    assert repo.bot_activity("x' OR '1'='1")["signals"].empty


def test_additive_migration_adds_new_columns(tmp_path):
    from sqlalchemy import create_engine, inspect
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = create_engine(url)
    with eng.begin() as c:     # a 'signals' table as created by the previous version (no timeframe column)
        c.exec_driver_sql("CREATE TABLE signals (id INTEGER PRIMARY KEY, signal_id VARCHAR(64), run_id VARCHAR(64), "
                          "environment VARCHAR(16), timestamp DATETIME, bot_id VARCHAR(32), symbol VARCHAR(16), "
                          "strategy VARCHAR(64), final_action VARCHAR(16), quantity INTEGER)")
    r = Repository(url)
    r.init()
    cols = {c["name"] for c in inspect(r.engine).get_columns("signals")}
    assert {"timeframe", "data_timestamp", "model_version"} <= cols
    assert "backtest_bot_results" in inspect(r.engine).get_table_names()
    assert r.migrate() == []                                     # idempotent


def test_paper_state_is_queryable_by_bot(repo):
    from tests.helpers import portfolio_with
    sleeves = {"SWING_V1": portfolio_with(20_000, {"NVDA": (10, 100.0)}),
               "INVESTOR_V1": portfolio_with(25_000, {"NVDA": (5, 90.0), "AAPL": (3, 50.0)})}
    fills = [{"order_id": "PB-1", "bot_id": "SWING_V1", "symbol": "NVDA", "side": "BUY", "quantity": 10,
              "price": 100.0, "reference_price": 99.5, "commission": 0.0, "timestamp": pd.Timestamp("2024-01-03")}]
    repo.save_paper_state("PAPER-1", pd.Timestamp("2024-01-03"), sleeves, fills, unallocated_cash=1000.0)
    repo.save_paper_state("PAPER-2", pd.Timestamp("2024-01-04"), sleeves, [], unallocated_cash=1000.0)
    pos = repo.read_df("positions")
    assert len(pos) == 3                                        # current positions replaced, not duplicated
    swing = repo.bot_activity("SWING_V1", "NVDA")
    assert len(swing["executions"]) == 1 and len(swing["positions"]) == 1
    assert swing["positions"].iloc[0]["quantity"] == 10
    inv = repo.bot_activity("INVESTOR_V1", "NVDA")
    assert inv["positions"].iloc[0]["quantity"] == 5 and inv["executions"].empty
    acct = repo.read_df("portfolio_snapshots", "bot_id = :b", {"b": "ACCOUNT"})
    assert len(acct) == 2 and acct.iloc[0]["equity"] == pytest.approx(1000 + 21_000 + 25_000 + 450 + 150)
