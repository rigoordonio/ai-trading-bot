"""Paper trading runs from Manila decide on the last COMPLETED New York session."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.execution.paper_trader import last_completed_session

MNL = ZoneInfo("Asia/Manila")
NY = ZoneInfo("America/New_York")


@pytest.mark.parametrize("local,expected", [
    (datetime(2026, 10, 20, 8, 15, tzinfo=MNL), "2026-10-19"),   # Tue 08:15 Manila -> Mon session
    (datetime(2026, 10, 24, 8, 15, tzinfo=MNL), "2026-10-23"),   # Sat 08:15 Manila -> Fri session
    (datetime(2026, 10, 19, 8, 15, tzinfo=MNL), "2026-10-16"),   # Mon 08:15 Manila = Sun NY -> Fri
    (datetime(2026, 11, 3, 8, 15, tzinfo=MNL), "2026-11-02"),    # after US DST change
    (datetime(2026, 10, 19, 15, 0, tzinfo=NY), "2026-10-16"),    # Mon 15:00 NY: Monday still trading
    (datetime(2026, 10, 19, 16, 45, tzinfo=NY), "2026-10-19"),   # Mon 16:45 NY: Monday complete
])
def test_last_completed_session(local, expected):
    assert last_completed_session(local) == pd.Timestamp(expected)


@pytest.mark.integration
@pytest.mark.slow
def test_live_run_from_manila_trades_on_last_us_session(tmp_path):
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
    for b in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=b.bot_id, model_version=b.ml_cfg.model_version,
                   feature_set_version=b.feature_version, features=b.ml_features, target={}, periods={}, metrics={},
                   model_name="mock")
    pt = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots)
    r = pt.run(now=datetime(2026, 6, 2, 8, 15, tzinfo=MNL))      # Tue 08:15 Manila
    assert r.as_of == "2026-06-01"                                # decided on Monday's NY session
    assert r.trading_enabled and r.submitted, r.blocking_reasons
    pt2 = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots)
    r2 = pt2.run(now=datetime(2026, 6, 2, 12, 0, tzinfo=MNL))    # same morning again -> refused, nothing sent
    assert r2.submitted == []


@pytest.mark.integration
@pytest.mark.slow
def test_latest_session_missing_from_provider_is_reported_and_retryable(tmp_path):
    """Regression (2026-10-20): Yahoo returned a blank row for Monday at 08:15
    Manila. The run must say 'data not available yet' (not 'already ran' as an
    error), decide nothing, and a later run with complete data must decide."""
    from src.bots.bot import build_bots
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model
    from tests.fakes import MockModel

    class Lagging(SyntheticProvider):
        cutoff = None

        def get_history(self, symbol, start=None, end=None, **kw):
            df = super().get_history(symbol, start, end, **kw)
            return df.loc[:self.cutoff] if self.cutoff is not None else df

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for b in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=b.bot_id, model_version=b.ml_cfg.model_version,
                   feature_set_version=b.feature_version, features=b.ml_features, target={}, periods={}, metrics={},
                   model_name="mock")
    PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots).run(pd.Timestamp("2026-06-01"))
    lag = Lagging()
    lag.cutoff = pd.Timestamp("2026-06-01")                     # Tuesday's run, but Monday... still only 06-01
    r = PaperTrader(cfg, tmp_path / "paper", lag, bots=bots).run(pd.Timestamp("2026-06-02"))
    assert r.trading_enabled and r.submitted == []
    assert any("not available yet" in x for x in r.blocking_reasons)
    assert not any(e.get("severity") == "ERROR" and "already ran" in str(e.get("detail")) for e in r.events)
    later = PaperTrader(cfg, tmp_path / "paper", SyntheticProvider(), bots=bots).run(pd.Timestamp("2026-06-02"))
    assert later.as_of == "2026-06-02" and not any("not available" in x for x in later.blocking_reasons)


def _paper_trade_module():
    import importlib.util
    import pathlib
    import sys
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location("paper_trade", pathlib.Path("scripts/paper_trade.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retry_rules():
    from src.execution.paper_trader import PaperRunReport
    rt = _paper_trade_module().retryable

    def rep(enabled=True, reasons=(), submitted=(), rejected=()):
        return PaperRunReport("R", "2026-06-01", enabled, list(reasons), list(submitted), list(rejected))
    assert rt(rep(reasons=["prices for the 2026-06-02 session are not available yet ..."]))
    assert rt(rep(reasons=["market data unavailable (ConnectionError: ...). Nothing was decided"]))
    assert rt(rep(False, ["HALT[broker]: ReconciliationError: reconciliation impossible: broker API unreachable",
                          "component UNHEALTHY: broker"]))
    assert not rt(rep())                                                     # normal run
    assert not rt(rep(reasons=["not available yet"], submitted=[{"x": 1}]))  # never after sending orders
    assert not rt(rep(False, ["HALT[execution_engine]: ReconciliationError: reconciliation failed: [...]"]))
    assert not rt(rep(reasons=["outside Alpaca's window ... Nothing was decided"]))   # waiting for 19:00 is not retry


@pytest.mark.integration
def test_internet_outage_is_explained_and_retried(tmp_path, monkeypatch):
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader

    class Down(SyntheticProvider):
        def get_history(self, *a, **k):
            raise ConnectionError("Max retries exceeded: nodename nor servname provided")

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "p"),
                                                       "retry_backoff_seconds": 0}}})
    r = PaperTrader(cfg, tmp_path / "paper", Down()).run(pd.Timestamp("2026-06-01"))
    assert r.submitted == [] and any("market data unavailable" in x for x in r.blocking_reasons)
    assert _paper_trade_module().retryable(r)


def test_scheduled_run_waits_and_retries_until_data_is_ready(tmp_path, monkeypatch):
    from src.execution.paper_trader import PaperRunReport
    mod = _paper_trade_module()
    script = [PaperRunReport("R1", "2026-06-01", True, ["prices ... not available yet"]),
              PaperRunReport("R2", "2026-06-01", True, ["market data unavailable (ConnectionError)"]),
              PaperRunReport("R3", "2026-06-02", True, [], [{"order": 1}])]
    made, slept = [], []

    class FakeTrader:
        def __init__(self, *a, **k):
            made.append(self)
            self.state_dir = tmp_path

        def run(self, as_of=None):
            return script.pop(0)

    monkeypatch.setattr(mod, "PaperTrader", FakeTrader)
    monkeypatch.setattr(mod, "get_provider", lambda name: None)
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr("sys.argv", ["paper_trade.py", "--retry-minutes", "1"])
    assert mod.main() == 0
    assert script == [] and slept == [60, 60] and len(made) == 3          # fresh trader each attempt
