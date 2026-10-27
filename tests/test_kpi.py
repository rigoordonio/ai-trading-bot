"""Local KPI page: data, rendering and the read-only localhost server."""
import json
import threading
import urllib.request

from src.dashboard.kpi_data import build_kpis
from src.dashboard.kpi_page import render_html
from tests.test_reviewer import NOW, steady, write_state


def _state(tmp_path):
    write_state(tmp_path, {"SWING_V1": steady(30, seed=1), "TREND_V1": steady(30, 40_000, seed=2),
                           "INVESTOR_V1": steady(30, seed=3)}, trades={"TREND_V1": [120, -40, 60]})
    from src.reviewer.review import Reviewer, ReviewerConfig
    Reviewer(tmp_path, ReviewerConfig(alerts={"notify_macos": False}), NOW).run()


def test_kpis_from_paper_state(tmp_path):
    _state(tmp_path)
    k = build_kpis(tmp_path, {"SWING_V1": 0.3, "TREND_V1": 0.4, "INVESTOR_V1": 0.3})
    assert [b["bot_id"] for b in k["bots"]] == ["SWING_V1", "TREND_V1", "INVESTOR_V1"]
    trend = k["bots"][1]
    assert trend["trades"] == 3 and abs(trend["win_rate"] - 2 / 3) < 1e-9 and trend["allocation"] == 0.4
    assert trend["grade"] != "—" and trend["status"] == "ACTIVE"
    assert len(k["history"]["dates"]) == 30 and k["history"]["series"]["SWING_V1"][0] == 100.0
    assert "SPY" in k["history"]["series"] and k["account"]["equity"] > 0


def test_page_renders_and_handles_empty_state(tmp_path):
    _state(tmp_path)
    page = render_html(build_kpis(tmp_path))
    for text in ("Trading bots KPI", "Swing", "Trend", "Investor", "Growth of 100", "Read-only page"):
        assert text in page
    empty = render_html(build_kpis(tmp_path / "nothing-here"))
    assert "no runs yet" in empty and "No positions yet" in empty


def test_server_is_local_and_read_only(tmp_path):
    import importlib.util
    import pathlib
    _state(tmp_path)
    spec = importlib.util.spec_from_file_location("kpi_server", pathlib.Path("scripts/kpi_server.py"))
    import sys
    sys.path.insert(0, "scripts")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(tmp_path, {}))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert b"Trading bots KPI" in urllib.request.urlopen(base + "/").read()
        k = json.loads(urllib.request.urlopen(base + "/api/kpis").read())
        assert k["bots"]
        req = urllib.request.Request(base + "/", data=b"x", method="POST")
        try:
            urllib.request.urlopen(req)
            raise AssertionError("POST must not be accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 501
    finally:
        srv.shutdown()


def _server_module():
    import importlib.util
    import pathlib
    import sys
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location("kpi_server", pathlib.Path("scripts/kpi_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_alpaca_section_read_only(tmp_path):
    from src.execution.alpaca_paper_broker import PAPER_URL, AlpacaPaperBroker
    from tests.fake_alpaca import FakeAlpaca
    _state(tmp_path)
    (tmp_path / "orders.json").write_text(json.dumps([{"order_id": "O1", "bot_id": "TREND_V1", "symbol": "MSFT",
                                                       "side": "BUY", "quantity": 18, "status": "ACKNOWLEDGED"}]))
    fake = FakeAlpaca(cash=90_000)
    fake.positions["MSFT"] = {"qty": 18, "price": 529.34}
    mod = _server_module()
    feed = mod.LiveFeed(ttl=0, broker=AlpacaPaperBroker("k", "s", PAPER_URL, session=fake, sleep=lambda s: None))
    snap, err, at = feed.get()
    assert err is None and snap["positions"][0]["symbol"] == "MSFT"
    from src.dashboard.kpi_data import live_view
    k = build_kpis(tmp_path)
    lv = live_view(snap, k, err, at)
    assert "TREND_V1" in lv["positions"][0]["owner"] and "booked at next run" in lv["positions"][0]["owner"]
    page = render_html(k, live=lv)
    assert "Live from Alpaca" in page and "Account value now" in page and "MSFT" in page
    assert not [c for c in fake.calls if c[0] != "GET"]                     # read-only: GET requests only


def test_live_section_degrades_gracefully(tmp_path, monkeypatch):
    _state(tmp_path)
    monkeypatch.setenv("BROKER", "simulated")
    mod = _server_module()
    snap, err, at = mod.LiveFeed(ttl=0).get()
    assert snap is None and "alpaca_paper" in err
    from src.dashboard.kpi_data import live_view
    page = render_html(build_kpis(tmp_path), live=live_view(snap, build_kpis(tmp_path), err, at))
    assert "Live data unavailable" in page and "Swing" in page
