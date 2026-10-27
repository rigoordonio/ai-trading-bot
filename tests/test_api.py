"""API smoke tests (requires fastapi + httpx)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)


def test_health_reports_real_money_disabled():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["real_money"] == "DISABLED"


def test_unknown_backtest_404():
    assert client.get("/backtests/NOPE").status_code == 404


def test_explain_returns_text_only():
    r = client.post("/explain", json={"symbol": "AAPL", "bot_id": "SWING_V1", "final_action": "BUY", "score": 0.7})
    assert r.status_code == 200 and set(r.json()) == {"explanation"}


def test_no_order_routes():
    paths = {getattr(r, "path", "") for r in app.routes}
    assert not any("order" in p and p != "/paper/orders" for p in paths)
    methods = {m for r in app.routes for m in getattr(r, "methods", set())}
    assert "DELETE" not in methods and "PUT" not in methods
