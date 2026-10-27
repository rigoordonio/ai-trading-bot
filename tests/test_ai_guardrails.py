"""The LLM layer can explain, never act. The API cannot create orders."""
import ast
import pathlib

from src.ai.explanation import DecisionFacts, explain, template_explanation
from src.ai.sentiment import categorise, score_headline

FORBIDDEN = ("src.execution", "src.risk", "src.portfolio", "src.trading_cycle", "src.safety")


def imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return out


def test_ai_package_cannot_reach_execution_or_risk():
    for p in pathlib.Path("src/ai").glob("*.py"):
        bad = [m for m in imports(p) if m.startswith(FORBIDDEN)]
        assert not bad, f"{p} imports {bad}"


def test_explanation_is_text_only(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    f = DecisionFacts("MSFT", "TREND_V1", "BUY", 0.72, 0.68, {"long_trend": 1.0, "relative_momentum": 0.8},
                      "PASS", "TREND_V1 BUY")
    out = explain(f)
    assert isinstance(out, str) and "MSFT" in out and "68%" in out and "PASS" in out
    assert not hasattr(out, "submit")


def test_llm_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")      # nothing listens here
    f = DecisionFacts("AAPL", "SWING_V1", "NO_TRADE", 0.3, None, {}, "BLOCKED", "risk blocked", "kill_switch")
    assert explain(f) == template_explanation(f)


def test_from_audit_tolerates_csv_round_trip_values():
    f = DecisionFacts.from_audit({"symbol": "X", "bot_id": "B", "final_action": "NO_TRADE", "score": float("nan"),
                                  "components": float("nan")})
    assert f.score is None and dict(f.components) == {}


def test_api_has_no_order_endpoints():
    tree = ast.parse(pathlib.Path("src/api/main.py").read_text())
    routes = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                    routes.append((d.func.attr, d.args[0].value if d.args else ""))
    posts = [r for r in routes if r[0] in ("post", "put", "patch", "delete")]
    assert posts == [("post", "/explain")]
    assert not any("order" in path.lower() for _, path in posts)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
           {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not used & {"submit_order", "OrderManager", "PaperBroker", "cancel_order"}
    assert not [m for m in imports(pathlib.Path("src/api/main.py")) if m.startswith(("src.execution.order",
                                                                                      "src.execution.paper"))]


def test_sentiment_is_bounded_and_deterministic():
    assert score_headline("Company beats estimates, record growth") > 0
    assert score_headline("Regulator opens probe; guidance cut") < 0
    assert categorise("Shares flat ahead of meeting") == "neutral"
    assert -1 <= score_headline("beat miss beat") <= 1
