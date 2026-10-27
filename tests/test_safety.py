"""Circuit breaker, API failure classification, retries, trading gate, severities."""
import pytest

from src.errors import (
    BrokerTimeoutError,
    CircuitOpenError,
    DataValidationError,
    ModelInferenceError,
    OrderValidationError,
    Severity,
    TradingSystemError,
)
from src.safety.api_errors import (
    APIError,
    FailureClass,
    MalformedResponseError,
    classify_exception,
    classify_status,
    parse_json_response,
    retry_call,
)
from src.safety.circuit_breaker import BreakerState, CircuitBreaker
from src.safety.health import HealthMonitor, HealthStatus, TradingGate


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ----------------------------------------------------------- circuit breaker
def test_circuit_breaker_full_cycle():
    clk = Clock()
    cb = CircuitBreaker("broker", failure_threshold=3, recovery_timeout=60, half_open_max_calls=1, clock=clk)
    for _ in range(2):
        cb.record_failure("boom")
    assert cb.state == BreakerState.CLOSED
    cb.record_failure("boom")
    assert cb.state == BreakerState.OPEN and not cb.allow()
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)
    clk.t = 61
    assert cb.state == BreakerState.HALF_OPEN
    assert cb.allow() and not cb.allow()          # exactly one probe
    cb.record_failure("probe failed")
    assert cb.state == BreakerState.OPEN
    clk.t = 200
    assert cb.call(lambda: "ok") == "ok"          # probe succeeds
    assert cb.state == BreakerState.CLOSED
    names = [e["event"] for e in cb.events]
    assert names == ["CIRCUIT_OPEN", "CIRCUIT_HALF_OPEN", "CIRCUIT_OPEN", "CIRCUIT_HALF_OPEN", "CIRCUIT_CLOSED"]


def test_success_resets_failure_count():
    cb = CircuitBreaker("x", 2)
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.state == BreakerState.CLOSED


def test_circuit_breaker_blocks_trading_gate():
    gate = TradingGate(HealthMonitor(), operator_enabled=True)
    cb = CircuitBreaker("market_data", 1)
    gate.breakers.append(cb)
    assert gate.trading_enabled
    cb.record_failure("vendor 503")
    assert not gate.trading_enabled
    assert any("circuit breaker OPEN" in r for r in gate.blocking_reasons())


# --------------------------------------------------------- API classification
@pytest.mark.parametrize("status,cls", [
    (400, FailureClass.NON_RETRYABLE), (401, FailureClass.TRADING_CRITICAL), (403, FailureClass.TRADING_CRITICAL),
    (404, FailureClass.NON_RETRYABLE), (429, FailureClass.RETRYABLE), (500, FailureClass.RETRYABLE),
    (502, FailureClass.RETRYABLE), (503, FailureClass.RETRYABLE)])
def test_http_status_classification(status, cls):
    assert classify_status(status) == cls
    with pytest.raises(APIError) as e:
        parse_json_response(status, '{"x": 1}')
    assert e.value.failure_class == cls
    assert (e.value.severity == Severity.CRITICAL) == (cls == FailureClass.TRADING_CRITICAL)


def test_timeout_connection_malformed_and_empty():
    assert classify_exception(TimeoutError()) == FailureClass.RETRYABLE
    assert classify_exception(ConnectionError()) == FailureClass.RETRYABLE
    assert classify_exception(BrokerTimeoutError("t")) == FailureClass.RETRYABLE
    with pytest.raises(MalformedResponseError):
        parse_json_response(200, "{not json")
    with pytest.raises(MalformedResponseError):
        parse_json_response(200, "")
    with pytest.raises(MalformedResponseError):
        parse_json_response(200, None)
    assert parse_json_response(200, '{"ok": true}') == {"ok": True}


def test_retry_exponential_backoff_with_limit():
    sleeps, calls = [], []

    def flaky():
        calls.append(1)
        raise APIError("HTTP 503", 503)

    with pytest.raises(APIError):
        retry_call(flaky, idempotent=True, max_attempts=3, base_delay=1.0, jitter=0.0, sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [1.0, 2.0]          # stops after 3 attempts


def test_retry_succeeds_and_does_not_retry_non_retryable():
    state = {"n": 0}

    def ok_on_second():
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError()
        return "done"

    assert retry_call(ok_on_second, idempotent=True, sleep=lambda s: None) == "done"
    calls = []
    with pytest.raises(APIError):
        retry_call(lambda: calls.append(1) or (_ for _ in ()).throw(APIError("bad", 400)), idempotent=True,
                   sleep=lambda s: None)
    assert len(calls) == 1


def test_order_post_never_blindly_retried():
    calls = []

    def post():
        calls.append(1)
        raise TimeoutError("POST /orders timed out")

    with pytest.raises(TimeoutError):
        retry_call(post, idempotent=False, max_attempts=5, sleep=lambda s: None)
    assert len(calls) == 1


def test_jitter_bounds():
    sleeps = []
    with pytest.raises(APIError):
        retry_call(lambda: (_ for _ in ()).throw(APIError("x", 500)), idempotent=True, max_attempts=4,
                   base_delay=1.0, jitter=0.25, sleep=sleeps.append)
    for got, base in zip(sleeps, [1, 2, 4], strict=True):
        assert 0.75 * base <= got <= 1.25 * base


# ------------------------------------------------------------- trading gate
def test_operator_switch_env(monkeypatch):
    monkeypatch.setenv("TRADING_ENABLED", "false")
    g = TradingGate()
    assert not g.trading_enabled and "TRADING_ENABLED=false" in g.blocking_reasons()[0]
    monkeypatch.setenv("TRADING_ENABLED", "true")
    assert TradingGate().trading_enabled


def test_critical_error_halts_until_validated_recovery():
    g = TradingGate(HealthMonitor(), operator_enabled=True)
    g.handle_error(ModelInferenceError("model file corrupted"))
    assert not g.trading_enabled
    assert g.health.components["model"].status == HealthStatus.UNHEALTHY
    assert g.recover(reconciliation_ok=True) is False            # model still unhealthy
    g.health.set("model", HealthStatus.HEALTHY, "reloaded")
    assert g.recover(reconciliation_ok=False) is False           # reconciliation must pass
    assert g.recover(reconciliation_ok=True) is True and g.trading_enabled


def test_non_critical_errors_do_not_halt_and_unknown_errors_do():
    g = TradingGate(HealthMonitor(), operator_enabled=True)
    g.handle_error(OrderValidationError("one bad order"))
    g.handle_error(TradingSystemError("dashboard glitch", severity=Severity.WARNING))
    assert g.trading_enabled
    g.handle_error(ZeroDivisionError("bug in a new feature"))     # unknown -> CRITICAL
    assert not g.trading_enabled


def test_severity_model():
    assert DataValidationError("current bar invalid").severity == Severity.CRITICAL
    assert OrderValidationError("x").severity == Severity.ERROR
    from src.errors import PortfolioStateError, ReconciliationError, RiskEngineError
    for exc in (PortfolioStateError, ReconciliationError, RiskEngineError):
        assert exc("x").severity == Severity.CRITICAL
