"""External API failure classification and a SAFE retry helper.

    RETRYABLE         429, 500, 502, 503, 504, timeouts, connection errors
    NON_RETRYABLE     400, 404, 422, malformed JSON, empty response
    TRADING_CRITICAL  401, 403 (credentials/permissions) -> halt trading

Order SUBMISSION is never retried by this helper (``idempotent=False``): a
timeout on a POST means the order state is UNKNOWN and must be reconciled
with the broker before anything else happens (see order_manager.py).
"""
from __future__ import annotations

import json
import random
import time
from enum import Enum
from typing import Any, Callable

from src.errors import BrokerError, BrokerTimeoutError, Severity
from src.logging_setup import get_logger

log = get_logger(__name__)


class FailureClass(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    TRADING_CRITICAL = "TRADING_CRITICAL"


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
CRITICAL_STATUS = {401, 403}


class APIError(BrokerError):
    def __init__(self, message: str, status_code: int | None = None, failure_class: FailureClass | None = None,
                 **kw):
        super().__init__(message, **kw)
        self.status_code = status_code
        self.failure_class = failure_class or classify_status(status_code)
        self.severity = Severity.CRITICAL if self.failure_class == FailureClass.TRADING_CRITICAL else Severity.ERROR


class MalformedResponseError(APIError):
    def __init__(self, message: str):
        super().__init__(message, None, FailureClass.NON_RETRYABLE)


def classify_status(status: int | None) -> FailureClass:
    if status is None:
        return FailureClass.RETRYABLE
    if status in CRITICAL_STATUS:
        return FailureClass.TRADING_CRITICAL
    if status in RETRYABLE_STATUS:
        return FailureClass.RETRYABLE
    return FailureClass.NON_RETRYABLE


def classify_exception(exc: BaseException) -> FailureClass:
    if isinstance(exc, APIError):
        return exc.failure_class
    if isinstance(exc, (TimeoutError, ConnectionError, BrokerTimeoutError)):
        return FailureClass.RETRYABLE
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return FailureClass.RETRYABLE
    return FailureClass.NON_RETRYABLE


def parse_json_response(status: int, text: str | None) -> Any:
    """Validate an HTTP response: raise a classified APIError for bad ones."""
    if status >= 400:
        detail = ""
        if text:
            try:
                body = json.loads(text)
                detail = str(body.get("message") or body.get("error") or body) if isinstance(body, dict) else text
            except json.JSONDecodeError:
                detail = text
        raise APIError(f"HTTP {status}" + (f": {detail[:300]}" if detail else ""), status)
    if text is None or not text.strip():
        raise MalformedResponseError("empty response body")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"malformed JSON: {exc}") from exc


def retry_call(fn: Callable[[], Any], *, idempotent: bool, max_attempts: int = 3, base_delay: float = 0.5,
               max_delay: float = 8.0, jitter: float = 0.25, sleep: Callable[[float], None] = time.sleep,
               what: str = "api call") -> Any:
    """Exponential backoff with jitter for RETRYABLE failures of IDEMPOTENT
    operations only. Never retries indefinitely."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            cls = classify_exception(exc)
            if not idempotent or cls != FailureClass.RETRYABLE or attempt >= max_attempts:
                if not idempotent and cls == FailureClass.RETRYABLE:
                    log.error("%s failed (%s) and is NOT idempotent: not retried", what, exc)
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 1 + random.uniform(-jitter, jitter)
            log.warning("%s failed (%s, %s), retry %d/%d in %.2fs", what, type(exc).__name__, cls.value,
                        attempt, max_attempts - 1, delay)
            sleep(max(0.0, delay))
