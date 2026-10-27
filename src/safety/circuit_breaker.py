"""Circuit breaker for external dependencies (broker, data vendor).

    CLOSED     normal; consecutive failures counted
       |  failures >= failure_threshold
       v
    OPEN       every call refused (CircuitOpenError); trading gate closed
       |  recovery_timeout elapsed
       v
    HALF_OPEN  up to half_open_max_calls probe calls allowed
       |  probe succeeds -> CLOSED      probe fails -> OPEN (timer restarts)

Every transition is recorded as an event; nothing happens silently.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable

from src.errors import CircuitOpenError
from src.logging_setup import get_logger

log = get_logger(__name__)


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 300.0,
                 half_open_max_calls: int = 1, clock: Callable[[], float] = time.monotonic):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.clock = clock
        self._state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at: float | None = None
        self.half_open_calls = 0
        self.events: list[dict] = []

    @classmethod
    def from_config(cls, name: str, c, clock=time.monotonic) -> "CircuitBreaker":
        return cls(name, c.failure_threshold, c.recovery_timeout_seconds, c.half_open_max_calls, clock)

    @property
    def state(self) -> BreakerState:
        if self._state == BreakerState.OPEN and self.opened_at is not None \
                and self.clock() - self.opened_at >= self.recovery_timeout:
            self._move(BreakerState.HALF_OPEN, "recovery timeout elapsed")
            self.half_open_calls = 0
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def _move(self, new: BreakerState, why: str) -> None:
        if new == self._state:
            return
        evt = {"event": f"CIRCUIT_{new.value}", "severity": "CRITICAL" if new == BreakerState.OPEN else "INFO",
               "component": self.name, "detail": why}
        self.events.append(evt)
        (log.critical if new == BreakerState.OPEN else log.warning)("circuit %s: %s -> %s (%s)",
                                                                    self.name, self._state.value, new.value, why)
        self._state = new

    def allow(self) -> bool:
        st = self.state
        if st == BreakerState.CLOSED:
            return True
        if st == BreakerState.HALF_OPEN and self.half_open_calls < self.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        if self._state == BreakerState.HALF_OPEN:
            self._move(BreakerState.CLOSED, "probe succeeded")

    def record_failure(self, why: str = "") -> None:
        self.failures += 1
        if self._state == BreakerState.HALF_OPEN:
            self.opened_at = self.clock()
            self._move(BreakerState.OPEN, f"probe failed: {why}")
        elif self._state == BreakerState.CLOSED and self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
            self._move(BreakerState.OPEN, f"{self.failures} consecutive failures: {why}")

    def call(self, fn: Callable, *args, **kwargs):
        if not self.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is {self.state.value}: call refused")
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            self.record_failure(f"{type(exc).__name__}: {exc}")
            raise
        self.record_success()
        return out
