"""Fault-injecting doubles for brokers, ledgers and models (deterministic)."""
from __future__ import annotations

from src.errors import BrokerTimeoutError, OrderRejectedError
from src.execution.order_manager import InMemoryLedger
from src.execution.paper_broker import PaperBroker


class FaultyBroker(PaperBroker):
    """PaperBroker with switchable failure modes.

    mode:
      None                      behave normally
      "timeout_after_accept"    the broker ACCEPTS the order, but the response is lost (timeout)
      "timeout_before_accept"   the request never reaches the broker (timeout)
      "reject"                  broker rejects (e.g. HTTP 400 business rejection)
      "error_after_accept"      broker accepts, then an unexpected exception is raised
      "query_down"              lookups (get_order_by_client_id / positions) fail
    """

    def __init__(self, *a, mode: str | None = None, **k):
        super().__init__(*a, **k)
        self.mode = mode
        self.submit_calls = 0

    def submit_order(self, order):
        self.submit_calls += 1
        if self.mode == "timeout_before_accept":
            raise BrokerTimeoutError("read timeout (request lost)")
        if self.mode == "reject":
            raise OrderRejectedError("broker rejected: insufficient buying power")
        bo = super().submit_order(order)
        if self.mode == "timeout_after_accept":
            raise BrokerTimeoutError("read timeout (response lost)")
        if self.mode == "error_after_accept":
            raise RuntimeError("connection reset by peer")
        return bo

    def get_order_by_client_id(self, client_order_id):
        if self.mode == "query_down":
            raise ConnectionError("broker API unreachable")
        return super().get_order_by_client_id(client_order_id)

    def get_positions(self):
        if self.mode == "query_down":
            raise ConnectionError("broker API unreachable")
        return super().get_positions()


class FaultyLedger(InMemoryLedger):
    def __init__(self, fail_intent: bool = False, fail_update: bool = False):
        super().__init__()
        self.fail_intent = fail_intent
        self.fail_update = fail_update

    def record_intent(self, order):
        if self.fail_intent:
            raise OSError("database connection refused")
        super().record_intent(order)

    def update(self, order):
        if self.fail_update:
            raise OSError("database write failed")
        super().update(order)


class MockModel:
    """sklearn-like model with programmable output."""

    def __init__(self, output=None, raise_exc: Exception | None = None):
        self.output = output
        self.raise_exc = raise_exc

    def predict_proba(self, X):
        import numpy as np
        if self.raise_exc:
            raise self.raise_exc
        if self.output is not None:
            p = np.full(len(X), self.output, dtype=float)
        else:
            p = np.full(len(X), 0.8)
        return np.column_stack([1 - p, p])
