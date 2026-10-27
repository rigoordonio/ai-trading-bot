"""TradingSignal: the one object every bot hands to the portfolio layer.

Two timestamps are kept apart on purpose:
    data_timestamp   the close of the last COMPLETED bar the decision used
                     (daily bar date; for a weekly bot, the decision day,
                     whose weekly features come from strictly earlier weeks)
    created_at       when the signal object was produced (wall clock, UTC)
The ORDER timestamp is separate again: it lives on the Order and is set
when the OrderManager submits it. A signal is never an order.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"

    @classmethod
    def from_action(cls, action: str | None) -> "SignalType":
        try:
            return cls(str(action).upper())
        except ValueError:
            return cls.NO_TRADE


def _clean(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class TradingSignal:
    signal_id: str
    bot_id: str
    strategy_version: str
    symbol: str
    data_timestamp: pd.Timestamp
    timeframe: str
    signal_type: SignalType
    signal_score: float | None
    model_probability: float | None
    expected_horizon: tuple[int, int]          # trading days (min, max)
    feature_set_version: str
    model_version: str | None
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @staticmethod
    def make_id(bot_id: str, symbol: str, data_timestamp: pd.Timestamp) -> str:
        """Deterministic: the same bot deciding on the same bar for the same
        symbol always yields the same id (re-runs do not create new signals)."""
        raw = f"{bot_id}|{symbol}|{pd.Timestamp(data_timestamp).date()}"
        return f"{bot_id}-{symbol}-{pd.Timestamp(data_timestamp):%Y%m%d}-" + hashlib.sha1(raw.encode()).hexdigest()[:6]

    @classmethod
    def create(cls, *, bot_id: str, strategy_version: str, symbol: str, data_timestamp: pd.Timestamp,
               timeframe: str, signal_type: SignalType, signal_score: float | None,
               model_probability: float | None, expected_horizon: tuple[int, int], feature_set_version: str,
               model_version: str | None, reason: str = "") -> "TradingSignal":
        ts = pd.Timestamp(data_timestamp)
        return cls(cls.make_id(bot_id, symbol, ts), bot_id, strategy_version, symbol, ts, timeframe, signal_type,
                   _clean(signal_score), _clean(model_probability), tuple(expected_horizon),  # type: ignore[arg-type]
                   feature_set_version, model_version, reason)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal_type"] = self.signal_type.value
        d["data_timestamp"] = str(self.data_timestamp.date())
        d["expected_horizon"] = list(self.expected_horizon)
        return d
