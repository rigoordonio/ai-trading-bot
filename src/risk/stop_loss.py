"""Trade-level loss controls, evaluated on each CLOSE; exits execute next open.

    fixed      close <= avg_cost * (1 - fixed_stop_pct)
    trailing   close <= highest_close_since_entry - trailing_atr_multiple * ATR
    time       held >= max_holding_days (calendar days)

Stops are hypotheses too: run the robustness tools with stops on/off and at
different widths. Tight stops often just add turnover and costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.portfolio import Position


@dataclass
class StopSignal:
    symbol: str
    rule: str
    detail: str


class StopLossManager:
    def __init__(self, enabled: bool = True, fixed_stop_pct: float | None = 0.10,
                 trailing_atr_multiple: float | None = 3.0, max_holding_days: int | None = None):
        self.enabled = enabled
        self.fixed = fixed_stop_pct
        self.trail = trailing_atr_multiple
        self.max_days = max_holding_days

    @classmethod
    def from_config(cls, c) -> "StopLossManager":
        return cls(c.enabled, c.fixed_stop_pct, c.trailing_atr_multiple, c.max_holding_days)

    def check(self, position: Position, close: float, atr: float | None, date: pd.Timestamp) -> StopSignal | None:
        if not self.enabled or close is None or close != close:
            return None
        if self.fixed is not None and close <= position.avg_cost * (1 - self.fixed):
            return StopSignal(position.symbol, "fixed_stop",
                              f"close {close:.2f} <= {1 - self.fixed:.0%} of entry {position.avg_cost:.2f}")
        if self.trail is not None and atr is not None and atr == atr and atr > 0:
            level = position.highest_close - self.trail * atr
            if close <= level:
                return StopSignal(position.symbol, "trailing_atr_stop",
                                  f"close {close:.2f} <= high {position.highest_close:.2f} - {self.trail}xATR")
        if self.max_days is not None and (date - pd.Timestamp(position.entry_date)).days >= self.max_days:
            return StopSignal(position.symbol, "time_stop", f"held >= {self.max_days} days")
        return None
