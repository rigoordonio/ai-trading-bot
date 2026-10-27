"""Portfolio drawdown monitor and kill switch.

Kill switch: once equity falls ``max_portfolio_drawdown`` below its running
peak, ALL NEW BUYS ARE BLOCKED. It latches: it stays on until reset
(manually, or automatically after N bars if configured). Sells are always
allowed, because reducing risk must never be blocked.

Daily loss limit: if today's equity change is worse than ``-max_daily_loss``,
new buys decided today are blocked (resets next session).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DrawdownState:
    peak: float = 0.0
    drawdown: float = 0.0
    kill_switch: bool = False
    kill_switch_since: pd.Timestamp | None = None
    bars_since_kill: int = 0
    daily_loss_breached: bool = False
    last_equity: float | None = None
    events: list[dict] = field(default_factory=list)


class DrawdownMonitor:
    def __init__(self, max_drawdown: float = 0.20, max_daily_loss: float | None = 0.04,
                 reset: str | dict = "manual", liquidate: bool = False):
        self.max_dd = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.reset_days = reset.get("days") if isinstance(reset, dict) else None
        self.liquidate = liquidate
        self.state = DrawdownState()

    @classmethod
    def from_config(cls, c) -> "DrawdownMonitor":
        return cls(c.max_portfolio_drawdown, c.max_daily_loss, c.kill_switch_reset, c.liquidate_on_kill_switch)

    def update(self, date: pd.Timestamp, equity: float) -> list[dict]:
        """Call once per bar with the closing equity. Returns new events."""
        st, events = self.state, []
        st.peak = max(st.peak, equity)
        st.drawdown = equity / st.peak - 1.0 if st.peak > 0 else 0.0

        st.daily_loss_breached = False
        if self.max_daily_loss is not None and st.last_equity:
            day_ret = equity / st.last_equity - 1.0
            if day_ret <= -self.max_daily_loss + 1e-12:
                st.daily_loss_breached = True
                events.append({"date": date, "event": "DAILY_LOSS_LIMIT", "severity": "WARNING",
                               "value": day_ret, "limit": -self.max_daily_loss,
                               "detail": "new buys blocked for this session"})
        st.last_equity = equity

        if st.kill_switch:
            st.bars_since_kill += 1
            if self.reset_days is not None and st.bars_since_kill >= self.reset_days:
                self._reset(date, equity, events, auto=True)
        elif -st.drawdown >= self.max_dd - 1e-12:      # exactly at the limit counts as a breach
            st.kill_switch, st.kill_switch_since, st.bars_since_kill = True, date, 0
            events.append({"date": date, "event": "KILL_SWITCH_ON", "severity": "CRITICAL",
                           "value": st.drawdown, "limit": -self.max_dd,
                           "detail": "portfolio drawdown limit breached: new trades blocked"})
        st.events.extend(events)
        return events

    def _reset(self, date, equity, events, auto: bool) -> None:
        st = self.state
        st.kill_switch, st.kill_switch_since, st.bars_since_kill = False, None, 0
        st.peak = equity      # restart the drawdown clock, otherwise it re-triggers immediately
        st.drawdown = 0.0
        events.append({"date": date, "event": "KILL_SWITCH_RESET", "severity": "INFO", "value": equity,
                       "limit": None, "detail": "automatic" if auto else "manual"})

    def manual_reset(self, date: pd.Timestamp, equity: float) -> list[dict]:
        events: list[dict] = []
        if self.state.kill_switch:
            self._reset(date, equity, events, auto=False)
            self.state.events.extend(events)
        return events

    def to_dict(self) -> dict:
        st = self.state
        return {"peak": st.peak, "drawdown": st.drawdown, "kill_switch": st.kill_switch,
                "kill_switch_since": str(st.kill_switch_since) if st.kill_switch_since is not None else None,
                "bars_since_kill": st.bars_since_kill, "last_equity": st.last_equity}

    def load_dict(self, d: dict | None) -> "DrawdownMonitor":
        if d:
            st = self.state
            st.peak, st.drawdown, st.kill_switch = float(d["peak"]), float(d["drawdown"]), bool(d["kill_switch"])
            st.kill_switch_since = pd.Timestamp(d["kill_switch_since"]) if d.get("kill_switch_since") else None
            st.bars_since_kill = int(d.get("bars_since_kill", 0))
            st.last_equity = d.get("last_equity")
        return self

    @property
    def buys_blocked(self) -> bool:
        return self.state.kill_switch or self.state.daily_loss_breached
