"""Component health and the global trading gate.

TRADING_ENABLED is the AND of:
    - the operator switch (env TRADING_ENABLED, default true)
    - no un-recovered CRITICAL halt
    - every critical component HEALTHY or DEGRADED (none UNHEALTHY)
    - no circuit breaker OPEN

Recovery is explicit: ``TradingGate.recover()`` clears halts only when every
critical component reports HEALTHY and the caller supplies a passing
reconciliation. A CRITICAL error never silently clears itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.errors import Severity
from src.logging_setup import get_logger

log = get_logger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


_RANK = {HealthStatus.HEALTHY: 0, HealthStatus.DEGRADED: 1, HealthStatus.UNHEALTHY: 2}
CRITICAL_COMPONENTS = ("market_data", "database", "model", "broker", "risk_engine", "execution_engine")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    detail: str = ""
    updated_at: datetime = field(default_factory=_now)


class HealthMonitor:
    def __init__(self, critical: tuple[str, ...] = CRITICAL_COMPONENTS):
        self.critical = set(critical)
        self.components: dict[str, ComponentHealth] = {c: ComponentHealth(c) for c in critical}
        self.last_seen: dict[str, str] = {}   # e.g. last_market_data, last_signal, last_order, last_reconciliation

    def set(self, name: str, status: HealthStatus, detail: str = "") -> None:
        prev = self.components.get(name)
        self.components[name] = ComponentHealth(name, status, detail)
        if prev is None or prev.status != status:
            lvl = log.warning if status != HealthStatus.HEALTHY else log.info
            lvl("health %s -> %s %s", name, status.value, detail)

    def mark(self, what: str, value: str | None = None) -> None:
        self.last_seen[what] = value or _now().isoformat(timespec="seconds")

    def disable_component(self, name: str) -> None:
        """A component that is intentionally not used (e.g. no ML model configured)."""
        self.critical.discard(name)
        self.components.pop(name, None)

    @property
    def overall(self) -> HealthStatus:
        worst = max((c.status for c in self.components.values()), key=lambda s: _RANK[s], default=HealthStatus.HEALTHY)
        return worst

    def critical_unhealthy(self) -> list[str]:
        return [n for n, c in self.components.items() if n in self.critical and c.status == HealthStatus.UNHEALTHY]

    def snapshot(self) -> dict:
        return {"overall": self.overall.value,
                "components": {n: {"status": c.status.value, "detail": c.detail,
                                   "updated_at": c.updated_at.isoformat(timespec="seconds")}
                               for n, c in self.components.items()},
                "last_seen": dict(self.last_seen)}


def env_trading_enabled() -> bool:
    return os.getenv("TRADING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Halt:
    reason: str
    component: str
    severity: str
    at: str


class TradingGate:
    def __init__(self, health: HealthMonitor | None = None, operator_enabled: bool | None = None):
        self.health = health or HealthMonitor()
        self.operator_enabled = env_trading_enabled() if operator_enabled is None else operator_enabled
        self.halts: list[Halt] = []
        self.breakers: list = []          # CircuitBreaker instances to consult
        self.events: list[dict] = []

    # ------------------------------------------------------------------ state
    @property
    def trading_enabled(self) -> bool:
        return not self.blocking_reasons()

    def blocking_reasons(self) -> list[str]:
        reasons = []
        if not self.operator_enabled:
            reasons.append("TRADING_ENABLED=false (operator kill switch)")
        reasons += [f"HALT[{h.component}]: {h.reason}" for h in self.halts]
        reasons += [f"component UNHEALTHY: {c}" for c in self.health.critical_unhealthy()]
        reasons += [f"circuit breaker OPEN: {b.name}" for b in self.breakers if b.is_open]
        return reasons

    # ---------------------------------------------------------------- actions
    def halt(self, reason: str, component: str = "system", severity: Severity = Severity.CRITICAL) -> dict:
        h = Halt(reason, component, severity.value, _now().isoformat(timespec="seconds"))
        self.halts.append(h)
        if component in self.health.components:
            self.health.set(component, HealthStatus.UNHEALTHY, reason)
        evt = {"event": "TRADING_HALTED", "severity": severity.value, "component": component, "detail": reason,
               "timestamp": h.at}
        self.events.append(evt)
        log.critical("TRADING HALTED [%s]: %s", component, reason)
        return evt

    def handle_error(self, exc: Exception, component: str | None = None) -> dict:
        """Record a trading-path error. CRITICAL -> halt; otherwise just an event."""
        sev = getattr(exc, "severity", Severity.CRITICAL)   # unknown exceptions are CRITICAL
        comp: str = component or getattr(exc, "component", None) or "system"
        detail = f"{type(exc).__name__}: {exc}"
        if sev == Severity.CRITICAL:
            return self.halt(detail, comp, Severity.CRITICAL)
        evt = {"event": "TRADING_ERROR", "severity": sev.value if isinstance(sev, Severity) else str(sev),
               "component": comp, "detail": detail, "timestamp": _now().isoformat(timespec="seconds")}
        self.events.append(evt)
        log.error("trading error [%s]: %s", comp, detail)
        return evt

    def recover(self, reconciliation_ok: bool, operator: str = "manual") -> bool:
        """Clear halts ONLY if every critical component is HEALTHY and the
        broker reconciliation passed. Returns whether trading is enabled."""
        unhealthy = [n for n, c in self.health.components.items()
                     if n in self.health.critical and c.status != HealthStatus.HEALTHY]
        if not reconciliation_ok or unhealthy:
            log.error("recovery refused: reconciliation_ok=%s, not healthy=%s", reconciliation_ok, unhealthy)
            return False
        self.halts.clear()
        self.events.append({"event": "TRADING_RECOVERED", "severity": "INFO", "component": "system",
                            "detail": f"recovered by {operator}", "timestamp": _now().isoformat(timespec="seconds")})
        return self.trading_enabled

    def status(self) -> dict:
        return {"trading_enabled": self.trading_enabled, "blocking_reasons": self.blocking_reasons(),
                "health": self.health.snapshot(),
                "circuit_breakers": {b.name: b.state.value for b in self.breakers}}
