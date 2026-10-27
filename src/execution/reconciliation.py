"""Internal books vs broker account (the broker is authoritative).

Checks
    missing position     we think we hold it, broker does not
    unexpected position  broker holds it, we do not
    quantity mismatch
    unknown open order   open at the broker, absent from our ledger
    duplicate order      two broker orders share one idempotency key
    cash mismatch        beyond a tolerance
    unresolved UNKNOWN   ledger orders still in UNKNOWN state

Any discrepancy -> TRADING_ENABLED = False with a CRITICAL event.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.errors import ReconciliationError
from src.execution.broker import BrokerAdapter
from src.execution.orders import OrderStatus
from src.safety.health import HealthStatus, TradingGate

OPEN_BROKER = {"ACKNOWLEDGED", "SUBMITTED", "PARTIALLY_FILLED"}


@dataclass
class ReconciliationReport:
    ok: bool
    discrepancies: list[dict] = field(default_factory=list)
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def reconcile(internal_positions: dict[str, int], internal_cash: float, ledger, broker: BrokerAdapter,
              gate: TradingGate | None = None, cash_tolerance: float = 1.0,
              attribution: dict[str, dict[str, int]] | None = None) -> ReconciliationReport:
    """Strategy virtual positions -> aggregate expected position -> broker.
    ``attribution`` (symbol -> {bot_id: qty}) is added to position
    discrepancies so a mismatch shows which bots' books are involved.
    Any discrepancy fails reconciliation and halts ALL new trading."""
    issues: list[dict] = []
    try:
        bpos = {s: p.quantity for s, p in broker.get_positions().items() if p.quantity != 0}
        acct = broker.get_account()
        borders = broker.get_orders()
    except Exception as exc:  # noqa: BLE001 - broker unreachable = cannot prove we are safe
        rep = ReconciliationReport(False, [{"type": "broker_unreachable", "detail": str(exc)}])
        if gate:
            gate.handle_error(ReconciliationError(f"reconciliation impossible: {exc}"), "broker")
        return rep

    ipos = {s: q for s, q in internal_positions.items() if q != 0}
    for s in sorted(set(ipos) | set(bpos)):
        i, b = ipos.get(s, 0), bpos.get(s, 0)
        by_bot = {"by_bot": dict(attribution.get(s, {}))} if attribution is not None else {}
        if i and not b:
            issues.append({"type": "missing_position", "symbol": s, "internal": i, "broker": 0, **by_bot})
        elif b and not i:
            issues.append({"type": "unexpected_position", "symbol": s, "internal": 0, "broker": b, **by_bot})
        elif i != b:
            issues.append({"type": "quantity_mismatch", "symbol": s, "internal": i, "broker": b, **by_bot})

    known_keys = {o.idempotency_key for o in ledger.all()}
    for bo in borders:
        if bo.status in OPEN_BROKER and bo.client_order_id not in known_keys:
            issues.append({"type": "unknown_open_order", "symbol": bo.symbol, "broker_order_id": bo.broker_order_id})
    dupes = [k for k, n in Counter(bo.client_order_id for bo in borders).items() if n > 1]
    for k in dupes:
        issues.append({"type": "duplicate_order", "client_order_id": k})
    for o in ledger.open_orders():
        if o.status == OrderStatus.UNKNOWN:
            issues.append({"type": "unresolved_unknown_order", "symbol": o.symbol, "order_id": o.order_id})
    if abs(acct.cash - internal_cash) > cash_tolerance:
        issues.append({"type": "cash_mismatch", "internal": internal_cash, "broker": acct.cash,
                       "difference": acct.cash - internal_cash})

    rep = ReconciliationReport(not issues, issues)
    if gate is not None:
        gate.health.mark("last_reconciliation", f"{rep.at} {'OK' if rep.ok else 'FAILED'}")
        if rep.ok:
            gate.health.set("execution_engine", HealthStatus.HEALTHY, "reconciled")
        else:
            gate.handle_error(ReconciliationError(f"reconciliation failed: {issues}"), "execution_engine")
    return rep
