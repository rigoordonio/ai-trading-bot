"""The Reviewer: grades every bot after each paper run and manages risk DOWN.

Reads only what the paper trader wrote (data/paper/*): per-bot daily
snapshots, closed trades, the order ledger and the run status. It never
talks to the broker and never places orders.

Output
    data/paper/reviewer_state.json   bot status + size multiplier (read by the paper trader)
    data/paper/review_latest.json    machine-readable report
    data/paper/reviews/review-YYYY-MM-DD.md   human-readable report

Statuses (risk can only go DOWN automatically):
    ACTIVE     x1.0 new-buy size
    PROBATION  x size_multiplier, lifted automatically on recovery
    PAUSED     x0 (no new buys; exits still run), lifted only by the user
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.config import CONFIG_DIR
from src.logging_setup import get_logger

log = get_logger(__name__)

ACTIVE, PROBATION, PAUSED = "ACTIVE", "PROBATION", "PAUSED"
STATE_FILE = "reviewer_state.json"


# ------------------------------------------------------------------ config
class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbationCfg(_S):
    max_drawdown: float = 0.08
    min_rolling_sharpe: float = -0.5
    size_multiplier: float = Field(0.5, ge=0.0, le=1.0)      # can never enlarge a trade


class PauseCfg(_S):
    max_drawdown: float = 0.12
    losing_streak: int = 6


class RecoverCfg(_S):
    max_drawdown: float = 0.04
    min_rolling_sharpe: float = 0.0


class AlertCfg(_S):
    missed_run_business_days: int = 2
    notify_macos: bool = True


class ReviewerConfig(_S):
    enabled: bool = True
    min_days: int = 20
    rolling_days: int = 40
    probation: ProbationCfg = Field(default_factory=ProbationCfg)
    pause: PauseCfg = Field(default_factory=PauseCfg)
    recover: RecoverCfg = Field(default_factory=RecoverCfg)
    alerts: AlertCfg = Field(default_factory=AlertCfg)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ReviewerConfig":
        p = Path(path) if path else CONFIG_DIR / "reviewer.yaml"
        raw = yaml.safe_load(p.read_text()) if p.exists() else {}
        return cls.model_validate(raw or {})


# --------------------------------------------------------------- reading state
def multipliers(state_dir: str | Path, bot_ids: list[str]) -> tuple[dict[str, tuple[float, str]], str | None]:
    """(bot -> (buy size multiplier, status)), error. Used by the paper trader.

    No file yet: everyone ACTIVE. A file that cannot be read: FAIL CLOSED,
    no new buys for any bot (exits still run) until it is fixed."""
    path = Path(state_dir) / STATE_FILE
    if not path.exists():
        return {b: (1.0, ACTIVE) for b in bot_ids}, None
    try:
        raw = json.loads(path.read_text())
        out = {}
        for b in bot_ids:
            rec = raw.get("bots", {}).get(b, {})
            m = float(rec.get("size_multiplier", 1.0))
            if not math.isfinite(m) or m < 0:
                raise ValueError(f"{b}: invalid size_multiplier {m!r}")
            out[b] = (min(m, 1.0), str(rec.get("status", ACTIVE)))
        return out, None
    except Exception as exc:  # noqa: BLE001 - unreadable reviewer state -> fail closed (no new buys)
        return {b: (0.0, "UNKNOWN") for b in bot_ids}, f"reviewer state unreadable ({exc}); no new buys"


# ------------------------------------------------------------------ metrics
@dataclass
class BotGrade:
    bot_id: str
    days: int
    equity: float | None
    start_equity: float | None
    total_return: float | None
    daily_pnl: float | None
    drawdown: float | None            # current, from peak (negative or 0)
    rolling_sharpe: float | None
    benchmark_return: float | None
    trades: int
    win_rate: float | None
    losing_streak: int
    open_positions: int
    grade: str
    status: str
    size_multiplier: float
    reasons: list[str] = field(default_factory=list)
    new_trades: list[dict] = field(default_factory=list)


def _sharpe(r: pd.Series) -> float | None:
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(252))


def _letter(sharpe: float | None, dd: float | None, cfg: ReviewerConfig, days: int) -> str:
    if days < cfg.min_days or sharpe is None:
        return f"learning ({days}/{cfg.min_days} days)"
    scale = [(1.0, "A"), (0.5, "B"), (0.0, "C"), (-0.5, "D")]
    letter = next((g for t, g in scale if sharpe >= t), "F")
    if dd is not None and -dd >= cfg.probation.max_drawdown and letter < "F":
        letter = chr(ord(letter) + 1) if letter != "D" else "F"
    return letter


def _losing_streak(trades: list[dict]) -> int:
    n = 0
    for t in reversed(trades):
        if float(t.get("pnl", 0.0)) < 0:
            n += 1
        else:
            break
    return n


# ------------------------------------------------------------------ review
class Reviewer:
    def __init__(self, state_dir: str | Path = "data/paper", cfg: ReviewerConfig | None = None,
                 now: datetime | None = None):
        self.dir = Path(state_dir)
        self.cfg = cfg or ReviewerConfig.load()
        self.now = now or datetime.now(timezone.utc)

    # ---------------------------------------------------------- persistence
    def _load_state(self) -> dict:
        p = self.dir / STATE_FILE
        if not p.exists():
            return {"bots": {}, "seen_closed_orders": []}
        return json.loads(p.read_text())

    def _save_state(self, state: dict) -> None:
        tmp = self.dir / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        os.replace(tmp, self.dir / STATE_FILE)

    def resume(self, bot_id: str) -> dict:
        """User action: lift a PAUSE (or probation). Drawdown is then measured
        from the resume date, so an old loss does not re-pause immediately."""
        state = self._load_state()
        rec = state.setdefault("bots", {}).setdefault(bot_id, {})
        rec.update(status=ACTIVE, size_multiplier=1.0, since=self.now.isoformat(timespec="seconds"),
                   reason="resumed by user", resumed_at=str(self.now.date()))
        self._save_state(state)
        return rec

    # ---------------------------------------------------------- main
    def run(self) -> dict:
        state = self._load_state()
        snaps = self._read_csv("snapshots.csv")
        sleeves = self._read_json("sleeves.json") or {}
        status = self._read_json("system_status.json") or {}
        orders = self._read_json("orders.json") or []
        bot_ids = [b for b in sleeves.get("sleeves", {})] or sorted(
            set(snaps["bot_id"]) - {"ACCOUNT"} if len(snaps) else [])
        alerts = self._operational_checks(snaps, status, orders, state)
        grades = [self._grade(b, snaps, sleeves, state) for b in bot_ids]
        account = self._account(snaps)

        changes = []
        for g in grades:
            prev = state["bots"].get(g.bot_id, {}).get("status", ACTIVE)
            if g.status != prev:
                changes.append(f"{g.bot_id}: {prev} -> {g.status} ({'; '.join(g.reasons)})")
            rec = state["bots"].setdefault(g.bot_id, {})
            if g.status != prev:
                rec["since"] = self.now.isoformat(timespec="seconds")
            rec.update(status=g.status, size_multiplier=g.size_multiplier, reason="; ".join(g.reasons),
                       grade=g.grade)
            rec["seen_trades"] = g.trades
        self._save_state(state)

        report = {"reviewed_at": self.now.isoformat(timespec="seconds"),
                  "as_of": str(snaps["date"].max()) if len(snaps) else None,
                  "account": account, "bots": [asdict(g) for g in grades], "alerts": alerts,
                  "status_changes": changes}
        (self.dir / "review_latest.json").write_text(json.dumps(report, indent=2, default=str))
        md = self.markdown(report)
        rdir = self.dir / "reviews"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"review-{report['as_of'] or self.now.date()}.md").write_text(md)
        if self.cfg.alerts.notify_macos and (changes or [a for a in alerts if a["level"] == "ALERT"]):
            self._notify(changes, alerts)
        return report

    # ---------------------------------------------------------- pieces
    def _read_csv(self, name: str) -> pd.DataFrame:
        p = self.dir / name
        if not p.exists():
            return pd.DataFrame(columns=["date", "bot_id", "equity"])
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.drop_duplicates(["date", "bot_id"], keep="last").sort_values("date")

    def _read_json(self, name: str) -> Any:
        p = self.dir / name
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None

    def _grade(self, bot_id: str, snaps: pd.DataFrame, sleeves: dict, state: dict) -> BotGrade:
        cfg = self.cfg
        rec = state["bots"].get(bot_id, {})
        prev_status = rec.get("status", ACTIVE)
        s = snaps[snaps["bot_id"] == bot_id] if len(snaps) else snaps
        eq = s.set_index("date")["equity"].astype(float) if len(s) else pd.Series(dtype=float)
        since = rec.get("resumed_at")
        eq_dd = eq[eq.index >= pd.Timestamp(since).date()] if since and len(eq) else eq
        days = len(eq)
        rets = eq.pct_change().dropna()
        dd = float(eq_dd.iloc[-1] / eq_dd.cummax().iloc[-1] - 1) if len(eq_dd) else None
        sharpe = _sharpe(rets.tail(cfg.rolling_days))
        bench_ret = None
        if len(s) and "benchmark_close" in s and s["benchmark_close"].notna().sum() >= 2:
            b = s["benchmark_close"].dropna().astype(float)
            bench_ret = float(b.iloc[-1] / b.iloc[0] - 1)
        trades = (sleeves.get("sleeves", {}).get(bot_id, {}) or {}).get("closed_trades", [])
        seen = int(rec.get("seen_trades", 0))
        new_trades = [{k: t.get(k) for k in ("symbol", "entry_date", "exit_date", "pnl", "return_pct",
                                              "holding_days", "exit_reason")} for t in trades[seen:]]
        wins = [t for t in trades if float(t.get("pnl", 0)) > 0]
        streak = _losing_streak(trades)
        positions = len((sleeves.get("sleeves", {}).get(bot_id, {}) or {}).get("positions", {}))

        status, reasons = prev_status, []
        if prev_status == PAUSED:
            reasons.append(rec.get("reason") or "paused; resume manually")
        elif dd is not None and -dd >= cfg.pause.max_drawdown:
            status, reasons = PAUSED, [f"drawdown {dd:.1%} beyond pause limit {cfg.pause.max_drawdown:.0%}"]
        elif streak >= cfg.pause.losing_streak:
            status, reasons = PAUSED, [f"{streak} losing trades in a row"]
        elif days >= cfg.min_days and ((dd is not None and -dd >= cfg.probation.max_drawdown) or
                                       (sharpe is not None and sharpe < cfg.probation.min_rolling_sharpe)):
            status = PROBATION
            reasons = [f"drawdown {dd:.1%}" if dd is not None else "",
                       f"rolling Sharpe {sharpe:.2f}" if sharpe is not None else ""]
            reasons = [r for r in reasons if r]
        elif prev_status == PROBATION:
            recovered = (dd is not None and -dd <= cfg.recover.max_drawdown and
                         (sharpe is None or sharpe >= cfg.recover.min_rolling_sharpe))
            if recovered:
                status, reasons = ACTIVE, ["recovered"]
            else:
                reasons = ["still below recovery thresholds"]
        else:
            status = ACTIVE
            reasons = ["within limits" if days >= cfg.min_days else "learning: not enough live data to judge"]
        mult = {ACTIVE: 1.0, PROBATION: cfg.probation.size_multiplier, PAUSED: 0.0}[status]
        return BotGrade(bot_id, days, float(eq.iloc[-1]) if days else None, float(eq.iloc[0]) if days else None,
                        float(eq.iloc[-1] / eq.iloc[0] - 1) if days and eq.iloc[0] > 0 else None,
                        float(eq.iloc[-1] - eq.iloc[-2]) if days > 1 else None, dd, sharpe, bench_ret,
                        len(trades), len(wins) / len(trades) if trades else None, streak, positions,
                        _letter(sharpe, dd, cfg, days), status, mult, reasons, new_trades)

    def _account(self, snaps: pd.DataFrame) -> dict:
        a = snaps[snaps["bot_id"] == "ACCOUNT"] if len(snaps) else snaps
        if not len(a):
            return {}
        eq = a.set_index("date")["equity"].astype(float)
        out = {"equity": float(eq.iloc[-1]), "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
               "daily_pnl": float(eq.iloc[-1] - eq.iloc[-2]) if len(eq) > 1 else None,
               "drawdown": float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1), "days": len(eq)}
        if "benchmark_close" in a and a["benchmark_close"].notna().sum() >= 2:
            b = a["benchmark_close"].dropna().astype(float)
            out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        return out

    def _operational_checks(self, snaps: pd.DataFrame, status: dict, orders: list, state: dict) -> list[dict]:
        from src.execution.paper_trader import last_completed_session

        alerts: list[dict] = []

        def add(level: str, what: str) -> None:
            alerts.append({"level": level, "detail": what})

        if not status:
            add("ALERT", "no run status found: has the bot run at all?")
        else:
            reasons = status.get("blocking_reasons") or []
            benign = [r for r in reasons if "Nothing was decided" in r]
            if not status.get("trading_enabled", False):
                add("ALERT", "trading HALTED: " + "; ".join(reasons))
            elif benign:
                add("WARN" if ("not available yet" in benign[0] or "unavailable" in benign[0]) else "INFO",
                    benign[0])
            rec = status.get("reconciliation") or {}
            if rec and not rec.get("ok", True):
                add("ALERT", f"books do not match the broker: {rec.get('discrepancies')}")
            for name, comp in (status.get("health", {}).get("components", {}) or {}).items():
                if name.startswith("bot:") and comp.get("status") != "HEALTHY":
                    add("WARN", f"{name[4:]} {comp.get('status')}: {comp.get('detail')}")
        if len(snaps):
            last = pd.Timestamp(snaps["date"].max())
            expected = last_completed_session(self.now)
            gap = len(pd.bdate_range(last, expected)) - 1
            if gap > self.cfg.alerts.missed_run_business_days:
                add("ALERT", f"last completed run is for {last.date()}, {gap} US sessions behind "
                             f"{expected.date()}: is the Mac on and the schedule loaded?")
        seen = set(state.get("seen_closed_orders", []))
        fresh = [o for o in orders if o.get("status") in ("CANCELLED", "REJECTED") and o["order_id"] not in seen]
        for o in fresh:
            add("WARN", f"order {o.get('bot_id') or o.get('strategy')} {o['side']} {o['quantity']} {o['symbol']} "
                        f"ended {o['status']} without (full) fill: {o.get('reject_reason') or 'expired/cancelled'}")
        state["seen_closed_orders"] = sorted(seen | {o["order_id"] for o in fresh})[-2000:]
        return alerts

    def _notify(self, changes: list[str], alerts: list[dict]) -> None:
        if sys.platform != "darwin" or os.getenv("TRADING_BOT_IGNORE_DOTENV") == "1":   # never from the test suite
            return
        msg = (changes[0] if changes else next(a["detail"] for a in alerts if a["level"] == "ALERT"))[:200]
        msg = msg.replace('"', "'")
        try:
            subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "Trading bot Reviewer"'],
                           timeout=10, check=False, capture_output=True)
        except Exception as exc:  # noqa: BLE001 - notification is best effort
            log.warning("notification failed: %s", exc)

    # ---------------------------------------------------------- report
    @staticmethod
    def markdown(r: dict) -> str:
        def pct(x):
            return "—" if x is None else f"{x:+.2%}"

        def money(x):
            return "—" if x is None else f"${x:,.0f}"

        lines = [f"# Reviewer report: {r['as_of']}", "", f"_Reviewed {r['reviewed_at']}_", ""]
        a = r.get("account") or {}
        if a:
            lines += ["## Account", f"Equity {money(a.get('equity'))} · total {pct(a.get('total_return'))} · "
                                    f"today {money(a.get('daily_pnl'))} · drawdown {pct(a.get('drawdown'))} · "
                                    f"SPY over same period {pct(a.get('benchmark_return'))}", ""]
        lines += ["## Bots", "", "| Bot | Grade | Status | Equity | Return | Today | Drawdown | Sharpe (rolling) | "
                                 "Trades | Win rate | Open |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in r["bots"]:
            sh = "—" if g["rolling_sharpe"] is None else f"{g['rolling_sharpe']:.2f}"
            wr = "—" if g["win_rate"] is None else f"{g['win_rate']:.0%}"
            lines.append(f"| {g['bot_id']} | {g['grade']} | {g['status']} (x{g['size_multiplier']:g}) | "
                         f"{money(g['equity'])} | {pct(g['total_return'])} | {money(g['daily_pnl'])} | "
                         f"{pct(g['drawdown'])} | {sh} | {g['trades']} | {wr} | {g['open_positions']} |")
        lines.append("")
        for g in r["bots"]:
            lines.append(f"- **{g['bot_id']}**: {'; '.join(g['reasons'])}")
        new = [(g["bot_id"], t) for g in r["bots"] for t in g["new_trades"]]
        if new:
            lines += ["", "## New closed trades", "", "| Bot | Symbol | P&L | Return | Days | Exit reason |",
                      "|---|---|---|---|---|---|"]
            for b, t in new:
                lines.append(f"| {b} | {t['symbol']} | {money(t['pnl'])} | {pct(t['return_pct'])} | "
                             f"{t['holding_days']} | {t['exit_reason']} |")
        lines += ["", "## Alerts", ""]
        lines += [f"- **{x['level']}** {x['detail']}" for x in r["alerts"]] or ["- none"]
        if r["status_changes"]:
            lines += ["", "## Status changes", ""] + [f"- {c}" for c in r["status_changes"]]
        return "\n".join(lines) + "\n"

    @staticmethod
    def summary(r: dict) -> str:
        out = [f"REVIEWER {r['as_of']}"]
        for g in r["bots"]:
            ret = "—" if g["total_return"] is None else f"{g['total_return']:+.2%}"
            out.append(f"  {g['bot_id']:<12} {g['status']:<9} grade {g['grade']:<22} return {ret}")
        for x in r["alerts"]:
            out.append(f"  [{x['level']}] {x['detail']}")
        for c in r["status_changes"]:
            out.append(f"  CHANGE {c}")
        return "\n".join(out)
