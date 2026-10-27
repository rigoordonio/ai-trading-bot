"""The Reviewer grades bots and can only REDUCE risk."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.reviewer.review import ACTIVE, PAUSED, PROBATION, Reviewer, ReviewerConfig, multipliers

NOW = datetime(2026, 12, 1, 1, 0, tzinfo=timezone.utc)        # Tue 08:00 Manila -> last session Mon 30 Nov


def write_state(d, equity: dict[str, list[float]], trades: dict[str, list[float]] | None = None,
                status: dict | None = None, orders: list | None = None, end="2026-11-30"):
    dates = pd.bdate_range(end=end, periods=max(len(v) for v in equity.values()))
    rows = []
    for b, eq in equity.items():
        for dt, e in zip(dates[-len(eq):], eq, strict=True):
            rows.append({"date": dt.date(), "bot_id": b, "equity": e, "cash": e, "positions_value": 0,
                         "n_positions": 0, "n_trades": 0, "realized_pnl": 0, "benchmark_close": 100 + len(rows)})
    acct = pd.DataFrame(rows).groupby("date", as_index=False)["equity"].sum()
    for _, r in acct.iterrows():
        rows.append({"date": r["date"], "bot_id": "ACCOUNT", "equity": r["equity"], "cash": 0, "positions_value": 0,
                     "n_positions": 0, "n_trades": 0, "realized_pnl": 0, "benchmark_close": 100.0})
    pd.DataFrame(rows).to_csv(d / "snapshots.csv", index=False)
    sleeves = {"sleeves": {b: {"cash": eq[-1], "positions": {}, "closed_trades": [
        {"symbol": "AAPL", "entry_date": "2026-10-01", "exit_date": "2026-10-10", "pnl": p, "return_pct": p / 1e4,
         "holding_days": 9, "exit_reason": "test"} for p in (trades or {}).get(b, [])]} for b, eq in equity.items()}}
    (d / "sleeves.json").write_text(json.dumps(sleeves))
    (d / "system_status.json").write_text(json.dumps(status or {"trading_enabled": True, "blocking_reasons": [],
                                                                "reconciliation": {"ok": True}, "health": {}}))
    (d / "orders.json").write_text(json.dumps(orders or []))


def steady(n, start=30_000.0, step=0.002, seed=1):
    rng = np.random.default_rng(seed)
    return list(start * np.cumprod(1 + step + rng.normal(0, 0.003, n)))


def test_learning_period_keeps_bots_active(tmp_path):
    write_state(tmp_path, {"SWING_V1": steady(5)})
    r = Reviewer(tmp_path, ReviewerConfig(), NOW).run()
    g = r["bots"][0]
    assert g["status"] == ACTIVE and g["size_multiplier"] == 1.0 and g["grade"].startswith("learning")
    assert (tmp_path / "reviews").exists() and (tmp_path / "review_latest.json").exists()


def test_good_bot_gets_a_grade(tmp_path):
    write_state(tmp_path, {"TREND_V1": steady(30)}, trades={"TREND_V1": [100, 200, -50]})
    g = Reviewer(tmp_path, ReviewerConfig(), NOW).run()["bots"][0]
    assert g["grade"] in ("A", "B") and g["status"] == ACTIVE and g["win_rate"] == pytest.approx(2 / 3)


def test_probation_halves_size_then_recovers(tmp_path):
    eq = steady(25) + [steady(25)[-1] * 0.90] * 3                  # -10%: probation, not pause
    write_state(tmp_path, {"SWING_V1": eq})
    g = Reviewer(tmp_path, ReviewerConfig(), NOW).run()["bots"][0]
    assert g["status"] == PROBATION and g["size_multiplier"] == 0.5
    m, err = multipliers(tmp_path, ["SWING_V1"])
    assert err is None and m["SWING_V1"] == (0.5, PROBATION)
    write_state(tmp_path, {"SWING_V1": eq + [eq[0] * 1.30] * 3})    # new high -> recovered
    g2 = Reviewer(tmp_path, ReviewerConfig(), NOW).run()["bots"][0]
    assert g2["status"] == ACTIVE and g2["size_multiplier"] == 1.0


def test_big_loss_pauses_even_while_learning_and_only_the_user_resumes(tmp_path):
    write_state(tmp_path, {"INVESTOR_V1": [30_000, 29_000, 26_000]})  # -13%
    r = Reviewer(tmp_path, ReviewerConfig(), NOW).run()
    assert r["bots"][0]["status"] == PAUSED and r["bots"][0]["size_multiplier"] == 0.0
    assert r["status_changes"]
    write_state(tmp_path, {"INVESTOR_V1": [30_000, 29_000, 26_000, 31_000]})   # recovered: still paused
    assert Reviewer(tmp_path, ReviewerConfig(), NOW).run()["bots"][0]["status"] == PAUSED
    Reviewer(tmp_path, ReviewerConfig(), NOW).resume("INVESTOR_V1")
    assert multipliers(tmp_path, ["INVESTOR_V1"])[0]["INVESTOR_V1"] == (1.0, ACTIVE)


def test_losing_streak_pauses(tmp_path):
    write_state(tmp_path, {"SWING_V1": steady(10)}, trades={"SWING_V1": [50] + [-10] * 6})
    assert Reviewer(tmp_path, ReviewerConfig(), NOW).run()["bots"][0]["status"] == PAUSED


def test_operational_alerts(tmp_path):
    orders = [{"order_id": "O1", "bot_id": "TREND_V1", "side": "BUY", "quantity": 14, "symbol": "META",
               "status": "CANCELLED", "reject_reason": None}]
    status = {"trading_enabled": False, "blocking_reasons": ["HALT[broker]: reconciliation failed"],
              "reconciliation": {"ok": False, "discrepancies": [{"symbol": "MSFT"}]},
              "health": {"components": {"bot:INVESTOR_V1": {"status": "DEGRADED", "detail": "no scores"}}}}
    write_state(tmp_path, {"TREND_V1": steady(3)}, status=status, orders=orders, end="2026-11-20")
    r = Reviewer(tmp_path, ReviewerConfig(), NOW).run()
    text = " | ".join(f"{a['level']} {a['detail']}" for a in r["alerts"])
    assert "HALTED" in text and "do not match" in text and "INVESTOR_V1 DEGRADED" in text
    assert "META ended CANCELLED" in text and "sessions behind" in text
    r2 = Reviewer(tmp_path, ReviewerConfig(), NOW).run()            # same expired order is reported once
    assert "META" not in " ".join(a["detail"] for a in r2["alerts"])


def test_reviewer_can_only_reduce_risk():
    with pytest.raises(Exception):
        ReviewerConfig(probation={"size_multiplier": 1.5})


def test_unreadable_state_fails_closed(tmp_path):
    (tmp_path / "reviewer_state.json").write_text("{corrupt")
    m, err = multipliers(tmp_path, ["SWING_V1", "TREND_V1"])
    assert err and all(v[0] == 0.0 for v in m.values())
    (tmp_path / "reviewer_state.json").write_text(json.dumps({"bots": {"SWING_V1": {"size_multiplier": 7}}}))
    assert multipliers(tmp_path, ["SWING_V1"])[0]["SWING_V1"][0] == 1.0      # capped: never above x1


def test_paused_bot_makes_no_new_buys_but_others_do():
    from tests.test_multi_bot import MON, Harness, day1_scores
    h = Harness()
    res_free = h.decide(MON, day1_scores())
    assert any(o.meta["bot_id"] == "SWING_V1" for o in res_free.orders)
    h2 = Harness()
    orig = h2.decide

    def decide_with_review(date, scores):
        from src.trading_cycle import BotRuntime
        real_init = BotRuntime.__init__

        def init(self, bot, *a, **k):
            real_init(self, bot, *a, **k)
            if bot.bot_id == "SWING_V1":
                self.buy_size_multiplier, self.reviewer_status = 0.0, PAUSED
            if bot.bot_id == "INVESTOR_V1":
                self.buy_size_multiplier, self.reviewer_status = 0.5, PROBATION
        BotRuntime.__init__ = init
        try:
            return orig(date, scores)
        finally:
            BotRuntime.__init__ = real_init
    res = decide_with_review(MON, day1_scores())
    bots = {o.meta["bot_id"]: o for o in res.orders}
    assert "SWING_V1" not in bots and "TREND_V1" in bots
    full_inv = [o for o in res_free.orders if o.meta["bot_id"] == "INVESTOR_V1"][0].quantity
    assert bots["INVESTOR_V1"].quantity == int(full_inv * 0.5)
    rec = [a for a in res.audit if a["bot_id"] == "SWING_V1" and a["symbol"] == "AAPL"][0]
    assert "reviewer" in rec["reason"] and rec["final_action"] == "NO_TRADE"


@pytest.mark.integration
@pytest.mark.slow
def test_paper_run_writes_history_and_obeys_reviewer(tmp_path):
    from src.bots.bot import build_bots
    from src.config import load_config
    from src.data.provider import SyntheticProvider
    from src.execution.paper_trader import PaperTrader
    from src.models.prediction import save_model
    from tests.fakes import MockModel

    cfg = load_config(overrides={"settings": {"data": {"provider": "synthetic", "raw_dir": str(tmp_path / "raw"),
                                                       "processed_dir": str(tmp_path / "proc")},
                                              "ml": {"artifacts_dir": str(tmp_path / "models")}}})
    bots = build_bots(cfg)
    for b in bots.values():
        save_model(MockModel(output=0.9), tmp_path / "models", bot_id=b.bot_id, model_version=b.ml_cfg.model_version,
                   feature_set_version=b.feature_version, features=b.ml_features, target={}, periods={},
                   metrics={}, model_name="mock")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "reviewer_state.json").write_text("{not json")          # corrupt -> no new buys at all
    r = PaperTrader(cfg, paper, SyntheticProvider(), bots=bots).run(pd.Timestamp("2026-06-01"))
    assert r.trading_enabled and not [o for o in r.submitted if o["side"] == "BUY"]
    assert any(e.get("event") == "REVIEWER_STATE_ERROR" for e in r.events)
    (paper / "reviewer_state.json").unlink()
    r2 = PaperTrader(cfg, paper, SyntheticProvider(), bots=bots).run(pd.Timestamp("2026-06-02"))
    assert [o for o in r2.submitted if o["side"] == "BUY"]
    snaps = pd.read_csv(paper / "snapshots.csv")
    assert "benchmark_close" in snaps and snaps["benchmark_close"].notna().any()
    dec = pd.read_csv(paper / "decisions" / "decisions-2026-06-02.csv")
    assert {"bot_id", "symbol", "action", "reason", "risk_status", "final_action"} <= set(dec.columns)
    rep = Reviewer(paper, ReviewerConfig(alerts={"notify_macos": False}), NOW).run()
    assert {g["bot_id"] for g in rep["bots"]} == set(bots)
