"""Multi-bot platform: contracts, attribution, conflicts, isolation, acceptance.

The deterministic harness below drives the REAL bots' risk configuration,
sleeve RiskEngines, MultiBotPortfolioManager, OrderValidator, TradingGate,
OrderManager, PaperBroker and reconciliation. Only the bots' scores are
scripted (controlled signals), so every expectation is exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bots.bot import build_bots
from src.bots.signal import SignalType, TradingSignal
from src.config import load_config
from src.errors import (
    ConfigurationError,
    DataValidationError,
    ModelCompatibilityError,
    TradingDisabledError,
)
from src.execution.order_manager import InMemoryLedger, OrderManager
from src.execution.orders import Fill, OrderStatus, Side
from src.execution.paper_broker import PaperBroker
from src.execution.reconciliation import reconcile
from src.execution.simulator import CostModel
from src.features.weekly import to_weekly
from src.portfolio.attribution import aggregate_positions, attribution_table, virtual_positions
from src.portfolio.multi_bot import MultiBotPortfolioManager
from src.portfolio.portfolio import Portfolio
from src.risk.drawdown import DrawdownMonitor
from src.risk.engine import RiskEngine
from src.risk.stop_loss import StopLossManager
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.health import HealthMonitor, TradingGate
from src.safety.order_validator import OrderValidator
from src.trading_cycle import BotRuntime, MarketState, TradingCycle
from tests.fakes import MockModel

SYMS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
CAPITAL = 100_000.0
MON, TUE, WED = pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-10")
NEXT_MON, NEXT_TUE = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16")
BUY_SCORE, EXIT_SCORE, NEUTRAL = 1.0, 0.0, 0.5


def make_cfg(**portfolio):
    return load_config(overrides={
        "settings": {"data": {"provider": "synthetic"}, "universe": {"symbols": SYMS, "benchmark": "SPY"}},
        "risk": {"portfolio": portfolio} if portfolio else {}})


# ---------------------------------------------------------------- harness
class Harness:
    """One account, three bots, scripted scores, real risk/order/broker path."""

    def __init__(self, cfg=None, price: float = 100.0, sleeve_cash: dict[str, float] | None = None):
        self.cfg = cfg or make_cfg()
        self.bots = build_bots(self.cfg)
        self.price = price
        self.gate = TradingGate(HealthMonitor(), operator_enabled=True)
        self.gate.health.disable_component("model")
        self.broker = PaperBroker(CAPITAL, CostModel(), trading_enabled=lambda: self.gate.trading_enabled,
                                  known_symbols=set(SYMS))
        self.ledger = InMemoryLedger()
        self.om = OrderManager(self.broker, self.ledger, self.gate, CircuitBreaker("broker", 3))
        self.sleeves = {b: Portfolio(cash=(sleeve_cash or {}).get(b, CAPITAL * bot.capital_allocation))
                        for b, bot in self.bots.items()}
        self.unallocated = CAPITAL - sum(p.cash for p in self.sleeves.values())
        self.broker.cash = CAPITAL
        self.risk = {b: RiskEngine(bot.risk_cfg, None, DrawdownMonitor.from_config(bot.risk_cfg.drawdown), b)
                     for b, bot in self.bots.items()}
        self.prm = MultiBotPortfolioManager.from_config(self.cfg.risk)
        self.prev_date: pd.Timestamp | None = None
        need = {c for bot in self.bots.values() for c in bot.signal_features}
        self.features = pd.DataFrame({c: 1.0 for c in need}, index=SYMS)
        self.features["close"] = price
        self.features["atr_14"] = 2.0
        self.features["realized_vol_20"] = 0.20
        self.features["realized_vol_60"] = 0.20
        self.features["dollar_volume_20"] = 1e10
        self.submitted: list = []
        self.cycle: TradingCycle | None = None

    def bar(self) -> pd.Series:
        p = self.price
        return pd.Series({"open": p, "high": p, "low": p, "close": p, "volume": 50_000_000.0})

    def signal_frame(self, bot_id: str, date: pd.Timestamp, scores: dict[str, float]):
        from src.strategies.base import SignalFrame
        idx = pd.MultiIndex.from_product([[date], SYMS], names=["date", "symbol"])
        s = pd.Series([scores.get(x, NEUTRAL) for x in SYMS], index=idx, dtype=float)
        t = self.bots[bot_id].cfg.thresholds
        return SignalFrame(bot_id, pd.DataFrame({"score": s}), t.buy, t.hold, t.exit)

    def decide(self, date: pd.Timestamp, scores: dict[str, dict[str, float]], broken: set[str] = frozenset()):
        """One decision cycle on ``date``'s close; returns the CycleResult."""
        runtimes = {}
        for b, bot in self.bots.items():
            sf = self.signal_frame(b, date, scores.get(b, {}))
            if b in broken:
                def boom(d, _b=b):
                    raise RuntimeError(f"{_b} strategy code crashed")
                sf.on = boom                                  # type: ignore[method-assign]
            runtimes[b] = BotRuntime(bot, sf, lambda d: self.features, self.sleeves[b], self.risk[b],
                                     StopLossManager.from_config(bot.risk_cfg.stops))
        self.cycle = TradingCycle(runtimes, self.prm, OrderValidator.from_config(self.cfg.risk), self.gate,
                                  self.cfg.risk.safety, set(SYMS), lambda: self.unallocated, "paper", "T",
                                  sleeves=self.sleeves)
        self.cycle._prev_date = self.prev_date
        for p in self.sleeves.values():
            p.mark_to_market({s: self.price for s in SYMS})
        bars = {s: self.bar() for s in SYMS}
        res = self.cycle.run(MarketState(date, bars, {s: date for s in SYMS}, {s: self.price for s in SYMS}))
        self.prev_date = date
        return res

    def submit(self, res):
        out = []
        for o in res.orders:
            out.append(self.om.submit(o))
        self.submitted += out
        return out

    def fill_at_open(self, date: pd.Timestamp):
        """Broker fills at the next open; fills are booked to the owning bot's sleeve."""
        fills = self.broker.process_open(date, {s: self.bar() for s in SYMS})
        for f in fills:
            o = self.ledger.get_by_key(f["client_order_id"])
            bot = o.meta["bot_id"]
            self.sleeves[bot].apply_fill(Fill(f["broker_order_id"], f["symbol"], Side(f["side"]), int(f["quantity"]),
                                              float(f["price"]), float(f["reference_price"]),
                                              float(f["commission"]), pd.Timestamp(f["timestamp"])),
                                         reason=o.reason)
        self.om.refresh_open_orders()
        return fills

    def reconcile(self):
        cash = self.unallocated + sum(p.cash for p in self.sleeves.values())
        return reconcile(aggregate_positions(self.sleeves), cash, self.ledger, self.broker, self.gate,
                         cash_tolerance=1.0, attribution=virtual_positions(self.sleeves))


def day1_scores():
    return {"SWING_V1": {"AAPL": BUY_SCORE}, "TREND_V1": {"MSFT": BUY_SCORE}, "INVESTOR_V1": {"AAPL": BUY_SCORE}}


# ======================================================== timeframe contracts
@pytest.fixture
def bots(cfg):
    return build_bots(cfg)


def test_swing_bot_uses_daily_data(bots, prices):
    b = bots["SWING_V1"]
    assert type(b).__name__ == "SwingBot" and b.timeframe == "daily" and b.weekly_pipeline is None
    assert not [f for f in b.ml_features if f.startswith("w_")]
    panel = b.compute_features(prices)
    assert not [c for c in panel.columns if c.startswith("w_")]
    # one row per DAILY session
    dates = panel.xs("AAPL", level="symbol").index
    assert pd.Series(dates).diff().dt.days.median() == 1
    weekly_bars = {s: to_weekly(df) for s, df in prices.items()}
    with pytest.raises(DataValidationError, match="DAILY"):
        b.compute_features(weekly_bars)
    with pytest.raises(ConfigurationError, match="daily"):          # the swing bot cannot be made weekly
        build_bots(load_config(overrides={"bots": {"bots": {"SWING_V1": {
            "timeframe": "1w", "weekly_features": {"return_windows": [4]}}}}}))


def test_trend_bot_uses_daily_data(bots, prices):
    b = bots["TREND_V1"]
    assert type(b).__name__ == "TrendBot" and b.timeframe == "daily" and b.weekly_pipeline is None
    assert not [f for f in b.ml_features if f.startswith("w_")]
    panel = b.compute_features(prices)
    assert "trend_strength" in panel.columns
    d = panel.xs("MSFT", level="symbol")
    exp = (d["sma_50"] / d["sma_200"] - 1) / d["realized_vol_60"]
    pd.testing.assert_series_equal(d["trend_strength"], exp, check_names=False)
    with pytest.raises(DataValidationError):
        b.compute_features({s: to_weekly(df) for s, df in prices.items()})


def test_investor_bot_uses_weekly_data(bots, prices):
    b = bots["INVESTOR_V1"]
    assert type(b).__name__ == "InvestorBot" and b.timeframe == "weekly" and b.weekly_pipeline is not None
    assert b.ml_features and all(f.startswith("w_") for f in b.ml_features)          # model: weekly only
    assert all(c.params.get("prefix") == "w_" for c in b.cfg.components.values())    # rules: weekly only
    panel = b.compute_features(prices)
    # w_close on a decision day is the close of the last COMPLETED weekly candle
    weekly = to_weekly(prices["AAPL"])
    t = pd.Timestamp("2019-06-12")                                                    # a Wednesday
    last_complete = weekly[weekly.index < t].iloc[-1]
    assert panel.loc[(t, "AAPL"), "w_close"] == pytest.approx(last_complete["close"])
    # bot-specific weekly features exist and are computed in weekly space
    assert {"w_drawdown_52", "w_trend_strength"} <= set(panel.columns)
    with pytest.raises(ConfigurationError, match="w_"):
        build_bots(load_config(overrides={"bots": {"bots": {"INVESTOR_V1": {"ml": {"features": ["ret_20d"]}}}}}))


@pytest.mark.parametrize("cutoff", ["2019-03-13", "2019-03-15", "2020-07-01"])   # Wed, Fri, Wed
def test_investor_does_not_use_incomplete_week(bots, prices, cutoff):
    b = bots["INVESTOR_V1"]
    t = pd.Timestamp(cutoff)
    full = b.compute_features(prices)
    trunc = b.compute_features({s: df.loc[:t] for s, df in prices.items()})        # data ends mid-week
    wcols = [c for c in full.columns if c.startswith("w_")]
    pd.testing.assert_frame_equal(full.xs(t, level="date")[wcols].sort_index(),
                                  trunc.xs(t, level="date")[wcols].sort_index())
    # the candle of the week containing t is never visible on t, even on its Friday
    wk = to_weekly(prices["AAPL"].loc[:t])
    current_week = wk.index[-1]
    assert full.loc[(t, "AAPL"), "w_close"] != pytest.approx(wk.loc[current_week, "close"]) or \
        wk.index[-1] < t


# ================================================================= models
def _save(tmp_path, bot, version=None, feature_version=None):
    from src.models.prediction import save_model
    return save_model(MockModel(output=0.7), tmp_path, bot_id=bot.bot_id,
                      model_version=version or bot.ml_cfg.model_version,
                      feature_set_version=feature_version or bot.feature_version, features=bot.ml_features,
                      target={"horizon_days": bot.ml_cfg.target_horizon_days},
                      periods={"train": ["2015-01-01", "2021-12-31"], "validation": ["2022-01-01", "2023-12-31"],
                               "final_fit": ["2015-01-01", "2023-12-31"]},
                      metrics={}, model_name="mock", model_id=bot.MODEL_ID)


def test_bot_model_version_matching(bots, tmp_path):
    import shutil

    from src.models.prediction import load_model
    s, t = bots["SWING_V1"], bots["TREND_V1"]
    d = _save(tmp_path, s)
    lm = load_model(tmp_path, "SWING_V1", s.ml_cfg.model_version, s.feature_version)
    meta = lm.meta
    for k in ("bot_id", "model_id", "model_version", "feature_set_version", "training_start", "training_end",
              "validation_period", "target_definition", "algorithm", "hyperparameters", "created_at"):
        assert meta.get(k) not in (None, ""), k
    assert meta["model_id"].startswith("swing_model_v1-swing-ml-1-")
    assert meta["training_end"] == "2023-12-31" and meta["validation_period"] == ["2022-01-01", "2023-12-31"]
    # SWING's artifact placed where TREND's model is expected -> rejected (bot_id mismatch)
    dst = tmp_path / "TREND_V1" / t.ml_cfg.model_version
    shutil.copytree(d, dst)
    with pytest.raises(ModelCompatibilityError, match="SWING_V1"):
        load_model(tmp_path, "TREND_V1", t.ml_cfg.model_version, t.feature_version)
    # wrong model_version inside metadata -> rejected
    _save(tmp_path / "v", s, version="swing-ml-2")
    (tmp_path / "v" / "SWING_V1" / "swing-ml-2").rename(tmp_path / "v" / "SWING_V1" / "swing-ml-1")
    with pytest.raises(ModelCompatibilityError):
        load_model(tmp_path / "v", "SWING_V1", "swing-ml-1", s.feature_version)
    # old artifacts without the required metadata are refused (retrain required)
    import json
    mp = tmp_path / "SWING_V1" / s.ml_cfg.model_version / "metadata.json"
    m = json.loads(mp.read_text())
    m.pop("model_id")
    mp.write_text(json.dumps(m))
    with pytest.raises(ModelCompatibilityError, match="retrain"):
        load_model(tmp_path, "SWING_V1", s.ml_cfg.model_version, s.feature_version)


def test_bot_feature_version_matching(bots, tmp_path, prices):
    from src.models.prediction import Predictor, load_model
    versions = {b.feature_version for b in bots.values()}
    assert len(versions) == 3
    s, t = bots["SWING_V1"], bots["TREND_V1"]
    _save(tmp_path, s)
    with pytest.raises(ModelCompatibilityError):
        load_model(tmp_path, "SWING_V1", s.ml_cfg.model_version, t.feature_version)
    lm = load_model(tmp_path, "SWING_V1", s.ml_cfg.model_version, s.feature_version)
    panel = s.compute_features(prices)
    with pytest.raises(ModelCompatibilityError):
        Predictor.from_loaded(lm).predict(panel, t.feature_version)
    assert Predictor.from_loaded(lm).predict(panel.tail(50), s.feature_version).notna().any()


# ================================================================ capital
def test_separate_capital_allocations():
    cfg = make_cfg()
    bots = build_bots(cfg)
    alloc = {b: x.capital_allocation for b, x in bots.items()}
    assert alloc == {"SWING_V1": 0.30, "TREND_V1": 0.40, "INVESTOR_V1": 0.30}
    # INVESTOR has almost no cash left in its sleeve; the account has plenty.
    h = Harness(cfg, sleeve_cash={"SWING_V1": 30_000, "TREND_V1": 40_000, "INVESTOR_V1": 500})
    res = h.decide(MON, day1_scores())
    by_bot = {o.meta["bot_id"]: o for o in res.orders}
    assert "SWING_V1" in by_bot and "TREND_V1" in by_bot
    inv = by_bot.get("INVESTOR_V1")
    assert inv is None or inv.quantity * h.price <= 500         # never spends another bot's cash
    h.submit(res)
    h.fill_at_open(TUE)
    for b, p in h.sleeves.items():
        assert p.cash >= -1e-6, b
    assert h.sleeves["INVESTOR_V1"].positions_value <= 500


def test_portfolio_exposure_limit():
    h = Harness(make_cfg(max_gross_exposure=0.08))
    res = h.decide(MON, day1_scores())
    total = sum(o.quantity * h.price for o in res.orders if o.side == Side.BUY)
    assert total <= 0.08 * CAPITAL + 1e-6
    rules = {e.get("rule") for e in res.events if e.get("event") in ("PORTFOLIO_RESIZE", "PORTFOLIO_BLOCK")}
    assert "max_account_gross_exposure" in rules


def test_strategy_concentration_limit():
    h = Harness(make_cfg(max_bot_weight=0.03))
    res = h.decide(MON, day1_scores())
    for o in res.orders:
        assert o.quantity * h.price <= 0.03 * CAPITAL + 1e-6
    assert any(e.get("rule") == "max_bot_weight" for e in res.events)


# ============================================================ attribution
def test_same_symbol_multi_bot_attribution():
    h = Harness()
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    vp = virtual_positions(h.sleeves)
    assert set(vp["AAPL"]) == {"SWING_V1", "INVESTOR_V1"} and set(vp["MSFT"]) == {"TREND_V1"}
    table = attribution_table(h.sleeves, {s: p.quantity for s, p in h.broker.get_positions().items()})
    assert table.loc["AAPL", "expected"] == vp["AAPL"]["SWING_V1"] + vp["AAPL"]["INVESTOR_V1"]
    assert (table["difference"] == 0).all()


def test_combined_position_matches_broker():
    h = Harness()
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    broker = {s: p.quantity for s, p in h.broker.get_positions().items()}
    assert broker == aggregate_positions(h.sleeves)
    rep = h.reconcile()
    assert rep.ok, rep.discrepancies
    assert h.gate.trading_enabled


def test_conflicting_bot_signals():
    h = Harness()
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    swing_qty = h.sleeves["SWING_V1"].quantity("AAPL")
    inv_qty = h.sleeves["INVESTOR_V1"].quantity("AAPL")
    # new week: SWING wants out of AAPL, INVESTOR still bullish, TREND wants AAPL too
    res = h.decide(NEXT_MON, {"SWING_V1": {"AAPL": EXIT_SCORE}, "INVESTOR_V1": {"AAPL": BUY_SCORE},
                              "TREND_V1": {"AAPL": BUY_SCORE, "MSFT": BUY_SCORE}})
    conflict = [e for e in res.events if e["event"] == "SIGNAL_CONFLICT"]
    assert conflict and conflict[0]["symbol"] == "AAPL"
    sells = [o for o in res.orders if o.side == Side.SELL and o.symbol == "AAPL"]
    assert len(sells) == 1 and sells[0].meta["bot_id"] == "SWING_V1" and sells[0].quantity == swing_qty
    sig = {(x.bot_id, x.symbol): x.signal_type for x in res.signals}
    assert sig[("SWING_V1", "AAPL")] == SignalType.EXIT
    assert sig[("INVESTOR_V1", "AAPL")] == SignalType.HOLD           # already holds: no add-on, no sell
    assert not [o for o in res.orders if o.meta["bot_id"] == "INVESTOR_V1" and o.side == Side.SELL]
    assert h.sleeves["INVESTOR_V1"].quantity("AAPL") == inv_qty


def test_swing_exit_does_not_close_investor_position():
    h = Harness()
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    inv_qty = h.sleeves["INVESTOR_V1"].quantity("AAPL")
    assert inv_qty > 0 and h.sleeves["SWING_V1"].quantity("AAPL") > 0
    res = h.decide(TUE, {"SWING_V1": {"AAPL": EXIT_SCORE}})
    h.submit(res)
    h.fill_at_open(WED)
    assert h.sleeves["SWING_V1"].quantity("AAPL") == 0
    assert h.sleeves["INVESTOR_V1"].quantity("AAPL") == inv_qty
    assert h.broker.get_positions()["AAPL"].quantity == inv_qty
    assert h.reconcile().ok


# ================================================================ failures
def test_reconciliation_failure_blocks_trading():
    h = Harness()
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    h.broker.positions["AAPL"].quantity += 7                      # broker disagrees with our books
    rep = h.reconcile()
    assert not rep.ok and not h.gate.trading_enabled
    d = [x for x in rep.discrepancies if x.get("symbol") == "AAPL"][0]
    assert set(d["by_bot"]) == {"SWING_V1", "INVESTOR_V1"}           # mismatch shows which bots' books
    res = h.decide(NEXT_MON, {"SWING_V1": {"NVDA": BUY_SCORE}, "TREND_V1": {"QQQ": BUY_SCORE},
                              "INVESTOR_V1": {"NVDA": BUY_SCORE}})
    assert res.orders == [] and res.rejected
    assert all("trading disabled" in (o.reject_reason or "") for o in res.rejected)
    n = len(h.broker.get_orders())
    with pytest.raises(TradingDisabledError):
        h.om.submit(res.rejected[0])
    assert len(h.broker.get_orders()) == n


def test_bot_failure_does_not_corrupt_other_bot_state():
    h = Harness()
    before = {b: (p.cash, dict(p.positions)) for b, p in h.sleeves.items()}
    res = h.decide(MON, day1_scores(), broken={"SWING_V1"})
    assert "SWING_V1" in res.disabled_bots
    assert any(e["event"] == "BOT_DISABLED" and e["bot_id"] == "SWING_V1" for e in res.events)
    assert h.gate.trading_enabled                                # account NOT halted
    assert {o.meta["bot_id"] for o in res.orders} == {"TREND_V1", "INVESTOR_V1"}
    assert h.sleeves["SWING_V1"].cash == before["SWING_V1"][0] and not h.sleeves["SWING_V1"].positions
    h.submit(res)
    h.fill_at_open(TUE)
    assert h.sleeves["TREND_V1"].quantity("MSFT") > 0 and h.sleeves["INVESTOR_V1"].quantity("AAPL") > 0
    assert h.sleeves["SWING_V1"].cash == before["SWING_V1"][0]
    assert h.reconcile().ok


def test_disabled_bot_still_counts_in_account_risk_and_keeps_stops():
    h = Harness(make_cfg(max_symbol_weight=0.07))
    h.submit(h.decide(MON, day1_scores()))
    h.fill_at_open(TUE)
    swing_val = h.sleeves["SWING_V1"].positions_value
    assert swing_val > 0
    # SWING disabled (e.g. model failed): its AAPL still uses the account-wide AAPL cap
    bots = h.bots
    runtimes = {b: BotRuntime(bot, h.signal_frame(b, NEXT_MON, {"AAPL": BUY_SCORE}), lambda d: h.features,
                              h.sleeves[b], h.risk[b], StopLossManager.from_config(bot.risk_cfg.stops))
                for b, bot in bots.items() if b != "SWING_V1"}
    runtimes["SWING_V1"] = BotRuntime.disabled(bots["SWING_V1"], h.sleeves["SWING_V1"], h.risk["SWING_V1"],
                                               StopLossManager.from_config(bots["SWING_V1"].risk_cfg.stops),
                                               "model failed")
    cyc = TradingCycle(runtimes, h.prm, OrderValidator.from_config(h.cfg.risk), h.gate, h.cfg.risk.safety,
                       set(SYMS), lambda: h.unallocated, "paper", "T", sleeves=h.sleeves)
    cyc._prev_date = TUE
    bars = {s: h.bar() for s in SYMS}
    res = cyc.run(MarketState(NEXT_MON, bars, {s: NEXT_MON for s in SYMS}, {s: h.price for s in SYMS}))
    assert "SWING_V1" in res.disabled_bots
    aapl_buys = sum(o.quantity * h.price for o in res.orders if o.symbol == "AAPL" and o.side == Side.BUY)
    held = sum(p.quantity("AAPL") for p in h.sleeves.values()) * h.price      # valued at the decision price
    assert held + aapl_buys <= 0.07 * CAPITAL + 1e-6               # SWING's shares were counted
    # stop-loss still protects the disabled bot's position
    h.price = 80.0                                                 # -20%: below SWING's 6% fixed stop
    for p in h.sleeves.values():
        p.mark_to_market({s: h.price for s in SYMS})
    bars = {s: h.bar() for s in SYMS}
    res2 = cyc.run(MarketState(NEXT_TUE, bars, {s: NEXT_TUE for s in SYMS}, {s: 100.0 for s in SYMS}))
    assert [o for o in res2.orders if o.meta["bot_id"] == "SWING_V1" and o.side == Side.SELL]


def test_portfolio_risk_failure_blocks_all_orders(monkeypatch):
    h = Harness()

    def broken(*a, **k):
        raise ZeroDivisionError("portfolio risk bug")
    monkeypatch.setattr(MultiBotPortfolioManager, "review", broken)
    res = h.decide(MON, day1_scores())
    assert res.orders == [] and len(res.rejected) >= 3
    assert not h.gate.trading_enabled
    monkeypatch.undo()
    res2 = h.decide(TUE, {"SWING_V1": {"NVDA": BUY_SCORE}})
    assert res2.orders == []                                       # halt persists until recovery


def test_signal_review_failure_blocks_all_orders(monkeypatch):
    h = Harness()
    monkeypatch.setattr(MultiBotPortfolioManager, "review_signals",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug")))
    res = h.decide(MON, day1_scores())
    assert res.orders == [] and not h.gate.trading_enabled


# ============================================================ look-ahead
@pytest.mark.parametrize("cutoff", ["2019-05-15", "2019-05-17", "2020-02-26"])
def test_future_data_does_not_change_historical_signal(bots, prices, cutoff):
    t = pd.Timestamp(cutoff)
    for b in bots.values():
        full = b.compute_features(prices)
        trunc = b.compute_features({s: df.loc[:t] for s, df in prices.items()})
        p_full = pd.Series(0.7, index=full.index)                 # same model output on both
        p_trunc = pd.Series(0.7, index=trunc.index)
        s_full = b.signal_frame(full, p_full).on(t)
        s_trunc = b.signal_frame(trunc, p_trunc).on(t)
        pd.testing.assert_frame_equal(s_full.sort_index(), s_trunc.sort_index(), check_dtype=False,
                                      obj=f"{b.bot_id} signal at {cutoff}")
        f_full = full.xs(t, level="date")[b.ml_features].sort_index()
        f_trunc = trunc.xs(t, level="date")[b.ml_features].sort_index()
        pd.testing.assert_frame_equal(f_full, f_trunc, obj=f"{b.bot_id} features at {cutoff}")


# ===================================================== final acceptance §24
def test_final_acceptance_scenario():
    h = Harness(make_cfg(max_symbol_weight=0.07))
    cfg = h.cfg
    # --- step 1: SWING BUY AAPL, TREND BUY MSFT, INVESTOR BUY AAPL
    res = h.decide(MON, day1_scores())
    assert {(o.meta["bot_id"], o.symbol) for o in res.orders} == {
        ("SWING_V1", "AAPL"), ("TREND_V1", "MSFT"), ("INVESTOR_V1", "AAPL")}
    # 5. every order passed BOTH risk levels (and is a registered risk approval)
    for o in res.orders:
        rec = o.meta["audit"]
        assert rec["risk_status"] in ("PASS", "RESIZED") and rec.get("portfolio_status") in ("PASS", "RESIZED")
        assert RiskEngine.verify(o)
    results = h.submit(res)
    assert all(r.submitted for r in results)
    h.fill_at_open(TUE)
    # 1. each bot respects its capital limit
    for b, bot in h.bots.items():
        p = h.sleeves[b]
        assert p.cash >= 0 and p.positions_value <= CAPITAL * bot.capital_allocation + 1e-6
        for pos in p.positions.values():
            assert pos.market_value <= bot.risk_cfg.limits.max_position_weight * p.equity + 1e-6
    # 2. aggregate AAPL exposure respects the account limit (resized, not blocked). The limit is
    #    enforced at the decision price; execution costs then move the fill price slightly.
    aapl_val = sum(p.quantity("AAPL") for p in h.sleeves.values()) * 100.0
    assert aapl_val <= cfg.risk.portfolio.max_symbol_weight * CAPITAL + 1e-6
    assert any(e.get("rule") == "max_symbol_weight" for e in res.events)
    # 3. virtual AAPL positions remain attributed separately
    vp = virtual_positions(h.sleeves)
    assert set(vp["AAPL"]) == {"SWING_V1", "INVESTOR_V1"}
    # 4. broker AAPL position == sum of strategy positions
    assert h.broker.get_positions()["AAPL"].quantity == sum(vp["AAPL"].values())
    assert h.reconcile().ok

    # --- step 2: SWING EXIT AAPL
    inv = vp["AAPL"]["INVESTOR_V1"]
    h.submit(h.decide(TUE, {"SWING_V1": {"AAPL": EXIT_SCORE}}))
    h.fill_at_open(WED)
    assert h.sleeves["SWING_V1"].quantity("AAPL") == 0                       # SWING allocation closes
    assert h.sleeves["INVESTOR_V1"].quantity("AAPL") == inv                  # INVESTOR intact
    assert h.broker.get_positions()["AAPL"].quantity == inv
    assert h.reconcile().ok and h.gate.trading_enabled

    # --- step 3: broker reconciliation failure -> ALL new orders blocked
    h.broker.positions["MSFT"].quantity -= 3
    assert not h.reconcile().ok and not h.gate.trading_enabled
    res3 = h.decide(NEXT_MON, {"SWING_V1": {"NVDA": BUY_SCORE}, "TREND_V1": {"QQQ": BUY_SCORE},
                               "INVESTOR_V1": {"NVDA": BUY_SCORE}})
    assert res3.orders == [] and len(res3.rejected) >= 3
    n_orders = len(h.broker.get_orders())
    for o in res3.rejected:
        assert o.status == OrderStatus.REJECTED
    assert len(h.broker.get_orders()) == n_orders


# ============================================================= interfaces
def test_trading_signal_object(bots):
    b = bots["TREND_V1"]
    s = b.make_signal("MSFT", pd.Timestamp("2024-01-08"), "BUY", 0.81, float("nan"), "test")
    assert isinstance(s, TradingSignal)
    assert s.signal_type == SignalType.BUY and s.model_probability is None and s.signal_score == 0.81
    assert s.bot_id == "TREND_V1" and s.timeframe == "daily" and s.expected_horizon == (20, 60)
    assert s.model_version == "trend-ml-1" and s.feature_set_version == b.feature_version
    assert s.signal_id == TradingSignal.make_id("TREND_V1", "MSFT", pd.Timestamp("2024-01-08"))
    assert SignalType.from_action("garbage") == SignalType.NO_TRADE
    inv = bots["INVESTOR_V1"].make_signal("AAPL", pd.Timestamp("2024-01-08"), "HOLD", 0.5, 0.6)
    assert inv.timeframe == "weekly" and inv.expected_horizon == (63, 252)
    d = s.to_dict()
    assert d["signal_type"] == "BUY" and d["data_timestamp"] == "2024-01-08"


def test_cycle_emits_signals_with_data_timestamp():
    h = Harness()
    res = h.decide(MON, day1_scores())
    assert {s.bot_id for s in res.signals} == set(h.bots)
    buy = [s for s in res.signals if s.signal_type == SignalType.BUY]
    assert {(s.bot_id, s.symbol) for s in buy} == {("SWING_V1", "AAPL"), ("TREND_V1", "MSFT"),
                                                   ("INVESTOR_V1", "AAPL")}
    assert all(s.data_timestamp == MON for s in res.signals)
    for o in res.orders:
        assert o.meta["signal_id"] == TradingSignal.make_id(o.meta["bot_id"], o.symbol, MON)
    # weekly bots do not regenerate signals within the same week
    res2 = h.decide(TUE, {})
    assert {s.bot_id for s in res2.signals} <= {"SWING_V1"}


def test_bot_contracts_reject_misconfiguration():
    with pytest.raises(ConfigurationError, match="not part of this bot"):
        build_bots(load_config(overrides={"bots": {"bots": {"SWING_V1": {"components": {
            "lt": {"type": "long_term_momentum", "weight": 0.1, "params": {"long_window": 52}}}}}}}))
    with pytest.raises(ConfigurationError, match="horizon"):
        build_bots(load_config(overrides={"bots": {"bots": {"TREND_V1": {"ml": {"target_horizon_days": 5}}}}}))
    with pytest.raises(ConfigurationError, match="fundamentals"):
        build_bots(load_config(overrides={"bots": {"bots": {"INVESTOR_V1": {"fundamentals": {"enabled": True}}}}}))
    with pytest.raises(ConfigurationError, match="unknown bot"):
        build_bots(load_config(), only=["LONGTERM_V1"])


def test_one_config_file_per_bot():
    from src.config import CONFIG_DIR
    files = sorted(p.name for p in (CONFIG_DIR / "bots").glob("*.yaml"))
    assert files == ["investor.yaml", "swing.yaml", "trend.yaml"]
    cfg = load_config()
    assert set(cfg.bots.bots) == {"SWING_V1", "TREND_V1", "INVESTOR_V1"}
    assert cfg.bots.bots["INVESTOR_V1"].timeframe == "weekly"                # "1w" alias
    assert cfg.bots.bots["INVESTOR_V1"].holding_period.in_trading_days() == (63, 252)
    assert not cfg.bots.bots["INVESTOR_V1"].fundamentals.enabled


def test_multi_bot_overview_table():
    from src.dashboard.data import OVERVIEW_ROWS, active_signal_counts, multi_bot_overview
    rows = []
    for i, d in enumerate(pd.bdate_range("2024-01-01", periods=5)):
        for b, e in (("SWING_V1", 30_000 + 100 * i), ("TREND_V1", 40_000 - 50 * i), ("ACCOUNT", 70_000 + 50 * i)):
            rows.append({"date": d, "bot_id": b, "equity": e, "cash": 1000.0, "positions_value": e - 1000,
                         "n_positions": 2, "n_trades": i})
    ov = multi_bot_overview(pd.DataFrame(rows), {"SWING_V1": 0.3, "TREND_V1": 0.4},
                            active_signal_counts([{"bot_id": "SWING_V1", "signal_type": "BUY"},
                                                  {"bot_id": "TREND_V1", "signal_type": "HOLD"}]))
    assert list(ov.index) == OVERVIEW_ROWS
    assert list(ov.columns) == ["SWING_V1", "TREND_V1", "TOTAL PORTFOLIO"]
    assert ov.at["Total P&L", "SWING_V1"] == 400 and ov.at["Daily P&L", "TREND_V1"] == -50
    assert ov.at["Capital allocation", "TOTAL PORTFOLIO"] == pytest.approx(0.7)
    assert ov.at["Active signals", "SWING_V1"] == 1 and ov.at["Active signals", "TREND_V1"] == 0
    assert "accuracy" not in " ".join(ov.index).lower()


def test_multi_bot_metrics_measure_diversification():
    from src.backtesting.metrics import multi_bot_metrics
    idx = pd.bdate_range("2024-01-01", periods=120, name="date")
    rng = np.random.default_rng(3)
    a = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    b = 40_000 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    be = {"A": pd.DataFrame({"equity": a}, index=idx), "B": pd.DataFrame({"equity": b}, index=idx)}
    acct = pd.DataFrame({"equity": a + b + 30_000, "positions_value": (a + b) * 0.5}, index=idx)
    m = multi_bot_metrics(acct, be, {"A": 0.3, "B": 0.4})
    assert abs(m["avg_pairwise_bot_correlation"]) < 0.3            # independent random walks
    assert m["diversification_ratio"] > 1.0                        # measured, not assumed
    assert 0 < m["capital_utilisation"] <= 1.5 and 0 < m["avg_gross_exposure"] < 1


def test_trend_candidate_rule_is_explicit():
    from src.bots.trend.strategy import bullish_structure
    df = pd.DataFrame({"close": [110, 110, 90, np.nan], "sma_50": [105, 95, 95, 100],
                       "sma_200": [100, 100, 100, 90], "ret_60d": [0.05, 0.05, 0.05, 0.05]})
    assert bullish_structure(df).tolist() == [True, False, False, False]
    assert bullish_structure(df, momentum_threshold=0.10).tolist() == [False, False, False, False]


def test_paper_history_window_is_long_enough_for_every_bot(cfg):
    """Regression: paper runs downloaded 900 days, too short for the investor's
    200-week market-trend feature, so INVESTOR_V1 could never score (all HOLD)."""
    from src.data.provider import SyntheticProvider
    end = pd.Timestamp("2026-10-16")
    start = end - pd.Timedelta(days=cfg.settings.data.paper_history_days)
    p = SyntheticProvider(seed=7)
    syms = set(cfg.settings.universe.symbols) | {cfg.settings.universe.benchmark}
    prices = {s: p.get_history(s, str(start.date()), str(end.date())) for s in syms}
    for b in build_bots(cfg).values():
        panel = b.compute_features(prices)
        last = panel.xs(panel.index.get_level_values("date").max(), level="date")
        cols = b.ml_features + b.signal_features
        assert not [c for c in cols if last[c].isna().all()], b.bot_id
