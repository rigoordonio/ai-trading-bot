"""Multi-bot architecture: independence, weekly candles without look-ahead, sleeves."""
import pandas as pd
import pytest

from src.bots.bot import build_bots
from src.config import load_config
from src.errors import ConfigurationError
from src.features.pipeline import FeatureConfig
from src.features.weekly import WeeklyFeaturePipeline, to_weekly


@pytest.fixture
def bots(cfg):
    return build_bots(cfg)


def test_three_independent_bots(bots):
    assert set(bots) == {"SWING_V1", "TREND_V1", "INVESTOR_V1"}
    feats = {b: tuple(x.ml_features) for b, x in bots.items()}
    assert len(set(feats.values())) == 3                              # independent ML features
    assert len({x.ml_cfg.target_horizon_days for x in bots.values()}) == 3   # independent targets
    assert len({x.ml_cfg.model_version for x in bots.values()}) == 3
    assert len({x.feature_version for x in bots.values()}) == 3
    comps = {b: tuple(sorted(x.components)) for b, x in bots.items()}
    assert len(set(comps.values())) == 3                              # independent strategy logic
    assert bots["INVESTOR_V1"].weekly_pipeline is not None and bots["SWING_V1"].weekly_pipeline is None


def test_independent_risk_parameters_and_capital(bots):
    methods = {b: x.risk_cfg.position_sizing.method for b, x in bots.items()}
    assert methods == {"SWING_V1": "atr", "TREND_V1": "volatility", "INVESTOR_V1": "fixed_fractional"}
    assert bots["SWING_V1"].risk_cfg.stops.max_holding_days == 30
    assert bots["INVESTOR_V1"].risk_cfg.stops.trailing_atr_multiple is None
    # shared (non-overridden) settings still come from risk.yaml
    assert bots["SWING_V1"].risk_cfg.drawdown.max_portfolio_drawdown == load_config().risk.drawdown \
        .max_portfolio_drawdown
    assert sum(b.capital_allocation for b in bots.values()) <= 1.0


def test_over_allocation_rejected():
    with pytest.raises(ConfigurationError, match="allocate"):
        load_config(overrides={"bots": {"bots": {"SWING_V1": {"capital_allocation": 0.9}}}})


def test_weekly_bars_aggregate_correctly():
    idx = pd.bdate_range("2024-01-01", "2024-01-12", name="date")    # two full weeks
    df = pd.DataFrame({"open": range(1, 11), "high": [x + 1 for x in range(1, 11)],
                       "low": [x - 0.5 for x in range(1, 11)], "close": [x + 0.5 for x in range(1, 11)],
                       "adj_close": [x + 0.5 for x in range(1, 11)], "volume": 100.0, "dividends": 0.0,
                       "splits": 0.0}, index=idx).astype(float)
    w = to_weekly(df)
    assert list(w.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert w.iloc[0]["open"] == 1 and w.iloc[0]["close"] == 5.5 and w.iloc[0]["high"] == 6
    assert w.iloc[0]["volume"] == 500


@pytest.mark.parametrize("cutoff", ["2019-03-13", "2019-03-15", "2020-07-01"])  # Wed, Fri, Wed
def test_weekly_features_have_no_lookahead(prices, cutoff):
    pipe = WeeklyFeaturePipeline(FeatureConfig(return_windows=[1, 4, 13], sma_windows=[10], ema_windows=[],
                                               realized_vol_windows=[13], high_52w_window=52,
                                               breakout_window=13, extra_breakout_windows=[],
                                               periods_per_year=52))
    full = pipe.compute(prices)
    t = pd.Timestamp(cutoff)
    trunc = pipe.compute({s: df.loc[:t] for s, df in prices.items()})
    a = full.xs(t, level="date").sort_index()
    b = trunc.xs(t, level="date").sort_index()
    pd.testing.assert_frame_equal(a, b)
    # the week containing t is never visible on t itself
    assert (a["week_end"] < t).all()


def test_bot_signal_frames_carry_versions(bots, prices):
    b = bots["TREND_V1"]
    sf = b.signal_frame(b.compute_features(prices))
    assert sf.meta["bot_id"] == "TREND_V1" and sf.meta["model_version"] == "trend-ml-1"
    assert sf.meta["strategy_version"] == "1.0.0" and sf.meta["feature_version"] == b.feature_version


@pytest.mark.slow
def test_multi_bot_backtest_uses_independent_sleeves(cfg, prices):
    from src.backtesting.engine import BacktestEngine, BotInput
    bots = build_bots(cfg)
    inputs = []
    for bot in bots.values():
        panel = bot.compute_features(prices)
        p = pd.Series(0.9, index=panel.index)      # fake OOS predictions: only to exercise wiring
        inputs.append(BotInput(bot, panel, bot.signal_frame(panel, p)))
    res = BacktestEngine(cfg, prices, inputs, start="2018-01-01", end="2019-06-30").run()
    cap = cfg.settings.backtest.initial_capital
    for b, eq in res.bot_equity.items():
        assert eq["equity"].iloc[0] == pytest.approx(cap * bots[b].capital_allocation, rel=0.02)
    assert set(res.bot_metrics) == set(bots)
    total = sum(e["equity"] for e in res.bot_equity.values()) + cap * (1 - sum(b.capital_allocation
                                                                             for b in bots.values()))
    pd.testing.assert_series_equal(total, res.equity["equity"], check_names=False)
    if len(res.audit):
        assert set(res.audit["bot_id"]) <= set(bots)
