"""Strategy behaviour on controlled, deterministic data + regression snapshot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.strategies.base import BUY, EXIT, HOLD, REDUCE, SignalFrame
from src.strategies.breakout import BreakoutStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from tests.conftest import make_bars
from tests.fixtures.make_strategy_regression import compute as compute_regression
from tests.fixtures.regression_data import regression_prices


@pytest.fixture
def reg_panel():
    return FeaturePipeline().compute(regression_prices())


def at(series, date, sym):
    return series.loc[(pd.Timestamp(date), sym)]


def test_buy_signal(reg_panel):
    s = TrendFollowingStrategy().score(reg_panel)
    d = reg_panel.index.get_level_values("date").unique()[290]  # inside the steady uptrend
    assert at(s, d, "UPDN") == 1.0
    sf = TrendFollowingStrategy().signal_frame(reg_panel)
    assert sf.decide(at(s, d, "UPDN"), held=False) == BUY


def test_exit_signal(reg_panel):
    s = TrendFollowingStrategy().score(reg_panel)
    d = reg_panel.index.get_level_values("date").unique()[340]   # during the crash
    assert at(s, d, "UPDN") < 1.0
    sf = TrendFollowingStrategy().signal_frame(reg_panel)
    assert sf.decide(at(s, d, "UPDN"), held=True) == EXIT


def test_hold_signal_stateful_mean_reversion():
    closes = list(np.linspace(100, 160, 260)) + [150, 142, 136, 133, 134, 136, 139, 143, 147, 152, 156]
    df = make_bars(closes, start="2019-01-01")
    panel = FeaturePipeline().compute({"X": df})
    s = MeanReversionStrategy({"rsi_threshold": 35, "zscore_threshold": -1.5}).score(panel).xs("X", level="symbol")
    entry = s[s == 1.0].index.min()
    assert entry is not pd.NaT and entry >= df.index[260]
    after = s.loc[entry:]
    # stays 1 (HOLD) until close >= SMA20, then exits with 0
    assert (after.iloc[:2] == 1.0).all()
    assert 0.0 in set(after.round(6))


def test_insufficient_history_gives_no_signal():
    df = make_bars(np.linspace(100, 130, 150))                   # < 200 bars: no SMA200
    panel = FeaturePipeline().compute({"X": df})
    s = TrendFollowingStrategy().score(panel)
    assert s.isna().all()
    sf = TrendFollowingStrategy().signal_frame(panel)
    assert sf.decide(s.iloc[-1], held=False) == HOLD
    assert sf.decide(s.iloc[-1], held=True) == HOLD               # missing data never forces an exit either


def test_decide_matrix():
    sf = SignalFrame("x", pd.DataFrame({"score": [0.5]}), 0.6, 0.4, 0.25)
    assert sf.decide(0.7, False) == BUY and sf.decide(0.5, False) == HOLD
    assert sf.decide(0.5, True) == HOLD and sf.decide(0.3, True) == REDUCE
    assert sf.decide(0.3, True, reduced=True) == HOLD and sf.decide(0.1, True) == EXIT
    assert sf.decide(float("nan"), True) == HOLD and sf.decide(None, False) == HOLD


def test_breakout_uses_prior_high_not_today(reg_panel):
    s = BreakoutStrategy({"lookback": 55, "exit_lookback": 20, "volume_ratio_min": 1.2}).score(reg_panel)
    assert s.notna().any()
    # by construction prior_high excludes today, so "close > prior high" is observable at the close
    ph = reg_panel["prior_high_55"].xs("WAVE", level="symbol")
    hi = reg_panel["high"].xs("WAVE", level="symbol")
    assert (ph.dropna() == hi.rolling(55).max().shift(1).dropna()).all()


def test_momentum_is_cross_sectional_and_bounded(reg_panel):
    s = MomentumStrategy({"top_n": 1}).score(reg_panel).dropna()
    assert ((s >= 0) & (s <= 1)).all()
    top = s.groupby(level="date").apply(lambda x: (x > 0.5).sum())
    assert (top <= 1).all()                                       # only top_n can exceed 0.5


def test_ensemble_weighting_and_required_component(reg_panel):
    from src.config import EnsembleConfig
    strats = {"trend_following": TrendFollowingStrategy(), "momentum": MomentumStrategy({"top_n": 1})}
    cfgE = EnsembleConfig(weights={"trend_following": 0.75, "momentum": 0.25})
    comps = EnsembleStrategy(strats, cfgE).component_scores(reg_panel)
    combined = EnsembleStrategy(strats, cfgE).combine(comps)
    row = comps.dropna().iloc[-1]
    assert combined.loc[comps.dropna().index[-1]] == pytest.approx(0.75 * row["trend_following"] +
                                                                   0.25 * row["momentum"])
    comps["ml"] = np.nan
    req = EnsembleStrategy(strats, EnsembleConfig(weights={"trend_following": 0.5, "momentum": 0.2, "ml": 0.3}),
                           required_components=("ml",))
    assert req.combine(comps).isna().all()                        # required ML missing -> no score


def test_weights_come_from_config_not_code():
    import inspect

    from src.strategies import ensemble
    src = inspect.getsource(ensemble)
    assert "0.30" not in src and "0.25" not in src


def test_strategy_regression_snapshot():
    """Detects UNINTENDED behaviour changes. If a change is intentional, rerun
    tests/fixtures/make_strategy_regression.py and review the diff."""
    expected = json.loads((Path(__file__).parent / "fixtures" / "strategy_regression.json").read_text())
    got = compute_regression()
    for name, exp in expected.items():
        assert set(got[name]) == set(exp), f"{name}: evaluated rows changed"
        diffs = {k: (exp[k], got[name][k]) for k in exp if abs(exp[k] - got[name][k]) > 1e-6}
        assert not diffs, f"{name}: {len(diffs)} signals changed, e.g. {list(diffs.items())[:3]}"
