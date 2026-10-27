import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.features import technical as ta
from src.features.pipeline import FeaturePipeline, ml_feature_columns
from tests.conftest import make_bars


# ---------------------------------------------------------------- indicators
def test_sma_and_ema_known_values():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ta.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert np.isnan(ta.sma(s, 3).iloc[1])
    e = ta.ema(s, 3)  # alpha = 0.5, seeded at first value
    assert e.iloc[2] == pytest.approx(0.5 * 3 + 0.5 * (0.5 * 2 + 0.5 * 1))


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ta.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert ta.rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    noisy = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 300).cumsum())
    r = ta.rsi(noisy, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_matches_independent_loop_implementation():
    closes = pd.Series(100 + np.random.default_rng(5).normal(0, 1, 120).cumsum())
    n = 14
    d = closes.diff().to_numpy()
    ag = al = None
    expected = []
    for i, x in enumerate(d):
        if i == 0:
            expected.append(np.nan)
            continue
        g, lo = max(x, 0.0), max(-x, 0.0)
        ag = g if ag is None else ag + (g - ag) / n
        al = lo if al is None else al + (lo - al) / n
        expected.append(np.nan if i < n else 100 - 100 / (1 + ag / al))
    got = ta.rsi(closes, n)
    mask = ~np.isnan(expected)
    assert np.allclose(got.to_numpy()[mask], np.array(expected)[mask])


def test_macd_relationship():
    s = pd.Series(100 + np.random.default_rng(2).normal(0, 1, 200).cumsum())
    m = ta.macd(s)
    pd.testing.assert_series_equal(m["macd"], ta.ema(s, 12) - ta.ema(s, 26), check_names=False)
    assert np.allclose((m["macd"] - m["macd_signal"]).dropna(), m["macd_hist"].dropna())


def test_atr_true_range_with_gap():
    df = make_bars([10, 10, 20, 20])
    df.iloc[2, df.columns.get_loc("open")] = 19.9
    df.iloc[2, df.columns.get_loc("low")] = 19.8      # gap up: today's range is tiny
    tr = ta.true_range(df["high"], df["low"], df["close"])
    # TR must use the previous close (10) because of the gap, not high-low (0.4)
    assert tr.iloc[2] == pytest.approx(df["high"].iloc[2] - 10)


def test_realized_vol_constant_returns_is_zero():
    s = pd.Series(100 * 1.01 ** np.arange(50))
    assert ta.realized_vol(s, 20).dropna().abs().max() < 1e-10


def test_bollinger_and_zscore():
    s = pd.Series(np.r_[np.full(19, 100.0), 110.0])
    bb = ta.bollinger(s, 20, 2.0)
    assert bb["zscore_20"].iloc[-1] == pytest.approx((110 - s.mean()) / s.std(ddof=0))
    assert bb["bb_upper"].iloc[-1] > bb["bb_mid"].iloc[-1] > bb["bb_lower"].iloc[-1]


def test_rolling_high_excludes_today():
    s = pd.Series([1.0, 2, 3, 10])
    assert ta.rolling_high(s, 3).iloc[-1] == 3.0          # today's 10 is not included
    assert ta.rolling_high(s, 3, include_today=True).iloc[-1] == 10.0


# ------------------------------------------------------------------ pipeline
def test_pipeline_columns_and_shape(prices):
    panel = FeaturePipeline().compute(prices)
    assert panel.index.names == ["date", "symbol"]
    required = ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
                "rsi_14", "macd", "macd_signal", "roc_10", "atr_14", "realized_vol_20", "realized_vol_60",
                "avg_volume_20", "volume_ratio", "volume_zscore", "dist_52w_high", "dist_sma_50", "dist_sma_200",
                "mkt_trend", "mkt_vol_20", "breadth_50", "risk_adj_mom_60", "prior_high_55"]
    missing = [c for c in required if c not in panel.columns]
    assert not missing
    last = panel.xs(panel.index.get_level_values("date").max(), level="date")
    assert last[required].notna().all().all()


def test_ml_features_exclude_price_levels(prices):
    cols = ml_feature_columns(FeaturePipeline().compute(prices))
    for bad in ("close", "sma_50", "ema_12", "atr_14", "avg_volume_20", "prior_high_55", "bb_upper"):
        assert bad not in cols
    for good in ("ret_20d", "rsi_14", "atr_pct", "dist_sma_200", "mkt_trend"):
        assert good in cols


@pytest.mark.parametrize("cutoff", ["2018-03-15", "2019-07-01", "2020-11-30"])
def test_no_lookahead_features_identical_on_truncated_history(prices, cutoff):
    """THE leakage test: features at date t computed with data up to t must
    equal features at t computed with the full history (which includes the
    future). Any difference means some feature peeked ahead."""
    pipe = FeaturePipeline()
    full = pipe.compute(prices)
    t = pd.Timestamp(cutoff)
    t = full.index.get_level_values("date")[full.index.get_level_values("date") <= t].max()
    truncated = pipe.compute({s: df.loc[:t] for s, df in prices.items()})
    a = full.xs(t, level="date").sort_index()
    b = truncated.xs(t, level="date").sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_future_shock_does_not_change_past_features(prices):
    pipe = FeaturePipeline()
    base = pipe.compute(prices)
    shocked = {s: df.copy() for s, df in prices.items()}
    t = shocked["AAPL"].index[600]
    for col in ("open", "high", "low", "close"):
        shocked["AAPL"].loc[shocked["AAPL"].index > t, col] *= 3.0
    after = pipe.compute(shocked)
    past = base.index.get_level_values("date") <= t
    pd.testing.assert_frame_equal(base[past], after[past])


def test_features_package_never_shifts_backwards():
    """Static guard: no shift(-n) anywhere in feature code."""
    for path in pathlib.Path("src/features").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "shift" and node.args:
                arg = node.args[0]
                neg = isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                assert not neg, f"negative shift (future data) in {path}"
