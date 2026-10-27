"""ML: leakage controls, chronological splits, walk-forward boundaries, model safety."""
import json

import numpy as np
import pandas as pd
import pytest

from src.backtesting.walk_forward import make_windows, walk_forward_predictions
from src.errors import FeatureSchemaError, ModelCompatibilityError, ModelInferenceError
from src.features.pipeline import FeaturePipeline
from src.models.prediction import Predictor, load_model, save_model
from src.models.training import build_dataset, candidate_models, date_slice, fit, make_target, train_select_evaluate
from src.safety.validation import validate_probabilities
from tests.fakes import MockModel

FEATS = ["ret_5d", "ret_20d", "rsi_14", "atr_pct", "dist_sma_50", "volume_ratio"]


@pytest.fixture
def panel(prices):
    return FeaturePipeline().compute(prices)


# ---------------------------------------------------------------- targets
def test_target_looks_forward_and_is_nan_at_the_end(panel):
    y = make_target(panel, 5, 0.0)
    close = panel["close"].unstack("symbol")["AAPL"]
    t = close.index[100]
    assert y.loc[(t, "AAPL")] == float(close.iloc[105] / close.iloc[100] - 1 > 0)
    last5 = close.index[-5:]
    assert y.loc[(last5, "AAPL")].isna().all()


def test_target_threshold_not_zero_by_default_in_bots(cfg):
    from src.config import load_config
    c = load_config()
    assert all(b.ml.target_threshold > 0 for b in c.bots.bots.values() if b.ml)


def test_target_never_used_as_feature(panel):
    from src.features.pipeline import ml_feature_columns
    cols = ml_feature_columns(panel)
    assert not any(c.startswith(("target", "fwd_")) for c in cols)


# ----------------------------------------------------------------- splits
def test_purge_removes_overlapping_labels(panel):
    ds = build_dataset(panel, FEATS, 5, 0.0)
    cal = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    tr = date_slice(ds, "2017-01-01", "2018-12-31", purge_days=5, calendar=cal)
    last_train = tr.y.index.get_level_values("date").max()
    # the last training label (t + 5 sessions) must end on/before the split boundary
    pos = cal.get_loc(last_train)
    assert cal[pos + 5] <= pd.Timestamp("2018-12-31")
    assert cal[pos + 6] > pd.Timestamp("2018-12-31")


def test_chronological_split_never_random(panel):
    split = {"train": ["2016-01-01", "2018-06-30"], "validation": ["2018-07-01", "2019-06-30"],
             "test": ["2019-07-01", None]}
    tr = train_select_evaluate(panel, FEATS, 5, 0.0, split, ["logistic_regression"])
    assert tr.periods["train"] == split["train"]
    assert pd.Timestamp(tr.periods["final_fit"][1]) < pd.Timestamp("2019-07-01")   # test never used to fit
    assert tr.test_metrics is not None and 0 <= tr.test_metrics["roc_auc"] <= 1


def test_scaler_is_fitted_on_training_rows_only(panel):
    ds = build_dataset(panel, FEATS, 5, 0.0)
    train = date_slice(ds, "2016-01-01", "2017-12-31", 5)
    model = fit(candidate_models(["logistic_regression"])["logistic_regression"], train)
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, train.X.mean().to_numpy())
    full_mean = ds.X.mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean)     # would be equal if fitted on everything


def test_walk_forward_windows_are_chronological_and_disjoint(panel):
    wins = make_windows("2016-01-01", "2020-12-31", 2, 1, 1, 1)
    assert len(wins) == 2
    for w in wins:
        assert w.train[1] < w.validation[0] <= w.validation[1] < w.test[0] <= w.test[1]
    assert wins[0].test[1] < wins[1].test[0]
    wf = walk_forward_predictions(panel, FEATS, 5, 0.0, wins, ["logistic_regression"])
    d = wf.predictions.index.get_level_values("date")
    assert d.min() >= wins[0].test[0] and d.max() <= wins[-1].test[1]     # OOS only
    assert ((wf.predictions >= 0) & (wf.predictions <= 1)).all()
    assert list(wf.windows["test_start"]) == [w.test[0] for w in wins]


def test_overlapping_windows_rejected(panel):
    from src.backtesting.walk_forward import Window
    t = pd.Timestamp
    w1 = Window((t("2016-01-01"), t("2017-12-31")), (t("2018-01-01"), t("2018-06-30")),
                (t("2018-07-01"), t("2019-06-30")))
    w2 = Window((t("2016-06-01"), t("2018-05-31")), (t("2018-06-01"), t("2018-12-31")),
                (t("2019-01-01"), t("2019-12-31")))
    with pytest.raises(ValueError, match="overlap"):
        walk_forward_predictions(panel, FEATS, 5, 0.0, [w1, w2], ["logistic_regression"])


# ---------------------------------------------------------- model safety
def _saved(tmp_path, panel, version="v1", feature_version="fv1"):
    ds = build_dataset(panel, FEATS, 5, 0.0)
    m = fit(candidate_models(["logistic_regression"])["logistic_regression"], ds)
    save_model(m, tmp_path, bot_id="BOT", model_version=version, feature_set_version=feature_version,
               features=FEATS, target={}, periods={}, metrics={}, model_name="lr")
    return m


def test_missing_model(tmp_path):
    with pytest.raises(ModelInferenceError, match="not found"):
        load_model(tmp_path, "BOT", "v1")


def test_corrupted_model(tmp_path, panel):
    _saved(tmp_path, panel)
    f = tmp_path / "BOT" / "v1" / "model.joblib"
    f.write_bytes(f.read_bytes()[:-50] + b"garbage")
    with pytest.raises(ModelInferenceError, match="checksum"):
        load_model(tmp_path, "BOT", "v1")


def test_wrong_model_version_and_feature_version(tmp_path, panel):
    _saved(tmp_path, panel)
    meta = tmp_path / "BOT" / "v1" / "metadata.json"
    j = json.loads(meta.read_text())
    with pytest.raises(ModelCompatibilityError):
        load_model(tmp_path, "BOT", "v1", expected_feature_version="fv2")
    j["model_version"] = "v0"
    meta.write_text(json.dumps(j))
    with pytest.raises(ModelCompatibilityError):
        load_model(tmp_path, "BOT", "v1")


def test_feature_version_mismatch_at_inference(tmp_path, panel):
    _saved(tmp_path, panel)
    pred = Predictor.from_loaded(load_model(tmp_path, "BOT", "v1", "fv1"))
    with pytest.raises(ModelCompatibilityError):
        pred.predict(panel, "fv2")
    assert pred.predict(panel.tail(50), "fv1").dropna().between(0, 1).all()


def test_wrong_feature_schema(panel):
    p = Predictor(MockModel(), FEATS, "fv1")
    with pytest.raises(FeatureSchemaError):
        p.predict(panel.drop(columns=["rsi_14"]), "fv1")
    swapped = Predictor(MockModel(), list(reversed(FEATS)), "fv1")
    out = swapped.predict(panel.tail(10), "fv1")          # Predictor selects its OWN column order
    assert out.notna().any()
    from src.safety.validation import assert_feature_schema
    with pytest.raises(FeatureSchemaError, match="order|differ"):
        assert_feature_schema(panel[FEATS], list(reversed(FEATS)))
    bad = panel[FEATS].tail(5).copy()
    bad["rsi_14"] = "high"
    with pytest.raises(FeatureSchemaError):
        assert_feature_schema(bad, FEATS)


@pytest.mark.parametrize("out", [float("nan"), float("inf"), -0.1, 1.2])
def test_nan_and_invalid_probability(panel, out):
    with pytest.raises(ModelInferenceError):
        Predictor(MockModel(output=out), FEATS, "fv1").predict(panel.tail(20), "fv1")


def test_inference_failure(panel):
    with pytest.raises(ModelInferenceError, match="inference failed"):
        Predictor(MockModel(raise_exc=MemoryError("oom")), FEATS, "fv1").predict(panel.tail(20), "fv1")


def test_rows_with_nan_features_are_not_guessed(panel):
    sub = panel.tail(20).copy()
    sub.iloc[0, sub.columns.get_loc("rsi_14")] = np.nan
    out = Predictor(MockModel(output=0.9), FEATS, "fv1").predict(sub, "fv1")
    assert np.isnan(out.iloc[0]) and out.iloc[1:].eq(0.9).all()


def test_validate_probabilities_helper():
    assert validate_probabilities([0.0, 0.5, 1.0]).tolist() == [0.0, 0.5, 1.0]
    for bad in ([0.5, np.nan], [1.0001], [-1e-9]):
        with pytest.raises(ModelInferenceError):
            validate_probabilities(bad)
    with pytest.raises(ModelInferenceError):
        validate_probabilities([0.5], n_expected=2)


def test_model_failure_never_becomes_a_buy(cfg, prices):
    """MODEL FAILURE -> NO TRADE: no predictions => combined score NaN => HOLD."""
    from src.bots.bot import build_bots
    bot = build_bots(cfg, ["SWING_V1"])["SWING_V1"]
    panel = bot.compute_features(prices)
    sf = bot.signal_frame(panel, ml_predictions=None)
    assert sf.scores["score"].isna().all()
    last = panel.index.get_level_values("date").max()
    for _sym, row in sf.on(last).iterrows():
        assert sf.decide(row["score"], held=False) == "HOLD"
