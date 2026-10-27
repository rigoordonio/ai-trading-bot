"""ML training with strict chronological discipline.

Target
    y_t = 1 if close_{t+h} / close_t - 1 > threshold else 0      (per symbol)
    The target LOOKS FORWARD by design; it is never used as a feature, and
    rows whose future is unknown get NaN (dropped from training).

Leakage controls
    * splits are chronological: train < validation < test, never random
    * PURGE: labels span h days, so the last h training dates would "see"
      validation prices. Those rows are dropped (same before the test set).
    * preprocessing (scaler) lives INSIDE an sklearn Pipeline, so it is fit
      on the training rows only, and re-fit per walk-forward window.
    * model choice / hyper-parameters are picked on VALIDATION; the test
      period is touched once, at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logging_setup import get_logger
from src.models.evaluation import classification_metrics

log = get_logger(__name__)


def make_target(panel: pd.DataFrame, horizon: int, threshold: float = 0.0, price_col: str = "close") -> pd.Series:
    """Forward-return label per (date, symbol); NaN when the future is unknown."""
    close = panel[price_col].unstack("symbol")
    fwd = close.shift(-horizon) / close - 1.0
    y = (fwd > threshold).astype(float).where(fwd.notna())
    return y.stack(future_stack=True).reindex(panel.index).rename("target")


def forward_returns(panel: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.Series:
    close = panel[price_col].unstack("symbol")
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.stack(future_stack=True).reindex(panel.index).rename(f"fwd_ret_{horizon}d")


def candidate_models(names: list[str], random_state: int = 42) -> dict[str, Pipeline]:
    out: dict[str, Pipeline] = {}
    for n in names:
        if n == "logistic_regression":
            est = LogisticRegression(C=0.1, max_iter=2000)
        elif n == "random_forest":
            est = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, n_jobs=-1,
                                         random_state=random_state)
        elif n == "gradient_boosting":
            est = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                 l2_regularization=1.0, random_state=random_state)
        elif n in ("lightgbm", "xgboost"):
            try:
                if n == "lightgbm":
                    from lightgbm import LGBMClassifier
                    est = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                         subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1)
                else:
                    from xgboost import XGBClassifier
                    est = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                        colsample_bytree=0.8, random_state=random_state, eval_metric="logloss")
            except ImportError:
                log.info("%s not installed: skipped", n)
                continue
        else:
            raise ValueError(f"unknown model '{n}'")
        out[n] = Pipeline([("scaler", StandardScaler()), ("model", est)])
    return out


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    fwd: pd.Series

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(panel: pd.DataFrame, features: list[str], horizon: int, threshold: float) -> Dataset:
    missing = [f for f in features if f not in panel.columns]
    if missing:
        raise KeyError(f"panel missing ML features {missing}")
    X = panel[features].astype(float).replace([np.inf, -np.inf], np.nan)
    y = make_target(panel, horizon, threshold)
    fwd = forward_returns(panel, horizon)
    ok = X.notna().all(axis=1) & y.notna()
    return Dataset(X[ok], y[ok], fwd[ok])


def date_slice(ds: Dataset, start, end, purge_days: int = 0, calendar: pd.DatetimeIndex | None = None) -> Dataset:
    """Rows with start <= date <= end. With ``purge_days`` the last N trading
    days before ``end`` are removed so labels cannot overlap the next period."""
    dates = ds.y.index.get_level_values("date")
    lo = pd.Timestamp(start) if start is not None else dates.min()
    hi = pd.Timestamp(end) if end is not None else dates.max()
    if purge_days > 0:
        cal = calendar if calendar is not None else pd.DatetimeIndex(sorted(set(dates)))
        cal = cal[cal <= hi]
        if len(cal) > purge_days:
            hi = cal[-purge_days - 1]
        else:
            hi = lo - pd.Timedelta(days=1)
    m = (dates >= lo) & (dates <= hi)
    return Dataset(ds.X[m], ds.y[m], ds.fwd[m])


@dataclass
class TrainResult:
    best_name: str
    model: Pipeline
    validation_scores: dict[str, dict]
    test_metrics: dict | None
    periods: dict[str, list[str | None]]
    features: list[str]
    n_train: int
    extra: dict = field(default_factory=dict)


def fit(model: Pipeline, ds: Dataset) -> Pipeline:
    if len(ds) < 100 or ds.y.nunique() < 2:
        raise ValueError(f"not enough training data ({len(ds)} rows, {ds.y.nunique()} classes)")
    model.fit(ds.X.to_numpy(), ds.y.to_numpy().astype(int))
    return model


def train_select_evaluate(panel: pd.DataFrame, features: list[str], horizon: int, threshold: float,
                          split: dict, model_names: list[str], selection_metric: str = "roc_auc",
                          purge_days: int | None = None, random_state: int = 42,
                          evaluate_test: bool = True) -> TrainResult:
    """Fit candidates on TRAIN, choose on VALIDATION, refit on TRAIN+VALIDATION,
    evaluate ONCE on TEST."""
    from sklearn.base import clone

    purge = horizon if purge_days is None else purge_days
    ds = build_dataset(panel, features, horizon, threshold)
    cal = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    (tr0, tr1), (va0, va1), (te0, te1) = split["train"], split["validation"], split["test"]
    train = date_slice(ds, tr0, tr1, purge, cal)
    val = date_slice(ds, va0, va1, purge, cal)
    scores: dict[str, dict] = {}
    for name, mdl in candidate_models(model_names, random_state).items():
        fitted = fit(mdl, train)
        p = fitted.predict_proba(val.X.to_numpy())[:, 1]
        scores[name] = classification_metrics(val.y.to_numpy(), p, val.fwd.to_numpy())
        log.info("validation %-20s auc=%.4f acc=%.4f", name, scores[name]["roc_auc"], scores[name]["accuracy"])
    if not scores:
        raise ValueError("no candidate models could be trained")
    best = max(scores, key=lambda n: np.nan_to_num(scores[n].get(selection_metric, -np.inf), nan=-np.inf))
    trval = date_slice(ds, tr0, va1, purge, cal)
    final = fit(clone(candidate_models([best], random_state)[best]), trval)
    importance = {}
    try:   # permutation importance on VALIDATION (model-agnostic, no test data used)
        from sklearn.inspection import permutation_importance

        sub = val.X.iloc[:5000]
        pi = permutation_importance(final, sub.to_numpy(), val.y.iloc[:5000].to_numpy().astype(int),
                                    n_repeats=3, random_state=random_state, scoring="roc_auc")
        importance = dict(sorted(zip(features, pi.importances_mean.tolist(), strict=True), key=lambda kv: -kv[1]))
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never affects trading
        log.warning("feature importance unavailable: %s", exc)
    test_metrics = None
    if evaluate_test:
        test = date_slice(ds, te0, te1, 0, cal)
        if len(test) > 0:
            pt = final.predict_proba(test.X.to_numpy())[:, 1]
            test_metrics = classification_metrics(test.y.to_numpy(), pt, test.fwd.to_numpy())
    periods = {"train": [tr0, tr1], "validation": [va0, va1], "test": [te0, te1],
               "final_fit": [tr0, str(trval.y.index.get_level_values('date').max().date()) if len(trval) else None]}
    return TrainResult(best, final, scores, test_metrics, periods, features, len(trval),
                       extra={"base_rate_train": float(train.y.mean()), "purge_days": purge,
                              "feature_importance": importance,
                              "threshold": threshold, "horizon": horizon})
