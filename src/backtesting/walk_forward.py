"""Walk-forward validation.

    |---- train ----|-- val --|-- test --|
              |---- train ----|-- val --|-- test --|        (step forward)

Per window: fit candidates on train (purged), pick on validation, refit on
train+validation (purged), predict the test window. Test windows never
overlap and are never used for any choice. The concatenated test-window
predictions are the ONLY predictions an ML backtest may use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.logging_setup import get_logger
from src.models.evaluation import classification_metrics
from src.models.training import build_dataset, candidate_models, date_slice, fit

log = get_logger(__name__)


@dataclass
class Window:
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


def make_windows(start, end, train_years=4, validation_years=1, test_years=1, step_years=1) -> list[Window]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    t0 = start
    while True:
        tr1 = t0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        va0 = tr1 + pd.Timedelta(days=1)
        va1 = va0 + pd.DateOffset(years=validation_years) - pd.Timedelta(days=1)
        te0 = va1 + pd.Timedelta(days=1)
        te1 = min(te0 + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), end)
        if te0 > end:
            break
        out.append(Window((t0, tr1), (va0, va1), (te0, te1)))
        t0 = t0 + pd.DateOffset(years=step_years)
    return out


@dataclass
class WalkForwardResult:
    predictions: pd.Series            # out-of-sample probabilities only
    windows: pd.DataFrame             # per-window choice + metrics
    oos_metrics: dict                 # aggregated across all test windows
    first_oos_date: pd.Timestamp | None


def walk_forward_predictions(panel: pd.DataFrame, features: list[str], horizon: int, threshold: float,
                             windows: list[Window], model_names: list[str], selection_metric: str = "roc_auc",
                             purge_days: int | None = None, random_state: int = 42) -> WalkForwardResult:
    purge = horizon if purge_days is None else purge_days
    ds = build_dataset(panel, features, horizon, threshold)
    cal = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    X_all = panel[features].astype(float).replace([np.inf, -np.inf], np.nan)
    preds, rows = [], []
    prev_test_end = None
    for w in windows:
        if prev_test_end is not None and w.test[0] <= prev_test_end:
            raise ValueError("walk-forward test windows overlap")
        train = date_slice(ds, *w.train, purge, cal)
        val = date_slice(ds, *w.validation, purge, cal)
        if len(train) < 200 or len(val) < 50 or train.y.nunique() < 2:
            log.warning("window %s skipped: not enough data", w.test)
            continue
        scores = {}
        for name, mdl in candidate_models(model_names, random_state).items():
            m = fit(mdl, train)
            scores[name] = classification_metrics(val.y.to_numpy(), m.predict_proba(val.X.to_numpy())[:, 1])
        best = max(scores, key=lambda n: np.nan_to_num(scores[n].get(selection_metric, -np.inf), nan=-np.inf))
        final = fit(clone(candidate_models([best], random_state)[best]), date_slice(ds, w.train[0], w.validation[1],
                                                                                    purge, cal))
        # Predict EVERY test-window row with complete features (labels not needed)
        dates = X_all.index.get_level_values("date")
        mask = (dates >= w.test[0]) & (dates <= w.test[1]) & X_all.notna().all(axis=1).to_numpy()
        if mask.any():
            p = final.predict_proba(X_all[mask].to_numpy())[:, 1]
            preds.append(pd.Series(p, index=X_all.index[mask]))
        test = date_slice(ds, *w.test, 0, cal)
        tm = classification_metrics(test.y.to_numpy(), final.predict_proba(test.X.to_numpy())[:, 1],
                                    test.fwd.to_numpy()) if len(test) and test.y.nunique() == 2 else {}
        rows.append({"train_start": w.train[0], "train_end": w.train[1], "val_end": w.validation[1],
                     "test_start": w.test[0], "test_end": w.test[1], "model": best,
                     "val_auc": scores[best].get("roc_auc"), "test_auc": tm.get("roc_auc"),
                     "test_ic": tm.get("ic_spearman"), "test_spread": tm.get("quintile_spread"), "n_test": len(test)})
        prev_test_end = w.test[1]
        log.info("WF window test %s..%s model=%s val_auc=%.3f test_auc=%s", w.test[0].date(), w.test[1].date(), best,
                 scores[best].get("roc_auc", np.nan), f"{tm.get('roc_auc', float('nan')):.3f}")
    pred = pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)
    pred.name = "ml_probability"
    oos = {}
    if len(pred):
        sub = date_slice(ds, pred.index.get_level_values("date").min(), pred.index.get_level_values("date").max())
        common = sub.y.index.intersection(pred.index)
        if len(common) and sub.y.loc[common].nunique() == 2:
            oos = classification_metrics(sub.y.loc[common].to_numpy(), pred.loc[common].to_numpy(),
                                         sub.fwd.loc[common].to_numpy())
    return WalkForwardResult(pred, pd.DataFrame(rows), oos,
                             pred.index.get_level_values("date").min() if len(pred) else None)
