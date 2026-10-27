"""Model evaluation: classification metrics AND trading-oriented metrics.

Accuracy alone is misleading: with a 55% base rate, predicting "up" always
scores 55%. What matters for trading is whether higher probabilities line
up with higher forward returns (ranking quality), so we also report:

    ic_spearman          rank correlation(probability, forward return)
    top_quintile_return  mean forward return of the top 20% probabilities
    quintile_spread      top-quintile minus bottom-quintile forward return
    top_quintile_hit     share of top-quintile picks with positive forward return
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score


def classification_metrics(y: np.ndarray, p: np.ndarray, fwd: np.ndarray | None = None,
                           threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "mean_probability": float(p.mean()) if len(p) else float("nan"),
    }
    if fwd is not None and len(fwd) >= 20:
        out.update(trading_metrics(p, np.asarray(fwd, dtype=float)))
    return out


def trading_metrics(p: np.ndarray, fwd: np.ndarray) -> dict:
    ok = np.isfinite(p) & np.isfinite(fwd)
    p, fwd = p[ok], fwd[ok]
    if len(p) < 20:
        return {}
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
    by_q = pd.Series(fwd).groupby(q.values).mean()
    top = fwd[q.values == 4]
    ic = spearmanr(p, fwd).statistic
    return {
        "ic_spearman": float(ic) if np.isfinite(ic) else 0.0,
        "top_quintile_return": float(by_q.get(4, np.nan)),
        "bottom_quintile_return": float(by_q.get(0, np.nan)),
        "quintile_spread": float(by_q.get(4, np.nan) - by_q.get(0, np.nan)),
        "top_quintile_hit": float((top > 0).mean()) if len(top) else float("nan"),
        "avg_forward_return": float(fwd.mean()),
        "quintile_returns": {int(k): float(v) for k, v in by_q.items()},
    }


def feature_importance(model, features: list[str]) -> pd.Series:
    est = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=features).sort_values(ascending=False)
