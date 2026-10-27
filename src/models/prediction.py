"""Model registry + safe inference.

Artifact layout (models/artifacts/<BOT_ID>/<model_version>/):
    model.joblib     the fitted sklearn Pipeline
    metadata.json    bot_id, model_version, feature_set_version, ordered
                     feature list, target definition, periods, metrics,
                     sha256 of model.joblib

Loading refuses: a missing file, a checksum mismatch (corruption), a wrong
model version, or an incompatible feature-set version. Inference refuses a
wrong column set/order and validates every probability. Any failure raises
ModelInferenceError -> the bot's ML component is NaN -> NO TRADE. A model
failure never becomes a default "buy" and never reuses old predictions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.errors import FeatureSchemaError, ModelCompatibilityError, ModelInferenceError
from src.safety.validation import assert_feature_schema, validate_probabilities


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_METADATA = ("bot_id", "model_id", "model_version", "feature_set_version", "features", "training_start",
                     "training_end", "validation_period", "target_definition", "algorithm", "hyperparameters",
                     "created_at", "sha256")


def _hyperparameters(model) -> dict:
    """JSON-safe hyperparameters of the final estimator (pipelines unwrapped)."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    try:
        params = est.get_params(deep=False)
    except Exception:  # noqa: BLE001 - mocks / custom models: record what we can
        return {"estimator": type(est).__name__}
    out: dict = {"estimator": type(est).__name__}
    for k, v in params.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
    return out


def save_model(model, directory: str | Path, *, bot_id: str, model_version: str, feature_set_version: str,
               features: list[str], target: dict, periods: dict, metrics: dict, model_name: str,
               model_id: str | None = None, hyperparameters: dict | None = None) -> Path:
    """Persist a model with the metadata required to load it safely.

    training_start/end describe the data the SAVED model was fitted on
    (train + validation after selection); validation_period is where the
    algorithm was chosen; the test period is reported in metrics only."""
    d = Path(directory) / bot_id / model_version
    d.mkdir(parents=True, exist_ok=True)
    mpath = d / "model.joblib"
    joblib.dump(model, mpath)
    digest = _sha256(mpath)
    fit_window = periods.get("final_fit") or periods.get("train") or [None, None]
    val = periods.get("validation") or [None, None]
    meta = {"bot_id": bot_id, "model_id": f"{model_id or bot_id.lower()}-{model_version}-{digest[:8]}",
            "model_version": model_version, "model_name": model_name, "algorithm": model_name,
            "hyperparameters": hyperparameters if hyperparameters is not None else _hyperparameters(model),
            "feature_set_version": feature_set_version, "features": list(features),
            "target_definition": {**target, "label": "1 if forward return over horizon_days > threshold else 0"},
            "target": target, "training_start": fit_window[0], "training_end": fit_window[1],
            "validation_period": list(val), "periods": periods, "metrics": metrics, "sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return d


@dataclass
class LoadedModel:
    model: object
    meta: dict

    @property
    def features(self) -> list[str]:
        return self.meta["features"]


def load_model(directory: str | Path, bot_id: str, model_version: str,
               expected_feature_version: str | None = None) -> LoadedModel:
    d = Path(directory) / bot_id / model_version
    mpath, jpath = d / "model.joblib", d / "metadata.json"
    if not mpath.exists() or not jpath.exists():
        raise ModelInferenceError(f"model artifact not found: {d}")
    try:
        meta = json.loads(jpath.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelInferenceError(f"model metadata unreadable: {exc}") from exc
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise ModelCompatibilityError(f"model metadata incomplete (missing {missing}); retrain with "
                                      "scripts/train_model.py")
    if meta.get("model_version") != model_version or meta.get("bot_id") != bot_id:
        raise ModelCompatibilityError(f"artifact is {meta.get('bot_id')}/{meta.get('model_version')}, "
                                      f"expected {bot_id}/{model_version}")
    if _sha256(mpath) != meta.get("sha256"):
        raise ModelInferenceError("model file checksum mismatch (corrupted or modified)")
    if expected_feature_version is not None and meta.get("feature_set_version") != expected_feature_version:
        raise ModelCompatibilityError(f"model trained on features {meta.get('feature_set_version')!r}, "
                                      f"pipeline produces {expected_feature_version!r}")
    try:
        model = joblib.load(mpath)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure = corrupted model
        raise ModelInferenceError(f"model could not be loaded: {type(exc).__name__}: {exc}") from exc
    if not hasattr(model, "predict_proba"):
        raise ModelInferenceError("loaded object has no predict_proba")
    return LoadedModel(model, meta)


class Predictor:
    """Wraps a model with schema + output validation."""

    def __init__(self, model, features: list[str], feature_set_version: str, model_version: str = ""):
        self.model = model
        self.features = list(features)
        self.feature_set_version = feature_set_version
        self.model_version = model_version

    @classmethod
    def from_loaded(cls, lm: LoadedModel) -> "Predictor":
        return cls(lm.model, lm.features, lm.meta["feature_set_version"], lm.meta["model_version"])

    def predict(self, panel: pd.DataFrame, feature_set_version: str) -> pd.Series:
        """Probabilities for rows with complete features; NaN for rows that
        cannot be scored (they then produce NO TRADE, not a guess)."""
        if feature_set_version != self.feature_set_version:
            raise ModelCompatibilityError(f"feature version {feature_set_version!r} != model "
                                          f"{self.feature_set_version!r}")
        missing = [c for c in self.features if c not in panel.columns]
        if missing:
            raise FeatureSchemaError(f"panel missing model features {missing}")
        X = panel[self.features]
        assert_feature_schema(X.replace([np.inf, -np.inf], np.nan), self.features)
        X = X.astype(float)
        if np.isinf(X.to_numpy()).any():
            raise FeatureSchemaError("infinite feature values")
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=panel.index, name="ml_probability")
        if ok.any():
            try:
                raw = self.model.predict_proba(X[ok].to_numpy())
            except Exception as exc:  # noqa: BLE001 - inference failure is a typed error
                raise ModelInferenceError(f"inference failed: {type(exc).__name__}: {exc}") from exc
            raw = np.asarray(raw)
            if raw.ndim != 2 or raw.shape[1] != 2:
                raise ModelInferenceError(f"unexpected predict_proba shape {raw.shape}")
            out[ok] = validate_probabilities(raw[:, 1], int(ok.sum()))
        return out
