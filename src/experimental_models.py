"""Explicit Sprint 7 candidates; healthy-only fitting and registered inputs."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .config import AppConfig
from .cross_validation import FlightSplit
from .errors import DataError
from .feature_sets import DIRECT_BASELINE_PREDICTORS, FEATURE_SETS, regression_predictors, validate_regression_predictors
from .models import make_hgb, regression_frame
from .provenance import require_healthy_source
from .reporting import format_json


@dataclass(frozen=True)
class Candidate:
    model: str
    feature_set: str

    @property
    def name(self):
        return f"{self.model}_{self.feature_set}"

    @property
    def regression(self):
        return self.model in ("hgb", "rf")

    def predictors(self, config):
        if self.regression:
            return list(regression_predictors(self.feature_set, config.as_dict()["features"]["rolling_windows_seconds"]))
        if self.model not in ("if", "ocsvm") or self.feature_set != "direct":
            raise DataError("Unknown Sprint 7 candidate.")
        if tuple(config.as_dict()["features"]["baseline_predictors"]) != DIRECT_BASELINE_PREDICTORS:
            raise DataError("Direct baseline predictors must match the registry exactly.")
        return list(DIRECT_BASELINE_PREDICTORS)


def candidates(mode):
    if mode == "ablation":
        return [Candidate(model, features) for model in ("hgb", "rf") for features in FEATURE_SETS]
    if mode == "compare-models":
        return [Candidate("hgb", "battery_d"), Candidate("rf", "battery_d"), Candidate("if", "direct"), Candidate("ocsvm", "direct")]
    raise DataError("Unknown Sprint 7 experiment.")


def candidate_config(config, candidate):
    values = config.as_dict()
    if candidate.regression:
        values["features"]["operational_set"] = candidate.feature_set
        values["features"]["regression_predictors"] = candidate.predictors(config)
    return AppConfig(config.source, config.root, values)


def frame(rows, config, candidate):
    if candidate.regression:
        return regression_frame(rows, candidate_config(config, candidate))
    import pandas as pd
    names = candidate.predictors(config)
    if any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) for r in rows for k in names):
        raise DataError("Direct baseline inputs must be finite registered features.")
    return pd.DataFrame([[r[k] for k in names] for r in rows], columns=names, dtype=float)


def fit_candidate(rows, config, split, candidate):
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from threadpoolctl import threadpool_limits
    split.validate()
    require_healthy_source(rows, config.as_dict()["data"]["healthy_status"])
    if {r["NamaFile"] for r in rows} != set(split.train) or any(r.get("reference_ready") is not True or r.get("split_role", "train") != "train" for r in rows):
        raise DataError("Candidate fitting requires exactly assigned healthy-ready train flights.")
    x = frame(rows, config, candidate)
    settings = config.as_dict()["models"]
    scaler, indices = None, np.arange(len(rows))
    if candidate.model == "hgb":
        model = make_hgb(candidate_config(config, candidate))
    elif candidate.model == "rf":
        model = RandomForestRegressor(**settings["random_forest"], random_state=config.seed)
    elif candidate.model == "if":
        model = IsolationForest(**settings["isolation_forest"], random_state=config.seed)
    elif candidate.model == "ocsvm":
        params = dict(settings["one_class_svm"])
        cap = params.pop("max_train_samples")
        if len(rows) > cap:
            indices = np.sort(np.random.default_rng(config.seed).choice(len(rows), cap, replace=False))
        scaler = StandardScaler().fit(x)  # all healthy training, never validation/test
        model = OneClassSVM(**params)
    else:
        raise DataError("Unknown candidate estimator.")
    target = np.array([r["voltage_sag_actual"] for r in rows]) if candidate.regression else None
    if candidate.regression and (target.dtype.kind not in "fi" or not np.isfinite(target).all()):
        raise DataError("Regression target must be finite healthy voltage sag.")
    with threadpool_limits(limits=1, user_api="openmp"):
        if candidate.regression:
            model.fit(x, target)
        else:
            model.fit(scaler.transform(x)[indices] if scaler else x)
    identities = [{key: r[key] for key in ("NamaFile", "Inst", "TimeUS", "source_record")} for r in rows]
    provenance = dict(train_flights=list(split.train), ready_training_samples=len(rows), model_fit_samples=len(indices),
                      fit_sample_indices=indices.tolist(), fit_sample_identities=[identities[i] for i in indices],
                      training_identity_sha256=hashlib.sha256(format_json({"rows": identities}).encode()).hexdigest(),
                      scaler_fit_samples=len(rows) if scaler else 0, scaler_fit_scope="all healthy-ready train only" if scaler else None,
                      sample_selection="sorted uniform without replacement, numpy default_rng seed" if scaler else "all ready train",
                      synthetic_rows=0, includes_low_current=True,
                      low_current_training_samples=sum(r["Curr"] < config.as_dict()["activity"]["min_current_a"] for r in rows))
    return dict(engine="battery_v2_sprint7", format_version=1, candidate=candidate.name, model_name=candidate.model,
                feature_set=candidate.feature_set, regression=candidate.regression, model=model, scaler=scaler,
                predictors=candidate.predictors(config), config=config.as_dict(), split=split.as_dict(),
                random_seed=config.seed, training_provenance=provenance,
                model_parameters=model.get_params(), currtot_unit="unverified", unit_conversion_applied=False,
                score_definition="max(0, actual sag - predicted sag)" if candidate.regression else "-decision_function (higher is more abnormal; signed)",
                score_unit="V" if candidate.regression else "model_score_not_volts")


def score_candidate(bundle, rows, config, role, *, healthy_cache=None):
    from threadpoolctl import threadpool_limits
    if role not in ("validation", "test"):
        raise DataError("Sprint 7 exports validation/test scores only.")
    split = bundle["split"]
    if {r["NamaFile"] for r in rows} != set(split[role]):
        raise DataError("Scoring flight does not match bundle role.")
    candidate = Candidate(bundle["model_name"], bundle["feature_set"])
    names = candidate.predictors(config)
    if bundle["predictors"] != names:
        raise DataError("Bundle predictor order mismatch.")
    ready = [r for r in rows if r["reference_ready"]]
    if healthy_cache is not None:
        if not candidate.regression or len(rows) != len(healthy_cache):
            raise DataError("Prediction reuse requires aligned regression inputs.")
        for row, base in zip(rows, healthy_cache):
            if any(row[k] != base[k] for k in (*names, "NamaFile", "Inst", "TimeUS", "source_record", "reference_ready")):
                raise DataError("Synthetic regression predictors changed; cannot reuse prediction.")
        values = iter(r["predicted_voltage_sag"] for r in healthy_cache if r["reference_ready"])
    else:
        x = frame(ready, config, candidate)
        with threadpool_limits(limits=1, user_api="openmp"):
            if candidate.regression:
                prediction = bundle["model"].predict(x)
            else:
                prediction = -bundle["model"].decision_function(bundle["scaler"].transform(x) if bundle["scaler"] else x)
        if not np.isfinite(prediction).all():
            raise DataError("Nonfinite candidate prediction.")
        values = iter(float(v) for v in prediction)
    result = []
    for row in rows:
        predicted = residual = score = None
        if row["reference_ready"]:
            value = next(values)
            if candidate.regression:
                predicted = value
                residual = row["voltage_sag_actual"] - value
                score = max(0., residual)
            else:
                score = value
        result.append({**row, "split_role": role, "used_for_fit": False,
                       "used_for_threshold": role == "validation" and row["reference_ready"], "unseen_by_model": True,
                       "predicted_voltage_sag": predicted, "residual_signed_v": residual, "anomaly_score": score})
    return result


def load_candidate_bundle(path: Path):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("engine") != "battery_v2_sprint7" or bundle.get("format_version") != 1:
        raise DataError("Unsupported Sprint 7 bundle.")
    candidate = Candidate(bundle["model_name"], bundle["feature_set"])
    config = AppConfig(path, path.parent, bundle["config"])
    expected = candidate.predictors(config)
    if expected != bundle["predictors"] or candidate.name != bundle["candidate"]:
        raise DataError("Bundle candidate/schema mismatch.")
    types = {"hgb": HistGradientBoostingRegressor, "rf": RandomForestRegressor, "if": IsolationForest, "ocsvm": OneClassSVM}
    if not isinstance(bundle["model"], types[candidate.model]):
        raise DataError("Bundle estimator mismatch.")
    source = bundle["scaler"] if candidate.model == "ocsvm" else bundle["model"]
    if list(source.feature_names_in_) != expected or (candidate.model == "ocsvm" and not isinstance(source, StandardScaler)):
        raise DataError("Fitted feature schema mismatch.")
    split = FlightSplit(*(tuple(bundle["split"][k]) for k in ("train", "validation", "test")))
    split.validate()
    if bundle["threshold"]["source_role"] != "validation" or set(bundle["threshold"]["source_flights"]) != set(split.validation):
        raise DataError("Bundle threshold provenance mismatch.")
    return bundle
