"""Healthy-only HGB Battery-D pilot; outer test remains reserved for Stage 6."""

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .config import AppConfig, project_path
from .cross_validation import FlightSplit, pilot_split
from .detectors import apply_single_path, calibrate_threshold, residual_score
from .errors import ConfigError, DataError
from .feature_sets import BATTERY_COLUMNS, feature_registry, validate_regression_predictors
from .features import feature_columns, generate_features
from .preprocessing import PROCESSED_COLUMNS, _csv
from .provenance import require_healthy_source
from .reporting import format_json

SCORE_COLUMNS = ("NamaFile", "Inst", "TimeUS", "source_record", "segment_id", "elapsed_time", "dt_seconds",
                 "split_role", "used_for_fit", "used_for_threshold", "unseen_by_model", "Curr", "Volt", "V_ref",
                 "reference_ready", "voltage_sag_actual", "predicted_voltage_sag", "residual_signed_v", "anomaly_score",
                 "anomaly_raw", "anomaly_duration_seconds", "anomaly_confirmed", "confirmed_episode_start", "detector_state")


def make_hgb(config: AppConfig):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # HGB auto early stopping can create a random row validation set at >10k
    # samples. Disable it so all reference-ready healthy training rows are fit.
    return HistGradientBoostingRegressor(**config.as_dict()["models"]["hist_gradient_boosting"],
                                         random_state=config.seed, early_stopping=False)


def regression_frame(rows: list[dict], config: AppConfig, predictors: list[str] | None = None):
    import pandas as pd
    settings = config.as_dict()["features"]
    names = validate_regression_predictors(predictors if predictors is not None else settings["regression_predictors"],
                                           settings["operational_set"], settings["rolling_windows_seconds"])
    for row in rows:
        if any(type(row.get(name)) not in (int, float) or not math.isfinite(row[name]) for name in names):
            raise DataError("Regression predictors must be finite numeric values from the feature registry.")
    return pd.DataFrame([[row[name] for name in names] for row in rows], columns=list(names), dtype=float)


def fit_hgb(rows: list[dict], config: AppConfig, split: FlightSplit):
    from threadpoolctl import threadpool_limits
    split.validate()
    require_healthy_source(rows, config.as_dict()["data"]["healthy_status"])
    if {r["NamaFile"] for r in rows} != set(split.train):
        raise DataError("HGB fitting must use exactly the assigned healthy training flights.")
    if any(r.get("reference_ready") is not True or r.get("split_role", "train") != "train" for r in rows):
        raise DataError("HGB fitting accepts only reference-ready training rows.")
    target = [r.get("voltage_sag_actual") for r in rows]
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in target):
        raise DataError("Healthy HGB target must be finite voltage sag.")
    frame = regression_frame(rows, config)
    model = make_hgb(config)
    try:
        with threadpool_limits(limits=1, user_api="openmp"):
            model.fit(frame, np.asarray(target, dtype=float))
    except (ValueError, TypeError) as exc:
        raise DataError(f"HGB fit failed: {exc}") from exc
    return model


def predict_hgb(model, rows: list[dict], config: AppConfig) -> np.ndarray:
    from threadpoolctl import threadpool_limits
    frame = regression_frame(rows, config)
    if list(getattr(model, "feature_names_in_", [])) != list(frame.columns):
        raise DataError("Fitted model feature order differs from the predictor registry.")
    if not rows:
        return np.array([], dtype=float)
    with threadpool_limits(limits=1, user_api="openmp"):
        prediction = model.predict(frame)
    if not np.isfinite(prediction).all():
        raise DataError("HGB produced nonfinite predictions.")
    return prediction


def score_rows(model, rows: list[dict], config: AppConfig, split_role: str) -> list[dict]:
    if split_role not in ("train", "validation", "test"):
        raise DataError("Unknown model scoring role.")
    ready_rows = [row for row in rows if row["reference_ready"]]
    predictions = iter(predict_hgb(model, ready_rows, config))
    scored = []
    for row in rows:
        predicted = float(next(predictions)) if row["reference_ready"] else None
        residual, score = residual_score(row["voltage_sag_actual"], predicted) if predicted is not None else (None, None)
        scored.append({**row, "split_role": split_role, "used_for_fit": split_role == "train" and row["reference_ready"],
                       "used_for_threshold": split_role == "validation" and row["reference_ready"],
                       "unseen_by_model": split_role != "train", "predicted_voltage_sag": predicted,
                       "residual_signed_v": residual, "anomaly_score": score})
    return scored


def load_model_bundle(path: Path) -> dict:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("engine") != "battery_v2" or bundle.get("format_version") != 1:
        raise DataError("Unsupported V2 model bundle.")
    cfg = bundle["config"]
    names = validate_regression_predictors(bundle["predictors"], cfg["features"]["operational_set"],
                                           cfg["features"]["rolling_windows_seconds"])
    if not isinstance(bundle["model"], HistGradientBoostingRegressor) or list(bundle["model"].feature_names_in_) != list(names):
        raise DataError("Bundle model/schema mismatch.")
    split = FlightSplit(*(tuple(bundle["split"][key]) for key in ("train", "validation", "test")))
    split.validate()
    if bundle["threshold"]["source_role"] != "validation" or set(bundle["threshold"]["source_flights"]) != set(split.validation):
        raise DataError("Bundle threshold lacks validation provenance.")
    return bundle


def _feature_input(summary: dict, config: AppConfig) -> tuple[list[dict], list[dict], dict, str]:
    output = Path(summary["output_directory"])
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    def read(name):
        value = (output / name).read_bytes()
        if hashlib.sha256(value).hexdigest() != manifest["outputs"][name]:
            raise DataError(f"Feature artifact checksum mismatch: {name}")
        return value

    reader = csv.DictReader(io.StringIO(read("battery_features.csv").decode("utf-8")), strict=True)
    expected = [*PROCESSED_COLUMNS, *feature_columns(config.as_dict()["features"]["rolling_windows_seconds"])]
    if reader.fieldnames != expected:
        raise DataError("Unexpected model input feature schema.")
    integers = {"TimeUS", "Inst", "source_record", "segment_id", "active_run_id", "reference_low_current_samples", "reference_lock_time_us"}
    booleans = {"is_segment_start", "active_by_current", "active_confirmed", "derivative_ready", "reference_ready"}
    text = {"NamaFile", "reference_status", "readiness"}
    nullable = {"active_run_id", "reference_lock_time_us", "V_ref", "voltage_sag_actual"}
    rows = []
    try:
        for raw in reader:
            row = {}
            for name, value in raw.items():
                if value == "" and name in nullable:
                    row[name] = None
                elif name in text:
                    row[name] = value
                elif name in booleans or name.startswith("Curr_roll_ready_"):
                    if value not in ("True", "False"):
                        raise DataError(f"Invalid feature boolean: {name}")
                    row[name] = value == "True"
                elif name in integers or name.startswith("Curr_roll_count_"):
                    row[name] = int(value)
                else:
                    row[name] = float(value)
                    if not math.isfinite(row[name]):
                        raise DataError(f"Nonfinite feature input: {name}")
            rows.append(row)
        coverage = list(csv.DictReader(io.StringIO(read("reference_coverage.csv").decode("utf-8"))))
        registry = json.loads(read("feature_registry.json"))
    except (ValueError, TypeError, csv.Error) as exc:
        raise DataError(f"Invalid feature artifact: {exc}") from exc
    if len(rows) != summary["retained_rows"]:
        raise DataError("Model input feature row count mismatch.")
    if registry != feature_registry(config.as_dict()["features"]["rolling_windows_seconds"], config.as_dict()["features"]["operational_set"]):
        raise DataError("Model input feature registry mismatch.")
    return rows, coverage, registry, hashlib.sha256(manifest_bytes).hexdigest()


def _write_files(output: Path, payloads: dict[str, bytes], config: AppConfig) -> None:
    for name, value in payloads.items():
        target = project_path(output / name, config.root)
        if target in (config.source, config.data_path):
            raise DataError("Pilot output collides with an input.")
        if target.exists() and target.read_bytes() != value:
            raise DataError(f"Existing pilot output differs: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        (output / name).write_bytes(value)


def baseline(config: AppConfig) -> dict:
    import joblib
    from .pilot_reporting import pilot_report, pilot_tables, validation_plot
    cfg = config.as_dict()
    if (cfg["features"]["operational_set"] != "battery_d" or cfg["features"]["rolling_windows_seconds"] != [.5, 1.0]
            or cfg["threshold"]["reference_percentile"] != 99.0
            or cfg["persistence"]["reference_duration_seconds"] != .5 or cfg["dual_path"]["enabled"]):
        raise ConfigError("Sprint 4 reference pilot requires Battery-D (0.5/1.0 s), validation percentile 99, persistence 0.5 s, dual-path disabled.")
    feature_summary = generate_features(config)
    if feature_summary["currtot_predictor_review_required"]:
        raise DataError("CurrTot decreases require documented review before Battery-D training; pilot stopped.")
    rows, coverage, registry, upstream_hash = _feature_input(feature_summary, config)
    eligible = [r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True"]
    split = pilot_split(eligible, cfg["evaluation"]["minimum_active_flights"])
    assignments = []
    for flight in coverage:
        name = flight["NamaFile"]
        role = ("train" if name in split.train else "validation" if name in split.validation
                else "reserved_test" if name in split.test else "excluded")
        assignments.append(dict(NamaFile=name, split_role=role, retained_samples=int(flight["retained_samples"]),
                                ready_samples=int(flight["ready_samples"]), v2_current_based_label=flight["v2_current_based_label"],
                                used_for_fit=role == "train", used_for_threshold=role == "validation",
                                unseen_by_model=role != "train", scored_in_pilot=role in ("train", "validation")))
    protocol = dict(experiment="sprint4_reference_pilot", evaluation_scope="historical_development_pilot",
                    split_rule="first_sorted_active_reference_ready_flight_test_next_validation_rest_train",
                    split=split.as_dict(), random_seed=config.seed, model_parameters=make_hgb(config).get_params(),
                    early_stopping_policy="disabled: no internal random row validation split", openmp_threads=1,
                    training_row_policy="all reference-ready rows of whole assigned healthy training flights, including low-current periods",
                    healthy_status=cfg["data"]["healthy_status"], synthetic_training_allowed=False,
                    predictors=registry["regression_sets"]["battery_d"], target="voltage_sag_actual",
                    threshold_percentile=99.0, persistence_seconds=.5, threshold_source="healthy_validation_only",
                    reserved_test_policy="no model prediction, threshold selection or pilot metrics on reserved test",
                    feature_manifest_sha256=upstream_hash, data_sha256=feature_summary["data_sha256"],
                    config_sha256=feature_summary["config_sha256"], dependency_versions=feature_summary["dependency_versions"])
    code_hashes = {f"src/{name}": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("models.py", "detectors.py", "persistence.py", "cross_validation.py", "provenance.py", "pilot_reporting.py")}
    identity = dict(protocol=protocol, code_sha256=code_hashes)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "baseline" / run_id, config.root)
    payloads = {"pilot_protocol.json": (format_json(protocol) + "\n").encode(),
                "pilot_assignments.csv": _csv(assignments), "config_used.yaml": yaml.safe_dump(cfg, sort_keys=False).encode(),
                "feature_registry.json": (format_json(registry) + "\n").encode()}
    # These concrete decisions are persisted BEFORE fitting or seeing scores.
    _write_files(output, payloads, config)
    completed = output / "manifest.json"
    if completed.exists():
        try:
            existing = json.loads(completed.read_text())
            if existing["run_id"] != run_id or existing["identity"] != identity:
                raise DataError("Existing pilot manifest identity mismatch.")
            for name, digest in existing["outputs"].items():
                target = project_path(output / name, config.root)
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise DataError(f"Existing pilot artifact checksum mismatch: {name}")
            return json.loads((output / "summary.json").read_text())
        except (ValueError, KeyError) as exc:
            raise DataError(f"Invalid completed pilot manifest: {exc}") from exc
    train_rows = [r for r in rows if r["NamaFile"] in split.train and r["reference_ready"]]
    model = fit_hgb(train_rows, config, split)
    training_counts = dict(sorted(Counter(r["NamaFile"] for r in train_rows).items()))
    train_fingerprint = [{name: r[name] for name in ("NamaFile", "Inst", "TimeUS", "source_record", "voltage_sag_actual", *protocol["predictors"])} for r in train_rows]
    fitted_provenance = dict(train_flights=list(split.train), ready_training_samples=len(train_rows),
                             samples_per_flight=training_counts,
                             low_current_training_samples=sum(r["Curr"] < cfg["activity"]["min_current_a"] for r in train_rows),
                             training_rows_sha256=hashlib.sha256(format_json({"rows": train_fingerprint}).encode()).hexdigest(),
                             healthy_status=cfg["data"]["healthy_status"], synthetic_rows=0,
                             early_stopping=False, fitted_iterations=int(model.n_iter_), scaler=None)
    train_scores = score_rows(model, [r for r in rows if r["NamaFile"] in split.train], config, "train")
    validation_scores = score_rows(model, [r for r in rows if r["NamaFile"] in split.validation], config, "validation")
    threshold = calibrate_threshold([r for r in validation_scores if r["reference_ready"]], split)
    threshold.update(persistence_seconds=.5, persistence_source="frozen_reference_protocol_not_tuned")
    decisions = apply_single_path(train_scores + validation_scores, threshold_v=threshold["threshold_v"],
                                  duration_seconds=.5, gap_threshold_seconds=config.gap_threshold_seconds)
    regression, alarms, by_role = pilot_tables(decisions, config.gap_threshold_seconds)
    summary = dict(status="OK", command="baseline", run_id=run_id, output_directory=str(output),
                   evaluation_scope=protocol["evaluation_scope"], split=split.as_dict(),
                   feature_run_id=feature_summary["run_id"], feature_directory=feature_summary["output_directory"],
                   feature_manifest_sha256=upstream_hash, data_sha256=feature_summary["data_sha256"],
                   random_seed=config.seed, predictors=protocol["predictors"],
                   training_samples=len(train_rows), validation_samples=threshold["source_samples"],
                   threshold_v=threshold["threshold_v"], threshold_percentile=99.0, persistence_seconds=.5,
                   metrics_by_role=by_role, reserved_test_evaluated=False, synthetic_evaluation_performed=False,
                   currtot_unit="unverified", model_training_performed=True,
                   detector_gate_status="NOT_ASSESSED_FULL_EVALUATION_PENDING", deployment_status="RESEARCH_PILOT_ONLY",
                   model_bundle=str(output / "model_bundle.joblib"))
    schema = dict(required_input_columns=list(BATTERY_COLUMNS), predictor_order=protocol["predictors"],
                  units={"TimeUS": "us_from_CSV_contract", "Volt": "V_from_CSV_contract", "Curr": "A_from_CSV_contract", "CurrTot": "unverified"},
                  unit_conversion_applied=False, reference_voltage=cfg["reference_voltage"],
                  time_policy=cfg["time"], instance_policy=cfg["data"]["instance_policy"],
                  reference_unavailable_policy="NOT_READY; exclude from fit, threshold and metric denominators")
    bundle = dict(format_version=1, engine="battery_v2", model=model, predictors=protocol["predictors"],
                  feature_registry=registry, config=cfg, input_schema_unit_policy=schema, split=split.as_dict(),
                  threshold=threshold, training_provenance=fitted_provenance, protocol=protocol,
                  feature_manifest_sha256=upstream_hash, data_sha256=feature_summary["data_sha256"], code_sha256=code_hashes)
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    payloads.update({"model_bundle.joblib": buffer.getvalue(), "threshold.json": (format_json(threshold) + "\n").encode(),
                     "training_provenance.json": (format_json(fitted_provenance) + "\n").encode(),
                     "schema_unit_policy.json": (format_json(schema) + "\n").encode(),
                     "scores.csv": _csv([{name: r[name] for name in SCORE_COLUMNS} for r in decisions], SCORE_COLUMNS),
                     "regression_per_flight.csv": _csv(regression), "healthy_alarm_per_flight.csv": _csv(alarms),
                     "summary.json": (format_json(summary) + "\n").encode(),
                     "pilot_report.md": pilot_report(summary, fitted_provenance, regression, alarms).encode("utf-8"),
                     "validation_trace.png": validation_plot([r for r in decisions if r["split_role"] == "validation"], threshold["threshold_v"])})
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise DataError("Input changed during HGB pilot; model outputs not written.")
    _write_files(output, payloads, config)
    manifest = dict(run_id=run_id, identity=identity,
                    outputs={name: hashlib.sha256(value).hexdigest() for name, value in sorted(payloads.items())})
    _write_files(output, {"manifest.json": (format_json(manifest) + "\n").encode()}, config)
    return summary
