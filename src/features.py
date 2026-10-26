"""Causal reference/feature state shared by batch processing and incremental use."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
from statistics import mean, median, pstdev

import yaml

from .config import AppConfig, project_path
from .errors import DataError
from .feature_sets import feature_registry, validate_regression_predictors, validate_windows, window_token
from .preprocessing import PROCESSED_COLUMNS, _csv, preprocess
from .reporting import format_json

BASE_FEATURE_COLUMNS = ("elapsed_time", "segment_elapsed_seconds", "dI_dt", "dV_dt", "derivative_ready",
                        "reference_status", "readiness", "reference_ready", "reference_low_current_samples",
                        "reference_lock_time_us", "V_ref", "voltage_sag_actual")
COVERAGE_COLUMNS = ("NamaFile", "Inst", "input_samples", "retained_samples", "v2_current_based_label",
                    "reference_status", "coverage_reason", "V_ref", "reference_low_current_samples",
                    "reference_lock_time_us", "warmup_samples", "reference_unavailable_samples", "ready_samples",
                    "sample_coverage_percentage", "observed_duration_seconds", "ready_observed_duration_seconds",
                    "excluded_unready_duration_seconds", "time_coverage_percentage", "activity_and_reference_eligible")


def feature_columns(windows: list[float]) -> tuple[str, ...]:
    validate_windows(windows)
    rolling = tuple(f"Curr_roll_{stat}_{window_token(w)}s" for w in windows
                    for stat in ("mean", "std", "count", "coverage", "ready"))
    return (*BASE_FEATURE_COLUMNS, *rolling)


@dataclass
class _StreamState:
    start_time_us: int
    segment_start_us: int
    segment_id: int = 0
    previous: dict | None = None
    low_current_volts: list[float] = field(default_factory=list)
    reference_status: str = "WARMUP"
    reference: float | None = None
    lock_time_us: int | None = None
    windows: dict[float, deque] = field(default_factory=dict)


class CausalFeatureBuilder:
    """Consume cleaned numeric rows, ordered within each flight/instance stream.

    Interleaved streams are supported. The outer pipeline rejects unselected
    multiple instances; this state machine still isolates them for future replay.
    No finalize operation uses the eventual flight length or rewrites old output.
    """

    def __init__(self, config: AppConfig):
        cfg = config.as_dict()
        self.gap_seconds = config.gap_threshold_seconds
        self.reference_config = cfg["reference_voltage"]
        self.windows = list(cfg["features"]["rolling_windows_seconds"])
        validate_windows(self.windows)
        self.states: dict[tuple[str, int], _StreamState] = {}

    def consume(self, row: dict) -> dict:
        if not isinstance(row.get("NamaFile"), str) or not row["NamaFile"].strip():
            raise DataError("Feature input requires a nonempty NamaFile.")
        for name in ("TimeUS", "Inst"):
            if type(row.get(name)) is not int or row[name] < 0:
                raise DataError(f"Feature input {name}: expected a nonnegative integer.")
        for name in ("Volt", "Curr", "CurrTot"):
            if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]):
                raise DataError(f"Feature input {name}: expected finite numeric telemetry.")
        key = row["NamaFile"], row["Inst"]
        time = row["TimeUS"]
        state = self.states.get(key)
        previous = state.previous if state else None
        if previous is not None and time <= previous["TimeUS"]:
            raise DataError(f"Feature timestamps must strictly increase within {key}.")
        dt = (time - previous["TimeUS"]) / 1e6 if previous else 0.0
        boundary = previous is None or dt > self.gap_seconds
        segment_id = (state.segment_id + int(boundary)) if previous else 0
        if "segment_id" in row and (type(row["segment_id"]) is not int or row["segment_id"] != segment_id):
            raise DataError(f"Feature segment_id disagrees with actual timestamps in {key}.")
        if "dt_seconds" in row and row["dt_seconds"] != (0.0 if boundary else dt):
            raise DataError(f"Feature dt_seconds disagrees with actual timestamps in {key}.")
        if "is_segment_start" in row and row["is_segment_start"] is not boundary:
            raise DataError(f"Feature is_segment_start disagrees with actual timestamps in {key}.")
        d_current = (row["Curr"] - previous["Curr"]) / dt if previous and not boundary else 0.0
        d_voltage = (row["Volt"] - previous["Volt"]) / dt if previous and not boundary else 0.0
        if not math.isfinite(d_current) or not math.isfinite(d_voltage):
            raise DataError("Nonfinite derivative from supplied telemetry; feature run rejected.")
        if state is None:
            state = _StreamState(time, time)
            self.states[key] = state
        if boundary:
            state.segment_start_us = time
            state.windows = {w: deque() for w in self.windows}
        elapsed = (time - state.start_time_us) / 1e6
        segment_elapsed = (time - state.segment_start_us) / 1e6
        reference = self.reference_config
        if state.reference_status == "WARMUP":
            if elapsed <= reference["warmup_seconds"]:
                if row["Curr"] <= reference["low_current_max_a"]:
                    state.low_current_volts.append(float(row["Volt"]))
            elif len(state.low_current_volts) >= reference["min_low_current_samples"]:
                state.reference = float(median(state.low_current_volts))
                state.reference_status = "READY"
                state.lock_time_us = time
            else:
                state.reference_status = "REFERENCE_UNAVAILABLE"
        ready = state.reference_status == "READY"
        item = {**row, "segment_id": segment_id, "dt_seconds": 0.0 if boundary else dt,
                "is_segment_start": boundary, "gap_before_seconds": dt if previous and boundary else 0.0,
                "elapsed_time": elapsed, "segment_elapsed_seconds": segment_elapsed,
                "dI_dt": d_current, "dV_dt": d_voltage, "derivative_ready": not boundary,
                "reference_status": state.reference_status, "readiness": "READY" if ready else "NOT_READY",
                "reference_ready": ready, "reference_low_current_samples": len(state.low_current_volts),
                "reference_lock_time_us": state.lock_time_us, "V_ref": state.reference,
                "voltage_sag_actual": state.reference - row["Volt"] if ready else None}
        for window, history in state.windows.items():
            # Half-open left endpoint and exact integer timestamp subtraction.
            while history and (time - history[0][0]) / 1e6 >= window:
                history.popleft()
            history.append((time, float(row["Curr"])))
            values = [value for _, value in history]
            suffix = window_token(window) + "s"
            item[f"Curr_roll_mean_{suffix}"] = float(mean(values))
            item[f"Curr_roll_std_{suffix}"] = float(pstdev(values)) if len(values) > 1 else 0.0
            item[f"Curr_roll_count_{suffix}"] = len(values)
            item[f"Curr_roll_coverage_{suffix}"] = min(segment_elapsed / window, 1.0)
            item[f"Curr_roll_ready_{suffix}"] = segment_elapsed >= window
        if item["voltage_sag_actual"] is not None and not math.isfinite(item["voltage_sag_actual"]):
            raise DataError("Nonfinite voltage sag from supplied telemetry; feature run rejected.")
        state.previous = {name: row[name] for name in ("TimeUS", "Curr", "Volt")}
        state.segment_id = segment_id
        return item


def build_features(rows: list[dict], config: AppConfig) -> list[dict]:
    """Batch deliberately uses the same causal state transitions as incremental use."""
    builder = CausalFeatureBuilder(config)
    return [builder.consume(row) for row in rows]


def reference_coverage(rows: list[dict], inventory: list[dict]) -> list[dict]:
    by_flight = {}
    for row in rows:
        by_flight.setdefault(row["NamaFile"], []).append(row)
    coverage = []
    for flight in inventory:
        records = by_flight.get(flight["NamaFile"], [])
        last = records[-1] if records else None
        observed_us = ready_us = 0
        for before, after in zip(records, records[1:]):
            if (before["Inst"], before["segment_id"]) == (after["Inst"], after["segment_id"]):
                interval = after["TimeUS"] - before["TimeUS"]
                observed_us += interval
                if before["reference_ready"] and after["reference_ready"]:
                    ready_us += interval
        ready_samples = sum(r["reference_ready"] for r in records)
        status = last["reference_status"] if last else "NO_RETAINED_SAMPLES"
        reason = {"WARMUP": "WARMUP_NOT_COMPLETED", "READY": "REFERENCE_LOCKED",
                  "REFERENCE_UNAVAILABLE": "INSUFFICIENT_LOW_CURRENT_DURING_FIXED_WARMUP",
                  "NO_RETAINED_SAMPLES": "NO_RETAINED_SAMPLES"}[status]
        active = flight["v2_current_based_label"] == "ACTIVE_LOAD"
        coverage.append(dict(NamaFile=flight["NamaFile"], Inst=last["Inst"] if last else None,
                             input_samples=int(flight["input_samples"]), retained_samples=len(records),
                             v2_current_based_label=flight["v2_current_based_label"], reference_status=status,
                             coverage_reason=reason, V_ref=last["V_ref"] if last else None,
                             reference_low_current_samples=last["reference_low_current_samples"] if last else 0,
                             reference_lock_time_us=last["reference_lock_time_us"] if last else None,
                             warmup_samples=sum(r["reference_status"] == "WARMUP" for r in records),
                             reference_unavailable_samples=sum(r["reference_status"] == "REFERENCE_UNAVAILABLE" for r in records),
                             ready_samples=ready_samples, sample_coverage_percentage=100 * ready_samples / len(records) if records else None,
                             observed_duration_seconds=observed_us / 1e6, ready_observed_duration_seconds=ready_us / 1e6,
                             excluded_unready_duration_seconds=(observed_us - ready_us) / 1e6,
                             time_coverage_percentage=100 * ready_us / observed_us if observed_us else None,
                             activity_and_reference_eligible=active and ready_us > 0 and ready_samples > 0))
    return coverage


def _preprocessed_input(summary: dict) -> tuple[list[dict], list[dict], str]:
    output = Path(summary["output_directory"])
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    def read(name):
        payload = (output / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["outputs"][name]:
            raise DataError(f"Preprocessing artifact checksum mismatch: {name}")
        return csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)

    reader = read("battery_segmented.csv")
    if reader.fieldnames != list(PROCESSED_COLUMNS):
        raise DataError("Unexpected processed CSV schema.")
    rows = []
    try:
        for raw in reader:
            row = {"NamaFile": raw["NamaFile"]}
            for field in ("TimeUS", "Inst", "source_record", "segment_id"):
                row[field] = int(raw[field])
            for field in ("Volt", "Curr", "CurrTot", "dt_seconds", "gap_before_seconds", "active_duration_seconds"):
                row[field] = float(raw[field])
            for field in ("is_segment_start", "active_by_current", "active_confirmed"):
                if raw[field] not in ("True", "False"):
                    raise DataError(f"Invalid boolean in processed CSV: {field}")
                row[field] = raw[field] == "True"
            row["active_run_id"] = int(raw["active_run_id"]) if raw["active_run_id"] else None
            rows.append(row)
        inventory = list(read("flight_inventory.csv"))
    except (ValueError, KeyError, csv.Error) as exc:
        raise DataError(f"Invalid preprocessing artifact: {exc}") from exc
    if len(rows) != summary["retained_rows"]:
        raise DataError("Preprocessing artifact row count mismatch.")
    return rows, inventory, hashlib.sha256(manifest_bytes).hexdigest()


def generate_features(config: AppConfig) -> dict:
    """Run preprocessing, then features/coverage/registry without training."""
    cfg = config.as_dict()
    settings = cfg["features"]
    validate_regression_predictors(settings["regression_predictors"], settings["operational_set"],
                                   settings["rolling_windows_seconds"])
    preprocessing = preprocess(config)
    rows, inventory, preprocessing_hash = _preprocessed_input(preprocessing)
    features = build_features(rows, config)
    coverage = reference_coverage(features, inventory)
    registry = feature_registry(settings["rolling_windows_seconds"], settings["operational_set"])
    code_hashes = {f"src/{name}": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("features.py", "feature_sets.py", "config.py", "preprocessing.py", "activity.py", "data_loader.py", "reporting.py")}
    versions = dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "PyYAML", "pytest", "matplotlib", "joblib")})
    config_bytes = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    identity = dict(preprocessing_manifest_sha256=preprocessing_hash, code_sha256=code_hashes,
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(), dependency_versions=versions)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "features" / run_id, config.root)
    ready_active = [r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"]]
    summary = dict(status="OK", command="features", run_id=run_id, output_directory=str(output),
                   **identity, data_sha256=preprocessing["data_sha256"], preprocessing_run_id=preprocessing["run_id"],
                   preprocessing_directory=preprocessing["output_directory"], input_rows=preprocessing["input_rows"],
                   retained_rows=len(features), flights=len(coverage), ready_samples=sum(r["ready_samples"] for r in coverage),
                   warmup_samples=sum(r["warmup_samples"] for r in coverage),
                   reference_unavailable_samples=sum(r["reference_unavailable_samples"] for r in coverage),
                   reference_available_flights=[r["NamaFile"] for r in coverage if r["reference_status"] == "READY"],
                   reference_unavailable_flights=[r["NamaFile"] for r in coverage if r["reference_status"] == "REFERENCE_UNAVAILABLE"],
                   warmup_incomplete_flights=[r["NamaFile"] for r in coverage if r["reference_status"] == "WARMUP"],
                   no_retained_sample_flights=[r["NamaFile"] for r in coverage if not r["retained_samples"]],
                   observed_duration_seconds=math.fsum(r["observed_duration_seconds"] for r in coverage),
                   ready_observed_duration_seconds=math.fsum(r["ready_observed_duration_seconds"] for r in coverage),
                   active_reference_ready_flights=ready_active,
                   reference_activity_gate_passed=len(ready_active) >= cfg["evaluation"]["minimum_active_flights"],
                   operational_predictors=registry["regression_sets"][settings["operational_set"]],
                   currtot_unit="unverified", currtot_predictor_review_required=preprocessing["currtot_predictor_review_required"],
                   model_training_performed=False, processed_csv=str(output / "battery_features.csv"))
    payloads = {
        "battery_features.csv": _csv(features, (*PROCESSED_COLUMNS, *feature_columns(settings["rolling_windows_seconds"]))),
        "reference_coverage.csv": _csv(coverage, COVERAGE_COLUMNS),
        "feature_registry.json": (format_json(registry) + "\n").encode(),
        "summary.json": (format_json(summary) + "\n").encode(),
        "config_used.yaml": config_bytes,
        "dependency_versions.json": (format_json(versions) + "\n").encode(),
        "feature_audit.md": _feature_audit(summary, coverage).encode("utf-8"),
    }
    manifest = dict(run_id=run_id, **identity, outputs={name: hashlib.sha256(payload).hexdigest()
                                                    for name, payload in sorted(payloads.items())})
    payloads["manifest.json"] = (format_json(manifest) + "\n").encode()
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise DataError("Input changed during feature generation; no feature output written.")
    for name, payload in payloads.items():
        target = project_path(output / name, config.root)
        if target in (config.data_path, config.source):
            raise DataError(f"Feature output collides with an input: {target}")
        if target.exists() and target.read_bytes() != payload:
            raise DataError(f"Existing feature output differs for run {run_id}: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    return summary


def _feature_audit(summary: dict, coverage: list[dict]) -> str:
    lines = ["# Sprint 3 — causal reference and features", "",
             f"Run `{summary['run_id']}`; preprocessing `{summary['preprocessing_run_id']}`.",
             f"Rows: {summary['retained_rows']}; ready: {summary['ready_samples']}; warm-up: {summary['warmup_samples']}; reference unavailable: {summary['reference_unavailable_samples']}.", "",
             "Reference collects accepted low-current samples at elapsed <= fixed warm-up duration. On the first sample strictly after that window, lock the median if enough samples exist; otherwise reference stays unavailable. No flight-length percentage, future sample selection or whole-flight fallback. V_ref remains locked across gaps and post-lock voltage changes.",
             "Derivative uses actual dt in seconds; first sample of every segment has dI_dt=dV_dt=0 and derivative_ready=false. Rolling uses (t-window,t], ddof=0, available past plus current, std=0 for one sample. Gap clears rolling/derivative history. Coverage is min(segment elapsed/window,1), not an estimate of sampling completeness. Readiness/count/coverage flags are diagnostics, not automatic predictors.",
             "Batch and incremental processing share CausalFeatureBuilder. elapsed_time is stream elapsed time, used only by Battery-E ablation. No distribution-dependent transformation is fitted. Regression predictors come exclusively from feature_registry.json; actual sag is a target, dV_dt only a direct-baseline input.", "",
             "## Reference coverage", "", "| Flight | Status | Low-current warm-up samples | V_ref V | Ready samples | Ready observed s |", "|---|---|---:|---:|---:|---:|"]
    for r in coverage:
        voltage = f"{r['V_ref']:.6f}" if r["V_ref"] is not None else "unavailable"
        lines.append(f"| {r['NamaFile']} | {r['reference_status']} | {r['reference_low_current_samples']} | {voltage} | {r['ready_samples']} | {r['ready_observed_duration_seconds']:.6f} |")
    lines += ["", "Coverage samples include only reference-ready rows; observed time includes intervals with both endpoints ready in the same segment. Gaps and warm-up/unavailable intervals are excluded from ready exposure. Truncated warm-up and fully dropped flights remain visible. Activity/reference eligibility does not assert model performance or hardware readiness.", "",
              "CurrTot remains unverified, without unit conversion. Healthy status is assumed from source. No training, synthetic fault experiments, detector thresholds or fitted model were produced.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)
