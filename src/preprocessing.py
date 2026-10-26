"""Battery-only cleaning/segmentation and reproducible Stage 2 audit output."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import yaml

from .activity import classify_activity, sensitivity_report
from .config import AppConfig, project_path
from .data_loader import load_battery_csv
from .errors import DataError
from .feature_sets import BATTERY_COLUMNS
from .reporting import format_json

PROCESSED_COLUMNS = (*BATTERY_COLUMNS, "source_record", "segment_id", "dt_seconds",
                     "is_segment_start", "gap_before_seconds", "active_by_current", "active_run_id",
                     "active_duration_seconds", "active_confirmed")
EVENT_COLUMNS = ("source_record", "NamaFile", "Inst", "TimeUS", "field", "reason", "action", "raw_value")
RUN_COLUMNS = ("NamaFile", "Inst", "segment_id", "active_run_id", "start_time_us", "end_time_us",
               "samples", "duration_seconds", "is_valid")


@dataclass
class CleaningResult:
    rows: list[dict]
    events: list[dict]
    dropped: list[dict]
    inventory: list[dict]
    errors: list[str]
    segments: list[dict]
    gaps: list[dict]


def _parse(value: str, integer: bool) -> int | float:
    """Parse identifiers exactly (no float rounding of microseconds)."""
    number = Decimal(value.strip())
    if not number.is_finite():
        raise ValueError("NONFINITE")
    if integer:
        if number < 0 or number != number.to_integral_value():
            raise ValueError("INVALID_NONNEGATIVE_INTEGER")
        return int(number)
    result = float(number)
    if not math.isfinite(result):
        raise ValueError("NONFINITE")
    return result


def clean_and_segment(source: list[dict[str, str]], gap_threshold_seconds: float) -> CleaningResult:
    """No imputation; diagnose raw identities before dropping invalid numerics.

    All accepted instances are audited before any telemetry row is dropped.
    Multiple instances and duplicate identities are blocking errors, even if
    another field in one of those rows would otherwise cause it to be dropped.
    """
    if not math.isfinite(gap_threshold_seconds) or gap_threshold_seconds <= 0:
        raise ValueError("gap threshold must be finite and positive")
    events, dropped, parsed, errors = [], [], [], []
    counts = Counter(r["NamaFile"] for r in source)
    instances = defaultdict(set)
    identities = set()
    previous_time = {}
    backwards = Counter()

    def event(index, raw, field, reason, action):
        events.append(dict(source_record=index, NamaFile=raw["NamaFile"], Inst=raw["Inst"], TimeUS=raw["TimeUS"],
                           field=field, reason=reason, action=action, raw_value=raw.get(field, "")))

    for index, raw in enumerate(source, start=2):
        row = {"NamaFile": raw["NamaFile"], "source_record": index}
        reasons = []
        for field in BATTERY_COLUMNS[1:]:
            try:
                row[field] = _parse(raw[field], field in ("TimeUS", "Inst"))
            except (InvalidOperation, ValueError, OverflowError) as exc:
                reason = "MISSING" if not raw[field].strip() else str(exc) if isinstance(exc, ValueError) else "NONNUMERIC"
                reasons.append(f"{field}:{reason}")
                event(index, raw, field, reason, "DROP")
            else:
                if field in ("Volt", "Curr", "CurrTot") and row[field] < 0:
                    event(index, raw, field, "NEGATIVE_VALUE", "RETAIN_AND_REPORT")
        if "Inst" in row:
            instances[raw["NamaFile"]].add(row["Inst"])
        if "Inst" in row and "TimeUS" in row:
            stream = raw["NamaFile"], row["Inst"]
            identity = *stream, row["TimeUS"]
            if identity in identities:
                errors.append(f"Duplicate timestamp in {stream}: {row['TimeUS']}")
                event(index, raw, "TimeUS", "DUPLICATE_TIMESTAMP", "REJECT_DATASET")
            identities.add(identity)
            if stream in previous_time and row["TimeUS"] < previous_time[stream]:
                backwards[raw["NamaFile"]] += 1
                event(index, raw, "TimeUS", "BACKWARD_IN_SOURCE_ORDER", "SORT_AND_REPORT")
            previous_time[stream] = row["TimeUS"]
        if reasons:
            dropped.append({**raw, "source_record": index, "drop_reasons": "|".join(reasons)})
        else:
            parsed.append(row)
    for flight, values in sorted(instances.items()):
        if len(values) > 1:
            errors.append(f"Multiple battery instances in {flight}: {sorted(values)}; explicit selection required.")
            events.append(dict(source_record=None, NamaFile=flight, Inst="|".join(map(str, sorted(values))),
                               TimeUS=None, field="Inst", reason="MULTIPLE_INSTANCES", action="REJECT_DATASET", raw_value=""))
    retained = Counter(r["NamaFile"] for r in parsed)
    inventory = [{"NamaFile": flight, "input_samples": count, "retained_samples": retained[flight],
                  "dropped_samples": count - retained[flight], "instances": "|".join(map(str, sorted(instances[flight]))),
                  "backward_timestamps_in_source_order": backwards[flight]} for flight, count in sorted(counts.items())]
    if errors:
        return CleaningResult([], events, dropped, inventory, errors, [], [])
    parsed.sort(key=lambda r: (r["NamaFile"], r["Inst"], r["TimeUS"]))
    previous = {}
    segments = {}
    gap_stats = defaultdict(list)
    decreases = Counter()
    for row in parsed:
        stream = row["NamaFile"], row["Inst"]
        before = previous.get(stream)
        interval = (row["TimeUS"] - before["TimeUS"]) / 1e6 if before else 0.0
        boundary = before is None or interval > gap_threshold_seconds
        segment_id = (before["segment_id"] + 1 if boundary else before["segment_id"]) if before else 0
        row.update(segment_id=segment_id, dt_seconds=0.0 if boundary else interval,
                   is_segment_start=boundary, gap_before_seconds=interval if before and boundary else 0.0)
        if before and row["CurrTot"] < before["CurrTot"]:
            decreases[row["NamaFile"]] += 1
            events.append(dict(source_record=row["source_record"], NamaFile=row["NamaFile"], Inst=row["Inst"],
                               TimeUS=row["TimeUS"], field="CurrTot", reason="CURRTOT_DECREASE",
                               action="RETAIN_REVIEW_BEFORE_BATTERY_B_TO_E", raw_value=source[row["source_record"] - 2]["CurrTot"]))
        if before:
            gap_stats[row["NamaFile"]].append(row["TimeUS"] - before["TimeUS"])
        key = *stream, segment_id
        if key not in segments:
            segments[key] = dict(NamaFile=stream[0], Inst=stream[1], segment_id=segment_id,
                                 start_time_us=row["TimeUS"], end_time_us=row["TimeUS"], samples=0,
                                 observed_duration_seconds=0.0)
        segment = segments[key]
        segment["end_time_us"] = row["TimeUS"]
        segment["samples"] += 1
        segment["observed_duration_seconds"] = (row["TimeUS"] - segment["start_time_us"]) / 1e6
        previous[stream] = row
    gaps = []
    for flight in sorted(counts):
        intervals = gap_stats[flight]
        valid = [t for t in intervals if t / 1e6 <= gap_threshold_seconds]
        excluded = [t for t in intervals if t / 1e6 > gap_threshold_seconds]
        flight_segments = [r for r in segments.values() if r["NamaFile"] == flight]
        gaps.append(dict(NamaFile=flight, gap_threshold_seconds=gap_threshold_seconds,
                         retained_samples=retained[flight], segments=len(flight_segments), gaps=len(excluded),
                         observed_duration_seconds=sum(valid) / 1e6, excluded_gap_seconds=sum(excluded) / 1e6,
                         wall_clock_span_seconds=sum(intervals) / 1e6, max_gap_seconds=max(excluded, default=0) / 1e6,
                         min_valid_dt_seconds=min(valid) / 1e6 if valid else None,
                         max_valid_dt_seconds=max(valid) / 1e6 if valid else None,
                         currtot_decreases=decreases[flight]))
    return CleaningResult(parsed, events, dropped, inventory, errors, list(segments.values()), gaps)


def _csv(rows: list[dict], columns: tuple | list | None = None) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _empty_activity(flight: str, settings: dict) -> dict:
    """Keep fully dropped flights visible, with undefined exposure percentage."""
    return dict(NamaFile=flight, samples=0, **settings, observed_duration_seconds=0.0,
                valid_active_duration_seconds=0.0, active_time_percentage=None, active_samples=0,
                active_sample_percentage=None, causal_confirmed_samples=0, active_runs=0, valid_active_runs=0,
                longest_valid_active_run_seconds=0.0, current_min_a=None, current_mean_a=None, current_std_a=None,
                current_p05_a=None, current_median_a=None, current_p95_a=None, current_max_a=None,
                v2_current_based_label="LOW_LOAD_OR_IDLE", activity_eligible=False, activity_reason="NO_RETAINED_SAMPLES")


def preprocess(config: AppConfig) -> dict:
    """Run Stage 2 and write content-addressed outputs, without changing raw."""
    cfg = config.as_dict()
    source_bytes = config.data_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = load_battery_csv(config.data_path)
    result = clean_and_segment(source, config.gap_threshold_seconds)
    provenance_path = config.root / "data/metadata/flight_inventory.csv"
    provenance_bytes = provenance_path.read_bytes() if provenance_path.is_file() else None
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest() if provenance_bytes is not None else None
    code_hashes = {f"src/{name}": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("preprocessing.py", "activity.py", "data_loader.py", "config.py", "feature_sets.py", "reporting.py")}
    config_bytes = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    signature = format_json(dict(data_sha256=source_hash, config_sha256=config_hash, code_sha256=code_hashes,
                                provenance_sha256=provenance_hash))
    run_id = hashlib.sha256(signature.encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "preprocessing" / run_id, config.root)
    payloads = {}

    def write(name, payload):
        payloads[name] = payload if isinstance(payload, bytes) else payload.encode("utf-8")

    write("config_used.yaml", config_bytes)
    write("data_quality_events.csv", _csv(result.events, EVENT_COLUMNS))
    write("dropped_rows.csv", _csv(result.dropped, (*BATTERY_COLUMNS, "source_record", "drop_reasons")))
    base_summary = dict(command="preprocess", run_id=run_id, data_sha256=source_hash,
                        config_sha256=config_hash, code_sha256=code_hashes, provenance_sha256=provenance_hash,
                        input_path=str(config.data_path), output_directory=str(output),
                        input_rows=len(source), dropped_rows=len(result.dropped),
                        currtot_unit="unverified", healthy_status=cfg["data"]["healthy_status"],
                        model_training_performed=False)
    if result.errors:
        summary = dict(base_summary, status="REJECTED", errors=result.errors,
                       diagnostic_inventory=result.inventory)
        write("summary.json", format_json(summary) + "\n")
    else:
        settings = {k: cfg["activity"][k] for k in ("min_current_a", "min_duration_seconds", "min_active_time_percentage")}
        rows, activity, runs = classify_activity(result.rows, **settings)
        seen = {r["NamaFile"] for r in activity}
        activity.extend(_empty_activity(r["NamaFile"], settings) for r in result.inventory if r["NamaFile"] not in seen)
        activity.sort(key=lambda r: r["NamaFile"])
        sensitivity, sensitivity_summary = sensitivity_report(result.rows, settings["min_active_time_percentage"])
        # Include zero-retained flights in every grid point as well.
        for grid in sensitivity_summary:
            grid_settings = {k: grid[k] for k in settings}
            sensitivity.extend(_empty_activity(r["NamaFile"], grid_settings) for r in result.inventory if r["NamaFile"] not in seen)
        sensitivity.sort(key=lambda r: (r["min_current_a"], r["min_duration_seconds"], r["NamaFile"]))
        # Historical labels are loaded only AFTER operational V2 labels have been computed.
        provenance = {}
        if provenance_bytes is not None:
            try:
                reader = csv.DictReader(io.StringIO(provenance_bytes.decode("utf-8-sig")), strict=True)
                if not {"NamaFile", "historical_v1_label"}.issubset(reader.fieldnames or []):
                    raise DataError(f"Invalid historical provenance schema: {provenance_path}")
                provenance = {r["NamaFile"]: r.get("historical_v1_label") or "UNKNOWN" for r in reader}
            except (UnicodeError, csv.Error) as exc:
                raise DataError(f"Cannot read historical provenance: {exc}") from exc
        by_flight = {r["NamaFile"]: r for r in activity}
        inventory = [{**r, "historical_v1_label": provenance.get(r["NamaFile"], "UNKNOWN"),
                      "v2_current_based_label": by_flight[r["NamaFile"]]["v2_current_based_label"],
                      "activity_eligible": by_flight[r["NamaFile"]]["activity_eligible"],
                      "healthy_status": cfg["data"]["healthy_status"]} for r in result.inventory]
        active = [r["NamaFile"] for r in activity if r["activity_eligible"]]
        event_counts = dict(sorted(Counter(e["reason"] for e in result.events).items()))
        summary = dict(base_summary, status="OK", retained_rows=len(rows), flights=len(inventory),
                       active_load_flights=active, low_load_or_idle_flights=[r["NamaFile"] for r in activity if not r["activity_eligible"]],
                       segments=len(result.segments), gaps=sum(r["gaps"] for r in result.gaps),
                       observed_duration_seconds=math.fsum(r["observed_duration_seconds"] for r in result.gaps),
                       excluded_gap_seconds=math.fsum(r["excluded_gap_seconds"] for r in result.gaps),
                       activity_gate_passed=len(active) >= cfg["evaluation"]["minimum_active_flights"],
                       reference_readiness="NOT_COMPUTED_STAGE_3", event_counts=event_counts,
                       currtot_predictor_review_required=event_counts.get("CURRTOT_DECREASE", 0) > 0,
                       processed_csv=str(output / "battery_segmented.csv"))
        write("battery_segmented.csv", _csv(rows, PROCESSED_COLUMNS))
        write("flight_inventory.csv", _csv(inventory))
        write("battery_activity.csv", _csv(activity))
        write("active_segments.csv", _csv(runs, RUN_COLUMNS))
        write("time_gap_summary.csv", _csv(result.gaps))
        segment_columns = ("NamaFile", "Inst", "segment_id", "start_time_us", "end_time_us", "samples", "observed_duration_seconds")
        write("segments.csv", _csv(result.segments, segment_columns))
        write("activity_sensitivity_per_flight.csv", _csv(sensitivity))
        write("activity_sensitivity_summary.csv", _csv(sensitivity_summary))
        write("unit_instance_diagnostics.json", format_json(dict(currtot_unit="unverified", unit_conversion_applied=False,
              unit_basis="TimeUS us, Volt V, Curr A follow supplied CSV contract; original logs absent",
              instance_policy=cfg["data"]["instance_policy"], per_flight=result.inventory,
              event_counts=event_counts, currtot_predictor_review_required=summary["currtot_predictor_review_required"])) + "\n")
        write("summary.json", format_json(summary) + "\n")
        write("data_audit.md", _audit_report(summary, activity, sensitivity_summary, settings))
    if config.data_path.read_bytes() != source_bytes:
        raise DataError("Input changed during preprocessing; no output written.")
    manifest = dict(run_id=run_id, provenance_sha256=provenance_hash,
                    outputs={name: hashlib.sha256(value).hexdigest() for name, value in sorted(payloads.items())})
    write("manifest.json", format_json(manifest) + "\n")
    # Refuse path escapes, raw-input collisions or differing content for the same run.
    for name, payload in payloads.items():
        target = project_path(output / name, config.root)
        if target == config.data_path or target == config.source:
            raise DataError(f"Output collides with an input: {target}")
        if target.exists() and target.read_bytes() != payload:
            raise DataError(f"Existing output differs for run {run_id}: {target}")
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (output / name).write_bytes(payload)
    except OSError as exc:
        raise DataError(f"Cannot write preprocessing outputs: {exc}") from exc
    if result.errors:
        raise DataError(f"Dataset rejected: {'; '.join(result.errors[:3])}. Diagnostics: {output / 'summary.json'}")
    return summary


def _audit_report(summary: dict, activity: list[dict], grid: list[dict], settings: dict) -> str:
    lines = ["# Sprint 2 — battery-only preprocessing and activity audit", "",
             f"Run `{summary['run_id']}`. Input {summary['input_rows']} rows; retained {summary['retained_rows']}; dropped {summary['dropped_rows']}.",
             f"{summary['flights']} flights; {summary['segments']} continuous segments; {summary['gaps']} gaps.",
             f"Observed duration: {summary['observed_duration_seconds']:.6f} s; excluded gaps: {summary['excluded_gap_seconds']:.6f} s.", "",
             "## Method", "",
             "Missing/nonfinite/invalid timestamp or instance rows are dropped with per-field reasons; no fill or interpolation. Sorting occurs after drop; dt/gaps are then recalculated. Duplicate timestamps and multiple instances reject the dataset. Negative telemetry is retained and reported. CurrTot decreases require review before Battery-B–E fitting.",
             f"Default activity: Curr >= {settings['min_current_a']} A, run duration >= {settings['min_duration_seconds']} s, flight active-time ratio >= {settings['min_active_time_percentage']}%.",
             "Observed duration sums intervals within continuous segments. Active exposure counts full durations of qualifying current-only runs, with both interval endpoints active. Run qualification and flight labels are retrospective audit outputs, never regression predictors. Row active_duration_seconds/active_confirmed are causal: no backfilling of the first second. Zero exposure is not eligible and its ratio is undefined (blank CSV/null JSON).",
             "Historical labels are provenance only, loaded after V2 classification. LOW_LOAD_OR_IDLE does not prove the UAV is on the ground. Minimum active-flight gate covers activity only; reference readiness is deferred to Stage 3.", "",
             "## Activity per flight", "", "| Flight | Observed s | Valid active s | Active % | Valid runs | V2 label |", "|---|---:|---:|---:|---:|---|"]
    for r in activity:
        percentage = f"{r['active_time_percentage']:.4f}" if r["active_time_percentage"] is not None else "undefined"
        lines.append(f"| {r['NamaFile']} | {r['observed_duration_seconds']:.4f} | {r['valid_active_duration_seconds']:.4f} | {percentage} | {r['valid_active_runs']} | {r['v2_current_based_label']} |")
    lines += ["", "## Frozen sensitivity audit", "", "No detection threshold or model parameter is selected here.", "",
              "| Current A | Min duration s | Active flights |", "|---:|---:|---:|"]
    lines += [f"| {r['min_current_a']} | {r['min_duration_seconds']} | {r['active_load_flights']} |" for r in grid]
    lines += ["", "## Unit and scope limits", "", "CurrTot remains unverified, without Ah/mAh conversion. TimeUS/Volt/Curr units follow the supplied CSV contract, without independent firmware/log verification. Healthy status is assumed from source; no measured battery fault ground truth exists. No reference voltage, causal features, model training or performance evaluation was performed.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)
