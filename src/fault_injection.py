"""Deterministic synthetic copies with explicit support and frozen reference."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import yaml

from .activity import classify_activity
from .config import AppConfig, project_path
from .errors import DataError
from .features import feature_columns, generate_features
from .preprocessing import PROCESSED_COLUMNS, _csv
from .provenance import SYNTHETIC_MARKERS
from .reporting import format_json

FAMILIES = ("increased_resistance", "progressive", "sudden_drop")
SEVERITIES = ("mild", "moderate", "severe")
MIN_POST_ONSET_SECONDS = 1.0  # Pre-model frozen experiment protocol, not detector tuning.
SYNTHETIC_COLUMNS = ("is_synthetic", "scenario_id", "fault_family", "fault_severity", "fault_start_position",
                     "fault_applicable", "fault_support", "fault_label", "fault_effect_present", "fault_phase",
                     "Volt_healthy", "voltage_sag_healthy", "dV_dt_healthy", "added_sag_v")


@dataclass
class FaultScenario:
    rows: list[dict]
    metadata: dict


def observability(max_added_sag_v: float, thresholds: dict) -> str:
    if not math.isfinite(max_added_sag_v) or max_added_sag_v < 0:
        raise DataError("Observability requires finite nonnegative added sag.")
    if max_added_sag_v < thresholds["very_low"]:
        return "VERY_LOW"
    if max_added_sag_v < thresholds["low"]:
        return "LOW"
    if max_added_sag_v < thresholds["medium"]:
        return "MEDIUM"
    return "HIGH"


def _validate_source(rows: list[dict], config: AppConfig, flight_id: str) -> None:
    previous = None
    locked_reference = None
    for row in rows:
        if row.get("NamaFile") != flight_id:
            raise DataError("A fault scenario must contain exactly one flight.")
        if SYNTHETIC_MARKERS.intersection(row):
            raise DataError("Cannot inject an already synthetic/scenario source.")
        for name in ("TimeUS", "Inst", "segment_id"):
            if type(row.get(name)) is not int or row[name] < 0:
                raise DataError(f"Fault input requires nonnegative integer {name}.")
        for name in ("Volt", "Curr", "CurrTot", "dV_dt"):
            if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]):
                raise DataError(f"Fault input requires finite {name}.")
        if previous:
            if row["Inst"] != previous["Inst"]:
                raise DataError("Multiple instances cannot be mixed in a fault scenario.")
            if row["TimeUS"] <= previous["TimeUS"]:
                raise DataError("Fault input timestamps must strictly increase.")
            boundary = (row["TimeUS"] - previous["TimeUS"]) / 1e6 > config.gap_threshold_seconds
            if row["segment_id"] != previous["segment_id"] + int(boundary):
                raise DataError("Fault input segment_id disagrees with actual timestamp gaps.")
        if type(row.get("reference_ready")) is not bool:
            raise DataError("Fault input reference_ready must be Boolean.")
        if row["reference_ready"]:
            if (type(row.get("V_ref")) not in (int, float) or not math.isfinite(row["V_ref"])
                    or type(row.get("voltage_sag_actual")) not in (int, float)
                    or not math.isclose(row["voltage_sag_actual"], row["V_ref"] - row["Volt"], abs_tol=1e-12)):
                raise DataError("Fault input requires an existing finite locked reference and consistent target.")
            if locked_reference is not None and row["V_ref"] != locked_reference:
                raise DataError("Fault source reference must already be locked, never recomputed across gaps.")
            locked_reference = row["V_ref"]
        elif locked_reference is not None:
            raise DataError("Fault source cannot become unready after reference lock.")
        previous = row


def _candidates(rows: list[dict], config: AppConfig, family: str) -> list[dict]:
    if not rows:
        return []
    if family != "sudden_drop":
        activity = config.as_dict()["activity"]
        _, _, runs = classify_activity(rows, **{key: activity[key] for key in
                                                ("min_current_a", "min_duration_seconds", "min_active_time_percentage")})
        return [dict(Inst=r["Inst"], segment_id=r["segment_id"], active_run_id=r["active_run_id"],
                     start_time_us=r["start_time_us"], end_time_us=r["end_time_us"], duration_seconds=r["duration_seconds"])
                for r in runs if r["is_valid"]]
    # The continuous READY portion can start after warm-up, inside a segment.
    groups = defaultdict(list)
    for row in rows:
        if row["reference_ready"]:
            groups[(row["Inst"], row["segment_id"])].append(row)
    return [dict(Inst=instance, segment_id=segment, active_run_id=None,
                 start_time_us=block[0]["TimeUS"], end_time_us=block[-1]["TimeUS"],
                 duration_seconds=(block[-1]["TimeUS"] - block[0]["TimeUS"]) / 1e6)
            for (instance, segment), block in groups.items()]


def inject_fault(rows: list[dict], config: AppConfig, *, family: str, severity: str,
                 start_position: float, flight_id: str | None = None) -> FaultScenario:
    """Modify only a copy. No model inference, fit or reference re-estimation.

    Support begins at the aligned onset, even if a ramp has zero effect there.
    first_effect_time_us records the first nonzero perturbation separately.
    """
    if family not in FAMILIES or severity not in SEVERITIES:
        raise DataError("Unknown synthetic fault family/severity.")
    if type(start_position) not in (float, int) or not math.isfinite(start_position) or not 0 < start_position < 1:
        raise DataError("Fault start position must lie strictly between 0 and 1.")
    flight_id = flight_id if flight_id is not None else rows[0]["NamaFile"] if rows else None
    if not isinstance(flight_id, str) or not flight_id.strip():
        raise DataError("Fault scenario requires a flight ID, including when no rows remain.")
    _validate_source(rows, config, flight_id)
    cfg = config.as_dict()
    settings = cfg["fault_injection"]
    key = dict(NamaFile=flight_id, family=family, severity=severity, start_position=float(start_position))
    scenario_id = hashlib.sha256(format_json(key).encode()).hexdigest()[:20]
    dose_key = {"increased_resistance": "resistance_ohm", "progressive": "progressive_max_delta_r_ohm", "sudden_drop": "sudden_drop_v"}[family]
    dose = settings[dose_key][severity]
    candidates = _candidates(rows, config, family)
    selected = min(candidates, key=lambda r: (-r["duration_seconds"], r["start_time_us"], r["Inst"], r["segment_id"])) if candidates else None
    metadata = dict(scenario_id=scenario_id, NamaFile=flight_id, fault_family=family, severity=severity,
                    start_position=float(start_position), configured_dose=float(dose),
                    dose_unit="V" if family == "sudden_drop" else "ohm", sudden_ramp_seconds=settings["sudden_ramp_seconds"] if family == "sudden_drop" else None,
                    selection_scope="continuous_ready_segment" if family == "sudden_drop" else "valid_active_run",
                    min_post_onset_seconds=MIN_POST_ONSET_SECONDS, fault_applicable=False,
                    evaluation_status="NOT_APPLICABLE", reason=None, selected_instance=None,
                    selected_segment_id=None, selected_active_run_id=None, segment_start_time_us=None,
                    segment_end_time_us=None, selected_segment_duration_seconds=None, requested_onset_time_us=None,
                    candidate_onset_time_us=None, post_onset_available_seconds=None, onset_time_us=None,
                    first_effect_time_us=None, fault_end_time_us=None, fault_duration_seconds=None,
                    current_at_onset_a=None, mean_current_after_fault_a=None, support_samples=0, effect_samples=0,
                    max_added_sag_v=None, mean_added_sag_v=None, observability="NOT_APPLICABLE",
                    frozen_v_ref=None, post_fault_original_start_time_us=None,
                    ending_policy="original_telemetry_at_next_sample_no_synthetic_recovery_ramp",
                    reference_policy="preserve_existing_locked_V_ref_and_readiness", is_synthetic=True)
    support = []
    if not rows:
        metadata["reason"] = "NO_RETAINED_SAMPLES"
    elif not any(r["reference_ready"] for r in rows):
        metadata["reason"] = "WARMUP_INCOMPLETE" if rows[-1].get("reference_status") == "WARMUP" else "REFERENCE_UNAVAILABLE"
    elif selected is None:
        metadata["reason"] = "NO_READY_SEGMENT" if family == "sudden_drop" else "NO_VALID_ACTIVE_SEGMENT"
    else:
        metadata.update(selected_instance=selected["Inst"], selected_segment_id=selected["segment_id"],
                        selected_active_run_id=selected["active_run_id"], segment_start_time_us=selected["start_time_us"],
                        segment_end_time_us=selected["end_time_us"], selected_segment_duration_seconds=selected["duration_seconds"])
        # Decimal avoids rounding an absolute timestamp through binary float.
        requested = Decimal(selected["start_time_us"]) + Decimal(str(start_position)) * (selected["end_time_us"] - selected["start_time_us"])
        block = [r for r in rows if r["Inst"] == selected["Inst"] and r["segment_id"] == selected["segment_id"]
                 and selected["start_time_us"] <= r["TimeUS"] <= selected["end_time_us"]]
        onset = next(r for r in block if Decimal(r["TimeUS"]) >= requested)
        available = (selected["end_time_us"] - onset["TimeUS"]) / 1e6
        metadata.update(requested_onset_time_us=str(requested), candidate_onset_time_us=onset["TimeUS"],
                        post_onset_available_seconds=available)
        tail = [r for r in block if r["TimeUS"] >= onset["TimeUS"]]
        if not onset["reference_ready"] or any(not r["reference_ready"] for r in tail):
            metadata["reason"] = "ONSET_NOT_REFERENCE_READY"
        elif available < MIN_POST_ONSET_SECONDS:
            metadata["reason"] = "INSUFFICIENT_POST_ONSET_SUPPORT"
        else:
            support = tail
            metadata.update(fault_applicable=True, evaluation_status="NOT_EVALUATED", reason="APPLICABLE",
                            onset_time_us=onset["TimeUS"], fault_end_time_us=support[-1]["TimeUS"],
                            fault_duration_seconds=available, current_at_onset_a=onset["Curr"],
                            mean_current_after_fault_a=float(mean(r["Curr"] for r in support)),
                            support_samples=len(support), frozen_v_ref=onset["V_ref"])
            metadata["post_fault_original_start_time_us"] = next((r["TimeUS"] for r in rows if r["TimeUS"] > support[-1]["TimeUS"]), None)
    support_times = {r["TimeUS"] for r in support}
    result, additions = [], []
    for row in rows:
        inside = row["TimeUS"] in support_times
        added = 0.0
        if inside:
            elapsed = (row["TimeUS"] - metadata["onset_time_us"]) / 1e6
            if family == "increased_resistance":
                added = row["Curr"] * dose
            elif family == "progressive":
                progress = min(1.0, max(0.0, elapsed / metadata["fault_duration_seconds"]))
                added = row["Curr"] * dose * progress
            else:
                ramp = settings["sudden_ramp_seconds"]
                added = dose * (min(1.0, elapsed / ramp) if ramp > 0 else 1.0)
            if not math.isfinite(added) or added < 0:
                raise DataError("Synthetic added sag must be finite and nonnegative.")
            additions.append(added)
            if added > 0 and metadata["first_effect_time_us"] is None:
                metadata["first_effect_time_us"] = row["TimeUS"]
        phase = ("NOT_APPLICABLE" if not metadata["fault_applicable"] else "FAULT_SUPPORT" if inside
                 else "PRE_FAULT" if row["TimeUS"] < metadata["onset_time_us"] else "POST_FAULT_ORIGINAL")
        item = {**row, "Volt_healthy": row["Volt"], "voltage_sag_healthy": row["voltage_sag_actual"],
                "dV_dt_healthy": row["dV_dt"], "Volt": row["Volt"] - added,
                "voltage_sag_actual": row["V_ref"] - (row["Volt"] - added) if row["reference_ready"] else None,
                "added_sag_v": float(added), "is_synthetic": True, "scenario_id": scenario_id,
                "fault_family": family, "fault_severity": severity, "fault_start_position": float(start_position),
                "fault_applicable": metadata["fault_applicable"], "fault_support": inside,
                "fault_label": int(inside), "fault_effect_present": added > 0, "fault_phase": phase}
        if not math.isfinite(item["Volt"]) or (item["voltage_sag_actual"] is not None and not math.isfinite(item["voltage_sag_actual"])):
            raise DataError("Synthetic voltage/target is nonfinite.")
        result.append(item)
    # Direct voltage-behaviour baseline must see the injected voltage derivative,
    # including return to original telemetry immediately after support ends.
    for index, row in enumerate(result):
        before = result[index - 1] if index else None
        boundary = before is None or row["segment_id"] != before["segment_id"]
        row["dV_dt"] = 0.0 if boundary else (row["Volt"] - before["Volt"]) / ((row["TimeUS"] - before["TimeUS"]) / 1e6)
        if not math.isfinite(row["dV_dt"]):
            raise DataError("Synthetic voltage derivative is nonfinite.")
    if support:
        metadata.update(max_added_sag_v=max(additions), mean_added_sag_v=float(mean(additions)),
                        effect_samples=sum(value > 0 for value in additions),
                        observability=observability(max(additions), settings["observability_max_sag_v"]))
    return FaultScenario(result, metadata)


def _write(path: Path, payload: bytes, config: AppConfig) -> None:
    path = project_path(path, config.root)
    if path in (config.data_path, config.source):
        raise DataError("Synthetic output collides with an input.")
    if path.exists():
        if path.read_bytes() != payload:
            raise DataError(f"Existing synthetic output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def generate_faults(config: AppConfig) -> dict:
    """Build an auditable scenario bank. Never fit/predict/evaluate a detector."""
    from .models import _feature_input
    from .fault_reporting import fault_report, fault_examples
    cfg = config.as_dict()
    feature_summary = generate_features(config)
    rows, coverage, _, upstream_hash = _feature_input(feature_summary, config)
    code = {f"src/{name}": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("fault_injection.py", "fault_reporting.py", "models.py", "provenance.py")}
    protocol = dict(stage=5, families=list(FAMILIES), severities=list(SEVERITIES),
                    start_positions=cfg["fault_injection"]["start_positions"], config_faults=cfg["fault_injection"],
                    min_post_onset_seconds=MIN_POST_ONSET_SECONDS, segment_tie_break="longest_duration_then_earliest_timestamp",
                    current_dependent_selection="longest valid active run; no silent shift if its aligned onset is not ready",
                    sudden_selection="longest continuous ready segment; low-load allowed",
                    onset_policy="first sample at/after fractional timestamp; ramp onset may have zero effect; first_effect recorded separately",
                    ending_policy="original telemetry at next sample; no synthetic recovery ramp",
                    reference_policy="copy existing V_ref/readiness; never recompute reference",
                    scope="scenario_generation_only_no_model_or_detector_evaluation",
                    not_applicable_policy="metadata/reason retained, no scenario CSV, no confusion-count eligibility",
                    feature_manifest_sha256=upstream_hash, data_sha256=feature_summary["data_sha256"],
                    config_sha256=feature_summary["config_sha256"], dependency_versions=feature_summary["dependency_versions"])
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "fault_injection" / run_id, config.root)
    synthetic = project_path(config.root / "data/synthetic" / run_id, config.root)
    completion = output / "manifest.json"
    if completion.exists():
        try:
            manifest = json.loads(completion.read_text())
            if manifest["run_id"] != run_id or manifest["identity"] != identity:
                raise DataError("Synthetic run identity mismatch.")
            for relative, digest in manifest["outputs"].items():
                if hashlib.sha256(project_path(relative, config.root).read_bytes()).hexdigest() != digest:
                    raise DataError(f"Synthetic artifact checksum mismatch: {relative}")
            return json.loads((output / "summary.json").read_text())
        except (ValueError, KeyError) as exc:
            raise DataError(f"Invalid completed synthetic manifest: {exc}") from exc
    _write(output / "scenario_protocol.json", (format_json(protocol) + "\n").encode(), config)
    _write(output / "config_used.yaml", yaml.safe_dump(cfg, sort_keys=False).encode(), config)
    by_flight = defaultdict(list)
    for row in rows:
        by_flight[row["NamaFile"]].append(row)
    active = sorted(r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True")
    low = sorted(r["NamaFile"] for r in coverage if r["v2_current_based_label"] == "LOW_LOAD_OR_IDLE" and r["reference_status"] == "READY")
    active_example = active[min(1, len(active) - 1)] if active else None
    low_example = low[0] if low else None
    position_example = sorted(cfg["fault_injection"]["start_positions"])[len(cfg["fault_injection"]["start_positions"]) // 2]
    examples, metadata, checksums = [], [], {}
    columns = (*PROCESSED_COLUMNS, *feature_columns(cfg["features"]["rolling_windows_seconds"]), *SYNTHETIC_COLUMNS)
    for flight in sorted(r["NamaFile"] for r in coverage):
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    scenario = inject_fault(by_flight[flight], config, family=family, severity=severity,
                                            start_position=position, flight_id=flight)
                    item = {**scenario.metadata, "scenario_csv": None, "scenario_sha256": None}
                    if item["fault_applicable"]:
                        target = synthetic / (item["scenario_id"] + ".csv.gz")
                        value = gzip.compress(_csv(scenario.rows, columns), compresslevel=6, mtime=0)
                        _write(target, value, config)
                        relative = target.relative_to(config.root).as_posix()
                        item.update(scenario_csv=relative, scenario_sha256=hashlib.sha256(value).hexdigest())
                        checksums[relative] = item["scenario_sha256"]
                        if severity == "severe" and position == position_example and (
                                flight == active_example or flight == low_example and family == "sudden_drop"):
                            examples.append(scenario)
                    metadata.append(item)
    applicable = [r for r in metadata if r["fault_applicable"]]
    reason_counts = dict(sorted(Counter(r["reason"] for r in metadata if not r["fault_applicable"]).items()))
    counts = []
    for family in FAMILIES:
        for severity in SEVERITIES:
            selected = [r for r in metadata if r["fault_family"] == family and r["severity"] == severity]
            valid = [r for r in selected if r["fault_applicable"]]
            counts.append(dict(fault_family=family, severity=severity, scenarios=len(selected), applicable=len(valid),
                               not_applicable=len(selected) - len(valid), **{label: sum(r["observability"] == label for r in valid)
                                                                          for label in ("VERY_LOW", "LOW", "MEDIUM", "HIGH")}))
    summary = dict(status="OK", command="inject-faults", run_id=run_id, output_directory=str(output),
                   synthetic_directory=str(synthetic), feature_run_id=feature_summary["run_id"],
                   feature_directory=feature_summary["output_directory"], feature_manifest_sha256=upstream_hash,
                   data_sha256=feature_summary["data_sha256"], flights=len(coverage), scenarios=len(metadata),
                   applicable_scenarios=len(applicable), not_applicable_scenarios=len(metadata) - len(applicable),
                   not_applicable_reasons=reason_counts, scenario_files=len(checksums),
                   synthetic_rows=sum(int(r["retained_samples"]) * sum(m["fault_applicable"] for m in metadata if m["NamaFile"] == r["NamaFile"]) for r in coverage),
                   observability_counts=dict(sorted(Counter(r["observability"] for r in applicable).items())),
                   observability_scope="max synthetic added sag; not battery health diagnosis", model_training_performed=False,
                   model_prediction_performed=False, detector_evaluation_performed=False,
                   recall_f1_delay_deadlines="NOT_MEASURED_STAGE_6", currtot_unit="unverified",
                   example_scenarios=[r.metadata["scenario_id"] for r in examples])
    applicability_columns = ("scenario_id", "NamaFile", "fault_family", "severity", "start_position", "fault_applicable", "evaluation_status", "reason",
                             "candidate_onset_time_us", "post_onset_available_seconds", "min_post_onset_seconds", "scenario_csv")
    observability_columns = ("scenario_id", "NamaFile", "fault_family", "severity", "start_position", "fault_applicable", "evaluation_status", "observability",
                             "max_added_sag_v", "mean_added_sag_v", "current_at_onset_a", "mean_current_after_fault_a")
    payloads = {"fault_manifest.csv": _csv(metadata),
                "fault_applicability.csv": _csv([{key: row[key] for key in applicability_columns} for row in metadata], applicability_columns),
                "fault_observability.csv": _csv([{key: row[key] for key in observability_columns} for row in metadata], observability_columns),
                "observability_summary.csv": _csv(counts), "summary.json": (format_json(summary) + "\n").encode(),
                "fault_report.md": fault_report(summary, counts).encode("utf-8")}
    if examples:
        payloads["example_faults.png"] = fault_examples(examples)
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise DataError("Input changed during fault generation; no completion manifest written.")
    for name, value in payloads.items():
        target = output / name
        _write(target, value, config)
        checksums[target.relative_to(config.root).as_posix()] = hashlib.sha256(value).hexdigest()
    for name in ("scenario_protocol.json", "config_used.yaml"):
        target = output / name
        checksums[target.relative_to(config.root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write(completion, (format_json(dict(run_id=run_id, identity=identity, outputs=dict(sorted(checksums.items())))) + "\n").encode(), config)
    return summary
