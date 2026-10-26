"""Frozen baseline replay, timestamp equivalence and monotonic latency evidence."""

from array import array
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from .config import project_path
from .data_loader import load_battery_csv
from .errors import DataError
from .evaluation import aggregate_scenarios, scenario_metrics
from .feature_sets import BATTERY_COLUMNS
from .features import feature_columns
from .low_load import load_sources as load_baseline_sources
from .models import SCORE_COLUMNS, _write_files
from .pilot_reporting import alarm_metrics
from .preprocessing import _csv
from .replay import StreamingDetector, TIMING_COLUMNS
from .research import same_metrics
from .research_sources import audit_inputs, encoded, load, sha, verify_outputs
from .residual_diagnostics import diagnose, diagnostic_columns

DECISION_COLUMNS = ("predicted_voltage_sag", "residual_signed_v", "anomaly_score", "anomaly_raw",
                    "anomaly_duration_seconds", "anomaly_confirmed", "confirmed_episode_start", "detector_state")


def read_scores(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    booleans = {"reference_ready", "anomaly_raw", "anomaly_confirmed", "confirmed_episode_start", "fault_support"}
    for r in rows:
        for name in ("TimeUS", "Inst", "source_record", "segment_id"):
            r[name] = int(r[name])
        for name in ("Volt", "Curr", "V_ref", "voltage_sag_actual", "predicted_voltage_sag", "residual_signed_v", "anomaly_score", "anomaly_duration_seconds"):
            r[name] = float(r[name]) if r[name] != "" else None
        for name in booleans.intersection(r):
            if r[name] not in ("True", "False"):
                raise DataError("Frozen score file contains an invalid Boolean.")
            r[name] = r[name] == "True"
    return rows


def load_replay_sources(config, *, baseline_summary=None, low_summary=None):
    research_reference = None
    pointer = config.artifacts_path / "sprint11/research_result.json"
    if baseline_summary is None and pointer.exists():
        research = load(pointer)
        directory = project_path(research["output_directory"], config.root)
        verify_outputs(directory, config)
        if load(directory / "summary.json") != research:
            raise DataError("Sprint 11 pointer differs from its completed research summary.")
        inventory = {r["command"]: r for r in load(directory / "source_inventory.json")}
        references = {}
        for command in ("cross-validate", "evaluate-low-load"):
            ref = inventory[command]
            source = project_path(ref["output_directory"], config.root)
            if sha(source / "manifest.json") != ref["manifest_sha256"]:
                raise DataError("Replay source differs from the Sprint 11 frozen source set.")
            value = load(source / ("results.json" if command == "cross-validate" else "summary.json"))
            references[command] = value["run_summary"] if command == "cross-validate" else value
        baseline_summary, low_summary = references["cross-validate"], references["evaluate-low-load"]
        research_reference = dict(run_id=research["run_id"], manifest_sha256=sha(directory / "manifest.json"))
    source, baseline, features, coverage, registry, bundles = (load_baseline_sources(config) if baseline_summary is None
        else load_baseline_sources(config, baseline_summary))
    if low_summary is None:
        path = config.artifacts_path / "sprint10/low_load_result.json"
        if not path.exists():
            raise DataError("Replay requires completed frozen low-load results; no automatic training/evaluation.")
        low_summary = load(path)
    low_dir = project_path(low_summary["output_directory"], config.root)
    if not low_dir.is_relative_to(config.artifacts_path / "low_load"):
        raise DataError("Replay low-load source leaves its artifact directory.")
    verify_outputs(low_dir, config)
    if (load(low_dir / "summary.json") != low_summary or low_summary["baseline_run_id"] != baseline["run_id"]
            or low_summary["feature_manifest_sha256"] != baseline["feature_manifest_sha256"]
            or not low_summary["all_low_load_unseen_by_detector"] or low_summary["threshold_changed"]):
        raise DataError("Replay low-load source has incompatible baseline/lineage/unseen policy.")
    return dict(source=source, baseline=baseline, features=features, coverage=coverage, registry=registry,
                bundles=bundles, low_summary=low_summary, low_dir=low_dir, research_reference=research_reference)


def compare_row(actual, expected, names, maxima):
    for name in names:
        a, b = actual[name], expected[name]
        if b is None or type(b) in (bool, str):
            equal = type(a) is type(b) and a == b
        else:
            equal = type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
            if equal:
                maxima[name] = max(maxima.get(name, 0.), abs(a - b))
        if not equal:
            raise DataError(f"Replay/batch mismatch {actual.get('NamaFile')}/{actual.get('TimeUS')}/{name}: {a!r} != {b!r}")


def latency_stats(values, budget_ns):
    if not values:
        return dict(samples=0, mean_ms=None, p50_ms=None, p95_ms=None, p99_ms=None, max_ms=None, over_nominal_budget_samples=0)
    v = np.asarray(values, dtype=np.uint64)
    return dict(samples=len(v), mean_ms=float(v.mean()) / 1e6, p50_ms=float(np.percentile(v, 50)) / 1e6,
                p95_ms=float(np.percentile(v, 95)) / 1e6, p99_ms=float(np.percentile(v, 99)) / 1e6,
                max_ms=int(v.max()) / 1e6, over_nominal_budget_samples=int((v > budget_ns).sum()))


def pool_healthy(records):
    n = sum(r["ready_samples"] for r in records)
    raw = sum(r["raw_alarm_samples"] for r in records)
    confirmed = sum(r["confirmed_alarm_samples"] for r in records)
    episodes = sum(r["confirmed_episodes"] for r in records)
    duration = sum(round(r["observed_ready_seconds"] * 1e6) for r in records) / 1e6
    return dict(ready_samples=n, raw_alarm_samples=raw, confirmed_alarm_samples=confirmed, confirmed_episodes=episodes,
                observed_ready_seconds=duration, raw_fpr=raw / n if n else None, confirmed_fpr=confirmed / n if n else None,
                confirmed_episodes_per_minute=60 * episodes / duration if duration else None)


def run_replay(config, *, sources=None):
    from .replay_reporting import replay_report, replay_plots
    audit = audit_inputs(config)
    sources = load_replay_sources(config) if sources is None else sources
    source, baseline = sources["source"], sources["baseline"]
    bundles = {fold: bundle for fold, bundle, *_ in sources["bundles"]}
    originals = {r["source_record"]: r for r in sources["features"]}
    source_policies = [dict(fold_id=f, bundle_sha256=sha(p), threshold_sha256=sha(t), split=b["split"], threshold=b["threshold"])
                       for f, b, p, t in sources["bundles"]]
    protocol = dict(experiment="sprint12_frozen_baseline_streaming_replay", scope="software_equivalence_and_host_latency_only",
                    raw_data_sha256=baseline["data_sha256"], baseline_run_id=baseline["run_id"],
                    baseline_manifest_sha256=sha(source / "manifest.json"), feature_manifest_sha256=baseline["feature_manifest_sha256"],
                    low_load_run_id=sources["low_summary"]["run_id"], low_load_manifest_sha256=sha(sources["low_dir"] / "manifest.json"),
                    research_reference=sources["research_reference"], policies=source_policies,
                    healthy_rule="all active flights at their own held-out fold; all low-load flights under frozen primary fold",
                    synthetic_rule="all frozen baseline outer cases, fresh feature/slope/timer state per case; NA skipped",
                    prediction_rule="healthy rows always uncached single-sample model calls; synthetic may reuse only exact ordered predictor values AND identical model object from already observed healthy replay",
                    cache_scope="prediction only; features, residual, slope, persistence and decisions always recomputed causally",
                    timing_clock="time.perf_counter_ns", timing_scope="in-process parse+features+prediction+diagnostics/detector; excludes model loading, disk I/O and export",
                    latency_policy="uncached healthy ready, unready and synthetic exact-input cache samples reported separately; first inference retained",
                    pacing="accelerated timestamp-driven; no sleep, nominal 10 Hz is a reporting budget, not imposed timing",
                    signal_delay_policy="telemetry onset to first uncontaminated confirmed sample inside support; persistence already included",
                    equivalence_tolerance=dict(absolute=1e-10, relative=1e-10), nominal_rate_hz=config.as_dict()["replay"]["nominal_rate_hz"],
                    deployment_enabled=False, emergency_candidate_replayed=False, threshold_changed=False,
                    current_dependency_versions=audit["current_dependency_versions"])
    code_root = Path(__file__).resolve().parents[1]
    code = {"src/" + n: sha(code_root / "src" / n) for n in
            ("replay.py", "replay_experiment.py", "replay_reporting.py", "features.py", "models.py", "preprocessing.py", "persistence.py", "residual_diagnostics.py", "detectors.py")}
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(encoded(identity)).hexdigest()[:16]
    output = project_path(config.artifacts_path / "replay" / run_id, config.root)
    if (output / "manifest.json").exists():
        manifest = verify_outputs(output, config)
        if manifest["identity"] != identity:
            raise DataError("Replay cache identity mismatch.")
        return load(output / "summary.json")
    hashes = {}

    def save(name, value):
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        _write_files(output, {name: value}, config)
        hashes[name] = hashlib.sha256(value).hexdigest()

    save("replay_protocol.json", encoded(protocol))
    save("config_used.yaml", yaml.safe_dump(config.as_dict(), sort_keys=False).encode())
    save("feature_registry.json", encoded(sources["registry"]))
    freeze = dict(hashes)
    save("pre_replay_freeze.json", encoded(freeze))
    raw = load_battery_csv(config.data_path)
    raw_groups = defaultdict(list)
    for record, row in enumerate(raw, 2):
        raw_groups[row["NamaFile"]].append(dict(row, source_record=record))
    cache_by_fold = {fold: {} for fold in bundles}
    maxima, jobs, health, faults, modes = {}, [], [], [], Counter()
    timings = defaultdict(lambda: {name: array("Q") for name in TIMING_COLUMNS})
    columns = (*BATTERY_COLUMNS, "source_record", "segment_id", "dt_seconds", "gap_before_seconds", "is_segment_start",
               *feature_columns(config.as_dict()["features"]["rolling_windows_seconds"]),
               *[c for c in DECISION_COLUMNS], *[c for c in diagnostic_columns(config.as_dict()["dual_path"]["slope_windows_seconds"]) if c != "dI_dt"],
               "accepted", "prediction_mode", "fault_support", *TIMING_COLUMNS)
    feature_names = ("segment_id", "dt_seconds", "gap_before_seconds", "is_segment_start", *feature_columns(config.as_dict()["features"]["rolling_windows_seconds"]))
    windows = config.as_dict()["dual_path"]["slope_windows_seconds"]
    diag_names = [n for n in diagnostic_columns(windows) if n != "dI_dt"]
    deadlines = config.as_dict()["evaluation"]["deadlines_seconds"]
    budget_ns = 1e9 / protocol["nominal_rate_hz"]

    def execute(job_id, fold, inputs, golden, cohort):
        if len(inputs) != len(golden):
            raise DataError("Replay source input/golden row count differs.")
        detector = StreamingDetector(bundles[fold], config, prediction_cache=cache_by_fold[fold], use_cache=cohort == "synthetic")
        decisions, previous, expected_features = [], None, []
        for inp, g in zip(inputs, golden):
            expected = dict(originals[g["source_record"]])
            if inp["NamaFile"] != expected["NamaFile"] or int(inp["TimeUS"]) != expected["TimeUS"]:
                raise DataError("Replay raw/golden/source identities differ.")
            expected["Volt"] = float(inp["Volt"])
            expected["voltage_sag_actual"] = expected["V_ref"] - expected["Volt"] if expected["reference_ready"] else None
            expected["dV_dt"] = ((expected["Volt"] - previous["Volt"]) / expected["dt_seconds"]
                                      if previous and not expected["is_segment_start"] else 0.)
            actual = detector.consume(inp)
            if not actual["accepted"]:
                raise DataError("Replay dropped a row retained in the frozen batch evaluation.")
            compare_row(actual, expected, feature_names, maxima)
            compare_row(actual, g, DECISION_COLUMNS, maxima)
            actual["fault_support"] = g.get("fault_support", False)
            decisions.append(actual)
            expected_features.append(dict(expected, **{n: g[n] for n in ("predicted_voltage_sag", "residual_signed_v", "anomaly_score")}))
            previous = expected
            modes[actual["prediction_mode"]] += 1
            timing_group = cohort + "/" + actual["prediction_mode"]
            for name in TIMING_COLUMNS:
                timings[timing_group][name].append(actual[name])
            assert actual["end_to_end_ns"] == sum(actual[n] for n in TIMING_COLUMNS[:-1])
        batch_diagnostics = diagnose(expected_features, windows, config.gap_threshold_seconds)
        for actual, expected in zip(decisions, batch_diagnostics):
            compare_row(actual, expected, diag_names, maxima)
        filename = f"traces/{cohort}/{job_id}.csv.gz"
        save(filename, gzip.compress(_csv([{n: r[n] for n in columns} for r in decisions], columns), compresslevel=6, mtime=0))
        jobs.append(dict(job_id=job_id, cohort=cohort, fold_id=fold, NamaFile=inputs[0]["NamaFile"] if inputs else None,
                         rows=len(decisions), reference_ready_samples=sum(r["reference_ready"] for r in decisions),
                         equivalence="PASS", trace_csv=filename, trace_sha256=hashes[filename]))
        return decisions

    with threadpool_limits(limits=1, user_api="openmp"):
        # All original flights are processed before synthetic cache lookups.
        for fold, bundle in sorted(bundles.items()):
            for flight in bundle["split"]["test"]:
                golden = read_scores(source / "folds" / fold / "healthy_test_scores.csv.gz")
                rows = execute(fold + "_" + flight, fold, raw_groups[flight], golden, "healthy_active")
                health.append(dict(cohort="healthy_active", fold_id=fold, NamaFile=flight,
                                   **alarm_metrics(rows, config.gap_threshold_seconds)))
        primary = sources["low_summary"]["primary_fold"]
        for flight in sorted(c["NamaFile"] for c in sources["coverage"] if c["v2_current_based_label"] == "LOW_LOAD_OR_IDLE"):
            golden = read_scores(sources["low_dir"] / "scores" / primary / (flight + ".csv.gz"))
            rows = execute(primary + "_" + flight, primary, raw_groups[flight], golden, "healthy_low_load")
            health.append(dict(cohort="healthy_low_load", fold_id=primary, NamaFile=flight,
                               **alarm_metrics(rows, config.gap_threshold_seconds)))
        with (source / "fault_per_scenario.csv").open(encoding="utf-8", newline="") as stream:
            planned = list(csv.DictReader(stream))
        for case in planned:
            applicable = case["fault_applicable"] == "True"
            metadata = dict(fault_applicable=applicable, onset_time_us=int(case["onset_time_us"]) if applicable else None,
                            fault_end_time_us=int(case["fault_end_time_us"]) if applicable else None)
            rows = []
            if applicable:
                golden_path = project_path(source / case["scores_csv"], config.root)
                if sha(golden_path) != case["scores_sha256"]:
                    raise DataError("Frozen synthetic batch score checksum mismatch.")
                golden = read_scores(golden_path)
                inputs = [{**{n: originals[g["source_record"]][n] for n in BATTERY_COLUMNS},
                           "source_record": g["source_record"], "Volt": g["Volt"]} for g in golden]
                rows = execute(case["fold_id"] + "_" + case["scenario_id"], case["fold_id"], inputs, golden, "synthetic")
            metrics = scenario_metrics(rows, metadata, deadlines, config.gap_threshold_seconds)
            for name in ("incident_detected", "preexisting_alarm", "evaluation_status"):
                expected = None if case[name] == "" else case[name] == "True" if name != "evaluation_status" else case[name]
                if metrics[name] != expected:
                    raise DataError("Replay incident classification differs from frozen batch.")
            for name in ("delay_seconds", "first_confirmation_time_us", "tp", "fp", "tn", "fn"):
                expected = float(case[name]) if case[name] != "" else None
                same_metrics({name: metrics[name]}, {name: expected}, "replay_case")
            warning = next((r for r in rows if r["TimeUS"] == metrics["first_confirmation_time_us"]), None)
            faults.append(dict(fold_id=case["fold_id"], scenario_id=case["scenario_id"], NamaFile=case["NamaFile"],
                               fault_family=case["fault_family"], severity=case["severity"], observability=case["observability"],
                               confirmation_processing_latency_ms=warning["end_to_end_ns"] / 1e6 if warning else None,
                               raw_run_start_time_us=warning["TimeUS"] - round(warning["anomaly_duration_seconds"] * 1e6) if warning else None,
                               persistence_elapsed_at_confirmation_seconds=warning["anomaly_duration_seconds"] if warning else None, **metrics))
    active = pool_healthy([r for r in health if r["cohort"] == "healthy_active"])
    low = pool_healthy([r for r in health if r["cohort"] == "healthy_low_load"])
    synthetic = aggregate_scenarios(faults, deadlines)
    same_metrics(active, baseline["healthy_active_pooled"], "replay_active")
    same_metrics(low, {k: v for k, v in sources["low_summary"]["primary_low_load"].items() if k in low}, "replay_low")
    same_metrics(synthetic, baseline["synthetic_pooled"], "replay_synthetic")
    latency = [dict(group=group, phase=name, nominal_budget_ms=budget_ns / 1e6, **latency_stats(values, budget_ns))
               for group, phases in sorted(timings.items()) for name, values in phases.items()]
    uncached = array("Q")
    for group, phases in timings.items():
        if group.startswith("healthy_") and group.endswith("/UNCACHED"):
            uncached.extend(phases["end_to_end_ns"])
    summary = dict(status="OK", command="replay", run_id=run_id, output_directory=str(output),
                   scope=protocol["scope"], baseline_run_id=baseline["run_id"], data_sha256=baseline["data_sha256"],
                   replay_jobs=len(jobs), replayed_rows=sum(j["rows"] for j in jobs), batch_stream_equivalence="PASS",
                   comparison_columns=len(set([*feature_names, *DECISION_COLUMNS, *diag_names])), numeric_max_absolute_errors=maxima,
                   prediction_mode_counts=dict(modes), uncached_healthy_end_to_end=latency_stats(uncached, budget_ns),
                   healthy_active=active, healthy_low_load=low, synthetic=synthetic,
                   model_training_performed=False, threshold_changed=False, operational_config_changed=False,
                   emergency_candidate_replayed=False, dual_promotion="NOT_PROMOTED", paced_replay_performed=False,
                   telemetry_signal_delay_includes_persistence=True, deployment_status="BASELINE_SOFTWARE_REPLAY_ONLY")
    save("jobs.csv", _csv(jobs))
    save("healthy_per_flight.csv", _csv(health))
    save("fault_per_scenario.csv", _csv(faults))
    save("latency_summary.csv", _csv(latency))
    save("equivalence.json", encoded(dict(status="PASS", rows=summary["replayed_rows"], jobs=len(jobs), compared_columns=summary["comparison_columns"], max_absolute_errors=maxima)))
    save("summary.json", encoded(summary))
    for name, value in replay_plots(summary, faults, latency).items():
        save(name, value)
    save("replay_report.md", replay_report(summary, protocol, latency).encode())
    if audit_inputs(config) != audit:
        raise DataError("Replay raw/config/protocol/dependencies changed during execution.")
    for name, digest in code.items():
        if sha(code_root / name) != digest:
            raise DataError("Replay code changed after the plan was frozen.")
    verify_outputs(source, config)
    verify_outputs(sources["low_dir"], config)
    if any(sha(output / n) != d for n, d in freeze.items()):
        raise DataError("Replay frozen model/policy plan changed.")
    _write_files(output, {"manifest.json": encoded(dict(run_id=run_id, identity=identity, outputs=hashes))}, config)
    return summary
