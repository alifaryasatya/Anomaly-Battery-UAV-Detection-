"""Read-only reuse of frozen reference detectors on whole low-load flights.

No fit, quantile calibration, synthetic fault evaluation, or policy selection
belongs in this stage. Multiple models score the same cohort as sensitivity
analyses; their exposures must never be added as independent flight exposure.
"""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .config import AppConfig, project_path
from .detectors import apply_single_path, residual_score
from .errors import DataError
from .evaluation import distribution
from .models import SCORE_COLUMNS, _feature_input, _write_files, load_model_bundle, predict_hgb
from .pilot_reporting import alarm_metrics
from .preprocessing import _csv
from .provenance import require_healthy_source
from .reporting import format_json

EXTRA_COLUMNS = ("fold_id", "unseen_by_detector", "source_training_flight", "source_validation_flight",
                 "used_for_policy_selection", "dI_dt", "outside_any_training_range")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value):
    return (format_json(value) + "\n").encode()


def verify_manifest(output, config):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["outputs"].items():
        target = project_path(output / name, config.root)
        if sha(target) != digest:
            raise DataError(f"Frozen artifact checksum mismatch: {target}")
    return manifest


def load_sources(config, baseline_summary=None):
    pointer = config.artifacts_path / "sprint6/cross_validation_result.json"
    if baseline_summary is None and not pointer.exists():
        raise DataError("Sprint 10 requires the completed frozen Sprint 6 cross_validation_result.json; no automatic refitting.")
    summary = baseline_summary if baseline_summary is not None else json.loads(pointer.read_text(encoding="utf-8"))
    source = project_path(summary["output_directory"], config.root)
    manifest = verify_manifest(source, config)
    if json.loads((source / "results.json").read_text(encoding="utf-8"))["run_summary"] != summary:
        raise DataError("Sprint 6 pointer differs from its verified summary.")
    protocol = manifest["identity"]["protocol"]
    if protocol["experiment"] != "sprint6_single_path_reference_loafo":
        raise DataError("Low-load evaluation requires the frozen single-path reference, not an unpromoted candidate.")
    feature_dir = project_path(config.artifacts_path / "features" / summary["feature_run_id"], config.root)
    verify_manifest(feature_dir, config)
    fs = json.loads((feature_dir / "summary.json").read_text(encoding="utf-8"))
    rows, coverage, registry, upstream = _feature_input(fs, config)
    if (upstream != summary["feature_manifest_sha256"] or sha(config.data_path) != summary["data_sha256"]
            or hashlib.sha256(yaml.safe_dump(config.as_dict(), sort_keys=False).encode()).hexdigest() != protocol["config_sha256"]):
        raise DataError("Frozen baseline data/config/features differ from current inputs.")
    require_healthy_source(rows, config.as_dict()["data"]["healthy_status"])
    bundles = []
    for assignment in protocol["assignments"]:
        fold = assignment["fold_id"]
        path = source / "folds" / fold / "model_bundle.joblib"
        bundle = load_model_bundle(path)
        threshold_path = path.parent / "threshold.json"
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        if (bundle["config"] != config.as_dict() or bundle["threshold"] != threshold
                or bundle["split"] != {k: assignment[k] for k in ("train", "validation", "test")}
                or bundle["fold_id"] != fold or bundle["data_sha256"] != summary["data_sha256"]
                or bundle["feature_manifest_sha256"] != upstream
                or threshold["percentile"] != 99 or threshold["persistence_seconds"] != .5):
            raise DataError("Frozen baseline bundle/policy/provenance mismatch.")
        bundles.append((fold, bundle, path, threshold_path))
    if not bundles or len({f for f, *_ in bundles}) != len(bundles):
        raise DataError("Missing or duplicate frozen reference folds.")
    return source, summary, rows, coverage, registry, sorted(bundles, key=lambda x: x[0])


def stats(values):
    values = [v for v in values if v is not None]
    result = distribution(values)
    result.update(p05=float(np.percentile(values, 5)) if values else None,
                  p95=float(np.percentile(values, 95)) if values else None)
    return result


def training_ranges(rows, bundle):
    ready = [r for r in rows if r["reference_ready"] and r["NamaFile"] in bundle["split"]["train"]]
    fingerprint = [{name: r[name] for name in ("NamaFile", "Inst", "TimeUS", "source_record", "voltage_sag_actual", *bundle["predictors"])} for r in ready]
    provenance = bundle["training_provenance"]
    if (len(ready) != provenance["ready_training_samples"]
            or hashlib.sha256(format_json({"rows": fingerprint}).encode()).hexdigest() != provenance["training_rows_sha256"]):
        raise DataError("Training domain rows differ from the frozen model provenance.")
    return {name: stats([r[name] for r in ready]) for name in bundle["predictors"]}


def outside(row, ranges):
    return any(row[name] < bounds["min"] or row[name] > bounds["max"] for name, bounds in ranges.items())


def score_low_load(rows, bundle, config, ranges):
    require_healthy_source(rows, config.as_dict()["data"]["healthy_status"])
    predictions = iter(predict_hgb(bundle["model"], [r for r in rows if r["reference_ready"]], config))
    scored = []
    for row in rows:
        ready, flight = row["reference_ready"], row["NamaFile"]
        predicted = float(next(predictions)) if ready else None
        residual, score = residual_score(row["voltage_sag_actual"], predicted) if ready else (None, None)
        training = flight in bundle["split"]["train"]
        validation = flight in bundle["split"]["validation"]
        scored.append(dict(row, split_role="low_load", used_for_fit=False, used_for_threshold=False,
                           used_for_policy_selection=False, unseen_by_model=not training,
                           unseen_by_detector=not (training or validation), source_training_flight=training,
                           source_validation_flight=validation, fold_id=bundle["fold_id"],
                           predicted_voltage_sag=predicted, residual_signed_v=residual, anomaly_score=score,
                           outside_any_training_range=outside(row, ranges) if ready else None))
    policy = bundle["threshold"]
    return apply_single_path(scored, threshold_v=policy["threshold_v"], duration_seconds=policy["persistence_seconds"],
                             gap_threshold_seconds=config.gap_threshold_seconds)


def pool(records):
    # Counts are pooled only WITHIN one model, over unique whole flights.
    result = {key: sum(r[key] for r in records) for key in
              ("ready_samples", "raw_alarm_samples", "confirmed_alarm_samples", "confirmed_episodes")}
    duration_us = sum(round(r["observed_ready_seconds"] * 1e6) for r in records)
    result["observed_ready_seconds"] = duration_us / 1e6
    n = result["ready_samples"]
    result.update(raw_fpr=result["raw_alarm_samples"] / n if n else None,
                  confirmed_fpr=result["confirmed_alarm_samples"] / n if n else None,
                  confirmed_episodes_per_minute=result["confirmed_episodes"] * 60e6 / duration_us if duration_us else None)
    return result


def episode_context(rows):
    episodes, current = [], []
    for row in rows:
        if current and (not row["anomaly_confirmed"] or row["confirmed_episode_start"]):
            episodes.append(current)
            current = []
        if row["anomaly_confirmed"]:
            current.append(row)
    if current:
        episodes.append(current)
    return [dict(fold_id=part[0]["fold_id"], NamaFile=part[0]["NamaFile"],
                 start_time_us=part[0]["TimeUS"], end_time_us=part[-1]["TimeUS"], confirmed_samples=len(part),
                 Curr_min_a=min(r["Curr"] for r in part), Curr_max_a=max(r["Curr"] for r in part),
                 abs_dI_dt_max_a_per_s=max(abs(r["dI_dt"]) for r in part),
                 residual_min_v=min(r["residual_signed_v"] for r in part),
                 residual_max_v=max(r["residual_signed_v"] for r in part)) for part in episodes]


def run_low_load(config: AppConfig, *, baseline_summary=None):
    from .low_load_reporting import plots, report
    source, source_summary, rows, coverage, registry, bundles = (load_sources(config) if baseline_summary is None
                                                               else load_sources(config, baseline_summary))
    by_flight = defaultdict(list)
    for row in rows:
        by_flight[row["NamaFile"]].append(row)
    cohort = sorted(c["NamaFile"] for c in coverage if c["v2_current_based_label"] == "LOW_LOAD_OR_IDLE")
    if not cohort or len(cohort) != len(set(cohort)):
        raise DataError("No unique LOW_LOAD_OR_IDLE flights available.")
    # This allocation rule is fixed before any low-load prediction.
    primary = bundles[0][0]
    sources, ranges_by_fold = [], {}
    for fold, bundle, path, threshold_path in bundles:
        assigned = set().union(*(set(v) for v in bundle["split"].values()))
        if assigned.intersection(cohort):
            raise DataError("Low-load cohort overlaps frozen train/validation/active-test assignments.")
        ranges_by_fold[fold] = training_ranges(rows, bundle)
        sources.append(dict(fold_id=fold, bundle_path=str(path.relative_to(config.root)), model_sha256=sha(path),
                            threshold_path=str(threshold_path.relative_to(config.root)), threshold_sha256=sha(threshold_path),
                            split=bundle["split"], threshold=bundle["threshold"]))
    protocol = dict(experiment="sprint10_frozen_reference_low_load", evaluation_scope="historical_development_low_load_false_alarm_only",
                    baseline_run_id=source_summary["run_id"], source_manifest_sha256=sha(source / "manifest.json"),
                    feature_manifest_sha256=source_summary["feature_manifest_sha256"], data_sha256=source_summary["data_sha256"],
                    cohort=cohort, primary_fold=primary, primary_rule="first sorted frozen fold; chosen before low-load scores",
                    sensitivity_rule="each frozen fold scores the same cohort separately; no pooled repeated exposure or winner selection",
                    model_training_performed=False, threshold_calibration_performed=False, policy_selection_performed=False,
                    synthetic_evaluation_performed=False, threshold_percentile=99, persistence_seconds=.5,
                    duration_policy="both endpoints ready, same stream/segment, actual dt <= gap threshold; exclude warmup/gaps",
                    domain_policy="descriptive marginal train-ready min/max only; not a detector or physical diagnosis",
                    example_rule="primary fold, first sorted flight with confirmed alarm, else raw alarm, else ready sample",
                    frozen_sources=sources, gap_threshold_seconds=config.gap_threshold_seconds,
                    healthy_status="assumed_from_source", currtot_unit="unverified")
    code_names = ("low_load.py", "low_load_reporting.py", "evaluation.py", "models.py", "pilot_reporting.py", "detectors.py", "persistence.py")
    identity = dict(protocol=protocol, code_sha256={"src/" + n: sha(Path(__file__).parent / n) for n in code_names})
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "low_load" / run_id, config.root)
    if (output / "manifest.json").exists():
        existing = verify_manifest(output, config)
        if existing["identity"] != identity:
            raise DataError("Low-load cache identity mismatch.")
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    frozen = {"protocol.json": json_bytes(protocol), "training_domain_ranges.json": json_bytes(ranges_by_fold),
              "config_used.yaml": yaml.safe_dump(config.as_dict(), sort_keys=False).encode(),
              "feature_registry.json": json_bytes(registry)}
    # Persist all source policies, hashes and train-only ranges BEFORE prediction.
    _write_files(output, frozen, config)
    freeze = {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
    _write_files(output, {"pre_evaluation_freeze.json": json_bytes(freeze)}, config)
    payloads, records, fold_records, responses, range_records, episodes = {}, [], [], [], [], []
    profiles = []
    fields = [*registry["regression_sets"]["battery_d"], "Volt", "voltage_sag_actual"]
    for c in coverage:
        flight = c["NamaFile"]
        raw = by_flight[flight]
        ready = [r for r in raw if r["reference_ready"]]
        for field in fields:
            subset = raw if field == "Volt" else ready
            profiles.append(dict(NamaFile=flight, v2_current_based_label=c["v2_current_based_label"],
                                 feature=field, sample_scope="all accepted" if field == "Volt" else "reference ready",
                                 initial_voltage_v=raw[0]["Volt"] if raw else None, V_ref=c["V_ref"],
                                 **stats([r[field] for r in subset])))
    primary_rows, sensitivity_example = {}, None
    for fold, bundle, *_ in bundles:
        fold_start = len(records)
        ranges = ranges_by_fold[fold]
        for flight in cohort:
            raw = by_flight[flight]
            if raw:
                decisions = score_low_load(raw, bundle, config, ranges)
            else:
                decisions = []
            ready = [r for r in decisions if r["reference_ready"]]
            metrics = alarm_metrics(decisions, config.gap_threshold_seconds)
            records.append(dict(fold_id=fold, NamaFile=flight, primary=fold == primary, total_samples=len(raw),
                                unseen_by_model=flight not in bundle["split"]["train"],
                                unseen_by_detector=flight not in bundle["split"]["train"] + bundle["split"]["validation"],
                                used_for_fit=False, used_for_threshold=False, used_for_policy_selection=False,
                                evaluation_status="EVALUATED" if ready else "NO_READY_EXPOSURE", threshold_v=bundle["threshold"]["threshold_v"],
                                outside_any_training_range_samples=sum(outside(r, ranges) for r in ready), **metrics))
            for field in ("predicted_voltage_sag", "residual_signed_v", "anomaly_score"):
                responses.append(dict(fold_id=fold, NamaFile=flight, response=field, **stats([r[field] for r in ready])))
            for name, bounds in ranges.items():
                n = sum(r[name] < bounds["min"] or r[name] > bounds["max"] for r in ready)
                range_records.append(dict(fold_id=fold, NamaFile=flight, predictor=name, training_min=bounds["min"],
                                          training_max=bounds["max"], ready_samples=len(ready), outside_samples=n,
                                          outside_fraction=n / len(ready) if ready else None))
            episodes.extend(episode_context(decisions))
            if fold != primary and sensitivity_example is None and any(r["anomaly_confirmed"] for r in decisions):
                sensitivity_example = (fold, flight, decisions, bundle["threshold"]["threshold_v"])
            columns = (*SCORE_COLUMNS, *EXTRA_COLUMNS)
            name = f"scores/{fold}/{flight}.csv.gz"
            payloads[name] = gzip.compress(_csv([{k: r[k] for k in columns} for r in decisions], columns), mtime=0)
            if fold == primary:
                primary_rows[flight] = decisions
        fold_records.append(dict(fold_id=fold, unique_flights=len(cohort), **pool(records[fold_start:])))
    primary_result = next(r for r in fold_records if r["fold_id"] == primary)
    summary = dict(status="OK", command="evaluate-low-load", run_id=run_id, output_directory=str(output),
                   evaluation_scope=protocol["evaluation_scope"], baseline_run_id=source_summary["run_id"],
                   data_sha256=protocol["data_sha256"], feature_manifest_sha256=protocol["feature_manifest_sha256"],
                   unique_low_load_flights=len(cohort), evaluated_model_flight_pairs=len(records), frozen_models=len(bundles),
                   primary_fold=primary, primary_low_load=primary_result,
                   sensitivity_across_frozen_models={k: distribution([r[k] for r in fold_records]) for k in
                       ("raw_fpr", "confirmed_fpr", "confirmed_episodes", "confirmed_episodes_per_minute")},
                   all_low_load_unseen_by_model=all(r["unseen_by_model"] for r in records),
                   all_low_load_unseen_by_detector=all(r["unseen_by_detector"] for r in records),
                   model_training_performed=False, threshold_calibration_performed=False, policy_selection_performed=False,
                   synthetic_evaluation_performed=False, threshold_changed=False, operational_dual_enabled=False,
                   deployment_status="RESEARCH_DOMAIN_DIAGNOSTICS_ONLY", currtot_unit="unverified")
    payloads.update({"per_model_flight.csv": _csv(records), "per_model_summary.csv": _csv(fold_records),
                     "flight_domain_profiles.csv": _csv(profiles), "response_distributions.csv": _csv(responses),
                     "outside_training_ranges.csv": _csv(range_records),
                     "episode_context.csv": _csv(episodes, ("fold_id", "NamaFile", "start_time_us", "end_time_us", "confirmed_samples",
                         "Curr_min_a", "Curr_max_a", "abs_dI_dt_max_a_per_s", "residual_min_v", "residual_max_v")),
                     "reference_coverage.csv": _csv(coverage), "summary.json": json_bytes(summary)})
    plot_payloads, example = plots(primary_rows, records, profiles, primary, bundles[0][1]["threshold"]["threshold_v"])
    summary["example_flight"] = example
    summary["example_status"] = ("CONFIRMED_FALSE_ALARM" if any(r["anomaly_confirmed"] for r in primary_rows.get(example, []))
                                 else "RAW_ALARM_ONLY" if any(r["anomaly_raw"] for r in primary_rows.get(example, []))
                                 else "NO_ALARM_CONTEXT_ONLY")
    summary["sensitivity_false_alarm_example"] = None
    if sensitivity_example:
        fold, flight, decisions, threshold = sensitivity_example
        extra, _ = plots({flight: decisions}, [r for r in records if r["fold_id"] == fold and r["NamaFile"] == flight], profiles, fold, threshold)
        plot_payloads["plots/sensitivity_false_alarm_context.png"] = extra["plots/example_false_alarm_context.png"]
        summary["sensitivity_false_alarm_example"] = dict(fold_id=fold, NamaFile=flight,
            selection_rule="first sorted sensitivity fold/flight with confirmed alarm; descriptive example only, added after initial diagnostic run")
    payloads["summary.json"] = json_bytes(summary)
    payloads.update(plot_payloads)
    payloads["research_report.md"] = report(summary, records, fold_records, profiles).encode()
    for name, digest in freeze.items():
        if sha(output / name) != digest:
            raise DataError("Pre-evaluation frozen policy/domain context changed.")
    verify_manifest(source, config)
    if sha(config.data_path) != summary["data_sha256"]:
        raise DataError("Raw input changed during low-load evaluation.")
    for name in payloads:
        (output / name).parent.mkdir(parents=True, exist_ok=True)
    _write_files(output, payloads, config)
    names = [*frozen, "pre_evaluation_freeze.json", *payloads]
    manifest = dict(run_id=run_id, identity=identity, outputs={n: sha(output / n) for n in sorted(names)})
    _write_files(output, {"manifest.json": json_bytes(manifest)}, config)
    return summary
