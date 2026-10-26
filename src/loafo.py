"""Frozen baseline LOAFO: calibrate all folds before any outer evaluation."""

from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .config import AppConfig, project_path
from .cross_validation import loafo_splits
from .detectors import apply_single_path, calibrate_threshold
from .errors import ConfigError, DataError
from .evaluation import aggregate_scenarios, deadline_name, distribution, ratio, scenario_metrics
from .fault_injection import FAMILIES, SEVERITIES, inject_fault
from .features import generate_features
from .models import SCORE_COLUMNS, _feature_input, _write_files, fit_hgb, make_hgb, score_rows
from .pilot_reporting import alarm_metrics, regression_metrics
from .preprocessing import _csv
from .reporting import format_json

SCENARIO_SCORE_COLUMNS = (*SCORE_COLUMNS, "scenario_id", "fault_family", "fault_severity", "fault_start_position",
                          "is_synthetic", "fault_applicable", "fault_support", "fault_label", "fault_effect_present", "fault_phase", "added_sag_v")


def _gzip_rows(rows: list[dict], columns) -> bytes:
    return gzip.compress(_csv([{key: r[key] for key in columns} for r in rows], columns), compresslevel=6, mtime=0)


def _pool_regression(records: list[dict]) -> dict:
    samples = sum(r["samples"] for r in records)
    absolute = math.fsum(r["sum_absolute_error_v"] for r in records)
    squared = math.fsum(r["sum_squared_error_v2"] for r in records)
    target = math.fsum(r["sum_target_v"] for r in records)
    target_squared = math.fsum(r["sum_squared_target_v2"] for r in records)
    variance = target_squared - target * target / samples if samples else 0.
    return dict(samples=samples, mae_v=absolute / samples if samples else None,
                rmse_v=math.sqrt(squared / samples) if samples else None,
                r2=1 - squared / variance if samples >= 2 and variance > 0 else None)


def _pool_healthy(records: list[dict]) -> dict:
    samples = sum(r["ready_samples"] for r in records)
    raw = sum(r["raw_alarm_samples"] for r in records)
    confirmed = sum(r["confirmed_alarm_samples"] for r in records)
    episodes = sum(r["confirmed_episodes"] for r in records)
    duration = math.fsum(r["observed_ready_seconds"] for r in records)
    return dict(ready_samples=samples, raw_alarm_samples=raw, confirmed_alarm_samples=confirmed,
                raw_fpr=ratio(raw, samples), confirmed_fpr=ratio(confirmed, samples), confirmed_episodes=episodes,
                observed_ready_seconds=duration, confirmed_episodes_per_minute=ratio(60 * episodes, duration))


def _summaries(regression: list[dict], healthy: list[dict], anomaly: list[dict], scenarios: list[dict], deadlines: list[float]) -> tuple[dict, list[dict]]:
    pooled = dict(regression=_pool_regression(regression), healthy_active=_pool_healthy(healthy),
                  synthetic=aggregate_scenarios(scenarios, deadlines))
    macro, table = {}, []
    groups = [("regression", regression, ("mae_v", "rmse_v", "r2")),
              ("healthy_active", healthy, ("raw_fpr", "confirmed_fpr", "confirmed_episodes_per_minute")),
              ("synthetic", anomaly, ("precision", "recall", "f1", "synthetic_sample_fpr", "incident_recall", "pre_fault_fpr",
                                      "delay_mean_seconds", "delay_median_seconds", *[deadline_name(d) + "_rate" for d in deadlines]))]
    for scope, records, names in groups:
        macro[scope] = {}
        for name in names:
            stats = distribution([r[name] for r in records])
            macro[scope][name] = stats
            table.append(dict(scope=scope, metric=name, **stats, pooled_value=pooled[scope][name],
                              fold_count=len(records), pooled_delay_detected_n=pooled["synthetic"]["delay_detected_n"] if name.startswith("delay_") else None))
    return dict(pooled=pooled, macro_per_fold=macro), table


def run_loafo(config: AppConfig) -> dict:
    import joblib
    from .cv_reporting import cv_report, fold_plot, deadline_plot, fault_plot
    cfg = config.as_dict()
    if (cfg["features"]["operational_set"] != "battery_d" or cfg["features"]["rolling_windows_seconds"] != [.5, 1.0]
            or cfg["threshold"]["reference_percentile"] != 99. or cfg["persistence"]["reference_duration_seconds"] != .5
            or cfg["dual_path"]["enabled"]):
        raise ConfigError("LOAFO reference baseline requires Battery-D, P99, 0.5 s persistence, dual-path disabled; tuning is a later stage.")
    feature_summary = generate_features(config)
    if feature_summary["currtot_predictor_review_required"]:
        raise DataError("CurrTot decreases require review before Battery-D LOAFO fitting.")
    rows, coverage, registry, upstream_hash = _feature_input(feature_summary, config)
    eligible = sorted(r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True")
    splits = loafo_splits(eligible, cfg["evaluation"]["minimum_active_flights"])
    deadlines = cfg["evaluation"]["deadlines_seconds"]
    protocol = dict(experiment="sprint6_single_path_reference_loafo", evaluation_scope="historical_development_cross_validation",
                    eligible_flights=eligible, assignments=[dict(fold_id=f"fold_{i:02d}", **split.as_dict()) for i, split in enumerate(splits)],
                    features=registry["regression_sets"]["battery_d"], model_parameters=make_hgb(config).get_params(),
                    random_seed=config.seed, openmp_threads=1, early_stopping=False, scaler=None,
                    train_policy="whole assigned healthy-assumed flights, reference-ready rows only, including low-current periods",
                    threshold_policy="P99 linear healthy-validation only", persistence_seconds=.5, candidate_selection="none: frozen reference baseline",
                    phase_order="fit/calibrate/freeze ALL folds, then outer healthy and synthetic evaluation",
                    incident_policy="new confirmed episode within support; carried confirmed alarm at onset contaminates entire scenario",
                    deadlines_seconds=deadlines, deadline_denominator="all applicable, including undetected and PREEXISTING_ALARM; NOT_APPLICABLE excluded",
                    fault_config=cfg["fault_injection"], sample_metric_policy="reference-ready samples; conventional confusion counts separate from incident success",
                    duration_policy="observed ready intervals only, both endpoints ready; no gaps/warm-up",
                    feature_manifest_sha256=upstream_hash, data_sha256=feature_summary["data_sha256"],
                    config_sha256=feature_summary["config_sha256"], dependency_versions=feature_summary["dependency_versions"])
    code = {f"src/{name}": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("loafo.py", "cross_validation.py", "evaluation.py", "cv_reporting.py", "models.py", "detectors.py", "persistence.py", "fault_injection.py", "provenance.py", "pilot_reporting.py")}
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "cross_validation" / run_id, config.root)
    completed = output / "manifest.json"
    if completed.exists():
        try:
            manifest = json.loads(completed.read_text())
            if manifest["run_id"] != run_id or manifest["identity"] != identity:
                raise DataError("LOAFO completed manifest identity mismatch.")
            for relative, digest in manifest["outputs"].items():
                if hashlib.sha256(project_path(output / relative, config.root).read_bytes()).hexdigest() != digest:
                    raise DataError(f"LOAFO artifact checksum mismatch: {relative}")
            return json.loads((output / "results.json").read_text())["run_summary"]
        except (ValueError, KeyError) as exc:
            raise DataError(f"Invalid LOAFO completed manifest: {exc}") from exc
    by_flight = defaultdict(list)
    for row in rows:
        by_flight[row["NamaFile"]].append(row)
    assignments = []
    for index, split in enumerate(splits):
        for flight in coverage:
            name = flight["NamaFile"]
            role = "train" if name in split.train else "validation" if name in split.validation else "test" if name in split.test else "excluded"
            assignments.append(dict(fold_id=f"fold_{index:02d}", NamaFile=name, split_role=role,
                                    retained_samples=int(flight["retained_samples"]), ready_samples=int(flight["ready_samples"]),
                                    v2_current_based_label=flight["v2_current_based_label"], reference_status=flight["reference_status"],
                                    used_for_fit=role == "train", used_for_threshold=role == "validation",
                                    unseen_by_model=role != "train", evaluated_outer=role == "test"))
    checksums = {}

    def save(name, value):
        project_path(output / name, config.root).parent.mkdir(parents=True, exist_ok=True)
        _write_files(output, {name: value}, config)
        checksums[name] = hashlib.sha256(value).hexdigest()

    save("experiment_protocol.json", (format_json(protocol) + "\n").encode())
    save("config_used.yaml", yaml.safe_dump(cfg, sort_keys=False).encode())
    save("feature_registry.json", (format_json(registry) + "\n").encode())
    save("fold_assignments.csv", _csv(assignments))
    save("reference_coverage.csv", (Path(feature_summary["output_directory"]) / "reference_coverage.csv").read_bytes())
    prepared, calibration = [], []
    # Crucial ordering: outer rows are never passed to score_rows in this phase.
    for index, split in enumerate(splits):
        fold = f"fold_{index:02d}"
        training = [r for flight in split.train for r in by_flight[flight] if r["reference_ready"]]
        model = fit_hgb(training, config, split)
        validation = score_rows(model, by_flight[split.validation[0]], config, "validation")
        threshold = calibrate_threshold([r for r in validation if r["reference_ready"]], split)
        threshold.update(persistence_seconds=.5, persistence_source="frozen_reference_protocol_not_tuned")
        selected = dict(fold_id=fold, split=split.as_dict(), threshold=threshold,
                        selected_before_outer=True, selection_scope="healthy_validation_only_fixed_reference_policy",
                        model_parameters=model.get_params(), predictors=protocol["features"], random_seed=config.seed,
                        feature_manifest_sha256=upstream_hash)
        train_fingerprint = [{key: r[key] for key in ("NamaFile", "Inst", "TimeUS", "source_record", "voltage_sag_actual", *protocol["features"])} for r in training]
        provenance = dict(train_flights=list(split.train), ready_training_samples=len(training),
                          samples_per_flight=dict(sorted(Counter(r["NamaFile"] for r in training).items())),
                          low_current_training_samples=sum(r["Curr"] < cfg["activity"]["min_current_a"] for r in training),
                          training_rows_sha256=hashlib.sha256(format_json({"rows": train_fingerprint}).encode()).hexdigest(),
                          healthy_status=cfg["data"]["healthy_status"], synthetic_rows=0, scaler=None,
                          early_stopping=False, fitted_iterations=int(model.n_iter_))
        schema = dict(required_input_columns=cfg["data"]["required_columns"], predictor_order=protocol["features"],
                      units={"TimeUS": "us_from_CSV_contract", "Volt": "V_from_CSV_contract", "Curr": "A_from_CSV_contract", "CurrTot": "unverified"},
                      reference_voltage=cfg["reference_voltage"], time_policy=cfg["time"], unit_conversion_applied=False,
                      reference_unavailable_policy="NOT_READY; excluded from fit/calibration/metrics")
        bundle = dict(format_version=1, engine="battery_v2", model=model, config=cfg, predictors=protocol["features"],
                      feature_registry=registry, split=split.as_dict(), threshold=threshold, training_provenance=provenance,
                      input_schema_unit_policy=schema, protocol=protocol, fold_id=fold,
                      data_sha256=feature_summary["data_sha256"], feature_manifest_sha256=upstream_hash, code_sha256=code)
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer, compress=3)
        save(f"folds/{fold}/selected_policy.json", (format_json(selected) + "\n").encode())
        save(f"folds/{fold}/threshold.json", (format_json(threshold) + "\n").encode())
        save(f"folds/{fold}/training_provenance.json", (format_json(provenance) + "\n").encode())
        save(f"folds/{fold}/model_bundle.joblib", buffer.getvalue())
        validation = apply_single_path(validation, threshold_v=threshold["threshold_v"], duration_seconds=.5, gap_threshold_seconds=config.gap_threshold_seconds)
        save(f"folds/{fold}/validation_scores.csv.gz", _gzip_rows(validation, SCORE_COLUMNS))
        calibration.append(dict(fold_id=fold, validation_flight=split.validation[0], threshold_v=threshold["threshold_v"],
                                **regression_metrics(validation), **alarm_metrics(validation, config.gap_threshold_seconds)))
        prepared.append((fold, split, model, threshold))
    freeze = {name: digest for name, digest in checksums.items() if name.endswith(("selected_policy.json", "threshold.json", "model_bundle.joblib"))}
    save("all_fold_policies_frozen.json", (format_json(dict(status="FROZEN_BEFORE_OUTER_EVALUATION", fold_count=len(prepared), files=freeze)) + "\n").encode())
    regression, healthy, anomaly, scenarios = [], [], [], []
    examples, first_failure = None, None
    example_position = sorted(cfg["fault_injection"]["start_positions"])[len(cfg["fault_injection"]["start_positions"]) // 2]
    for fold, split, model, threshold in prepared:
        test = by_flight[split.test[0]]
        scored = score_rows(model, test, config, "test")
        decision = apply_single_path(scored, threshold_v=threshold["threshold_v"], duration_seconds=.5, gap_threshold_seconds=config.gap_threshold_seconds)
        save(f"folds/{fold}/healthy_test_scores.csv.gz", _gzip_rows(decision, SCORE_COLUMNS))
        ready = [r for r in decision if r["reference_ready"]]
        target = np.array([r["voltage_sag_actual"] for r in ready])
        residual = np.array([r["residual_signed_v"] for r in ready])
        regression.append(dict(fold_id=fold, test_flight=split.test[0], unseen_by_model=True, used_for_threshold=False,
                               **regression_metrics(decision), sum_absolute_error_v=float(np.sum(np.abs(residual))),
                               sum_squared_error_v2=float(np.sum(residual ** 2)), sum_target_v=float(np.sum(target)),
                               sum_squared_target_v2=float(np.sum(target ** 2))))
        healthy.append(dict(fold_id=fold, test_flight=split.test[0], unseen_by_model=True, used_for_threshold=False,
                            **alarm_metrics(decision, config.gap_threshold_seconds)))
        local = []
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(test, config, family=family, severity=severity, start_position=position)
                    meta = generated.metadata
                    evaluated = []
                    score_path = None
                    if meta["fault_applicable"]:
                        evaluated = apply_single_path(score_rows(model, generated.rows, config, "test"),
                                                       threshold_v=threshold["threshold_v"], duration_seconds=.5,
                                                       gap_threshold_seconds=config.gap_threshold_seconds)
                        score_path = f"folds/{fold}/scenario_scores/{meta['scenario_id']}.csv.gz"
                        save(score_path, _gzip_rows(evaluated, SCENARIO_SCORE_COLUMNS))
                    metrics = scenario_metrics(evaluated, meta, deadlines, config.gap_threshold_seconds)
                    record = {**meta, "fold_id": fold, "test_flight": split.test[0], "validation_flight": split.validation[0],
                              "threshold_v": threshold["threshold_v"], "persistence_seconds": .5, "unseen_by_model": True,
                              "used_for_threshold": False, **metrics, "scores_csv": score_path,
                              "scores_sha256": checksums[score_path] if score_path else None}
                    local.append(record)
                    scenarios.append(record)
                    if evaluated and fold == "fold_00" and family == "progressive" and severity == "severe" and position == example_position:
                        examples = (evaluated, record)
                    if evaluated and not metrics["incident_detected"] and first_failure is None:
                        first_failure = (evaluated, record)
        anomaly.append(dict(fold_id=fold, test_flight=split.test[0], threshold_v=threshold["threshold_v"], **aggregate_scenarios(local, deadlines)))
    result, summary_table = _summaries(regression, healthy, anomaly, scenarios, deadlines)
    abrupt = [r for r in scenarios if r["fault_family"] in ("increased_resistance", "sudden_drop") and r["severity"] == "severe"]
    gradual = [r for r in scenarios if r["fault_family"] == "progressive" and r["observability"] in ("MEDIUM", "HIGH")]
    target_groups = dict(abrupt_severe=aggregate_scenarios(abrupt, deadlines), gradual_observable=aggregate_scenarios(gradual, deadlines))
    checks = {}
    for name, key, goal in (("abrupt_severe", deadline_name(1.) + "_rate", cfg["research_targets"]["abrupt_severe_rate_le_1s"]),
                            ("gradual_observable", deadline_name(5.) + "_rate", cfg["research_targets"]["gradual_observable_rate_le_5s"])):
        rate = target_groups[name].get(key)
        checks[name] = dict(rate=rate, target=goal, applicable=target_groups[name]["applicable_scenarios"],
                            status="MET" if rate is not None and rate >= goal else "NOT_MET")
    run_summary = dict(status="OK", command="cross-validate", run_id=run_id, output_directory=str(output),
                       evaluation_scope=protocol["evaluation_scope"], folds=len(splits), active_flights=eligible,
                       feature_run_id=feature_summary["run_id"], data_sha256=feature_summary["data_sha256"],
                       feature_manifest_sha256=upstream_hash, random_seed=config.seed, model_training_performed=True,
                       threshold_policy="healthy-validation P99 per fold", persistence_seconds=.5,
                       all_policies_frozen_before_outer=True, outer_test_evaluated=True,
                       regression_pooled=result["pooled"]["regression"], healthy_active_pooled=result["pooled"]["healthy_active"],
                       synthetic_pooled=result["pooled"]["synthetic"], research_target_checks=checks,
                       low_load_evaluated=False, currtot_unit="unverified", deployment_status="RESEARCH_BASELINE_ONLY",
                       operational_gate_status="NOT_ASSESSED_FULL_PROTOCOL_PENDING")
    full_result = dict(run_summary=run_summary, **result, target_groups=target_groups, research_target_checks=checks)
    save("regression_per_fold.csv", _csv(regression))
    save("healthy_active_per_fold.csv", _csv(healthy))
    save("anomaly_per_fold.csv", _csv(anomaly))
    save("fault_per_scenario.csv", _csv(scenarios))
    save("failed_cases.csv", _csv([r for r in scenarios if not r["incident_detected"]], list(scenarios[0])))
    save("validation_calibration_per_fold.csv", _csv(calibration))
    save("summary.csv", _csv(summary_table))
    save("results.json", (format_json(full_result) + "\n").encode())
    save("research_report.md", cv_report(full_result, regression, healthy, anomaly, scenarios).encode("utf-8"))
    save("per_fold_metrics.png", fold_plot(regression, healthy, anomaly))
    save("severity_deadlines.png", deadline_plot(scenarios, deadlines))
    if examples:
        save("fault_example.png", fault_plot(*examples))
    if first_failure:
        save("failed_fault_example.png", fault_plot(*first_failure))
    # Selection is immutable even after outer results become available.
    if any(hashlib.sha256((output / name).read_bytes()).hexdigest() != digest for name, digest in freeze.items()):
        raise DataError("Fold model/selected policy changed during outer evaluation.")
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != run_summary["data_sha256"]:
        raise DataError("Raw data changed during LOAFO; no completion manifest written.")
    _write_files(output, {"manifest.json": (format_json(dict(run_id=run_id, identity=identity, outputs=dict(sorted(checksums.items())))) + "\n").encode()}, config)
    return run_summary
