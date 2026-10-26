"""Fixed-policy Sprint 7 ablation and model comparison, without winner promotion."""

from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .config import project_path
from .cross_validation import loafo_splits
from .errors import ConfigError, DataError
from .evaluation import aggregate_scenarios, deadline_name, distribution, scenario_metrics
from .experimental_models import candidates, fit_candidate, load_candidate_bundle, score_candidate
from .fault_injection import FAMILIES, SEVERITIES, inject_fault
from .features import generate_features
from .loafo import SCENARIO_SCORE_COLUMNS, _gzip_rows, _pool_healthy, _pool_regression
from .models import SCORE_COLUMNS, _feature_input, _write_files
from .persistence import TimePersistence
from .pilot_reporting import alarm_metrics, regression_metrics
from .preprocessing import _csv
from .provenance import require_healthy_source
from .reporting import format_json


def calibrate(rows, split, bundle):
    require_healthy_source(rows, bundle["config"]["data"]["healthy_status"])
    ready = [r for r in rows if r["reference_ready"]]
    if not ready or {r["NamaFile"] for r in ready} != set(split.validation) or any(r["split_role"] != "validation" for r in rows):
        raise DataError("Calibration requires assigned healthy validation only.")
    scores = [r["anomaly_score"] for r in ready]
    if any(type(s) not in (float, int) or not math.isfinite(s) for s in scores):
        raise DataError("Calibration scores must be finite.")
    fingerprint = [{k: r[k] for k in ("NamaFile", "Inst", "TimeUS", "source_record", "anomaly_score")} for r in ready]
    return dict(method="healthy_validation_percentile", percentile=99., quantile_method="numpy.percentile_linear",
                threshold_v=float(np.percentile(scores, 99, method="linear")) if bundle["regression"] else None,
                threshold=float(np.percentile(scores, 99, method="linear")), source_role="validation",
                source_flights=list(split.validation), source_samples=len(ready),
                source_sha256=hashlib.sha256(format_json({"rows": fingerprint}).encode()).hexdigest(),
                score_definition=bundle["score_definition"], score_unit=bundle["score_unit"],
                comparison="score > threshold", persistence_seconds=.5, selection="fixed protocol, not tuned")


def decide(rows, threshold, config):
    if not math.isfinite(threshold):
        raise DataError("Nonfinite threshold.")
    timer = TimePersistence(.5, config.gap_threshold_seconds)
    result = []
    for row in rows:
        ready = row["reference_ready"]
        score = row["anomaly_score"]
        if type(ready) is not bool or (ready and (type(score) not in (int, float) or not math.isfinite(score))):
            raise DataError("Ready candidate score must be finite (direct scores may be signed).")
        decision = timer.consume(flight=row["NamaFile"], instance=row["Inst"], segment_id=row["segment_id"],
                                 time_us=row["TimeUS"], ready=ready, raw=bool(ready and score > threshold))
        result.append({**row, **decision})
    return result


def summarize(specs, regression, healthy, anomaly, scenarios, deadlines):
    results, summary = {}, []
    for spec in specs:
        name = spec.name
        reg = [r for r in regression if r["candidate"] == name]
        health = [r for r in healthy if r["candidate"] == name]
        inc = [r for r in anomaly if r["candidate"] == name]
        syn = [r for r in scenarios if r["candidate"] == name]
        pooled = dict(regression=_pool_regression(reg) if reg else None, healthy_active=_pool_healthy(health),
                      synthetic=aggregate_scenarios(syn, deadlines))
        groups = [("healthy_active", health, ("raw_fpr", "confirmed_fpr", "confirmed_episodes_per_minute")),
                  ("synthetic", inc, ("precision", "recall", "f1", "synthetic_sample_fpr", "incident_recall", "pre_fault_fpr",
                                      "delay_mean_seconds", "delay_median_seconds", *[deadline_name(d) + "_rate" for d in deadlines]))]
        if reg:
            groups.insert(0, ("regression", reg, ("mae_v", "rmse_v", "r2")))
        macro = {}
        for scope, rows, keys in groups:
            macro[scope] = {}
            for key in keys:
                stats = distribution([r[key] for r in rows])
                macro[scope][key] = stats
                summary.append(dict(candidate=name, scope=scope, metric=key, **stats, pooled_value=pooled[scope][key], fold_count=len(rows),
                                    pooled_delay_detected_n=pooled["synthetic"]["delay_detected_n"] if key.startswith("delay_") else None))
        results[name] = dict(model=spec.model, feature_set=spec.feature_set, pooled=pooled, macro_per_fold=macro)
    return results, summary


def run_comparison(config, mode):
    import joblib
    from .comparison_reporting import comparison_report, comparison_plot
    cfg = config.as_dict()
    if (cfg["features"]["operational_set"] != "battery_d" or cfg["features"]["rolling_windows_seconds"] != [.5, 1.]
            or cfg["threshold"]["reference_percentile"] != 99. or cfg["persistence"]["reference_duration_seconds"] != .5
            or cfg["dual_path"]["enabled"]):
        raise ConfigError("Sprint 7 requires frozen Battery-D reference config, P99, 0.5 s, dual-path off; no tuning.")
    feature_summary = generate_features(config)
    if feature_summary["currtot_predictor_review_required"]:
        raise DataError("CurrTot decreases require review before comparison.")
    rows, coverage, registry, upstream = _feature_input(feature_summary, config)
    eligible = sorted(r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True")
    splits = loafo_splits(eligible, cfg["evaluation"]["minimum_active_flights"])
    specs = candidates(mode)
    protocol = dict(experiment=f"sprint7_{mode}", evaluation_scope="historical_development_comparison",
                    interpretation="development selection only; no independent holdout or automatic winner/promotion",
                    candidates=[dict(candidate=s.name, model=s.model, feature_set=s.feature_set, predictors=s.predictors(config)) for s in specs],
                    assignments=[dict(fold_id=f"fold_{i:02d}", **s.as_dict()) for i, s in enumerate(splits)],
                    model_parameters=cfg["models"], random_seed=config.seed, threshold_percentile=99., persistence_seconds=.5,
                    candidate_selection="none; evaluate all predeclared candidates, no automatic winner",
                    phase_order="fit/calibrate/freeze ALL candidate-fold policies before ANY outer evaluation in this experiment",
                    feature_policy="registered sets only; elapsed_time Battery-E ablation only; regression voltage predictors forbidden",
                    train_policy="whole assigned healthy-assumed flights, reference-ready rows incl idle; no synthetic",
                    scaler_policy="OCSVM StandardScaler on all healthy-ready training only; no scaler for other models",
                    ocsvm_sampling="uniform without replacement default_rng(seed), sorted indices; cap config; save source identities",
                    score_policy="HGB/RF positive sag residual V; IF/OCSVM -decision_function, signed model-specific score, not V",
                    inference_reuse="regression prediction reuse only after exact registered predictor/identity/readiness equality; direct models rescore injected voltage behavior",
                    paired_policy="identical folds, source features/readiness, scenarios/support and applicable denominators",
                    deadlines_seconds=cfg["evaluation"]["deadlines_seconds"], fault_config=cfg["fault_injection"],
                    incident_policy="new confirmation in support only; carried warning contaminates whole incident",
                    deadline_denominator="all applicable incl misses and PREEXISTING_ALARM; exclude NOT_APPLICABLE",
                    feature_manifest_sha256=upstream, data_sha256=feature_summary["data_sha256"],
                    config_sha256=feature_summary["config_sha256"], dependency_versions=feature_summary["dependency_versions"],
                    low_load_evaluated=False, currtot_unit="unverified")
    code = {p.relative_to(Path(__file__).parent.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((Path(__file__).parent).glob("*.py"))}
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    folder = "ablation" if mode == "ablation" else "model_comparison"
    output = project_path(config.artifacts_path / folder / run_id, config.root)
    complete = output / "manifest.json"
    if complete.exists():
        try:
            manifest = json.loads(complete.read_text())
            if manifest["identity"] != identity or manifest["run_id"] != run_id:
                raise DataError("Comparison manifest identity mismatch.")
            for name, digest in manifest["outputs"].items():
                if hashlib.sha256(project_path(output / name, config.root).read_bytes()).hexdigest() != digest:
                    raise DataError(f"Comparison artifact checksum mismatch: {name}")
            return json.loads((output / "results.json").read_text())["run_summary"]
        except (KeyError, ValueError) as exc:
            raise DataError(f"Invalid comparison manifest: {exc}") from exc
    hashes = {}

    def save(name, payload):
        project_path(output / name, config.root).parent.mkdir(parents=True, exist_ok=True)
        _write_files(output, {name: payload}, config)
        hashes[name] = hashlib.sha256(payload).hexdigest()

    def save_json(name, payload):
        save(name, (format_json(payload) + "\n").encode())

    save_json("experiment_protocol.json", protocol)
    save_json("feature_registry.json", registry)
    save("config_used.yaml", yaml.safe_dump(cfg, sort_keys=False).encode())
    save("reference_coverage.csv", (Path(feature_summary["output_directory"]) / "reference_coverage.csv").read_bytes())
    assignments = []
    by_flight = defaultdict(list)
    for row in rows:
        by_flight[row["NamaFile"]].append(row)
    for i, split in enumerate(splits):
        for flight in coverage:
            name = flight["NamaFile"]
            role = "train" if name in split.train else "validation" if name in split.validation else "test" if name in split.test else "excluded"
            assignments.append(dict(fold_id=f"fold_{i:02d}", NamaFile=name, split_role=role, ready_samples=int(flight["ready_samples"]),
                                    retained_samples=int(flight["retained_samples"]), reference_status=flight["reference_status"],
                                    v2_current_based_label=flight["v2_current_based_label"], used_for_fit=role == "train",
                                    used_for_threshold=role == "validation", evaluated_outer=role == "test"))
    save("fold_assignments.csv", _csv(assignments))
    calibration, prepared = [], []
    # Phase 1 never scores outer rows. Bundles are unloaded between fits.
    for spec in specs:
        for i, split in enumerate(splits):
            fold = f"fold_{i:02d}"
            prefix = f"candidates/{spec.name}/{fold}"
            training = [r for flight in split.train for r in by_flight[flight] if r["reference_ready"]]
            bundle = fit_candidate(training, config, split, spec)
            scored = score_candidate(bundle, by_flight[split.validation[0]], config, "validation")
            threshold = calibrate(scored, split, bundle)
            bundle.update(threshold=threshold, feature_registry=registry, reference_policy=cfg["reference_voltage"],
                          fold_id=fold, feature_manifest_sha256=upstream, data_sha256=feature_summary["data_sha256"], code_sha256=code)
            selected = {k: bundle[k] for k in ("candidate", "model_name", "feature_set", "predictors", "model_parameters", "split", "threshold", "score_definition", "score_unit")}
            selected.update(selected_before_outer=True, selected_by="fixed predeclared plan; no candidate tuning", seed=config.seed)
            save_json(prefix + "/selected_policy.json", selected)
            save_json(prefix + "/threshold.json", threshold)
            save_json(prefix + "/training_provenance.json", bundle["training_provenance"])
            buffer = io.BytesIO()
            joblib.dump(bundle, buffer, compress=3)
            save(prefix + "/model_bundle.joblib", buffer.getvalue())
            decision = decide(scored, threshold["threshold"], config)
            save(prefix + "/validation_scores.csv.gz", _gzip_rows(decision, SCORE_COLUMNS))
            calibration.append(dict(candidate=spec.name, fold_id=fold, validation_flight=split.validation[0],
                                    threshold=threshold["threshold"], score_unit=bundle["score_unit"],
                                    **alarm_metrics(decision, config.gap_threshold_seconds)))
            prepared.append((spec, fold, split, prefix))
            del bundle, scored, decision, buffer
    freeze = {name: digest for name, digest in hashes.items() if name.endswith(("selected_policy.json", "threshold.json", "model_bundle.joblib"))}
    save_json("all_policies_frozen.json", dict(status="FROZEN_BEFORE_OUTER_EVALUATION", candidate_folds=len(prepared), files=freeze))
    regression, healthy, anomaly, scenarios = [], [], [], []
    # Pairing signatures must match across models and feature sets, incl NA cases.
    signatures = {}
    for spec, fold, split, prefix in prepared:
        bundle = load_candidate_bundle(output / prefix / "model_bundle.joblib")
        threshold = bundle["threshold"]["threshold"]
        test = by_flight[split.test[0]]
        scored = score_candidate(bundle, test, config, "test")
        decision = decide(scored, threshold, config)
        save(prefix + "/healthy_test_scores.csv.gz", _gzip_rows(decision, SCORE_COLUMNS))
        tags = dict(candidate=spec.name, model=spec.model, feature_set=spec.feature_set, fold_id=fold, test_flight=split.test[0])
        ready = [r for r in scored if r["reference_ready"]]
        if spec.regression:
            y = np.array([r["voltage_sag_actual"] for r in ready])
            e = np.array([r["residual_signed_v"] for r in ready])
            regression.append(dict(**tags, **regression_metrics(scored), sum_absolute_error_v=float(np.sum(np.abs(e))),
                                   sum_squared_error_v2=float(np.sum(e ** 2)), sum_target_v=float(np.sum(y)), sum_squared_target_v2=float(np.sum(y ** 2))))
        health = dict(**tags, unseen_by_model=True, used_for_threshold=False, **alarm_metrics(decision, config.gap_threshold_seconds))
        healthy.append(health)
        local = []
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(test, config, family=family, severity=severity, start_position=position)
                    meta = generated.metadata
                    key = (fold, meta["scenario_id"])
                    signature = tuple(meta.get(k) for k in ("fault_applicable", "onset_time_us", "fault_end_time_us", "support_samples", "observability", "frozen_v_ref", "reason"))
                    if key in signatures and signatures[key] != signature:
                        raise DataError("Candidate scenario pairing/denominator changed.")
                    signatures[key] = signature
                    evaluated, score_path = [], None
                    if meta["fault_applicable"]:
                        evaluated = decide(score_candidate(bundle, generated.rows, config, "test", healthy_cache=scored if spec.regression else None), threshold, config)
                        score_path = prefix + f"/scenario_scores/{meta['scenario_id']}.csv.gz"
                        save(score_path, _gzip_rows(evaluated, SCENARIO_SCORE_COLUMNS))
                    metrics = scenario_metrics(evaluated, meta, cfg["evaluation"]["deadlines_seconds"], config.gap_threshold_seconds)
                    record = dict(meta, **tags, validation_flight=split.validation[0], threshold=threshold,
                                  score_unit=bundle["score_unit"], persistence_seconds=.5, unseen_by_model=True, used_for_threshold=False)
                    record.update(metrics, scores_csv=score_path, scores_sha256=hashes[score_path] if score_path else None)
                    local.append(record)
                    scenarios.append(record)
        anomaly.append(dict(**tags, **aggregate_scenarios(local, cfg["evaluation"]["deadlines_seconds"])))
        del bundle, scored, decision, evaluated, generated
    candidate_results, summary = summarize(specs, regression, healthy, anomaly, scenarios, cfg["evaluation"]["deadlines_seconds"])
    paired = []
    for spec in specs:
        if spec.name == "hgb_battery_d":
            continue
        for i in range(len(splits)):
            fold = f"fold_{i:02d}"
            reference = next(r for r in anomaly if r["candidate"] == "hgb_battery_d" and r["fold_id"] == fold)
            case = next(r for r in anomaly if r["candidate"] == spec.name and r["fold_id"] == fold)
            h_ref = next(r for r in healthy if r["candidate"] == "hgb_battery_d" and r["fold_id"] == fold)
            h_case = next(r for r in healthy if r["candidate"] == spec.name and r["fold_id"] == fold)
            record = dict(candidate=spec.name, reference_candidate="hgb_battery_d", fold_id=fold, test_flight=case["test_flight"],
                          applicable_scenarios=case["applicable_scenarios"], ready_samples=h_case["ready_samples"],
                          healthy_fpr_delta=h_case["confirmed_fpr"] - h_ref["confirmed_fpr"],
                          episodes_per_minute_delta=h_case["confirmed_episodes_per_minute"] - h_ref["confirmed_episodes_per_minute"])
            for metric in ("f1", "incident_recall", *[deadline_name(d) + "_rate" for d in cfg["evaluation"]["deadlines_seconds"]]):
                record[metric + "_delta"] = case[metric] - reference[metric] if case[metric] is not None and reference[metric] is not None else None
            paired.append(record)
    overview = {name: dict(healthy_confirmed_fpr=r["pooled"]["healthy_active"]["confirmed_fpr"],
                           healthy_episodes_per_minute=r["pooled"]["healthy_active"]["confirmed_episodes_per_minute"],
                           incident_detected=r["pooled"]["synthetic"]["detected_scenarios"],
                           applicable=r["pooled"]["synthetic"]["applicable_scenarios"],
                           sample_f1=r["pooled"]["synthetic"]["f1"],
                           regression_rmse_v=r["pooled"]["regression"]["rmse_v"] if r["pooled"]["regression"] else None)
                for name, r in candidate_results.items()}
    run_summary = dict(status="OK", command=mode, run_id=run_id, output_directory=str(output),
                       evaluation_scope=protocol["evaluation_scope"], folds=len(splits), candidates=len(specs), candidate_folds=len(prepared),
                       candidate_names=[s.name for s in specs], active_flights=eligible, feature_run_id=feature_summary["run_id"],
                       data_sha256=feature_summary["data_sha256"], feature_manifest_sha256=upstream,
                       all_policies_frozen_before_outer=True, paired_scenarios_verified=True, scenario_evaluations=len(scenarios),
                       overview=overview, selected_winner=None, operational_config_changed=False, low_load_evaluated=False,
                       deployment_status="RESEARCH_COMPARISON_ONLY", currtot_unit="unverified")
    full = dict(run_summary=run_summary, candidates=candidate_results)
    for name, data in (("regression_per_fold.csv", regression), ("healthy_active_per_fold.csv", healthy),
                       ("anomaly_per_fold.csv", anomaly), ("fault_per_scenario.csv", scenarios),
                       ("validation_calibration_per_fold.csv", calibration), ("summary.csv", summary), ("paired_per_fold.csv", paired)):
        save(name, _csv(data))
    save("failed_cases.csv", _csv([r for r in scenarios if not r["incident_detected"]], list(scenarios[0])))
    save_json("results.json", full)
    save("research_report.md", comparison_report(full, scenarios).encode())
    save("comparison.png", comparison_plot(full))
    if any(hashlib.sha256((output / name).read_bytes()).hexdigest() != digest for name, digest in freeze.items()):
        raise DataError("Comparison selection/model changed after outer evaluation.")
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != run_summary["data_sha256"]:
        raise DataError("Raw changed during comparison.")
    _write_files(output, {"manifest.json": (format_json(dict(run_id=run_id, identity=identity, outputs=dict(sorted(hashes.items())))) + "\n").encode()}, config)
    return run_summary
