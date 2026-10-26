"""Frozen 20-policy HGB-D search on validation only; paired outer diagnostics."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

import joblib
import numpy as np
import yaml

from .config import project_path
from .cross_validation import loafo_splits
from .detectors import apply_single_path, calibrate_threshold
from .errors import ConfigError, DataError
from .evaluation import aggregate_scenarios, scenario_metrics
from .experimental_models import Candidate, fit_candidate, load_candidate_bundle, score_candidate
from .fault_injection import FAMILIES, SEVERITIES, inject_fault
from .features import generate_features
from .loafo import SCENARIO_SCORE_COLUMNS, _gzip_rows, _summaries
from .models import SCORE_COLUMNS, _feature_input, _write_files
from .pilot_reporting import alarm_metrics, regression_metrics
from .preprocessing import _csv
from .reporting import format_json

HEALTH_COLUMNS = (*SCORE_COLUMNS, "used_for_policy_selection")
FAULT_COLUMNS = (*SCENARIO_SCORE_COLUMNS, "used_for_policy_selection")
RANK_FIELDS = ("abrupt_severe_le_1s_rate", "gradual_observable_le_5s_rate", "incident_recall", "f1",
               "delay_median_seconds", "delay_mean_seconds")


def policy_id(percentile, duration):
    return "p" + format(percentile, "g").replace(".", "p") + "_d" + format(duration, "g").replace(".", "p")


def rank_key(row):
    """Minimize tuple: four descending metrics, two ascending delays, parameters."""
    def finite(value):
        return value is not None and math.isfinite(value)
    return (*[-row[k] if finite(row[k]) else math.inf for k in RANK_FIELDS[:4]],
            *[row[k] if finite(row[k]) else math.inf for k in RANK_FIELDS[4:]],
            row["percentile"], row["persistence_seconds"])


def healthy_eligible(row, reference):
    keys = ("healthy_confirmed_fpr", "healthy_episodes_per_minute")
    if any(r.get(k) is None or not math.isfinite(r[k]) for r in (row, reference) for k in keys):
        return False, "UNDEFINED_HEALTHY_GATE"
    reasons = [k.upper() + "_EXCEEDS_REFERENCE" for k in keys if row[k] > reference[k]]
    return not reasons, ";".join(reasons) if reasons else "HEALTHY_GATES_PASS"


def select_policy(rows, validation_flight):
    """Input contains validation metrics only; no outer-derived comparison."""
    if not rows or any(r.get("source_role") != "validation" or r.get("validation_flight") != validation_flight
                       or r.get("used_for_policy_selection") is not True for r in rows):
        raise DataError("Policy selection accepts assigned validation metrics only.")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise DataError("Duplicate policy candidate.")
    reference = next((r for r in rows if r["percentile"] == 99. and r["persistence_seconds"] == .5), None)
    if reference is None:
        raise DataError("Reference P99/0.5 s must be in the candidate budget.")
    eligible = []
    for r in rows:
        r["healthy_eligible"], r["eligibility_reason"] = healthy_eligible(r, reference)
        if r["healthy_eligible"]:
            eligible.append(r)
    eligible.sort(key=rank_key)
    chosen = eligible[0] if eligible else reference
    for r in rows:
        r["eligible_rank"] = next((i + 1 for i, e in enumerate(eligible) if e is r), None)
        r["selected"] = r is chosen
    return dict(chosen, selection_reason="LEXICOGRAPHIC_VALIDATION_RANK" if eligible else "NO_ELIGIBLE_CANDIDATE_REFERENCE_RETAINED",
                eligible_candidates=len(eligible), candidates_evaluated=len(rows),
                ranking_fields=list(RANK_FIELDS), selected_before_outer=True)


def selection_metrics(records, deadlines):
    aggregate = aggregate_scenarios(records, deadlines)
    abrupt = aggregate_scenarios([r for r in records if r["fault_family"] in ("increased_resistance", "sudden_drop") and r["severity"] == "severe"], deadlines)
    gradual = aggregate_scenarios([r for r in records if r["fault_family"] == "progressive" and r["observability"] in ("MEDIUM", "HIGH")], deadlines)
    return dict(aggregate, abrupt_severe_le_1s_rate=abrupt["deadline_le_1s_rate"],
                abrupt_severe_applicable=abrupt["applicable_scenarios"], abrupt_severe_successes=abrupt["deadline_le_1s_successes"],
                gradual_observable_le_5s_rate=gradual["deadline_le_5s_rate"],
                gradual_observable_applicable=gradual["applicable_scenarios"], gradual_observable_successes=gradual["deadline_le_5s_successes"])


def label_scores(rows, role, synthetic=False):
    # Synthetic validation supports secondary policy ranking, never quantiles.
    return [{**r, "used_for_threshold": bool(role == "validation" and not synthetic and r["reference_ready"]),
             "used_for_policy_selection": bool(role == "validation" and r["reference_ready"])} for r in rows]


def decide(rows, threshold, duration, config):
    return apply_single_path(rows, threshold_v=threshold, duration_seconds=duration, gap_threshold_seconds=config.gap_threshold_seconds)


def run_tuning(config):
    from .tuning_reporting import tuning_report, tuning_plots
    cfg = config.as_dict()
    percentiles, durations = cfg["threshold"]["candidates"], cfg["persistence"]["candidates_seconds"]
    if (cfg["features"]["operational_set"] != "battery_d" or cfg["features"]["rolling_windows_seconds"] != [.5, 1.]
            or cfg["threshold"]["reference_percentile"] != 99. or cfg["persistence"]["reference_duration_seconds"] != .5
            or cfg["dual_path"]["enabled"] or len(percentiles) != 4 or len(durations) != 5
            or 99. not in percentiles or .5 not in durations):
        raise ConfigError("Sprint 8 requires HGB Battery-D, fixed P99/0.5 reference, 4x5 configured grid including reference, dual-path off.")
    features = generate_features(config)
    if features["currtot_predictor_review_required"]:
        raise DataError("CurrTot decreases require review before tuning.")
    rows, coverage, registry, upstream = _feature_input(features, config)
    eligible = sorted(r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True")
    splits = loafo_splits(eligible, cfg["evaluation"]["minimum_active_flights"])
    policies = [dict(candidate_id=policy_id(p, d), percentile=p, persistence_seconds=d) for p in sorted(percentiles) for d in sorted(durations)]
    spec = Candidate("hgb", "battery_d")
    deadlines = cfg["evaluation"]["deadlines_seconds"]
    protocol = dict(experiment="sprint8_validation_threshold_persistence", evaluation_scope="historical_development_validation_selection",
                    model="hgb_battery_d", model_parameters=cfg["models"]["hist_gradient_boosting"], random_seed=config.seed,
                    features=spec.predictors(config), early_stopping=False, scaler=None, openmp_threads=1,
                    assignments=[dict(fold_id=f"fold_{i:02d}", **s.as_dict()) for i, s in enumerate(splits)],
                    candidate_budget_per_fold=len(policies), policies=policies, reference_policy=dict(percentile=99., persistence_seconds=.5),
                    quantile_source="healthy assigned validation only; linear percentile; synthetic never calibrated",
                    secondary_source="27 fixed synthetic copies of assigned validation only, all candidates evaluated",
                    healthy_gate="reject confirmed sample FPR OR confirmed episodes/min above validation reference; undefined gate ineligible",
                    ranking=dict(descending=list(RANK_FIELDS[:4]), ascending=list(RANK_FIELDS[4:]), final_tie=["percentile", "persistence_seconds"],
                                 undefined="rates/F1 rank below defined zero; delay +infinity", fallback="reference if no eligible"),
                    phase_order="fit and select ALL folds using validation, save/freeze all policies and inputs, then evaluate paired outer reference/selected",
                    train_policy="whole assigned healthy-ready training flights incl idle, no synthetic, fixed model/feature hyperparameters",
                    incident_policy="new confirmation within support; carried warning contaminates whole incident; misses remain deadline failures",
                    prediction_reuse="exact registered predictor/identity/readiness equality before reusing synthetic HGB prediction",
                    no_outer_reselection=True, no_automatic_deployment=True, feature_manifest_sha256=upstream,
                    data_sha256=features["data_sha256"], config_sha256=features["config_sha256"], fault_config=cfg["fault_injection"],
                    deadlines_seconds=deadlines, dependency_versions=features["dependency_versions"], currtot_unit="unverified")
    code = {p.relative_to(Path(__file__).parent.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "tuning" / run_id, config.root)
    if (output / "manifest.json").exists():
        try:
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["run_id"] != run_id or manifest["identity"] != identity:
                raise DataError("Tuning manifest identity mismatch.")
            for name, digest in manifest["outputs"].items():
                if hashlib.sha256(project_path(output / name, config.root).read_bytes()).hexdigest() != digest:
                    raise DataError(f"Tuning artifact checksum mismatch: {name}")
            return json.loads((output / "results.json").read_text())["run_summary"]
        except (KeyError, ValueError) as exc:
            raise DataError(f"Invalid tuning manifest: {exc}") from exc
    hashes = {}

    def save(name, value):
        project_path(output / name, config.root).parent.mkdir(parents=True, exist_ok=True)
        _write_files(output, {name: value}, config)
        hashes[name] = hashlib.sha256(value).hexdigest()

    def save_json(name, value):
        save(name, (format_json(value) + "\n").encode())

    save_json("experiment_protocol.json", protocol)
    save_json("feature_registry.json", registry)
    save("config_used.yaml", yaml.safe_dump(cfg, sort_keys=False).encode())
    save("reference_coverage.csv", (Path(features["output_directory"]) / "reference_coverage.csv").read_bytes())
    by_flight = defaultdict(list)
    for r in rows:
        by_flight[r["NamaFile"]].append(r)
    assignments = []
    for i, split in enumerate(splits):
        for flight in coverage:
            name = flight["NamaFile"]
            role = "train" if name in split.train else "validation" if name in split.validation else "test" if name in split.test else "excluded"
            assignments.append(dict(fold_id=f"fold_{i:02d}", NamaFile=name, split_role=role,
                                    ready_samples=int(flight["ready_samples"]), retained_samples=int(flight["retained_samples"]),
                                    reference_status=flight["reference_status"], activity_label=flight["v2_current_based_label"]))
    save("fold_assignments.csv", _csv(assignments))
    grid, validation_cases, selected = [], [], []
    # No outer flight is scored or injected in this phase.
    for i, split in enumerate(splits):
        fold, prefix = f"fold_{i:02d}", f"folds/fold_{i:02d}"
        train = [r for flight in split.train for r in by_flight[flight] if r["reference_ready"]]
        bundle = fit_candidate(train, config, split, spec)
        validation = label_scores(score_candidate(bundle, by_flight[split.validation[0]], config, "validation"), "validation")
        thresholds = {p: calibrate_threshold([r for r in validation if r["reference_ready"]], split, percentile=p) for p in percentiles}
        save_json(prefix + "/threshold_candidates.json", {format(p, "g"): t for p, t in thresholds.items()})
        base = decide(validation, thresholds[99.]["threshold_v"], .5, config)
        save(prefix + "/healthy_validation_scores.csv.gz", _gzip_rows(base, HEALTH_COLUMNS))
        health, local = {}, defaultdict(list)
        for policy in policies:
            decision = decide(validation, thresholds[policy["percentile"]]["threshold_v"], policy["persistence_seconds"], config)
            health[policy["candidate_id"]] = alarm_metrics(decision, config.gap_threshold_seconds)
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(by_flight[split.validation[0]], config, family=family, severity=severity, start_position=position)
                    meta = generated.metadata
                    scored, path = [], None
                    if meta["fault_applicable"]:
                        scored = label_scores(score_candidate(bundle, generated.rows, config, "validation", healthy_cache=validation), "validation", synthetic=True)
                        path = prefix + f"/validation_scenario_scores/{meta['scenario_id']}.csv.gz"
                        save(path, _gzip_rows(decide(scored, thresholds[99.]["threshold_v"], .5, config), FAULT_COLUMNS))
                    for policy in policies:
                        evaluated = decide(scored, thresholds[policy["percentile"]]["threshold_v"], policy["persistence_seconds"], config) if scored else []
                        record = dict(meta, **policy, fold_id=fold, validation_flight=split.validation[0], source_role="validation",
                                      used_for_policy_selection=True, used_for_threshold=False, threshold_v=thresholds[policy["percentile"]]["threshold_v"],
                                      scores_csv=path, scores_sha256=hashes[path] if path else None)
                        record.update(scenario_metrics(evaluated, meta, deadlines, config.gap_threshold_seconds))
                        local[policy["candidate_id"]].append(record)
                        validation_cases.append(record)
        candidates = []
        for policy in policies:
            h = health[policy["candidate_id"]]
            candidates.append(dict(**policy, fold_id=fold, validation_flight=split.validation[0], source_role="validation",
                                   used_for_policy_selection=True, threshold_v=thresholds[policy["percentile"]]["threshold_v"],
                                   healthy_ready_samples=h["ready_samples"], healthy_raw_fpr=h["raw_fpr"],
                                   healthy_raw_alarm_samples=h["raw_alarm_samples"], healthy_confirmed_alarm_samples=h["confirmed_alarm_samples"],
                                   healthy_confirmed_fpr=h["confirmed_fpr"], healthy_confirmed_episodes=h["confirmed_episodes"],
                                   healthy_observed_ready_seconds=h["observed_ready_seconds"], healthy_episodes_per_minute=h["confirmed_episodes_per_minute"],
                                   **selection_metrics(local[policy["candidate_id"]], deadlines)))
        chosen = select_policy(candidates, split.validation[0])
        grid.extend(candidates)
        threshold = dict(thresholds[chosen["percentile"]], threshold=chosen["threshold_v"], persistence_seconds=chosen["persistence_seconds"],
                         policy_selection_source="healthy and synthetic validation only", selected_before_outer=True)
        reference = dict(thresholds[99.], threshold=thresholds[99.]["threshold_v"], persistence_seconds=.5)
        bundle.update(threshold=threshold, experiment_stage=8, fold_id=fold, feature_registry=registry,
                      feature_manifest_sha256=upstream, data_sha256=features["data_sha256"], code_sha256=code)
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer, compress=3)
        save(prefix + "/model_bundle.joblib", buffer.getvalue())
        save_json(prefix + "/training_provenance.json", bundle["training_provenance"])
        save_json(prefix + "/selected_policy.json", dict(selection=chosen, threshold=threshold, split=split.as_dict()))
        save_json(prefix + "/reference_policy.json", reference)
        selected.append(dict(fold_id=fold, split=split.as_dict(), selection=chosen, threshold=threshold, reference=reference))
        del bundle, validation, scored, base, evaluated, generated, buffer
    save("threshold_persistence_grid.csv", _csv(grid))
    save("validation_fault_per_candidate.csv", _csv(validation_cases))
    save_json("selected_per_fold.json", selected)
    freeze = dict(hashes)
    save_json("all_policies_frozen.json", dict(status="FROZEN_BEFORE_ANY_OUTER_EVALUATION", folds=len(splits), files=freeze))
    regression, healthy, anomaly, scenarios = [], [], [], []
    for entry in selected:
        fold, split = entry["fold_id"], entry["split"]
        prefix = "folds/" + fold
        bundle = load_candidate_bundle(output / prefix / "model_bundle.joblib")
        test = by_flight[split["test"][0]]
        scored = label_scores(score_candidate(bundle, test, config, "test"), "test")
        for policy, payload in (("reference", entry["reference"]), ("selected", entry["threshold"])):
            decision = decide(scored, payload["threshold_v"], payload["persistence_seconds"], config)
            save(prefix + f"/{policy}/healthy_test_scores.csv.gz", _gzip_rows(decision, HEALTH_COLUMNS))
            tags = dict(policy=policy, fold_id=fold, test_flight=split["test"][0], unseen_by_model=True, used_for_policy_selection=False)
            ready = [r for r in scored if r["reference_ready"]]
            y, e = np.array([r["voltage_sag_actual"] for r in ready]), np.array([r["residual_signed_v"] for r in ready])
            regression.append(dict(**tags, **regression_metrics(scored), sum_absolute_error_v=float(np.sum(np.abs(e))),
                                   sum_squared_error_v2=float(np.sum(e ** 2)), sum_target_v=float(np.sum(y)), sum_squared_target_v2=float(np.sum(y ** 2))))
            healthy.append(dict(**tags, **alarm_metrics(decision, config.gap_threshold_seconds)))
        local = {"reference": [], "selected": []}
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(test, config, family=family, severity=severity, start_position=position)
                    meta = generated.metadata
                    synthetic = label_scores(score_candidate(bundle, generated.rows, config, "test", healthy_cache=scored), "test", synthetic=True) if meta["fault_applicable"] else []
                    for policy, payload in (("reference", entry["reference"]), ("selected", entry["threshold"])):
                        evaluated, path = [], None
                        if synthetic:
                            evaluated = decide(synthetic, payload["threshold_v"], payload["persistence_seconds"], config)
                            path = prefix + f"/{policy}/scenario_scores/{meta['scenario_id']}.csv.gz"
                            save(path, _gzip_rows(evaluated, FAULT_COLUMNS))
                        record = dict(meta, policy=policy, fold_id=fold, test_flight=split["test"][0], validation_flight=split["validation"][0],
                                      source_role="test", used_for_policy_selection=False, used_for_threshold=False, unseen_by_model=True,
                                      threshold_v=payload["threshold_v"], persistence_seconds=payload["persistence_seconds"],
                                      scores_csv=path, scores_sha256=hashes[path] if path else None)
                        record.update(scenario_metrics(evaluated, meta, deadlines, config.gap_threshold_seconds))
                        scenarios.append(record)
                        local[policy].append(record)
        for policy in local:
            anomaly.append(dict(policy=policy, fold_id=fold, test_flight=split["test"][0], **selection_metrics(local[policy], deadlines)))
    results, summaries = {}, []
    for policy in ("reference", "selected"):
        subsets = [[r for r in data if r["policy"] == policy] for data in (regression, healthy, anomaly, scenarios)]
        result, table = _summaries(*subsets, deadlines)
        results[policy] = result
        summaries.extend(dict(policy=policy, **r) for r in table)
    paired = []
    for entry in selected:
        fold = entry["fold_id"]
        h = {p: next(r for r in healthy if r["fold_id"] == fold and r["policy"] == p) for p in ("reference", "selected")}
        a = {p: next(r for r in anomaly if r["fold_id"] == fold and r["policy"] == p) for p in ("reference", "selected")}
        row = dict(fold_id=fold, test_flight=entry["split"]["test"][0], validation_flight=entry["split"]["validation"][0],
                   selected_percentile=entry["selection"]["percentile"], selected_persistence_seconds=entry["selection"]["persistence_seconds"],
                   healthy_fpr_delta=h["selected"]["confirmed_fpr"] - h["reference"]["confirmed_fpr"],
                   healthy_episodes_per_minute_delta=h["selected"]["confirmed_episodes_per_minute"] - h["reference"]["confirmed_episodes_per_minute"])
        for k in ("f1", "incident_recall", "delay_mean_seconds", "delay_median_seconds", *[f"deadline_le_{int(d)}s_rate" for d in deadlines]):
            row[k + "_delta"] = a["selected"][k] - a["reference"][k] if a["selected"][k] is not None and a["reference"][k] is not None else None
        paired.append(row)
    summary = dict(status="OK", command="tune", run_id=run_id, output_directory=str(output), evaluation_scope=protocol["evaluation_scope"],
                   folds=len(splits), candidate_budget_per_fold=len(policies), candidates_evaluated=len(grid),
                   validation_scenario_evaluations=len(validation_cases), outer_scenario_evaluations=len(scenarios),
                   active_flights=eligible, feature_run_id=features["run_id"], feature_manifest_sha256=upstream, data_sha256=features["data_sha256"],
                   all_policies_frozen_before_outer=True, outer_used_for_selection=False,
                   selected_policies=[dict(fold_id=r["fold_id"], percentile=r["selection"]["percentile"], persistence_seconds=r["selection"]["persistence_seconds"],
                                           eligible_candidates=r["selection"]["eligible_candidates"]) for r in selected],
                   reference_pooled=results["reference"]["pooled"], selected_pooled=results["selected"]["pooled"],
                   operational_config_changed=False, deployment_status="RESEARCH_TUNED_CANDIDATE_ONLY", low_load_evaluated=False, currtot_unit="unverified")
    full = dict(run_summary=summary, policies=results)
    for name, data in (("regression_per_fold.csv", regression), ("healthy_active_per_fold.csv", healthy), ("anomaly_per_fold.csv", anomaly),
                       ("fault_per_scenario.csv", scenarios), ("paired_per_fold.csv", paired), ("summary.csv", summaries)):
        save(name, _csv(data))
    save("failed_cases.csv", _csv([r for r in scenarios if not r["incident_detected"]], list(scenarios[0])))
    save_json("results.json", full)
    save("research_report.md", tuning_report(full, selected, paired).encode())
    for name, payload in tuning_plots(grid, healthy, paired).items():
        save(name, payload)
    if any(hashlib.sha256((output / name).read_bytes()).hexdigest() != digest for name, digest in freeze.items()):
        raise DataError("Validation selection inputs/model/policies changed after outer evaluation.")
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise DataError("Raw data changed during tuning.")
    _write_files(output, {"manifest.json": (format_json(dict(run_id=run_id, identity=identity, outputs=dict(sorted(hashes.items())))) + "\n").encode()}, config)
    return summary
