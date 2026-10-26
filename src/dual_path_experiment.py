"""Predeclared Path A selection, healthy shortlist and synthetic refinement."""

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from .config import project_path
from .cross_validation import loafo_splits
from .detectors import calibrate_threshold
from .dual_detector import (PATH_COLUMNS, PARAMETERS, apply_dual, attribution_metrics, candidate_grid,
                            calibrate_slopes, choose_path_a, choose_refined, promotion_decision, shortlist)
from .errors import ConfigError, DataError
from .evaluation import scenario_metrics
from .experimental_models import Candidate, fit_candidate, load_candidate_bundle, score_candidate
from .fault_injection import FAMILIES, SEVERITIES, inject_fault
from .features import generate_features
from .loafo import SCENARIO_SCORE_COLUMNS, _gzip_rows, _summaries
from .models import SCORE_COLUMNS, _feature_input, _write_files
from .pilot_reporting import alarm_metrics, regression_metrics
from .preprocessing import _csv
from .reporting import format_json
from .residual_diagnostics import diagnose, diagnostic_columns
from .single_path_tuning import RANK_FIELDS, label_scores, selection_metrics


def health_fields(rows, config):
    values = alarm_metrics(rows, config.gap_threshold_seconds)
    return dict(healthy_ready_samples=values["ready_samples"], healthy_confirmed_fpr=values["confirmed_fpr"],
                healthy_episodes_per_minute=values["confirmed_episodes_per_minute"], healthy_confirmed_alarm_samples=values["confirmed_alarm_samples"],
                healthy_confirmed_episodes=values["confirmed_episodes"], healthy_observed_ready_seconds=values["observed_ready_seconds"])


def baseline_context(rows, fold):
    """Observed baseline-warning episode context; no threshold-based transient label."""
    groups, current = [], []
    for r in rows:
        if r["confirmed_episode_start"]:
            if current:
                groups.append(current)
            current = []
        if r["anomaly_confirmed"]:
            current.append(r)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [dict(fold_id=fold, validation_flight=g[0]["NamaFile"], source_role="validation", episode=i,
                 start_time_us=g[0]["TimeUS"], end_time_us=g[-1]["TimeUS"], confirmed_samples=len(g),
                 current_min_a=min(r["Curr"] for r in g), current_max_a=max(r["Curr"] for r in g),
                 max_abs_dI_dt_a_per_s=max(abs(r["dI_dt"]) for r in g),
                 residual_min_v=min(r["residual_signed_v"] for r in g), residual_max_v=max(r["residual_signed_v"] for r in g)) for i, g in enumerate(groups)]


def run_dual_experiment(config):
    from .dual_reporting import dual_report, promotion_report, dual_plots
    cfg = config.as_dict()
    dc = cfg["dual_path"]
    windows = dc["slope_windows_seconds"]
    policies = candidate_grid(config)
    if (cfg["features"]["operational_set"] != "battery_d" or cfg["features"]["rolling_windows_seconds"] != [.5, 1.]
            or cfg["threshold"]["reference_percentile"] != 99 or cfg["persistence"]["reference_duration_seconds"] != .5
            or dc["enabled"] or len(policies) != 144 or not 1 <= dc["refinement_max_candidates"] <= 3
            or len(windows) != 3 or len(dc["high_percentiles"]) != 2 or len(dc["low_percentiles"]) != 3
            or len(dc["slope_percentiles"]) != 4 or len(dc["path_b_persistence_seconds"]) != 2):
        raise ConfigError("Sprint 9 requires fixed HGB-D/P99/0.5 reference, operational dual off, 144 configured pairs and shortlist <=3.")
    feature_summary = generate_features(config)
    if feature_summary["currtot_predictor_review_required"]:
        raise DataError("CurrTot decreases require review before dual-path training.")
    rows, coverage, registry, upstream = _feature_input(feature_summary, config)
    eligible = sorted(r["NamaFile"] for r in coverage if r["activity_and_reference_eligible"] == "True")
    splits = loafo_splits(eligible, cfg["evaluation"]["minimum_active_flights"])
    spec, deadlines = Candidate("hgb", "battery_d"), cfg["evaluation"]["deadlines_seconds"]
    protocol = dict(experiment="sprint9_causal_residual_dual_path", evaluation_scope="historical_development_dual_path_selection",
                    assignments=[dict(fold_id=f"fold_{i:02d}", **s.as_dict()) for i, s in enumerate(splits)],
                    features=spec.predictors(config), model_parameters=cfg["models"]["hist_gradient_boosting"], seed=config.seed,
                    early_stopping=False, scaler=None, openmp_threads=1, dual_config=dc, candidate_budget_per_fold=144,
                    reference="fixed HGB-D P99/0.5 s operational baseline, not outer-chosen Sprint 8 candidate",
                    slope="signed residual OLS in (t-window,t], >=2 unique timestamps, local time origin; reset gap/segment/unready",
                    delta="adjacent signed residual difference V and actual-time rate V/s, no value until previous ready in same continuity segment",
                    path_a_selection="healthy-validation standalone A: lowest confirmed FPR, episodes/min, then ascending high percentile; persistence .5 s",
                    healthy_grid="evaluate all 144 pairs for audit; only selected Path A is shortlist eligible; no slope-ready calibration => ineligible",
                    shortlist="healthy union FPR AND episodes/min <= fixed reference; sort FPR, episodes/min, ascending full parameter tuple; cap configured <=3",
                    refinement="all 27 fixed synthetic validation copies on shortlisted pairs only; need >=1 applicable case; no unrefined candidate may win",
                    refinement_rank=dict(descending=list(RANK_FIELDS[:4]), ascending=list(RANK_FIELDS[4:]), final_tie=list(PARAMETERS),
                                         undefined="rate/F1 below defined zero; delay +infinity"),
                    fallback="fixed P99/0.5 single-path when no safely refined dual candidate",
                    quantiles="positive residual high/low and signed slope percentiles from assigned healthy validation only; linear interpolation",
                    timers="independent A .5 s and B .3/.5 s; OR confirmed union; union episode only false->true or gap; no timer on combined raw",
                    attribution="A_ONLY/B_ONLY/BOTH/NONE per confirmed sample; first_confirmation_path only union starts; incident path only if uncontaminated",
                    duration_column="anomaly_duration_seconds=max(individual A,B raw elapsed), diagnostic only, never union persistence",
                    phase_order="fit/calibrate/select/refine ALL folds, save/freeze models and validation inputs before any outer score",
                    promotion_gate="paired arithmetic macro means over ALL folds, every gate value defined; healthy FPR/episode no increase, F1/all deadlines no decrease, mean detected-incident delay strictly decreases; >=1 refined dual fold",
                    no_outer_reselection=True, no_operational_enable=True, diagnostic_transient_context="numeric current/dI_dt at baseline validation warning episodes; no causal or physical attribution",
                    fault_config=cfg["fault_injection"], deadlines_seconds=deadlines, feature_manifest_sha256=upstream,
                    data_sha256=feature_summary["data_sha256"], config_sha256=feature_summary["config_sha256"], dependency_versions=feature_summary["dependency_versions"])
    code = {p.relative_to(Path(__file__).parent.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = dict(protocol=protocol, code_sha256=code)
    run_id = hashlib.sha256(format_json(identity).encode()).hexdigest()[:16]
    output = project_path(config.artifacts_path / "dual_path" / run_id, config.root)
    if (output / "manifest.json").exists():
        try:
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["run_id"] != run_id or manifest["identity"] != identity:
                raise DataError("Dual-path manifest identity mismatch.")
            for name, digest in manifest["outputs"].items():
                if hashlib.sha256(project_path(output / name, config.root).read_bytes()).hexdigest() != digest:
                    raise DataError(f"Dual-path artifact checksum mismatch: {name}")
            return json.loads((output / "results.json").read_text())["run_summary"]
        except (KeyError, ValueError) as exc:
            raise DataError(f"Invalid dual-path manifest: {exc}") from exc
    hashes = {}

    def save(name, payload):
        project_path(output / name, config.root).parent.mkdir(parents=True, exist_ok=True)
        _write_files(output, {name: payload}, config)
        hashes[name] = hashlib.sha256(payload).hexdigest()

    def save_json(name, payload):
        save(name, (format_json(payload) + "\n").encode())

    health_columns = (*SCORE_COLUMNS, "used_for_policy_selection", *diagnostic_columns(windows), *PATH_COLUMNS)
    fault_columns = (*SCENARIO_SCORE_COLUMNS, "used_for_policy_selection", *diagnostic_columns(windows), *PATH_COLUMNS)
    save_json("experiment_protocol.json", protocol)
    save_json("feature_registry.json", registry)
    save("config_used.yaml", yaml.safe_dump(cfg, sort_keys=False).encode())
    save("reference_coverage.csv", (Path(feature_summary["output_directory"]) / "reference_coverage.csv").read_bytes())
    by_flight = defaultdict(list)
    for r in rows:
        by_flight[r["NamaFile"]].append(r)
    assignments = []
    for i, split in enumerate(splits):
        for flight in coverage:
            name = flight["NamaFile"]
            role = "train" if name in split.train else "validation" if name in split.validation else "test" if name in split.test else "excluded"
            assignments.append(dict(fold_id=f"fold_{i:02d}", NamaFile=name, split_role=role, ready_samples=int(flight["ready_samples"]),
                                    retained_samples=int(flight["retained_samples"]), reference_status=flight["reference_status"]))
    save("fold_assignments.csv", _csv(assignments))
    all_grid, refined, validation_cases, validation_manifest, selections, contexts = [], [], [], [], [], []
    validation_example = None
    for i, split in enumerate(splits):
        fold, prefix = f"fold_{i:02d}", f"folds/fold_{i:02d}"
        training = [r for flight in split.train for r in by_flight[flight] if r["reference_ready"]]
        bundle = fit_candidate(training, config, split, spec)
        base = label_scores(score_candidate(bundle, by_flight[split.validation[0]], config, "validation"), "validation")
        validation = diagnose(base, windows, config.gap_threshold_seconds)
        percentiles = sorted({99., *dc["high_percentiles"], *dc["low_percentiles"]})
        thresholds = {p: calibrate_threshold([r for r in base if r["reference_ready"]], split, percentile=p) for p in percentiles}
        slopes = calibrate_slopes(validation, split, windows, dc["slope_percentiles"])
        reference = dict(mode="REFERENCE", high_percentile=99., threshold_high_v=thresholds[99.]["threshold_v"], path_a_persistence_seconds=.5,
                         low_percentile=None, slope_percentile=None, slope_window_seconds=None, path_b_persistence_seconds=None,
                         threshold_low_v=None, threshold_slope_v_per_s=None)
        ref_decision = apply_dual(validation, reference, config.gap_threshold_seconds)
        reference_health = health_fields(ref_decision, config)
        save(prefix + "/healthy_validation_scores.csv.gz", _gzip_rows(ref_decision, health_columns))
        contexts.extend(baseline_context(ref_decision, fold))
        save_json(prefix + "/residual_thresholds.json", {format(p, "g"): t for p, t in thresholds.items()})
        save_json(prefix + "/slope_thresholds.json", list(slopes.values()))
        path_as = []
        for p in dc["high_percentiles"]:
            policy = dict(reference, high_percentile=p, threshold_high_v=thresholds[p]["threshold_v"])
            path_as.append(dict(high_percentile=p, source_role="validation", validation_flight=split.validation[0],
                                **health_fields(apply_dual(validation, policy, config.gap_threshold_seconds), config)))
        chosen_a = choose_path_a(path_as, reference_health, split.validation[0])
        save_json(prefix + "/path_a_selection.json", dict(candidates=path_as, selected=chosen_a, reference_health=reference_health))
        grid = []
        for params in policies:
            slope = slopes[(params["slope_window_seconds"], params["slope_percentile"])]["threshold_v_per_s"]
            policy = dict(params, mode="DUAL", threshold_high_v=thresholds[params["high_percentile"]]["threshold_v"],
                          threshold_low_v=thresholds[params["low_percentile"]]["threshold_v"], threshold_slope_v_per_s=slope,
                          path_a_persistence_seconds=.5)
            health = health_fields(apply_dual(validation, policy, config.gap_threshold_seconds), config) if slope is not None else dict.fromkeys(reference_health)
            grid.append(dict(policy, fold_id=fold, source_role="validation", validation_flight=split.validation[0], **health))
        short = shortlist(grid, reference_health, chosen_a["high_percentile"] if chosen_a else None, split.validation[0], dc["refinement_max_candidates"])
        short = [dict(r) for r in short]  # keep healthy grid schema independent from refinement metrics
        local = {r["candidate_id"]: [] for r in short}
        # Generate diagnostic validation copies for all planned cases, even if no shortlist.
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(by_flight[split.validation[0]], config, family=family, severity=severity, start_position=position)
                    meta, synthetic, path = generated.metadata, [], None
                    if meta["fault_applicable"]:
                        synthetic = diagnose(label_scores(score_candidate(bundle, generated.rows, config, "validation", healthy_cache=base), "validation", True), windows, config.gap_threshold_seconds)
                        path = prefix + f"/validation_scenario_scores/{meta['scenario_id']}.csv.gz"
                        save(path, _gzip_rows(apply_dual(synthetic, reference, config.gap_threshold_seconds), fault_columns))
                    validation_manifest.append(dict(meta, fold_id=fold, source_role="validation", validation_flight=split.validation[0],
                                                    scores_csv=path, scores_sha256=hashes[path] if path else None))
                    for policy in short:
                        decision = apply_dual(synthetic, policy, config.gap_threshold_seconds) if synthetic else []
                        metrics = scenario_metrics(decision, meta, deadlines, config.gap_threshold_seconds)
                        record = dict(meta, fold_id=fold, candidate_id=policy["candidate_id"], source_role="validation", validation_flight=split.validation[0],
                                      used_for_policy_selection=True, used_for_threshold=False, scores_csv=path, scores_sha256=hashes[path] if path else None)
                        record.update(metrics, **attribution_metrics(decision, metrics))
                        local[policy["candidate_id"]].append(record)
                        validation_cases.append(record)
        for policy in short:
            metrics = selection_metrics(local[policy["candidate_id"]], deadlines)
            policy.update(metrics, synthetic_refined=metrics["applicable_scenarios"] > 0)
            next(r for r in grid if r["candidate_id"] == policy["candidate_id"])["synthetic_refined"] = policy["synthetic_refined"]
            refined.append(dict(policy))
        chosen = choose_refined([r for r in short if r["synthetic_refined"]], split.validation[0])
        selected_policy = dict(chosen) if chosen else dict(reference, mode="SINGLE_FALLBACK")
        reason = "HEALTHY_SHORTLIST_AND_SYNTHETIC_VALIDATION_RANK" if chosen else "NO_SAFE_SYNTHETIC_REFINED_DUAL_REFERENCE_RETAINED"
        all_grid.extend(grid)
        entry = dict(fold_id=fold, split=split.as_dict(), reference=reference, selected_policy=selected_policy, selection_reason=reason,
                     selected_path_a_high_percentile=chosen_a["high_percentile"] if chosen_a else None,
                     shortlisted_ids=[r["candidate_id"] for r in short], selected_before_outer=True)
        selections.append(entry)
        threshold = dict(thresholds[selected_policy["high_percentile"]], threshold=selected_policy["threshold_high_v"], persistence_seconds=.5)
        bundle.update(threshold=threshold, dual_policy=selected_policy, experiment_stage=9, fold_id=fold, feature_registry=registry,
                      feature_manifest_sha256=upstream, data_sha256=feature_summary["data_sha256"], code_sha256=code)
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer, compress=3)
        save(prefix + "/model_bundle.joblib", buffer.getvalue())
        save_json(prefix + "/training_provenance.json", bundle["training_provenance"])
        save_json(prefix + "/selected_policy.json", entry)
        if i == 0:
            validation_example = (apply_dual(validation, selected_policy, config.gap_threshold_seconds), selected_policy, reference)
        del bundle, base, validation, ref_decision, synthetic, generated, buffer
    save("validation_healthy_grid.csv", _csv(all_grid))
    refined_columns = list(all_grid[0])  # explicit schema for an empty shortlist
    save("validation_refined_candidates.csv", _csv(refined, list(refined[0]) if refined else refined_columns))
    validation_columns = list(validation_cases[0]) if validation_cases else ["fold_id", "candidate_id", "scenario_id", "source_role", "evaluation_status"]
    save("validation_fault_per_candidate.csv", _csv(validation_cases, validation_columns))
    save("validation_fault_manifest.csv", _csv(validation_manifest))
    context_columns = ["fold_id", "validation_flight", "source_role", "episode", "start_time_us", "end_time_us", "confirmed_samples", "current_min_a", "current_max_a", "max_abs_dI_dt_a_per_s", "residual_min_v", "residual_max_v"]
    save("baseline_validation_episode_context.csv", _csv(contexts, context_columns))
    save_json("selected_per_fold.json", selections)
    freeze = dict(hashes)
    save_json("all_policies_frozen.json", dict(status="FROZEN_BEFORE_ANY_OUTER_SCORE", folds=len(splits), files=freeze))
    regression, healthy, anomaly, scenarios = [], [], [], []
    example = None
    for entry in selections:
        fold, split = entry["fold_id"], entry["split"]
        prefix = "folds/" + fold
        bundle = load_candidate_bundle(output / prefix / "model_bundle.joblib")
        test = by_flight[split["test"][0]]
        base = label_scores(score_candidate(bundle, test, config, "test"), "test")
        diagnostics = diagnose(base, windows, config.gap_threshold_seconds)
        for policy, payload in (("reference", entry["reference"]), ("selected", entry["selected_policy"])):
            decision = apply_dual(diagnostics, payload, config.gap_threshold_seconds)
            save(prefix + f"/{policy}/healthy_test_scores.csv.gz", _gzip_rows(decision, health_columns))
            tags = dict(policy=policy, fold_id=fold, test_flight=split["test"][0], unseen_by_model=True, used_for_policy_selection=False)
            ready = [r for r in base if r["reference_ready"]]
            y, e = np.array([r["voltage_sag_actual"] for r in ready]), np.array([r["residual_signed_v"] for r in ready])
            regression.append(dict(**tags, **regression_metrics(base), sum_absolute_error_v=float(np.sum(np.abs(e))),
                                   sum_squared_error_v2=float(np.sum(e ** 2)), sum_target_v=float(np.sum(y)), sum_squared_target_v2=float(np.sum(y ** 2))))
            healthy.append(dict(**tags, **alarm_metrics(decision, config.gap_threshold_seconds), **attribution_metrics(decision)))
        local = {"reference": [], "selected": []}
        for family in FAMILIES:
            for severity in SEVERITIES:
                for position in cfg["fault_injection"]["start_positions"]:
                    generated = inject_fault(test, config, family=family, severity=severity, start_position=position)
                    meta = generated.metadata
                    synthetic = diagnose(label_scores(score_candidate(bundle, generated.rows, config, "test", healthy_cache=base), "test", True), windows, config.gap_threshold_seconds) if meta["fault_applicable"] else []
                    for policy, payload in (("reference", entry["reference"]), ("selected", entry["selected_policy"])):
                        evaluated, path = [], None
                        if synthetic:
                            evaluated = apply_dual(synthetic, payload, config.gap_threshold_seconds)
                            path = prefix + f"/{policy}/scenario_scores/{meta['scenario_id']}.csv.gz"
                            save(path, _gzip_rows(evaluated, fault_columns))
                        metrics = scenario_metrics(evaluated, meta, deadlines, config.gap_threshold_seconds)
                        record = dict(meta, policy=policy, fold_id=fold, test_flight=split["test"][0], validation_flight=split["validation"][0],
                                      source_role="test", used_for_policy_selection=False, used_for_threshold=False, unseen_by_model=True,
                                      mode=payload["mode"], threshold_v=payload["threshold_high_v"], scores_csv=path, scores_sha256=hashes[path] if path else None)
                        record.update(metrics, **attribution_metrics(evaluated, metrics))
                        local[policy].append(record)
                        scenarios.append(record)
                        if fold == "fold_00" and policy == "selected" and family == "progressive" and severity == "severe" and position == sorted(cfg["fault_injection"]["start_positions"])[1] and evaluated:
                            example = (evaluated, payload, entry["reference"])
        for policy, records in local.items():
            anomaly.append(dict(policy=policy, fold_id=fold, test_flight=split["test"][0], **selection_metrics(records, deadlines)))
        del bundle, base, diagnostics, synthetic, evaluated, generated
    results, summary_rows = {}, []
    for policy in ("reference", "selected"):
        subsets = [[r for r in data if r["policy"] == policy] for data in (regression, healthy, anomaly, scenarios)]
        result, summary = _summaries(*subsets, deadlines)
        results[policy] = result
        summary_rows.extend(dict(policy=policy, **r) for r in summary)
    dual_folds = sum(e["selected_policy"]["mode"] == "DUAL" for e in selections)
    promotion = promotion_decision(results, len(splits), dual_folds)
    paired = []
    for entry in selections:
        fold = entry["fold_id"]
        h = {p: next(r for r in healthy if r["fold_id"] == fold and r["policy"] == p) for p in ("reference", "selected")}
        a = {p: next(r for r in anomaly if r["fold_id"] == fold and r["policy"] == p) for p in ("reference", "selected")}
        row = dict(fold_id=fold, test_flight=entry["split"]["test"][0], mode=entry["selected_policy"]["mode"],
                   healthy_fpr_delta=h["selected"]["confirmed_fpr"] - h["reference"]["confirmed_fpr"],
                   healthy_episodes_per_minute_delta=h["selected"]["confirmed_episodes_per_minute"] - h["reference"]["confirmed_episodes_per_minute"])
        for key in ("f1", "incident_recall", "delay_mean_seconds", "delay_median_seconds", *[f"deadline_le_{int(d)}s_rate" for d in deadlines]):
            row[key + "_delta"] = a["selected"][key] - a["reference"][key] if a["selected"][key] is not None and a["reference"][key] is not None else None
        paired.append(row)
    summary = dict(status="OK", command="tune-dual-path", run_id=run_id, output_directory=str(output), evaluation_scope=protocol["evaluation_scope"],
                   folds=len(splits), candidate_pairs_evaluated=len(all_grid), shortlisted_refined_candidates=len(refined),
                   validation_candidate_scenario_evaluations=len(validation_cases), outer_scenario_evaluations=len(scenarios),
                   dual_selected_folds=dual_folds, single_fallback_folds=len(splits) - dual_folds,
                   active_flights=eligible, feature_run_id=feature_summary["run_id"], feature_manifest_sha256=upstream,
                   data_sha256=feature_summary["data_sha256"], all_policies_frozen_before_outer=True, outer_used_for_selection=False,
                   reference_pooled=results["reference"]["pooled"], selected_pooled=results["selected"]["pooled"], promotion_status=promotion["status"],
                   operational_config_changed=False, operational_dual_path_enabled=False, low_load_evaluated=False,
                   currtot_unit="unverified", deployment_status="RESEARCH_DUAL_PATH_ONLY")
    full = dict(run_summary=summary, policies=results, promotion=promotion)
    for name, data in (("regression_per_fold.csv", regression), ("healthy_active_per_fold.csv", healthy), ("anomaly_per_fold.csv", anomaly),
                       ("fault_per_scenario.csv", scenarios), ("paired_per_fold.csv", paired), ("summary.csv", summary_rows)):
        save(name, _csv(data))
    save("failed_cases.csv", _csv([r for r in scenarios if not r["incident_detected"]], list(scenarios[0])))
    save_json("promotion_decision.json", promotion)
    save("promotion_decision.md", promotion_report(promotion).encode())
    save_json("results.json", full)
    save("research_report.md", dual_report(full, selections, contexts).encode())
    for name, payload in dual_plots(healthy, selections, validation_example, example).items():
        save(name, payload)
    if any(hashlib.sha256((output / name).read_bytes()).hexdigest() != digest for name, digest in freeze.items()):
        raise DataError("Dual selection inputs/models/policies changed after outer evaluation.")
    if hashlib.sha256(config.data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise DataError("Raw data changed during dual-path evaluation.")
    _write_files(output, {"manifest.json": (format_json(dict(run_id=run_id, identity=identity, outputs=dict(sorted(hashes.items())))) + "\n").encode()}, config)
    return summary
