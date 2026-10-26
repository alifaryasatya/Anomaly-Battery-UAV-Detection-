"""Research-final orchestration with immutable source reuse and factual reports.

"Final" describes a consolidated research package, not a new final holdout,
operational detector promotion, replay verification or completed Sprint 13.
"""

from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path

import yaml

from .config import AppConfig, project_path
from .errors import DataError
from .evaluation import aggregate_scenarios, deadline_name
from .models import _write_files
from .preprocessing import _csv
from .research_sources import (MODEL_STAGES, STAGES, audit_inputs, discover, encoded, load,
                               resolve_all, sha, verify_outputs, verify_source)


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def same_metrics(actual, expected, label):
    for key, value in expected.items():
        other = actual.get(key)
        if value is None or type(value) in (str, bool):
            equal = other == value
        else:
            equal = type(other) in (int, float) and math.isclose(other, value, rel_tol=1e-10, abs_tol=1e-10)
        if not equal:
            raise DataError(f"Consolidated/source metrics disagree: {label}/{key}")


def validate_chain(resolved):
    feature = resolved["features"]["summary"]
    feature_sha = resolved["features"]["reference"]["manifest_sha256"]
    if feature["preprocessing_manifest_sha256"] != resolved["preprocess"]["reference"]["manifest_sha256"]:
        raise DataError("Research preprocessing/features lineage mismatch.")
    active = resolved["cross-validate"]["summary"]["active_flights"]
    for command, item in resolved.items():
        summary = item["summary"]
        if command not in ("preprocess", "features") and summary.get("feature_manifest_sha256") != feature_sha:
            raise DataError(f"Research feature lineage mismatch: {command}")
        if summary.get("feature_run_id", feature["run_id"]) != feature["run_id"]:
            raise DataError(f"Research feature run mismatch: {command}")
        if "active_flights" in summary and summary["active_flights"] != active:
            raise DataError(f"Research active flight assignment mismatch: {command}")
    baseline = resolved["cross-validate"]["content"]["pooled"]
    for command in ("tune", "tune-dual-path"):
        for scope, metrics in baseline.items():
            same_metrics(resolved[command]["summary"]["reference_pooled"][scope], metrics, command + "/reference/" + scope)
    for command in ("ablation", "compare-models"):
        item = resolved[command]["content"]["candidates"]["hgb_battery_d"]["pooled"]
        for scope, metrics in baseline.items():
            same_metrics(item[scope], metrics, command + "/hgb_d/" + scope)
    low = resolved["evaluate-low-load"]["summary"]
    if low["baseline_run_id"] != resolved["cross-validate"]["summary"]["run_id"]:
        raise DataError("Low-load source does not match consolidated frozen baseline.")
    if (not low["all_low_load_unseen_by_model"] or not low["all_low_load_unseen_by_detector"]
            or low["threshold_changed"] or low["policy_selection_performed"]):
        raise DataError("Low-load results violate unseen/frozen-policy scope.")
    promotion = resolved["tune-dual-path"]["content"]["promotion"]
    if promotion["status"] != resolved["tune-dual-path"]["summary"]["promotion_status"]:
        raise DataError("Dual-path promotion summary/decision mismatch.")
    if promotion["gate_passed"] != all(c["passed"] for c in promotion["checks"].values()):
        raise DataError("Dual-path promotion does not agree with its frozen gate checks.")
    return promotion


def typed_scenario(raw, deadlines):
    result = dict(raw)
    bool_names = ("fault_applicable", "incident_detected", "preexisting_alarm", *[deadline_name(d) for d in deadlines])
    number_names = ("tp", "fp", "tn", "fn", "delay_seconds", "pre_fault_samples", "pre_fault_confirmed_alarm_samples")
    for name in bool_names:
        value = raw[name]
        if value not in ("True", "False", ""):
            raise DataError(f"Invalid scenario boolean: {name}")
        result[name] = None if value == "" else value == "True"
    for name in number_names:
        try:
            result[name] = float(raw[name]) if raw[name] != "" else None
        except (TypeError, ValueError) as exc:
            raise DataError(f"Invalid scenario numeric value: {name}") from exc
    if type(result["fault_applicable"]) is not bool:
        raise DataError("Scenario applicability must be explicit.")
    if result["fault_applicable"]:
        for name in ("incident_detected", "preexisting_alarm", *[deadline_name(d) for d in deadlines]):
            if type(result[name]) is not bool:
                raise DataError("Applicable scenario lacks defined incident/deadline values.")
        for name in number_names:
            value = result[name]
            if name != "delay_seconds" and (value is None or not math.isfinite(value) or value < 0 or not value.is_integer()):
                raise DataError("Applicable scenario lacks valid confusion/exposure counts.")
        if result["incident_detected"] and (result["delay_seconds"] is None or not math.isfinite(result["delay_seconds"]) or result["delay_seconds"] < 0):
            raise DataError("Detected scenario lacks a valid delay.")
    return result


def consolidate_groups(resolved, deadlines):
    grouped, all_series, failures = [], [], []
    dimensions = (("overall", None), ("flight", "NamaFile"), ("family", "fault_family"), ("severity", "severity"),
                  ("position", "start_position"), ("observability", "observability"))
    for command in ("cross-validate", "ablation", "compare-models", "tune", "tune-dual-path"):
        item = resolved[command]
        path = Path(item["summary"]["output_directory"]) / "fault_per_scenario.csv"
        series = defaultdict(list)
        for raw in table(path):
            series[raw.get("candidate", raw.get("policy", "reference"))].append(typed_scenario(raw, deadlines))
        expected_series = ({"reference"} if command == "cross-validate" else
                           set(item["content"]["candidates"]) if command in ("ablation", "compare-models") else {"reference", "selected"})
        if set(series) != expected_series:
            raise DataError(f"Scenario series missing or unexpected: {command}")
        for name, records in sorted(series.items()):
            identities = [(r["fold_id"], r["scenario_id"]) for r in records]
            if len(set(identities)) != len(identities):
                raise DataError(f"Duplicate scenario within experimental series: {command}/{name}")
            total = aggregate_scenarios(records, deadlines)
            expected = (item["content"]["pooled"]["synthetic"] if command == "cross-validate" else
                        item["content"]["candidates"][name]["pooled"]["synthetic"] if command in ("ablation", "compare-models") else
                        item["content"]["policies"][name]["pooled"]["synthetic"])
            same_metrics(total, expected, command + "/" + name)
            all_series.append(dict(experiment=command, series=name, **total))
            for dimension, key in dimensions:
                groups = defaultdict(list)
                for record in records:
                    groups[record[key] if key else "ALL"].append(record)
                for value, part in sorted(groups.items()):
                    grouped.append(dict(experiment=command, series=name, dimension=dimension, group=value,
                                        **aggregate_scenarios(part, deadlines)))
            for r in records:
                if not r["fault_applicable"] or not r["incident_detected"]:
                    failures.append(dict(experiment=command, series=name, scenario_id=r["scenario_id"], fold_id=r["fold_id"],
                                         NamaFile=r["NamaFile"], fault_family=r["fault_family"], severity=r["severity"],
                                         start_position=r["start_position"], observability=r["observability"],
                                         fault_applicable=r["fault_applicable"], evaluation_status=r["evaluation_status"],
                                         incident_detected=r["incident_detected"], preexisting_alarm=r["preexisting_alarm"],
                                         delay_seconds=r["delay_seconds"], reason=r["reason"], detection_reason=r["detection_reason"]))
    return grouped, all_series, failures


def model_registry(resolved, config):
    records = []
    for command, item in resolved.items():
        protocol = item["manifest"].get("identity", {}).get("protocol", {})
        candidates = {c["candidate"]: c for c in protocol.get("candidates", [])}
        for name, digest in item["manifest"]["outputs"].items():
            if not name.endswith("model_bundle.joblib"):
                continue
            parts = Path(name).parts
            candidate = parts[1] if parts[0] == "candidates" else "hgb_battery_d"
            predictors = candidates.get(candidate, {}).get("predictors", config.as_dict()["features"]["regression_predictors"])
            records.append(dict(experiment=command, candidate=candidate, fold_id=parts[-2] if len(parts) > 1 else "pilot",
                                bundle_path=str((Path(item["summary"]["output_directory"]) / name).relative_to(config.root)),
                                sha256=digest, predictors="|".join(predictors),
                                source_manifest_sha256=item["reference"]["manifest_sha256"], copied_or_refitted=False))
    return records


def run_research(config: AppConfig):
    from .research_reporting import make_plots, research_report
    audit = audit_inputs(config)
    cfg = config.as_dict()
    if cfg["dual_path"]["enabled"] or cfg["features"]["operational_set"] != "battery_d":
        raise DataError("Research orchestration preserves the Battery-D single-path operational baseline; dual-path remains experimental.")
    base = project_path(config.artifacts_path / "research_evaluation", config.root)
    config_key = hashlib.sha256(encoded(dict(config=cfg, data_sha256=audit["inspection"]["data_sha256"],
                                             freeze_sha256=audit["freeze_manifest_sha256"]))).hexdigest()[:16]
    catalog_path = base / "sources" / (config_key + ".json")
    choices = discover(config, catalog_path)
    code_root = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(code_root)).replace("\\", "/"): sha(p) for p in
            [code_root / "main.py", *sorted((code_root / "src").glob("*.py"))]}
    plan = dict(experiment="sprint11_research_final_orchestration", evaluation_scope="historical_development_research_consolidation",
                config=cfg, data_sha256=audit["inspection"]["data_sha256"], freeze_manifest_sha256=audit["freeze_manifest_sha256"],
                experiment_protocol_sha256=audit["experiment_protocol_sha256"], dependency_versions=audit["current_dependency_versions"],
                stage_order=[s[0] for s in STAGES], source_rule="catalog, else explicit historical pointer, else build missing; corrupt results fail without rebuilding",
                selection_rule="fixed engine/config candidate budgets; validation-only selection, ALL candidates/folds frozen before each experiment's outer evaluation",
                cross_experiment_rule="sequential historical development experiments, not a global prospective holdout; no outer-selected winner or policy revision",
                low_load_rule="same unique cohort under frozen baseline folds; no repeated exposure pooling", operational_config_changed=False,
                independent_final_holdout=False, replay_evaluated=False, code_sha256=code)
    plan_id = hashlib.sha256(encoded(plan)).hexdigest()[:16]
    plan_dir = base / "plans" / plan_id
    if not (plan_dir / "orchestration_plan.json").exists():
        _write_files(plan_dir, {"orchestration_plan.json": encoded(plan)}, config)
    elif load(plan_dir / "orchestration_plan.json") != plan:
        raise DataError("Frozen orchestration plan differs.")
    resolved, trace = resolve_all(config, audit, choices)
    promotion = validate_chain(resolved)
    references = [resolved[s[0]]["reference"] for s in STAGES]
    identity = dict(plan=plan, sources=references)
    run_id = hashlib.sha256(encoded(identity)).hexdigest()[:16]
    output = project_path(base / run_id, config.root)
    if (output / "manifest.json").exists():
        existing = verify_outputs(output, config)
        if existing["identity"] != identity:
            raise DataError("Research completion cache identity mismatch.")
        if not catalog_path.exists():
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            _write_files(catalog_path.parent, {catalog_path.name: encoded({s[0]: resolved[s[0]]["summary"] for s in STAGES})}, config)
        return load(output / "summary.json")
    selection_freeze = {r["command"]: dict(run_id=r["run_id"], manifest_sha256=r["manifest_sha256"]) for r in references}
    initial = {"orchestration_plan.json": encoded(plan), "source_inventory.json": encoded(references),
               "source_inventory.csv": _csv(references), "source_results_frozen.json": encoded(selection_freeze),
               "execution_trace.csv": _csv(trace), "input_audit.json": encoded(audit),
               "config_used.yaml": yaml.safe_dump(cfg, sort_keys=False).encode(),
               "dependency_versions.json": encoded(audit["current_dependency_versions"])}
    _write_files(output, initial, config)
    # Reports consume only this fully verified, fixed set of source results.
    deadlines = cfg["evaluation"]["deadlines_seconds"]
    grouped, series, failures = consolidate_groups(resolved, deadlines)
    registry = model_registry(resolved, config)
    baseline = resolved["cross-validate"]["summary"]
    low = resolved["evaluate-low-load"]["summary"]
    result = dict(evaluation_scope=plan["evaluation_scope"], data_audit=audit["inspection"],
                  preprocessing=resolved["preprocess"]["summary"], features=resolved["features"]["summary"],
                  fault_bank=resolved["inject-faults"]["summary"], baseline=baseline,
                  baseline_macro_per_fold=resolved["cross-validate"]["content"]["macro_per_fold"],
                  ablation=resolved["ablation"]["summary"], model_comparison=resolved["compare-models"]["summary"],
                  tuning=resolved["tune"]["summary"], dual_path=resolved["tune-dual-path"]["summary"],
                  promotion_decision=promotion, low_load=low, operational_baseline="frozen HGB Battery-D P99 / 0.5 s single-path",
                  selected_new_winner=None, independent_final_holdout=False, replay_evaluated=False,
                  synthetic_series=series, source_runs=references)
    low_models = table(Path(low["output_directory"]) / "per_model_summary.csv")
    result["low_load_per_model"] = [{k: v if k == "fold_id" else float(v) if v != "" else None for k, v in r.items()} for r in low_models]
    checks = [dict(check="audit_and_source_integrity", status="PASS", detail="Frozen raw/config/protocol and all source output hashes verified"),
              dict(check="feature_and_reference_lineage", status="PASS", detail="Same preprocessing/features, active flight assignments and reference baseline"),
              dict(check="validation_before_outer", status="PASS", detail="All five outer experiments have verified immutable pre-outer freeze records"),
              dict(check="recomputed_scenario_totals", status="PASS", detail=f"{len(series)} experimental series reconcile to their source summaries"),
              dict(check="low_load_unseen_and_exposure", status="PASS", detail="Explicit source membership; unique cohort exposure retained separately per model"),
              dict(check="dual_promotion", status="PASS" if promotion["gate_passed"] else "FAIL", detail=promotion["status"]),
              dict(check="independent_final_holdout", status="WARNING", detail="Historical development data; no untouched final holdout"),
              dict(check="currtot_units_and_healthy_labels", status="WARNING", detail="CurrTot unverified; healthy assumed_from_source"),
              dict(check="replay_and_hardware", status="NOT_EVALUATED", detail="Sprint 12 replay and hardware readiness not established")]
    for name, check in baseline["research_target_checks"].items():
        checks.append(dict(check="baseline_target_" + name, status="PASS" if check["status"] == "MET" else "FAIL",
                           detail=f"rate={check['rate']}; target={check['target']}; applicable={check['applicable']}"))
    summary = dict(status="OK", command="research-evaluation", run_id=run_id, output_directory=str(output),
                   evaluation_scope=plan["evaluation_scope"], data_sha256=plan["data_sha256"], source_stages=len(references),
                   verified_source_files=sum(r["verified_files"] for r in references), model_bundle_references=len(registry),
                   source_stages_built_at_creation=[r["command"] for r in trace if r["action"] == "BUILT_MISSING_STAGE"],
                   training_stages_built_at_creation=[r["command"] for r in trace if r["training_stage_built"]],
                   synthetic_series=len(series), grouped_metric_rows=len(grouped), failed_or_not_applicable_cases=len(failures),
                   baseline_healthy_active=baseline["healthy_active_pooled"], baseline_synthetic=baseline["synthetic_pooled"],
                   baseline_target_checks=baseline["research_target_checks"], low_load_primary=low["primary_low_load"],
                   low_load_model_sensitivity=low["sensitivity_across_frozen_models"], dual_promotion=promotion["status"],
                   operational_config_changed=False, operational_dual_path_enabled=False, selected_new_winner=None,
                   independent_final_holdout=False, replay_evaluated=False, deployment_status="RESEARCH_CONSOLIDATION_ONLY")
    payloads = {"results.json": encoded(result), "summary.json": encoded(summary), "gate_checks.csv": _csv(checks),
                "synthetic_series.csv": _csv(series), "synthetic_group_metrics.csv": _csv(grouped),
                "failed_or_not_applicable_cases.csv": _csv(failures, ("experiment", "series", "scenario_id", "fold_id", "NamaFile", "fault_family", "severity", "start_position", "observability", "fault_applicable", "evaluation_status", "incident_detected", "preexisting_alarm", "delay_seconds", "reason", "detection_reason")),
                "model_bundle_registry.csv": _csv(registry),
                "feature_registry.json": (Path(resolved["features"]["summary"]["output_directory"]) / "feature_registry.json").read_bytes()}
    for command, source_name, target_name in (
        ("preprocess", "flight_inventory.csv", "dataset/flight_inventory.csv"),
        ("preprocess", "time_gap_summary.csv", "dataset/time_gap_summary.csv"),
        ("preprocess", "battery_activity.csv", "dataset/battery_activity.csv"),
        ("features", "reference_coverage.csv", "dataset/reference_coverage.csv"),
        ("inject-faults", "fault_applicability.csv", "synthetic/fault_applicability.csv"),
        ("inject-faults", "fault_observability.csv", "synthetic/fault_observability.csv"),
        ("cross-validate", "healthy_active_per_fold.csv", "baseline/healthy_active_per_fold.csv"),
        ("cross-validate", "regression_per_fold.csv", "baseline/regression_per_fold.csv"),
        ("evaluate-low-load", "per_model_summary.csv", "low_load/per_model_summary.csv"),
        ("evaluate-low-load", "per_model_flight.csv", "low_load/per_model_flight.csv"),
        ("tune", "selected_per_fold.json", "selection/single_path_selected_per_fold.json"),
        ("tune-dual-path", "selected_per_fold.json", "selection/dual_path_selected_per_fold.json"),
        ("tune-dual-path", "promotion_decision.json", "selection/promotion_decision.json")):
        source_path = Path(resolved[command]["summary"]["output_directory"]) / source_name
        payloads[target_name] = source_path.read_bytes()
    for command, target in (("ablation", "ablation_overview.csv"), ("compare-models", "model_comparison_overview.csv")):
        payloads[target] = _csv([dict(candidate=name, **values) for name, values in resolved[command]["summary"]["overview"].items()])
    payloads.update(make_plots(result, grouped))
    payloads["research_report.md"] = research_report(summary, result, checks).encode()
    # Detect a changing source/data/config during aggregation, before completion.
    if audit_inputs(config) != audit:
        raise DataError("Research inputs or dependency versions changed during orchestration.")
    for name, digest in code.items():
        if sha(code_root / name) != digest:
            raise DataError("Research engine code changed after the orchestration plan was frozen.")
    for stage in STAGES:
        checked = verify_source(stage, resolved[stage[0]]["summary"], config, audit)
        if checked["reference"] != resolved[stage[0]]["reference"]:
            raise DataError("Research source identity changed during reporting.")
    for name, data in initial.items():
        if (output / name).read_bytes() != data:
            raise DataError("Research source-results freeze changed during reporting.")
    for name in payloads:
        (output / name).parent.mkdir(parents=True, exist_ok=True)
    _write_files(output, payloads, config)
    outputs = {n: sha(output / n) for n in sorted([*initial, *payloads])}
    _write_files(output, {"manifest.json": encoded(dict(run_id=run_id, identity=identity, outputs=outputs))}, config)
    # A new catalog indexes completed sources only; historical sprint pointers
    # are never overwritten, even when the orchestrator built missing stages.
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_files(catalog_path.parent, {catalog_path.name: encoded({s[0]: resolved[s[0]]["summary"] for s in STAGES})}, config)
    return summary
