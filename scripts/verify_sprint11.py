"""Independent stdlib audit of the consolidated research package.

No project functions, model fitting, calibration or candidate selection.
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
DEADLINES = (1, 2, 5, 10, 30)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(value, expected):
    if expected is None:
        assert value in (None, ""), (value, expected)
    else:
        assert value not in (None, "") and math.isclose(float(value), expected, rel_tol=1e-10, abs_tol=1e-10), (value, expected)


def rate(n, d):
    return n / d if d else None


def calculate(records):
    app = [r for r in records if r["fault_applicable"] == "True"]
    detected = [r for r in app if r["incident_detected"] == "True"]
    counts = {k: sum(int(float(r[k])) for r in app) for k in ("tp", "fp", "tn", "fn")}
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    delay = [float(r["delay_seconds"]) for r in detected]
    contaminated = sum(r["preexisting_alarm"] == "True" for r in app)
    pre = sum(int(float(r["pre_fault_samples"])) for r in app)
    pre_alarm = sum(int(float(r["pre_fault_confirmed_alarm_samples"])) for r in app)
    result = dict(scenarios=len(records), applicable_scenarios=len(app), not_applicable_scenarios=len(records) - len(app),
                  detected_scenarios=len(detected), undetected_scenarios=len(app) - len(detected),
                  preexisting_alarm_scenarios=contaminated, undetected_without_preexisting=len(app) - len(detected) - contaminated,
                  incident_recall=rate(len(detected), len(app)), **counts, precision=rate(tp, tp + fp), recall=rate(tp, tp + fn),
                  f1=rate(2 * tp, 2 * tp + fp + fn), synthetic_sample_fpr=rate(fp, fp + tn),
                  delay_detected_n=len(delay), delay_mean_seconds=mean(delay) if delay else None,
                  delay_median_seconds=median(delay) if delay else None, delay_std_seconds=pstdev(delay) if delay else None,
                  pre_fault_samples=pre, pre_fault_confirmed_alarm_samples=pre_alarm, pre_fault_fpr=rate(pre_alarm, pre))
    for d in DEADLINES:
        hits = sum(r[f"deadline_le_{d}s"] == "True" for r in app)
        result[f"deadline_le_{d}s_successes"] = hits
        result[f"deadline_le_{d}s_rate"] = rate(hits, len(app))
    return result


def verify():
    summary = load(ROOT / "artifacts/sprint11/research_result.json")
    output = Path(summary["output_directory"])
    manifest = load(output / "manifest.json")
    for name, digest in manifest["outputs"].items():
        assert sha(output / name) == digest, name
    assert summary == load(output / "summary.json") and manifest["run_id"] == summary["run_id"]
    results = load(output / "results.json")
    inventory = load(output / "source_inventory.json")
    assert inventory == manifest["identity"]["sources"] == results["source_runs"]
    assert len(inventory) == summary["source_stages"] == 10
    lookup = {r["command"]: Path(r["output_directory"]) for r in inventory}
    manifests = {}
    verified_source_files = 0
    for source in inventory:
        path = Path(source["output_directory"])
        assert sha(path / "manifest.json") == source["manifest_sha256"]
        m = load(path / "manifest.json")
        manifests[source["command"]] = m
        for name, digest in m["outputs"].items():
            target = ROOT / name if source["command"] == "inject-faults" else path / name
            assert sha(target) == digest, target
        verified_source_files += len(m["outputs"])
        assert len(m["outputs"]) == source["verified_files"]
        if source["pre_outer_frozen_files"]:
            name = "all_fold_policies_frozen.json" if source["command"] == "cross-validate" else "all_policies_frozen.json"
            freeze = load(path / name)
            assert len(freeze["files"]) == source["pre_outer_frozen_files"]
            for name, digest in freeze["files"].items():
                assert m["outputs"][name] == digest
    assert verified_source_files == summary["verified_source_files"]
    features = load(lookup["features"] / "summary.json")
    assert features["preprocessing_manifest_sha256"] == sha(lookup["preprocess"] / "manifest.json")
    assert Path(features["preprocessing_directory"]) == lookup["preprocess"]
    assert sha(ROOT / "data/raw/dataset_baterai.csv") == summary["data_sha256"]
    for name, digest in load(ROOT / "artifacts/audit/freeze_manifest.json")["files"].items():
        assert sha(ROOT / name) == digest
    source_groups = {}
    source_case_count = 0
    failures = set()
    for command in ("cross-validate", "ablation", "compare-models", "tune", "tune-dual-path"):
        rows = table(lookup[command] / "fault_per_scenario.csv")
        source_case_count += len(rows)
        for r in rows:
            series = r.get("candidate", r.get("policy", "reference"))
            key = command, series
            source_groups.setdefault(key, []).append(r)
            if r["fault_applicable"] != "True" or r["incident_detected"] != "True":
                failures.add((command, series, r["fold_id"], r["scenario_id"]))
    groups = table(output / "synthetic_group_metrics.csv")
    dimensions = dict(flight="NamaFile", family="fault_family", severity="severity", position="start_position", observability="observability")
    expected_group_keys = set()
    for (command, series), rows in source_groups.items():
        expected_group_keys.add((command, series, "overall", "ALL"))
        for dimension, name in dimensions.items():
            expected_group_keys.update((command, series, dimension, r[name]) for r in rows)
    actual_keys = [(r["experiment"], r["series"], r["dimension"], r["group"]) for r in groups]
    assert len(set(actual_keys)) == len(actual_keys) and set(actual_keys) == expected_group_keys
    for record in groups:
        rows = source_groups[(record["experiment"], record["series"])]
        if record["dimension"] != "overall":
            rows = [r for r in rows if r[dimensions[record["dimension"]]] == record["group"]]
        for key, value in calculate(rows).items():
            close(record[key], value)
    for record in table(output / "synthetic_series.csv"):
        expected = calculate(source_groups[(record["experiment"], record["series"])])
        for key, value in expected.items():
            close(record[key], value)
    actual_failures = {(r["experiment"], r["series"], r["fold_id"], r["scenario_id"]) for r in table(output / "failed_or_not_applicable_cases.csv")}
    assert actual_failures == failures and len(failures) == summary["failed_or_not_applicable_cases"]
    models = table(output / "model_bundle_registry.csv")
    expected_models = {(command, name, digest) for command, m in manifests.items() for name, digest in m["outputs"].items() if name.endswith("model_bundle.joblib")}
    assert len(models) == len(expected_models) == summary["model_bundle_references"]
    for model in models:
        path = ROOT / model["bundle_path"]
        relative = str(path.relative_to(lookup[model["experiment"]])).replace("\\", "/")
        assert (model["experiment"], relative, model["sha256"]) in expected_models
        assert sha(path) == model["sha256"] and model["copied_or_refitted"] == "False"
    promotion = load(lookup["tune-dual-path"] / "promotion_decision.json")
    assert promotion == results["promotion_decision"] == load(output / "selection/promotion_decision.json")
    assert all(c["passed"] for c in promotion["checks"].values()) == promotion["gate_passed"]
    assert summary["dual_promotion"] == promotion["status"]
    checks = {r["check"]: r for r in table(output / "gate_checks.csv")}
    assert checks["dual_promotion"]["status"] == ("PASS" if promotion["gate_passed"] else "FAIL")
    assert checks["replay_and_hardware"]["status"] == "NOT_EVALUATED"
    baseline = load(lookup["cross-validate"] / "results.json")["run_summary"]
    assert summary["baseline_healthy_active"] == baseline["healthy_active_pooled"]
    assert summary["baseline_synthetic"] == baseline["synthetic_pooled"]
    assert summary["baseline_target_checks"] == baseline["research_target_checks"]
    low = load(lookup["evaluate-low-load"] / "summary.json")
    assert low["baseline_run_id"] == baseline["run_id"]
    assert summary["low_load_primary"] == low["primary_low_load"]
    assert summary["low_load_model_sensitivity"] == low["sensitivity_across_frozen_models"]
    assert (output / "low_load/per_model_summary.csv").read_bytes() == (lookup["evaluate-low-load"] / "per_model_summary.csv").read_bytes()
    assert len(source_groups) == summary["synthetic_series"]
    assert len(groups) == summary["grouped_metric_rows"]
    for name in ("operational_config_changed", "operational_dual_path_enabled", "independent_final_holdout", "replay_evaluated"):
        assert summary[name] is False
    assert summary["selected_new_winner"] is None
    text = (output / "research_report.md").read_text(encoding="utf-8")
    assert summary["dual_promotion"] in text and "CurrTot remains unverified" in text
    assert "does not diagnose the underlying internal physical failure mechanism" in text
    assert "must not be interpreted as measured internal battery failures" in text
    return dict(status="PASS", run_id=summary["run_id"], project_functions_imported=False, fitting_or_selection_performed=False,
                source_stages=len(inventory), verified_source_files=verified_source_files, source_scenario_case_records=source_case_count,
                separate_experimental_series=len(source_groups), independently_recomputed_group_rows=len(groups),
                failed_or_na_cases=len(failures), frozen_model_bundle_references=len(models),
                verified_research_outputs=len(manifest["outputs"]), promotion_and_undefined_gates_preserved=True,
                historical_development_scope_preserved=True)


if __name__ == "__main__":
    result = verify()
    (ROOT / "artifacts/sprint11/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
