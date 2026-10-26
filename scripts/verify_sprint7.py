"""Independent stdlib audit of every Sprint 7 exported decision and denominator.

No project model, detector or metrics code is imported. Run after both CLI runs.
"""

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def yes(v):
    assert v in ("True", "False"), v
    return v == "True"


def close(value, expected):
    if expected is None:
        assert value is None or value == "", value
    else:
        assert math.isclose(float(value), expected, rel_tol=1e-8, abs_tol=1e-9), (value, expected)


def rate(a, b):
    return a / b if b else None


def audit_decisions(rows, threshold, regression):
    prior = start = None
    confirmed_before = False
    exposure = 0.
    for r in rows:
        time, ready = int(r["TimeUS"]), yes(r["reference_ready"])
        continuous = prior is not None and all(r[k] == prior[k] for k in ("NamaFile", "Inst", "segment_id")) and 0 < time - int(prior["TimeUS"]) <= 500000
        if continuous and ready and yes(prior["reference_ready"]):
            exposure += (time - int(prior["TimeUS"])) / 1e6
        if not continuous:
            start, confirmed_before = None, False
        if ready:
            score = float(r["anomaly_score"])
            assert math.isfinite(score)
            if regression:
                error = float(r["voltage_sag_actual"]) - float(r["predicted_voltage_sag"])
                close(r["residual_signed_v"], error)
                close(score, max(0., error))
            else:
                assert not r["predicted_voltage_sag"] and not r["residual_signed_v"]
        else:
            assert not r["anomaly_score"] and not r["predicted_voltage_sag"]
        raw = ready and float(r["anomaly_score"]) > threshold
        if not raw:
            start = None
        elif start is None:
            start = time
        duration = (time - start) / 1e6 if start is not None else 0.
        confirmed = raw and duration >= .5
        assert yes(r["anomaly_raw"]) == raw
        close(r["anomaly_duration_seconds"], duration)
        assert yes(r["anomaly_confirmed"]) == confirmed
        assert yes(r["confirmed_episode_start"]) == (confirmed and not confirmed_before)
        assert r["detector_state"] == ("NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw else "NORMAL")
        assert r["used_for_fit"] == "False" and yes(r["unseen_by_model"])
        confirmed_before, prior = confirmed, r
    return exposure


def audit_run(summary):
    output = Path(summary["output_directory"])
    manifest = load(output / "manifest.json")
    for name, digest in manifest["outputs"].items():
        path = (output / name).resolve()
        assert path.is_relative_to(output)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    frozen = load(output / "all_policies_frozen.json")
    assert len(frozen["files"]) == 3 * summary["candidate_folds"]
    for name, digest in frozen["files"].items():
        assert manifest["outputs"][name] == digest
    protocol = load(output / "experiment_protocol.json")
    assert protocol["threshold_percentile"] == 99 and protocol["persistence_seconds"] == .5
    assert summary["selected_winner"] is None and not summary["operational_config_changed"]
    all_scenarios = table(output / "fault_per_scenario.csv")
    scenarios = defaultdict(list)
    signatures = {}
    for r in all_scenarios:
        scenarios[(r["candidate"], r["fold_id"])].append(r)
        signature = tuple(r[k] for k in ("fault_applicable", "onset_time_us", "fault_end_time_us", "support_samples", "observability", "frozen_v_ref"))
        assert signatures.setdefault(r["scenario_id"], signature) == signature
    healthy_table = table(output / "healthy_active_per_fold.csv")
    reg_table = table(output / "regression_per_fold.csv")
    results = load(output / "results.json")
    checked = 0
    for spec in protocol["candidates"]:
        name = spec["candidate"]
        regression = spec["model"] in ("hgb", "rf")
        h_samples = h_confirmed = h_raw = h_episodes = 0
        h_duration = 0.
        for split in protocol["assignments"]:
            fold = split["fold_id"]
            train, val, test = (set(split[k]) for k in ("train", "validation", "test"))
            assert len(train) == 4 and not (train & val or train & test or val & test)
            folder = output / "candidates" / name / fold
            provenance = load(folder / "training_provenance.json")
            assert set(provenance["train_flights"]) == train and provenance["synthetic_rows"] == 0
            ids = provenance["fit_sample_identities"]
            assert len(ids) == provenance["model_fit_samples"] and {r["NamaFile"] for r in ids} == train
            indices = provenance["fit_sample_indices"]
            assert indices == sorted(set(indices)) and max(indices) < provenance["ready_training_samples"]
            if spec["model"] == "ocsvm":
                assert len(ids) <= 2500 and provenance["scaler_fit_samples"] == provenance["ready_training_samples"]
            else:
                assert len(ids) == provenance["ready_training_samples"] and provenance["scaler_fit_samples"] == 0
            threshold = load(folder / "threshold.json")
            validation = table(folder / "validation_scores.csv.gz")
            assert {r["NamaFile"] for r in validation} == val and set(threshold["source_flights"]) == val
            ready_val = [r for r in validation if yes(r["reference_ready"])]
            scores = sorted(float(r["anomaly_score"]) for r in ready_val)
            rank = .99 * (len(scores) - 1)
            index = math.floor(rank)
            close(threshold["threshold"], scores[index] + (rank - index) * (scores[math.ceil(rank)] - scores[index]))
            fingerprint = [{"NamaFile": r["NamaFile"], "Inst": int(r["Inst"]), "TimeUS": int(r["TimeUS"]),
                            "source_record": int(r["source_record"]), "anomaly_score": float(r["anomaly_score"])} for r in ready_val]
            digest = hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            assert digest == threshold["source_sha256"]
            audit_decisions(validation, threshold["threshold"], regression)
            original = table(folder / "healthy_test_scores.csv.gz")
            assert {r["NamaFile"] for r in original} == test
            duration = audit_decisions(original, threshold["threshold"], regression)
            assert all(r["used_for_threshold"] == "False" and r["split_role"] == "test" for r in original)
            checked += len(original) + len(validation)
            ready = [r for r in original if yes(r["reference_ready"])]
            health = next(r for r in healthy_table if r["candidate"] == name and r["fold_id"] == fold)
            assert int(health["ready_samples"]) == len(ready)
            close(health["observed_ready_seconds"], duration)
            h_samples += len(ready)
            h_duration += duration
            for key, flag in (("raw_alarm_samples", "anomaly_raw"), ("confirmed_alarm_samples", "anomaly_confirmed"), ("confirmed_episodes", "confirmed_episode_start")):
                assert int(health[key]) == sum(yes(r[flag]) for r in ready)
            h_raw += int(health["raw_alarm_samples"])
            h_confirmed += int(health["confirmed_alarm_samples"])
            h_episodes += int(health["confirmed_episodes"])
            if regression:
                reg = next(r for r in reg_table if r["candidate"] == name and r["fold_id"] == fold)
                errors = [float(r["residual_signed_v"]) for r in ready]
                close(reg["mae_v"], mean(abs(e) for e in errors))
                close(reg["rmse_v"], math.sqrt(mean(e * e for e in errors)))
            else:
                assert not any(r["candidate"] == name for r in reg_table)
            base = {r["source_record"]: r for r in original}
            for meta in scenarios[(name, fold)]:
                if not yes(meta["fault_applicable"]):
                    assert meta["evaluation_status"] == "NOT_APPLICABLE"
                    assert all(not meta[k] for k in ("tp", "fp", "tn", "fn", "scores_csv", "delay_seconds", "incident_detected"))
                    continue
                rows = table(output / meta["scores_csv"])
                checked += len(rows)
                audit_decisions(rows, threshold["threshold"], regression)
                onset, end = int(meta["onset_time_us"]), int(meta["fault_end_time_us"])
                for r in rows:
                    b = base[r["source_record"]]
                    assert r["V_ref"] == b["V_ref"] and r["Curr"] == b["Curr"] and r["reference_ready"] == b["reference_ready"]
                    if regression:
                        assert r["predicted_voltage_sag"] == b["predicted_voltage_sag"]
                    assert yes(r["fault_support"]) == (onset <= int(r["TimeUS"]) <= end)
                    if yes(r["reference_ready"]):
                        close(r["voltage_sag_actual"], float(b["voltage_sag_actual"]) + float(r["added_sag_v"]))
                ready = [r for r in rows if yes(r["reference_ready"])]
                support = [r for r in ready if yes(r["fault_support"])]
                existing = yes(support[0]["anomaly_confirmed"]) and not yes(support[0]["confirmed_episode_start"])
                fresh = next((r for r in support if yes(r["confirmed_episode_start"])), None)
                detected = not existing and fresh is not None
                assert yes(meta["preexisting_alarm"]) == existing and yes(meta["incident_detected"]) == detected
                assert meta["evaluation_status"] == ("PREEXISTING_ALARM" if existing else "DETECTED" if detected else "UNDETECTED")
                delay = (int(fresh["TimeUS"]) - onset) / 1e6 if detected else None
                close(meta["delay_seconds"], delay)
                counts = dict(tp=0, fp=0, tn=0, fn=0)
                for r in ready:
                    label, alarm = yes(r["fault_support"]), yes(r["anomaly_confirmed"])
                    counts["tp" if label and alarm else "fn" if label else "fp" if alarm else "tn"] += 1
                for k, count in counts.items():
                    assert int(meta[k]) == count
                close(meta["precision"], rate(counts["tp"], counts["tp"] + counts["fp"]))
                close(meta["f1"], rate(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]))
                for d in protocol["deadlines_seconds"]:
                    key = "deadline_le_" + format(d, "g").replace(".", "p") + "s"
                    assert yes(meta[key]) == (detected and delay <= d)
                pre = [r for r in ready if int(r["TimeUS"]) < onset]
                assert int(meta["pre_fault_samples"]) == len(pre)
                assert int(meta["pre_fault_confirmed_alarm_samples"]) == sum(yes(r["anomaly_confirmed"]) for r in pre)
        pooled = results["candidates"][name]["pooled"]
        assert h_samples == pooled["healthy_active"]["ready_samples"] == 26870
        close(pooled["healthy_active"]["confirmed_fpr"], h_confirmed / h_samples)
        close(pooled["healthy_active"]["raw_fpr"], h_raw / h_samples)
        close(pooled["healthy_active"]["observed_ready_seconds"], h_duration)
        close(pooled["healthy_active"]["confirmed_episodes_per_minute"], h_episodes * 60 / h_duration)
        local = [r for r in all_scenarios if r["candidate"] == name and yes(r["fault_applicable"])]
        syn = pooled["synthetic"]
        detected = [r for r in local if yes(r["incident_detected"])]
        assert syn["applicable_scenarios"] == len(local) and syn["detected_scenarios"] == len(detected)
        assert syn["preexisting_alarm_scenarios"] == sum(yes(r["preexisting_alarm"]) for r in local)
        counts = {k: sum(int(r[k]) for r in local) for k in ("tp", "fp", "tn", "fn")}
        for k, count in counts.items():
            assert syn[k] == count
        close(syn["f1"], rate(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]))
        delays = [float(r["delay_seconds"]) for r in detected]
        close(syn["delay_mean_seconds"], mean(delays) if delays else None)
        close(syn["delay_median_seconds"], median(delays) if delays else None)
        close(syn["delay_std_seconds"], pstdev(delays) if delays else None)
        for d in protocol["deadlines_seconds"]:
            key = "deadline_le_" + format(d, "g").replace(".", "p") + "s"
            successes = sum(yes(r[key]) for r in local)
            assert syn[key + "_successes"] == successes
            close(syn[key + "_rate"], rate(successes, len(local)))
    per_scope = {"regression": reg_table, "healthy_active": healthy_table, "synthetic": table(output / "anomaly_per_fold.csv")}
    for r in table(output / "summary.csv"):
        values = [float(x[r["metric"]]) for x in per_scope[r["scope"]] if x["candidate"] == r["candidate"] and x[r["metric"]] != ""]
        assert int(r["n_defined"]) == len(values)
        for key, value in dict(mean=mean(values) if values else None, median=median(values) if values else None,
                               std=pstdev(values) if values else None, min=min(values) if values else None, max=max(values) if values else None).items():
            close(r[key], value)
    failures = table(output / "failed_cases.csv")
    assert {(r["candidate"], r["scenario_id"]) for r in failures} == {(r["candidate"], r["scenario_id"]) for r in all_scenarios if r["incident_detected"] != "True"}
    return dict(run_id=summary["run_id"], status="PASS", candidate_folds=summary["candidate_folds"],
                scenario_evaluations=len(all_scenarios), unique_paired_scenarios=len(signatures), failed_cases=len(failures),
                score_rows_verified=checked, output_hashes_verified=len(manifest["outputs"]), frozen_hashes_verified=len(frozen["files"]))


def main():
    summaries = [load(ROOT / "artifacts/sprint7" / name) for name in ("ablation_result.json", "comparison_result.json")]
    results = [audit_run(s) for s in summaries]
    # RF parallel floating-point reductions can differ in the final bits.
    # Counts/policies must match exactly; continuous metrics use tight tolerance.
    a, b = (load(Path(s["output_directory"]) / "results.json")["candidates"] for s in summaries)
    differences = []

    def consistent(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                consistent(left[key], right[key])
        elif type(left) is float:
            assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)
            differences.append(abs(left - right))
        else:
            assert left == right, (left, right)

    for name in ("hgb_battery_d", "rf_battery_d"):
        consistent(a[name], b[name])
    source = load(ROOT / "artifacts/sprint6/cross_validation_result.json")
    assert a["hgb_battery_d"]["pooled"]["healthy_active"] == source["healthy_active_pooled"]
    assert a["hgb_battery_d"]["pooled"]["synthetic"] == source["synthetic_pooled"]
    current = hashlib.sha256((ROOT / "data/raw/dataset_baterai.csv").read_bytes()).hexdigest()
    assert all(s["data_sha256"] == current for s in summaries)
    result = dict(status="PASS", runs=results, score_rows_verified=sum(r["score_rows_verified"] for r in results),
                  cross_command_hgbd_rfd_metrics_numerically_consistent=True,
                  cross_command_max_float_difference=max(differences, default=0.), hgbd_matches_sprint6=True,
                  project_code_imported=False, fitting_or_prediction_performed=False,
                  checks="all hashes, pre-outer freeze, split/provenance, exact validation P99/fingerprint, signed score/timestamp persistence, synthetic reference/masks/regression prediction equality, confusion, incident/contamination/delay/deadlines, pre-fault counts, pooled/macro statistics, complete failures, raw preservation")
    (ROOT / "artifacts/sprint7/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
