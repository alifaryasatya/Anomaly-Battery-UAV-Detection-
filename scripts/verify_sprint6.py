"""Independent stdlib audit of exported LOAFO decisions and metric denominators.

Run after main.py cross-validate and saving artifacts/sprint6/cross_validation_result.json.
No project metric, detector, model or fitting code is imported.
"""

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


def yes(value):
    assert value in ("True", "False")
    return value == "True"


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10), (actual, expected)


def decisions(rows, threshold):
    """Rebuild continuous exceedance runs directly from exported timestamps."""
    start = previous = None
    was_confirmed = False
    exposure = 0.
    for row in rows:
        time = int(row["TimeUS"])
        ready = yes(row["reference_ready"])
        continuous = previous is not None and all(row[k] == previous[k] for k in ("NamaFile", "Inst", "segment_id")) and 0 < time - int(previous["TimeUS"]) <= 500000
        if continuous and ready and yes(previous["reference_ready"]):
            exposure += (time - int(previous["TimeUS"])) / 1e6
        if not continuous:
            start, was_confirmed = None, False
        if ready:
            residual = float(row["voltage_sag_actual"]) - float(row["predicted_voltage_sag"])
            close(row["residual_signed_v"], residual)
            close(row["anomaly_score"], max(0., residual))
        else:
            assert all(not row[k] for k in ("anomaly_score", "predicted_voltage_sag", "residual_signed_v"))
        raw = ready and float(row["anomaly_score"]) > threshold
        if not raw:
            start = None
        elif start is None:
            start = time
        duration = (time - start) / 1e6 if start is not None else 0.
        confirmed = raw and duration >= .5
        assert yes(row["anomaly_raw"]) == raw
        close(row["anomaly_duration_seconds"], duration)
        assert yes(row["anomaly_confirmed"]) == confirmed
        assert yes(row["confirmed_episode_start"]) == (confirmed and not was_confirmed)
        assert row["detector_state"] == ("NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw else "NORMAL")
        was_confirmed, previous = confirmed, row
    return exposure


def main():
    summary = load(ROOT / "artifacts/sprint6/cross_validation_result.json")
    output = Path(summary["output_directory"])
    manifest = load(output / "manifest.json")
    for relative, digest in manifest["outputs"].items():
        path = (output / relative).resolve()
        assert path.is_relative_to(output)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    freeze = load(output / "all_fold_policies_frozen.json")
    assert freeze["fold_count"] == 6 and len(freeze["files"]) == 18
    for name, digest in freeze["files"].items():
        assert manifest["outputs"][name] == digest
    protocol = load(output / "experiment_protocol.json")
    assert len(protocol["features"]) == 7 and protocol["early_stopping"] is False
    assert protocol["candidate_selection"].startswith("none:")
    assert hashlib.sha256((ROOT / "data/raw/dataset_baterai.csv").read_bytes()).hexdigest() == summary["data_sha256"]
    assignments = table(output / "fold_assignments.csv")
    healthy_records = table(output / "healthy_active_per_fold.csv")
    healthy, checked_rows, thresholds = {}, 0, {}
    targets, errors = [], []
    for assignment in protocol["assignments"]:
        fold = assignment["fold_id"]
        train, val, test = (set(assignment[k]) for k in ("train", "validation", "test"))
        assert len(train) == 4 and len(val) == len(test) == 1
        assert not (train & val or train & test or val & test)
        roles = {r["NamaFile"]: r["split_role"] for r in assignments if r["fold_id"] == fold}
        assert {f for f, role in roles.items() if role == "train"} == train
        assert {f for f, role in roles.items() if role == "validation"} == val
        assert {f for f, role in roles.items() if role == "test"} == test
        folder = output / "folds" / fold
        threshold = load(folder / "threshold.json")
        thresholds[fold] = threshold["threshold_v"]
        validation = table(folder / "validation_scores.csv.gz")
        assert {r["NamaFile"] for r in validation} == val
        scores = sorted(float(r["anomaly_score"]) for r in validation if yes(r["reference_ready"]))
        rank = .99 * (len(scores) - 1)
        index = math.floor(rank)
        quantile = scores[index] + (rank - index) * (scores[math.ceil(rank)] - scores[index])
        close(threshold["threshold_v"], quantile)
        assert threshold["source_samples"] == len(scores)
        assert set(threshold["source_flights"]) == val
        decisions(validation, quantile)
        rows = table(folder / "healthy_test_scores.csv.gz")
        assert {r["NamaFile"] for r in rows} == test
        for r in rows:
            assert r["split_role"] == "test" and yes(r["unseen_by_model"])
            assert not yes(r["used_for_fit"]) and not yes(r["used_for_threshold"])
        duration = decisions(rows, quantile)
        checked_rows += len(rows) + len(validation)
        ready = [r for r in rows if yes(r["reference_ready"])]
        metrics = next(r for r in healthy_records if r["fold_id"] == fold)
        assert int(metrics["ready_samples"]) == len(ready)
        for field, flag in (("raw_alarm_samples", "anomaly_raw"), ("confirmed_alarm_samples", "anomaly_confirmed"), ("confirmed_episodes", "confirmed_episode_start")):
            assert int(metrics[field]) == sum(yes(r[flag]) for r in ready)
        close(metrics["observed_ready_seconds"], duration)
        healthy[fold] = {r["source_record"]: r for r in rows}
        targets.extend(float(r["voltage_sag_actual"]) for r in ready)
        errors.extend(float(r["residual_signed_v"]) for r in ready)
    assert len(targets) == 26870
    regression = summary["regression_pooled"]
    close(regression["mae_v"], mean(abs(e) for e in errors))
    close(regression["rmse_v"], math.sqrt(mean(e * e for e in errors)))
    target_mean = mean(targets)
    close(regression["r2"], 1 - sum(e * e for e in errors) / sum((y - target_mean) ** 2 for y in targets))
    records = table(output / "fault_per_scenario.csv")
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    delays, hits, contaminated = [], {float(d): 0 for d in protocol["deadlines_seconds"]}, 0
    applicable = 0
    for meta in records:
        if not yes(meta["fault_applicable"]):
            assert meta["evaluation_status"] == "NOT_APPLICABLE" and not meta["scores_csv"]
            assert all(not meta[k] for k in (*counts, "incident_detected", "delay_seconds"))
            continue
        applicable += 1
        rows = table(output / meta["scores_csv"])
        checked_rows += len(rows)
        decisions(rows, thresholds[meta["fold_id"]])
        onset, end = int(meta["onset_time_us"]), int(meta["fault_end_time_us"])
        ready = [r for r in rows if yes(r["reference_ready"])]
        for r in rows:
            base = healthy[meta["fold_id"]][r["source_record"]]
            for key in ("V_ref", "reference_ready", "predicted_voltage_sag", "Curr"):
                assert r[key] == base[key], (meta["scenario_id"], key)
            assert yes(r["fault_support"]) == (onset <= int(r["TimeUS"]) <= end)
            if yes(r["reference_ready"]):
                close(r["voltage_sag_actual"], float(base["voltage_sag_actual"]) + float(r["added_sag_v"]))
        support = [r for r in ready if yes(r["fault_support"])]
        existing = yes(support[0]["anomaly_confirmed"]) and not yes(support[0]["confirmed_episode_start"])
        fresh = next((r for r in support if yes(r["confirmed_episode_start"])), None)
        detected = not existing and fresh is not None
        contaminated += existing
        assert yes(meta["preexisting_alarm"]) == existing
        assert yes(meta["incident_detected"]) == detected
        assert meta["evaluation_status"] == ("PREEXISTING_ALARM" if existing else "DETECTED" if detected else "UNDETECTED")
        delay = (int(fresh["TimeUS"]) - onset) / 1e6 if detected else None
        if detected:
            close(meta["delay_seconds"], delay)
            delays.append(delay)
        else:
            assert not meta["delay_seconds"] and not meta["first_confirmation_time_us"]
        for d in hits:
            success = detected and delay <= d
            assert yes(meta["deadline_le_" + format(d, "g").replace(".", "p") + "s"]) == success
            hits[d] += success
        local = dict.fromkeys(counts, 0)
        for r in ready:
            label, alarm = yes(r["fault_support"]), yes(r["anomaly_confirmed"])
            local["tp" if label and alarm else "fn" if label else "fp" if alarm else "tn"] += 1
        for k, count in local.items():
            assert int(meta[k]) == count
            counts[k] += count
        pre = [r for r in ready if int(r["TimeUS"]) < onset]
        assert int(meta["pre_fault_samples"]) == len(pre)
        assert int(meta["pre_fault_confirmed_alarm_samples"]) == sum(yes(r["anomaly_confirmed"]) for r in pre)
    pooled = summary["synthetic_pooled"]
    assert pooled["applicable_scenarios"] == applicable
    assert pooled["detected_scenarios"] == len(delays) and pooled["preexisting_alarm_scenarios"] == contaminated
    for k, count in counts.items():
        assert pooled[k] == count
    close(pooled["f1"], 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"]))
    close(pooled["delay_mean_seconds"], mean(delays))
    close(pooled["delay_median_seconds"], median(delays))
    close(pooled["delay_std_seconds"], pstdev(delays))
    for d, count in hits.items():
        key = "deadline_le_" + format(d, "g").replace(".", "p") + "s"
        assert pooled[key + "_successes"] == count
        close(pooled[key + "_rate"], count / applicable)
    failures = table(output / "failed_cases.csv")
    assert {r["scenario_id"] for r in failures} == {r["scenario_id"] for r in records if r["incident_detected"] != "True"}
    result = dict(status="PASS", run_id=summary["run_id"], output_hashes_verified=len(manifest["outputs"]),
                  frozen_selection_hashes_verified=len(freeze["files"]), folds=6, planned_scenarios=len(records),
                  applicable_scenarios=applicable, detected_scenarios=len(delays), failed_scenarios=len(failures),
                  score_rows_verified=checked_rows, unique_healthy_ready_samples=len(targets),
                  checks="disjoint roles, validation-only linear P99, frozen policy hashes, positive residual, independent timestamp persistence, unchanged synthetic predictions/reference/current, support masks, conventional confusion, incident contamination/delay, all-applicable deadlines, pre-fault counts, pooled regression/F1/delay, complete failure visibility, raw hash",
                  project_code_imported=False, model_fit_or_predict_performed=False)
    (ROOT / "artifacts/sprint6/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
