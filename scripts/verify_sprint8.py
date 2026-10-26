"""Independent stdlib replay of the full validation grid and paired outer results.

No project fitting, detector, selection or metrics functions are imported.
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
DEADLINES = (1, 2, 5, 10, 30)
RANK = ("abrupt_severe_le_1s_rate", "gradual_observable_le_5s_rate", "incident_recall", "f1", "delay_median_seconds", "delay_mean_seconds")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def yes(s):
    assert s in ("True", "False"), s
    return s == "True"


def close(value, expected):
    if expected is None:
        assert value is None or value == "", value
    else:
        assert math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-10), (value, expected)


def ratio(a, b):
    return a / b if b else None


def score_input(path, role, synthetic=False):
    rows = table(path)
    typed = []
    for r in rows:
        ready = yes(r["reference_ready"])
        assert r["split_role"] == role and r["used_for_fit"] == "False" and yes(r["unseen_by_model"])
        assert yes(r["used_for_policy_selection"]) == (ready and role == "validation")
        assert yes(r["used_for_threshold"]) == (ready and role == "validation" and not synthetic)
        score = None
        if ready:
            error = float(r["voltage_sag_actual"]) - float(r["predicted_voltage_sag"])
            close(r["residual_signed_v"], error)
            score = max(0., error)
            close(r["anomaly_score"], score)
        else:
            assert not r["predicted_voltage_sag"] and not r["anomaly_score"]
        typed.append(dict(time=int(r["TimeUS"]), segment=r["segment_id"], ready=ready, score=score,
                          support=yes(r["fault_support"]) if synthetic else False, original=r))
    return typed


def simulate(rows, threshold, duration, meta=None, check_exports=False):
    prior = start = None
    before = False
    exposure = 0.
    n = raw_n = confirmed_n = episodes = 0
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    support_n = support_raw = pre_n = pre_raw = pre_confirm = 0
    existing, fresh = False, None
    onset, end = (int(meta[k]) for k in ("onset_time_us", "fault_end_time_us")) if meta else (None, None)
    support_times = []
    for row in rows:
        time, ready = row["time"], row["ready"]
        continuous = prior is not None and row["segment"] == prior["segment"] and 0 < time - prior["time"] <= 500000
        if continuous and ready and prior["ready"]:
            exposure += (time - prior["time"]) / 1e6
        if not continuous:
            start, before = None, False
        raw = ready and row["score"] > threshold
        if not raw:
            start = None
        elif start is None:
            start = time
        elapsed = (time - start) / 1e6 if start is not None else 0.
        confirmed = bool(raw and elapsed >= duration)
        episode = confirmed and not before
        if check_exports:
            r = row["original"]
            assert yes(r["anomaly_raw"]) == raw and yes(r["anomaly_confirmed"]) == confirmed
            assert yes(r["confirmed_episode_start"]) == episode
            close(r["anomaly_duration_seconds"], elapsed)
            assert r["detector_state"] == ("NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw else "NORMAL")
        if ready:
            n += 1
            raw_n += raw
            confirmed_n += confirmed
            episodes += episode
            if meta:
                label = row["support"]
                assert label == (onset <= time <= end)
                counts["tp" if label and confirmed else "fn" if label else "fp" if confirmed else "tn"] += 1
                if label:
                    support_times.append(time)
                    if support_n == 0:
                        existing = confirmed and not episode
                    support_n += 1
                    support_raw += raw
                    if episode and fresh is None:
                        fresh = time
                if time < onset:
                    pre_n += 1
                    pre_raw += raw
                    pre_confirm += confirmed
        before, prior = confirmed, row
    result = dict(ready_samples=n, raw_alarm_samples=raw_n, confirmed_alarm_samples=confirmed_n,
                  raw_fpr=ratio(raw_n, n), confirmed_fpr=ratio(confirmed_n, n), confirmed_episodes=episodes,
                  observed_ready_seconds=exposure, confirmed_episodes_per_minute=ratio(60 * episodes, exposure))
    if meta:
        assert support_times[0] == onset and support_times[-1] == end
        detected = fresh is not None and not existing
        delay = (fresh - onset) / 1e6 if detected else None
        result.update(counts, preexisting_alarm=existing, incident_detected=detected, delay_seconds=delay,
                      evaluation_status="PREEXISTING_ALARM" if existing else "DETECTED" if detected else "UNDETECTED",
                      first_confirmation_time_us=fresh if detected else None, candidate_new_confirmation_time_us=fresh,
                      raw_fault_support_samples=support_raw, evaluated_samples=n, fault_support_samples=support_n,
                      pre_fault_samples=pre_n, pre_fault_raw_alarm_samples=pre_raw, pre_fault_confirmed_alarm_samples=pre_confirm,
                      pre_fault_raw_fpr=ratio(pre_raw, pre_n), pre_fault_fpr=ratio(pre_confirm, pre_n),
                      precision=ratio(counts["tp"], counts["tp"] + counts["fp"]), recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
                      f1=ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
                      synthetic_sample_fpr=ratio(counts["fp"], counts["fp"] + counts["tn"]),
                      **{f"deadline_le_{d}s": bool(detected and delay <= d) for d in DEADLINES})
    return result


def verify_metrics(actual, expected):
    for k, value in expected.items():
        if k not in actual:
            continue
        if isinstance(value, str):
            assert actual[k] == value, (k, actual[k], value)
        elif type(value) is bool:
            assert yes(actual[k]) == value, (k, actual[k], value)
        else:
            close(actual[k], value)


def aggregate(records):
    applicable = [r for r in records if r.get("applicable", True)]
    detected = [r for r in applicable if r["incident_detected"]]
    counts = {k: sum(r[k] for r in applicable) for k in ("tp", "fp", "tn", "fn")}
    delays = [r["delay_seconds"] for r in detected]
    contaminated = sum(r["preexisting_alarm"] for r in applicable)
    pre_n = sum(r["pre_fault_samples"] for r in applicable)
    pre_alarm = sum(r["pre_fault_confirmed_alarm_samples"] for r in applicable)
    result = dict(scenarios=len(records), applicable_scenarios=len(applicable), not_applicable_scenarios=len(records) - len(applicable),
                  detected_scenarios=len(detected), undetected_scenarios=len(applicable) - len(detected),
                  preexisting_alarm_scenarios=contaminated, undetected_without_preexisting=len(applicable) - len(detected) - contaminated,
                  incident_recall=ratio(len(detected), len(applicable)), **counts,
                  precision=ratio(counts["tp"], counts["tp"] + counts["fp"]), recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
                  f1=ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
                  synthetic_sample_fpr=ratio(counts["fp"], counts["fp"] + counts["tn"]),
                  delay_detected_n=len(delays), delay_mean_seconds=mean(delays) if delays else None,
                  delay_median_seconds=median(delays) if delays else None, delay_std_seconds=pstdev(delays) if delays else None,
                  pre_fault_samples=pre_n, pre_fault_confirmed_alarm_samples=pre_alarm, pre_fault_fpr=ratio(pre_alarm, pre_n))
    for d in DEADLINES:
        key = f"deadline_le_{d}s"
        successes = sum(r[key] for r in applicable)
        result[key + "_successes"] = successes
        result[key + "_rate"] = ratio(successes, len(applicable))
    return result


def rank(row):
    return (*[-row[k] if row[k] is not None else math.inf for k in RANK[:4]],
            *[row[k] if row[k] is not None else math.inf for k in RANK[4:]], row["percentile"], row["persistence_seconds"])


def audit_scenario(meta, rows, threshold, duration, base, exports):
    if meta["fault_applicable"] == "False":
        assert not meta["scores_csv"]
        assert all(not meta[k] for k in ("tp", "fp", "tn", "fn", "incident_detected", "delay_seconds", "deadline_le_1s"))
        return dict(applicable=False, fault_family=meta["fault_family"], severity=meta["severity"], observability=meta["observability"])
    if exports:
        for row in rows:
            r = row["original"]
            b = base[r["source_record"]]
            for k in ("predicted_voltage_sag", "V_ref", "Curr", "reference_ready"):
                assert r[k] == b[k], (meta["scenario_id"], k)
            if row["ready"]:
                close(r["voltage_sag_actual"], float(b["voltage_sag_actual"]) + float(r["added_sag_v"]))
    computed = simulate(rows, threshold, duration, meta, check_exports=exports)
    verify_metrics(meta, computed)
    return dict(computed, applicable=True, fault_family=meta["fault_family"], severity=meta["severity"], observability=meta["observability"])


def main():
    summary = load(ROOT / "artifacts/sprint8/tuning_result.json")
    output = Path(summary["output_directory"])
    manifest = load(output / "manifest.json")
    for name, digest in manifest["outputs"].items():
        path = (output / name).resolve()
        assert path.is_relative_to(output) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    frozen = load(output / "all_policies_frozen.json")
    for name, digest in frozen["files"].items():
        assert manifest["outputs"][name] == digest
    protocol = load(output / "experiment_protocol.json")
    selected = load(output / "selected_per_fold.json")
    grid = table(output / "threshold_persistence_grid.csv")
    validation_cases = table(output / "validation_fault_per_candidate.csv")
    outer_cases = table(output / "fault_per_scenario.csv")
    health_outer = table(output / "healthy_active_per_fold.csv")
    all_replayed = exported = 0
    outer_recomputed = defaultdict(list)
    for entry in selected:
        fold, split = entry["fold_id"], entry["split"]
        train, val, test = (set(split[k]) for k in ("train", "validation", "test"))
        assert len(train) == 4 and not (train & val or train & test or val & test)
        folder = output / "folds" / fold
        val_rows = score_input(folder / "healthy_validation_scores.csv.gz", "validation")
        assert {r["original"]["NamaFile"] for r in val_rows} == val
        base = {r["original"]["source_record"]: r["original"] for r in val_rows}
        scores = sorted(r["score"] for r in val_rows if r["ready"])
        candidates = [r for r in grid if r["fold_id"] == fold]
        assert len(candidates) == 20
        thresholds = load(folder / "threshold_candidates.json")
        computed = {}
        for r in candidates:
            assert r["source_role"] == "validation" and r["validation_flight"] in val and yes(r["used_for_policy_selection"])
            p, d = float(r["percentile"]), float(r["persistence_seconds"])
            position = p / 100 * (len(scores) - 1)
            index = math.floor(position)
            threshold = scores[index] + (position - index) * (scores[math.ceil(position)] - scores[index])
            close(r["threshold_v"], threshold)
            payload = thresholds[format(p, "g")]
            assert set(payload["source_flights"]) == val and payload["source_role"] == "validation" and payload["source_samples"] == len(scores)
            fingerprint = [{"NamaFile": row["original"]["NamaFile"], "Inst": int(row["original"]["Inst"]), "TimeUS": row["time"],
                            "source_record": int(row["original"]["source_record"]), "anomaly_score": row["score"]} for row in val_rows if row["ready"]]
            assert hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest() == payload["source_sha256"]
            h = simulate(val_rows, threshold, d, check_exports=p == 99. and d == .5)
            all_replayed += len(val_rows)
            mapping = dict(healthy_ready_samples="ready_samples", healthy_raw_fpr="raw_fpr", healthy_raw_alarm_samples="raw_alarm_samples",
                           healthy_confirmed_alarm_samples="confirmed_alarm_samples", healthy_confirmed_fpr="confirmed_fpr",
                           healthy_confirmed_episodes="confirmed_episodes", healthy_observed_ready_seconds="observed_ready_seconds",
                           healthy_episodes_per_minute="confirmed_episodes_per_minute")
            for key, other in mapping.items():
                close(r[key], h[other])
            computed[r["candidate_id"]] = dict(percentile=p, persistence_seconds=d, threshold_v=threshold, scenarios=[],
                                                 healthy_confirmed_fpr=h["confirmed_fpr"], healthy_episodes_per_minute=h["confirmed_episodes_per_minute"])
        exported += len(val_rows)
        groups = defaultdict(list)
        for r in validation_cases:
            if r["fold_id"] == fold:
                assert r["source_role"] == "validation" and r["validation_flight"] in val
                assert r["used_for_threshold"] == "False" and yes(r["used_for_policy_selection"])
                groups[r["scenario_id"]].append(r)
        assert len(groups) == 27
        for cases in groups.values():
            assert len(cases) == 20
            sample = cases[0]
            rows = score_input(output / sample["scores_csv"], "validation", synthetic=True) if sample["scores_csv"] else []
            exported += len(rows)
            if rows:
                assert {r["original"]["NamaFile"] for r in rows} == val
                reference_meta = next(r for r in cases if float(r["percentile"]) == 99. and float(r["persistence_seconds"]) == .5)
                audit_scenario(reference_meta, rows, thresholds["99"]["threshold_v"], .5, base, True)
            for r in cases:
                c = computed[r["candidate_id"]]
                value = audit_scenario(r, rows, c["threshold_v"], c["persistence_seconds"], base, False)
                c["scenarios"].append(value)
                all_replayed += len(rows)
        for r in candidates:
            c = computed[r["candidate_id"]]
            metrics = aggregate(c["scenarios"])
            abrupt = aggregate([s for s in c["scenarios"] if s["fault_family"] in ("increased_resistance", "sudden_drop") and s["severity"] == "severe"])
            gradual = aggregate([s for s in c["scenarios"] if s["fault_family"] == "progressive" and s["observability"] in ("MEDIUM", "HIGH")])
            c.update(metrics, abrupt_severe_le_1s_rate=abrupt["deadline_le_1s_rate"], abrupt_severe_applicable=abrupt["applicable_scenarios"],
                     abrupt_severe_successes=abrupt["deadline_le_1s_successes"], gradual_observable_le_5s_rate=gradual["deadline_le_5s_rate"],
                     gradual_observable_applicable=gradual["applicable_scenarios"], gradual_observable_successes=gradual["deadline_le_5s_successes"])
            verify_metrics(r, c)
        reference = next(c for c in computed.values() if c["percentile"] == 99 and c["persistence_seconds"] == .5)
        good = []
        for r in candidates:
            c = computed[r["candidate_id"]]
            eligible = all(c[k] is not None and reference[k] is not None and c[k] <= reference[k] for k in ("healthy_confirmed_fpr", "healthy_episodes_per_minute"))
            assert yes(r["healthy_eligible"]) == eligible
            if eligible:
                good.append((r["candidate_id"], c))
        good.sort(key=lambda pair: rank(pair[1]))
        chosen_id, chosen = good[0] if good else next((key, c) for key, c in computed.items() if c is reference)
        assert entry["selection"]["candidate_id"] == chosen_id and entry["selection"]["eligible_candidates"] == len(good)
        for r in candidates:
            assert yes(r["selected"]) == (r["candidate_id"] == chosen_id)
            close(r["eligible_rank"], next((i + 1 for i, (key, _) in enumerate(good) if key == r["candidate_id"]), None))
        for policy, threshold in (("reference", entry["reference"]), ("selected", entry["threshold"])):
            original = score_input(folder / policy / "healthy_test_scores.csv.gz", "test")
            assert {r["original"]["NamaFile"] for r in original} == test
            h = simulate(original, threshold["threshold_v"], threshold["persistence_seconds"], check_exports=True)
            verify_metrics(next(r for r in health_outer if r["fold_id"] == fold and r["policy"] == policy), h)
            exported += len(original)
            all_replayed += len(original)
            originals = {r["original"]["source_record"]: r["original"] for r in original}
            for meta in [r for r in outer_cases if r["fold_id"] == fold and r["policy"] == policy]:
                assert meta["source_role"] == "test" and meta["used_for_policy_selection"] == meta["used_for_threshold"] == "False"
                rows = score_input(output / meta["scores_csv"], "test", synthetic=True) if meta["scores_csv"] else []
                exported += len(rows)
                all_replayed += len(rows)
                value = audit_scenario(meta, rows, threshold["threshold_v"], threshold["persistence_seconds"], originals, True)
                outer_recomputed[policy].append(value)
    results = load(output / "results.json")
    for policy in ("reference", "selected"):
        recomputed = aggregate(outer_recomputed[policy])
        pooled = results["policies"][policy]["pooled"]["synthetic"]
        for k, value in recomputed.items():
            close(pooled[k], value)
    original = load(ROOT / "artifacts/sprint6/cross_validation_result.json")
    assert results["policies"]["reference"]["pooled"] == dict(regression=original["regression_pooled"], healthy_active=original["healthy_active_pooled"], synthetic=original["synthetic_pooled"])
    assert results["policies"]["reference"]["pooled"]["regression"] == results["policies"]["selected"]["pooled"]["regression"]
    failed = table(output / "failed_cases.csv")
    assert {(r["policy"], r["scenario_id"]) for r in failed} == {(r["policy"], r["scenario_id"]) for r in outer_cases if r["incident_detected"] != "True"}
    assert hashlib.sha256((ROOT / "data/raw/dataset_baterai.csv").read_bytes()).hexdigest() == summary["data_sha256"]
    result = dict(status="PASS", run_id=summary["run_id"], output_hashes_verified=len(manifest["outputs"]),
                  pre_outer_hashes_verified=len(frozen["files"]), policies_independently_selected=len(selected),
                  grid_candidates_verified=len(grid), validation_candidate_scenarios_verified=len(validation_cases),
                  outer_scenarios_verified=len(outer_cases), exported_score_rows_verified=exported,
                  policy_row_decisions_replayed=all_replayed, reference_matches_sprint6=True,
                  project_code_imported=False, fitting_or_prediction_performed=False,
                  checks="hash/freeze/splits, healthy-only linear quantiles/fingerprints, validation-only flags, full grid timestamp decisions, conventional/incident/contamination/NA/deadline metrics, pooled aggregation, independent healthy screening and lexicographic/tie selection, paired outer decisions, complete failures, raw preservation")
    (ROOT / "artifacts/sprint8/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
