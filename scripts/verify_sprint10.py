"""Independent low-load score, timer, exposure, domain and saved-object audit.

Does not import project modules, fit an estimator, or calibrate a threshold.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if expected is None:
        assert actual in (None, ""), (actual, expected)
    else:
        assert actual not in (None, "") and math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-11), (actual, expected)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    i = int(position)
    return values[i] + (position - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def stats(values):
    if not values:
        return dict(n_defined=0, mean=None, std=None, median=None, min=None, max=None, p05=None, p95=None)
    return dict(n_defined=len(values), mean=mean(values), std=pstdev(values), median=median(values),
                min=min(values), max=max(values), p05=quantile(values, .05), p95=quantile(values, .95))


def check_stats(record, values):
    for key, value in stats(values).items():
        close(record[key], value)


def check_manifest(directory):
    manifest = load(directory / "manifest.json")
    for name, value in manifest["outputs"].items():
        assert digest(directory / name) == value, directory / name
    return manifest


def replay(rows, threshold, duration, gap_us):
    start = prior = None
    was_confirmed = False
    n = raw_count = confirmed_count = episodes = exposure = 0
    for r in rows:
        t, segment, ready = int(r["TimeUS"]), int(r["segment_id"]), r["reference_ready"] == "True"
        boundary = prior is None or segment != int(prior["segment_id"]) or t - int(prior["TimeUS"]) > gap_us
        if prior and not boundary and ready and prior["reference_ready"] == "True":
            assert t > int(prior["TimeUS"])
            exposure += t - int(prior["TimeUS"])
        if boundary:
            start, was_confirmed = None, False
        raw = ready and float(r["anomaly_score"]) > threshold
        start = (t if start is None else start) if raw else None
        elapsed = (t - start) / 1e6 if start is not None else 0.
        confirmed = raw and elapsed >= duration
        episode = confirmed and not was_confirmed
        state = "NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw else "NORMAL"
        assert (r["anomaly_raw"] == "True") == raw
        assert (r["anomaly_confirmed"] == "True") == confirmed
        assert (r["confirmed_episode_start"] == "True") == episode
        assert r["detector_state"] == state
        close(r["anomaly_duration_seconds"], elapsed)
        n += ready
        raw_count += raw
        confirmed_count += confirmed
        episodes += episode
        prior, was_confirmed = r, confirmed
    return dict(ready_samples=n, raw_alarm_samples=raw_count, confirmed_alarm_samples=confirmed_count,
                confirmed_episodes=episodes, observed_ready_seconds=exposure / 1e6,
                raw_fpr=raw_count / n if n else None, confirmed_fpr=confirmed_count / n if n else None,
                confirmed_episodes_per_minute=episodes * 60e6 / exposure if exposure else None)


def verify():
    import joblib
    import pandas as pd
    from threadpoolctl import threadpool_limits

    summary = load(ROOT / "artifacts/sprint10/low_load_result.json")
    run = Path(summary["output_directory"])
    manifest = check_manifest(run)
    protocol = load(run / "protocol.json")
    source_summary = load(ROOT / "artifacts/sprint6/cross_validation_result.json")
    source = Path(source_summary["output_directory"])
    check_manifest(source)
    assert digest(source / "manifest.json") == protocol["source_manifest_sha256"]
    assert digest(ROOT / "data/raw/dataset_baterai.csv") == summary["data_sha256"]
    feature_dir = ROOT / "artifacts/features" / source_summary["feature_run_id"]
    check_manifest(feature_dir)
    assert digest(feature_dir / "manifest.json") == summary["feature_manifest_sha256"]
    features = table(feature_dir / "battery_features.csv")
    lookup = {(r["NamaFile"], int(r["source_record"])): r for r in features}
    groups = defaultdict(list)
    for r in features:
        groups[r["NamaFile"]].append(r)
    coverage = {r["NamaFile"]: r for r in table(run / "reference_coverage.csv")}
    cohort = sorted(f for f, r in coverage.items() if r["v2_current_based_label"] == "LOW_LOAD_OR_IDLE")
    assert cohort == protocol["cohort"] and len(cohort) == summary["unique_low_load_flights"]
    assert summary["primary_fold"] == sorted(s["fold_id"] for s in protocol["frozen_sources"])[0]
    for name, value in load(run / "pre_evaluation_freeze.json").items():
        assert digest(run / name) == value
    metric_records = table(run / "per_model_flight.csv")
    domains = table(run / "outside_training_ranges.csv")
    responses = table(run / "response_distributions.csv")
    ranges = load(run / "training_domain_ranges.json")
    episode_records = table(run / "episode_context.csv")
    computed, decision_rows, prediction_rows = defaultdict(list), 0, 0
    for src in protocol["frozen_sources"]:
        fold = src["fold_id"]
        path = ROOT / src["bundle_path"]
        assert digest(path) == src["model_sha256"]
        assert digest(ROOT / src["threshold_path"]) == src["threshold_sha256"]
        bundle = joblib.load(path)
        assert bundle["threshold"] == src["threshold"] == load(ROOT / src["threshold_path"])
        assert src["threshold"]["percentile"] == 99 and src["threshold"]["persistence_seconds"] == .5
        assert bundle["split"] == src["split"]
        assert not set(cohort).intersection(set().union(*(set(v) for v in src["split"].values())))
        train = [r for r in features if r["NamaFile"] in src["split"]["train"] and r["reference_ready"] == "True"]
        for name in bundle["predictors"]:
            check_stats(ranges[fold][name], [float(r[name]) for r in train])
        for flight in cohort:
            rows = table(run / f"scores/{fold}/{flight}.csv.gz")
            assert len(rows) == len(groups[flight])
            ready = [r for r in rows if r["reference_ready"] == "True"]
            x = pd.DataFrame([[float(lookup[(flight, int(r["source_record"]))][n]) for n in bundle["predictors"]] for r in ready], columns=bundle["predictors"])
            with threadpool_limits(limits=1, user_api="openmp"):
                predictions = iter(bundle["model"].predict(x)) if ready else iter(())
            for row in rows:
                original = lookup[(flight, int(row["source_record"]))]
                assert row["split_role"] == "low_load" and row["fold_id"] == fold
                assert row["used_for_fit"] == row["used_for_threshold"] == row["used_for_policy_selection"] == "False"
                assert row["unseen_by_model"] == row["unseen_by_detector"] == "True"
                assert row["source_training_flight"] == row["source_validation_flight"] == "False"
                for name in ("Volt", "Curr", "dI_dt", "V_ref", "voltage_sag_actual"):
                    close(row[name], float(original[name]) if original[name] else None)
                if row["reference_ready"] == "True":
                    predicted = float(next(predictions))
                    residual = float(original["voltage_sag_actual"]) - predicted
                    close(row["predicted_voltage_sag"], predicted)
                    close(row["residual_signed_v"], residual)
                    close(row["anomaly_score"], max(0, residual))
                    outside = any(float(original[n]) < b["min"] or float(original[n]) > b["max"] for n, b in ranges[fold].items())
                    assert (row["outside_any_training_range"] == "True") == outside
                else:
                    assert row["predicted_voltage_sag"] == row["residual_signed_v"] == row["anomaly_score"] == row["outside_any_training_range"] == ""
            metrics = replay(rows, src["threshold"]["threshold_v"], .5, round(protocol["gap_threshold_seconds"] * 1e6))
            exported = next(r for r in metric_records if r["fold_id"] == fold and r["NamaFile"] == flight)
            for key, value in metrics.items():
                close(exported[key], value)
            assert exported["unseen_by_model"] == exported["unseen_by_detector"] == "True"
            for name, bounds in ranges[fold].items():
                r = next(r for r in domains if r["fold_id"] == fold and r["NamaFile"] == flight and r["predictor"] == name)
                n = sum(float(lookup[(flight, int(row["source_record"]))][name]) < bounds["min"] or float(lookup[(flight, int(row["source_record"]))][name]) > bounds["max"] for row in ready)
                close(r["outside_samples"], n)
                close(r["outside_fraction"], n / len(ready) if ready else None)
            for name in ("predicted_voltage_sag", "residual_signed_v", "anomaly_score"):
                r = next(r for r in responses if r["fold_id"] == fold and r["NamaFile"] == flight and r["response"] == name)
                check_stats(r, [float(row[name]) for row in ready])
            starts = [r for r in rows if r["confirmed_episode_start"] == "True"]
            assert len(starts) == len([r for r in episode_records if r["fold_id"] == fold and r["NamaFile"] == flight])
            for start in starts:
                ep = next(r for r in episode_records if r["fold_id"] == fold and r["NamaFile"] == flight and int(r["start_time_us"]) == int(start["TimeUS"]))
                part = [r for r in rows if int(ep["start_time_us"]) <= int(r["TimeUS"]) <= int(ep["end_time_us"])]
                assert all(r["anomaly_confirmed"] == "True" for r in part)
                close(ep["confirmed_samples"], len(part))
                for name, values in (("Curr", [float(r["Curr"]) for r in part]), ("residual", [float(r["residual_signed_v"]) for r in part])):
                    unit = "a" if name == "Curr" else "v"
                    close(ep[name + "_min_" + unit], min(values))
                    close(ep[name + "_max_" + unit], max(values))
                close(ep["abs_dI_dt_max_a_per_s"], max(abs(float(r["dI_dt"])) for r in part))
            computed[fold].append(metrics)
            decision_rows += len(rows)
            prediction_rows += len(ready)
    profiles = table(run / "flight_domain_profiles.csv")
    for r in profiles:
        subset = groups[r["NamaFile"]]
        close(r["initial_voltage_v"], float(subset[0]["Volt"]) if subset else None)
        if r["feature"] != "Volt":
            subset = [s for s in subset if s["reference_ready"] == "True"]
        check_stats(r, [float(s[r["feature"]]) for s in subset])
    for record in table(run / "per_model_summary.csv"):
        parts = computed[record["fold_id"]]
        combined = {k: sum(r[k] for r in parts) for k in ("ready_samples", "raw_alarm_samples", "confirmed_alarm_samples", "confirmed_episodes", "observed_ready_seconds")}
        n, exposure = combined["ready_samples"], combined["observed_ready_seconds"]
        combined.update(raw_fpr=combined["raw_alarm_samples"] / n if n else None,
                        confirmed_fpr=combined["confirmed_alarm_samples"] / n if n else None,
                        confirmed_episodes_per_minute=60 * combined["confirmed_episodes"] / exposure if exposure else None)
        for key, value in combined.items():
            close(record[key], value)
            if record["fold_id"] == summary["primary_fold"]:
                close(summary["primary_low_load"][key], value)
    assert len(metric_records) == len(cohort) * len(protocol["frozen_sources"])
    assert not any(summary[k] for k in ("model_training_performed", "threshold_calibration_performed", "policy_selection_performed", "synthetic_evaluation_performed", "threshold_changed", "operational_dual_enabled"))
    return dict(status="PASS", run_id=summary["run_id"], project_functions_imported=False, fitting_performed=False,
                unique_low_load_flights=len(cohort), model_flight_pairs=len(metric_records), decision_rows=decision_rows,
                saved_bundle_predictions=prediction_rows, domain_profiles=len(profiles), response_profiles=len(responses),
                marginal_range_comparisons=len(domains), confirmed_episode_contexts=len(episode_records),
                verified_artifact_files=len(manifest["outputs"]), primary_allocation="first sorted frozen fold, unchanged",
                repeated_exposure_pooled_as_independent=False)


if __name__ == "__main__":
    result = verify()
    (ROOT / "artifacts/sprint10/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
