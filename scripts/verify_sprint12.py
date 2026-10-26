"""Independent raw-feature/state/latency audit; no project function imports."""

from array import array
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import runpy
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts/verify_sprint11.py"))
calculate = BASE["calculate"]
PHASES = ("parse_ns", "features_ns", "prediction_ns", "diagnostics_detector_ns", "end_to_end_ns")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def yes(value):
    assert value in ("True", "False")
    return value == "True"


def close(actual, expected):
    if expected is None:
        assert actual in (None, ""), (actual, expected)
    else:
        assert actual not in (None, "") and math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def quantile(values, q):
    position = (len(values) - 1) * q
    i = int(position)
    return values[i] + (position - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def verify():
    import joblib
    import pandas as pd
    from threadpoolctl import threadpool_limits

    summary = load(ROOT / "artifacts/sprint12/replay_result.json")
    out = Path(summary["output_directory"])
    manifest = load(out / "manifest.json")
    for name, digest in manifest["outputs"].items():
        assert sha(out / name) == digest, name
    assert load(out / "summary.json") == summary
    protocol = load(out / "replay_protocol.json")
    assert protocol == manifest["identity"]["protocol"]
    cfg = __import__("yaml").safe_load((out / "config_used.yaml").read_text(encoding="utf-8"))
    source = ROOT / "artifacts/cross_validation" / protocol["baseline_run_id"]
    assert sha(source / "manifest.json") == protocol["baseline_manifest_sha256"]
    baseline = load(source / "results.json")["run_summary"]
    features_dir = ROOT / "artifacts/features" / baseline["feature_run_id"]
    assert sha(features_dir / "manifest.json") == protocol["feature_manifest_sha256"]
    low = ROOT / "artifacts/low_load" / protocol["low_load_run_id"]
    assert sha(low / "manifest.json") == protocol["low_load_manifest_sha256"]
    assert sha(ROOT / cfg["data"]["path"]) == protocol["raw_data_sha256"]
    for directory in (source, features_dir, low):
        for name, digest in load(directory / "manifest.json")["outputs"].items():
            assert sha(directory / name) == digest
    for name, digest in load(out / "pre_replay_freeze.json").items():
        assert sha(out / name) == digest
    models, thresholds = {}, {}
    for policy in protocol["policies"]:
        path = source / "folds" / policy["fold_id"] / "model_bundle.joblib"
        assert sha(path) == policy["bundle_sha256"]
        assert sha(path.parent / "threshold.json") == policy["threshold_sha256"]
        b = joblib.load(path)
        assert b["threshold"] == policy["threshold"] and b["split"] == policy["split"]
        assert b["threshold"]["percentile"] == 99 and b["threshold"]["persistence_seconds"] == .5
        models[policy["fold_id"]] = b
        thresholds[policy["fold_id"]] = b["threshold"]["threshold_v"]
    raw = {i: r for i, r in enumerate(table(ROOT / cfg["data"]["path"]), 2)}
    cases = {r["fold_id"] + "_" + r["scenario_id"]: r for r in table(source / "fault_per_scenario.csv")}
    case_results = {r["fold_id"] + "_" + r["scenario_id"]: r for r in table(out / "fault_per_scenario.csv")}
    jobs = table(out / "jobs.csv")
    reference_features = {int(r["source_record"]): r for r in table(features_dir / "battery_features.csv")}
    timings = defaultdict(lambda: {n: array("Q") for n in PHASES})
    cache, counts, health = {}, Counter(), []
    prediction_rows = row_count = slope_values = 0
    synthetic_started = False
    for job in jobs:
        cohort, fold = job["cohort"], job["fold_id"]
        is_fault = cohort == "synthetic"
        if is_fault:
            synthetic_started = True
        else:
            assert not synthetic_started
        rows = table(out / job["trace_csv"])
        assert sha(out / job["trace_csv"]) == job["trace_sha256"]
        assert len(rows) == int(job["rows"])
        names = models[fold]["predictors"]
        if cohort == "healthy_active":
            golden_path = source / "folds" / fold / "healthy_test_scores.csv.gz"
            assert job["NamaFile"] in models[fold]["split"]["test"]
        elif cohort == "healthy_low_load":
            golden_path = low / "scores" / fold / (job["NamaFile"] + ".csv.gz")
            assert fold == load(low / "summary.json")["primary_fold"]
        else:
            case = cases[job["job_id"]]
            golden_path = source / case["scores_csv"]
            assert sha(golden_path) == case["scores_sha256"] and case["fault_applicable"] == "True"
        golden = table(golden_path)
        assert len(golden) == len(rows)
        predictions = iter(())
        if not is_fault:
            ready = [r for r in rows if yes(r["reference_ready"])]
            if ready:
                frame = pd.DataFrame([[float(r[n]) for n in names] for r in ready], columns=names)
                with threadpool_limits(limits=1, user_api="openmp"):
                    predictions = iter(models[fold]["model"].predict(frame))
        history, residuals, low_volts = deque(), deque(), []
        prior = None
        reference, locked = None, None
        status = "WARMUP"
        stream_start = int(rows[0]["TimeUS"]) if rows else None
        segment, segment_start, run_start, was_confirmed = 0, stream_start, None, False
        n = raw_count = confirmed_count = episodes = exposure_us = 0
        tp = fp = tn = fn = pre_n = pre_alarm = 0
        carried, first_warning = False, None
        for r, g in zip(rows, golden):
            t, source_record = int(r["TimeUS"]), int(r["source_record"])
            original = raw[source_record]
            assert r["NamaFile"] == original["NamaFile"] == g["NamaFile"] and int(original["TimeUS"]) == t == int(g["TimeUS"])
            assert int(r["Inst"]) == int(original["Inst"]) == int(g["Inst"])
            for name in ("Curr", "CurrTot"):
                close(r[name], float(original[name]))
            voltage, current = float(r["Volt"]), float(r["Curr"])
            close(voltage, float(g["Volt"]))
            dt_us = t - int(prior["TimeUS"]) if prior else 0
            assert prior is None or dt_us > 0
            gap = prior is None or dt_us > 500000
            if gap:
                segment += int(prior is not None)
                segment_start, run_start, was_confirmed = t, None, False
                history.clear()
                residuals.clear()
            elapsed, segment_elapsed = (t - stream_start) / 1e6, (t - segment_start) / 1e6
            assert int(r["segment_id"]) == segment and yes(r["is_segment_start"]) == gap
            close(r["dt_seconds"], 0 if gap else dt_us / 1e6)
            close(r["gap_before_seconds"], dt_us / 1e6 if prior and gap else 0)
            close(r["elapsed_time"], elapsed)
            close(r["segment_elapsed_seconds"], segment_elapsed)
            if status == "WARMUP":
                if elapsed <= 2:
                    if current <= 1:
                        low_volts.append(voltage)
                elif len(low_volts) >= 5:
                    reference, status, locked = median(low_volts), "READY", t
                else:
                    status = "REFERENCE_UNAVAILABLE"
            ready = status == "READY"
            assert r["reference_status"] == status and yes(r["reference_ready"]) == ready
            assert r["readiness"] == ("READY" if ready else "NOT_READY")
            close(r["reference_lock_time_us"], locked)
            close(r["reference_low_current_samples"], len(low_volts))
            close(r["V_ref"], reference)
            target = reference - voltage if ready else None
            close(r["voltage_sag_actual"], target)
            close(r["dI_dt"], (current - float(prior["Curr"])) / (dt_us / 1e6) if prior and not gap else 0)
            close(r["dV_dt"], (voltage - float(prior["Volt"])) / (dt_us / 1e6) if prior and not gap else 0)
            assert yes(r["derivative_ready"]) == (not gap)
            history.append((t, current))
            while history and t - history[0][0] >= 1000000:
                history.popleft()
            for w, token in ((.5, "0p5s"), (1., "1p0s")):
                values = [v for time, v in history if t - time < w * 1e6]
                avg = math.fsum(values) / len(values)
                std = math.sqrt(math.fsum((v - avg) ** 2 for v in values) / len(values))
                close(r["Curr_roll_mean_" + token], avg)
                close(r["Curr_roll_std_" + token], std)
                close(r["Curr_roll_count_" + token], len(values))
                close(r["Curr_roll_coverage_" + token], min(segment_elapsed / w, 1))
                assert yes(r["Curr_roll_ready_" + token]) == (segment_elapsed >= w)
            values = tuple(float(r[name]) for name in names)
            key = fold, values
            predicted, residual = None, None
            if ready:
                if is_fault and r["prediction_mode"] == "EXACT_INPUT_CACHE":
                    assert key in cache
                    predicted = cache[key]
                elif not is_fault:
                    assert r["prediction_mode"] == "UNCACHED"
                    predicted = float(next(predictions))
                    cache[key] = predicted
                    prediction_rows += 1
                else:
                    assert r["prediction_mode"] == "UNCACHED" and key not in cache
                    with threadpool_limits(limits=1, user_api="openmp"):
                        predicted = float(models[fold]["model"].predict(pd.DataFrame([values], columns=names))[0])
                    cache[key] = predicted
                    prediction_rows += 1
                residual = target - predicted
            else:
                assert r["prediction_mode"] == "NOT_READY"
            close(r["predicted_voltage_sag"], predicted)
            close(r["residual_signed_v"], residual)
            close(r["anomaly_score"], max(0, residual) if ready else None)
            delta_ready = ready and prior is not None and yes(prior["reference_ready"]) and not gap
            delta = residual - float(prior["residual_signed_v"]) if delta_ready else None
            assert yes(r["residual_delta_ready"]) == delta_ready
            close(r["residual_delta_v"], delta)
            close(r["residual_delta_v_per_s"], delta / (dt_us / 1e6) if delta_ready else None)
            if ready:
                residuals.append((t, residual))
            else:
                residuals.clear()
            while residuals and t - residuals[0][0] >= 1000000:
                residuals.popleft()
            for w, token in ((.3, "0p3s"), (.5, "0p5s"), (1., "1p0s")):
                subset = [(time, v) for time, v in residuals if t - time < w * 1e6]
                m, slope = len(subset), None
                if m >= 2:
                    x, y = [(time - t) / 1e6 for time, _ in subset], [v for _, v in subset]
                    sx, sy = math.fsum(x), math.fsum(y)
                    slope = (m * math.fsum(a * b for a, b in zip(x, y)) - sx * sy) / (m * math.fsum(a * a for a in x) - sx * sx)
                    slope_values += 1
                close(r["residual_slope_" + token], slope)
                close(r["residual_slope_count_" + token], m)
                assert yes(r["residual_slope_ready_" + token]) == (m >= 2)
            raw_alarm = ready and max(0, residual) > thresholds[fold]
            run_start = (t if run_start is None else run_start) if raw_alarm else None
            duration = (t - run_start) / 1e6 if run_start is not None else 0
            confirmed = raw_alarm and duration >= .5
            episode = confirmed and not was_confirmed
            state = "NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw_alarm else "NORMAL"
            assert yes(r["anomaly_raw"]) == raw_alarm and yes(r["anomaly_confirmed"]) == confirmed
            assert yes(r["confirmed_episode_start"]) == episode and r["detector_state"] == state
            close(r["anomaly_duration_seconds"], duration)
            for name in ("predicted_voltage_sag", "residual_signed_v", "anomaly_score", "anomaly_duration_seconds"):
                close(r[name], float(g[name]) if g[name] else None)
            for name in ("anomaly_raw", "anomaly_confirmed", "confirmed_episode_start", "detector_state"):
                assert r[name] == g[name]
            if prior and not gap and ready and yes(prior["reference_ready"]):
                exposure_us += dt_us
            n += ready
            raw_count += raw_alarm
            confirmed_count += confirmed
            episodes += episode
            if is_fault and ready:
                support = yes(r["fault_support"])
                assert support == yes(g["fault_support"])
                tp += support and confirmed
                fp += not support and confirmed
                tn += not support and not confirmed
                fn += support and not confirmed
                onset = int(case["onset_time_us"])
                if t == onset:
                    carried = confirmed and not episode
                if support and episode and first_warning is None:
                    first_warning = r
                if t < onset:
                    pre_n += 1
                    pre_alarm += confirmed
            for name in PHASES:
                value = int(r[name])
                assert value >= 0
                timings[cohort + "/" + r["prediction_mode"]][name].append(value)
            assert int(r["end_to_end_ns"]) == sum(int(r[name]) for name in PHASES[:-1])
            assert yes(r["accepted"])
            counts[r["prediction_mode"]] += 1
            prior, was_confirmed = r, confirmed
            row_count += 1
        metrics = dict(ready_samples=n, raw_alarm_samples=raw_count, confirmed_alarm_samples=confirmed_count,
                       confirmed_episodes=episodes, observed_ready_seconds=exposure_us / 1e6,
                       raw_fpr=raw_count / n if n else None, confirmed_fpr=confirmed_count / n if n else None,
                       confirmed_episodes_per_minute=episodes * 60e6 / exposure_us if exposure_us else None)
        if not is_fault:
            health.append(dict(cohort=cohort, **metrics))
        else:
            c = case_results[job["job_id"]]
            detected = first_warning is not None and not carried
            assert yes(c["incident_detected"]) == detected and yes(c["preexisting_alarm"]) == carried
            for name, value in dict(tp=tp, fp=fp, tn=tn, fn=fn, pre_fault_samples=pre_n, pre_fault_confirmed_alarm_samples=pre_alarm).items():
                close(c[name], value)
            delay = (int(first_warning["TimeUS"]) - int(case["onset_time_us"])) / 1e6 if detected else None
            close(c["delay_seconds"], delay)
            close(c["confirmation_processing_latency_ms"], int(first_warning["end_to_end_ns"]) / 1e6 if detected else None)
            close(c["persistence_elapsed_at_confirmation_seconds"], float(first_warning["anomaly_duration_seconds"]) if detected else None)
            close(c["raw_run_start_time_us"], int(first_warning["TimeUS"]) - round(float(first_warning["anomaly_duration_seconds"]) * 1e6) if detected else None)
    assert row_count == summary["replayed_rows"] and dict(counts) == summary["prediction_mode_counts"]
    assert len(jobs) == summary["replay_jobs"]
    assert len([r for r in cases.values() if r["fault_applicable"] == "True"]) == len([j for j in jobs if j["cohort"] == "synthetic"])
    for case_id, case in cases.items():
        if case["fault_applicable"] == "False":
            assert case_id not in {j["job_id"] for j in jobs}
            assert case_results[case_id]["delay_seconds"] == case_results[case_id]["incident_detected"] == ""
    for key, value in calculate(list(case_results.values())).items():
        close(summary["synthetic"][key], value)
    for record in table(out / "latency_summary.csv"):
        values = sorted(timings[record["group"]][record["phase"]])
        close(record["samples"], len(values))
        close(record["mean_ms"], math.fsum(values) / len(values) / 1e6)
        for q in (50, 95, 99):
            close(record[f"p{q}_ms"], quantile(values, q / 100) / 1e6)
        close(record["max_ms"], max(values) / 1e6)
        close(record["over_nominal_budget_samples"], sum(v > float(record["nominal_budget_ms"]) * 1e6 for v in values))
    for key, cohort in (("healthy_active", "healthy_active"), ("healthy_low_load", "healthy_low_load")):
        subset = [r for r in health if r["cohort"] == cohort]
        for name in ("ready_samples", "raw_alarm_samples", "confirmed_alarm_samples", "confirmed_episodes", "observed_ready_seconds"):
            close(summary[key][name], math.fsum(r[name] for r in subset))
    assert summary["batch_stream_equivalence"] == "PASS"
    assert not summary["model_training_performed"] and not summary["threshold_changed"] and not summary["emergency_candidate_replayed"]
    assert not summary["operational_config_changed"] and not summary["paced_replay_performed"]
    assert summary["telemetry_signal_delay_includes_persistence"]
    return dict(status="PASS", run_id=summary["run_id"], project_functions_imported=False, fitting_performed=False,
                rows_independently_replayed=row_count, jobs=len(jobs), saved_model_ready_predictions=prediction_rows,
                cached_predictions_verified=counts["EXACT_INPUT_CACHE"], independently_checked_ols_values=slope_values,
                raw_reference_rolling_derivative_timers_states_and_latency_verified=True,
                signal_delay_and_persistence_counted_once=True, research_emergency_gate_preserved=True,
                latency_percentile_rows=len(table(out / "latency_summary.csv")))


if __name__ == "__main__":
    result = verify()
    (ROOT / "artifacts/sprint12/independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
