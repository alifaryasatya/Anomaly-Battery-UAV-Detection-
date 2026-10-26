"""Independent stdlib residual OLS, dual timer, selection and promotion audit."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import runpy
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts/verify_sprint8.py"))
load, table, yes, close = (BASE[k] for k in ("load", "table", "yes", "close"))
score_input, aggregate, verify_metrics = (BASE[k] for k in ("score_input", "aggregate", "verify_metrics"))
DEADLINES = (1, 2, 5, 10, 30)
PARAMS = ("high_percentile", "low_percentile", "slope_percentile", "slope_window_seconds", "path_b_persistence_seconds")
RANK = ("abrupt_severe_le_1s_rate", "gradual_observable_le_5s_rate", "incident_recall", "f1", "delay_median_seconds", "delay_mean_seconds")


def slope_names(w):
    token = str(float(w)).replace(".", "p")
    return f"residual_slope_{token}s", f"residual_slope_count_{token}s", f"residual_slope_ready_{token}s"


def diagnostics(rows, windows):
    history, prior = [], None
    for row in rows:
        r, t = row["original"], row["time"]
        boundary = prior is None or row["segment"] != prior["segment"] or t - prior["time"] > 500000
        if boundary or not row["ready"]:
            history = []
        delta_ready = row["ready"] and prior is not None and prior["ready"] and not boundary
        assert yes(r["residual_delta_ready"]) == delta_ready
        delta = float(r["residual_signed_v"]) - float(prior["original"]["residual_signed_v"]) if delta_ready else None
        close(r["residual_delta_v"], delta)
        close(r["residual_delta_v_per_s"], delta / ((t - prior["time"]) / 1e6) if delta_ready else None)
        if row["ready"]:
            history.append((t, float(r["residual_signed_v"])))
        history = [(time, value) for time, value in history if t - time < round(max(windows) * 1e6)]
        for w in windows:
            subset = [(time, value) for time, value in history if t - time < round(w * 1e6)]
            name, count, ready = slope_names(w)
            assert int(r[count]) == len(subset) and yes(r[ready]) == (len(subset) >= 2)
            expected = None
            if len(subset) >= 2:
                x, y = [(time - t) / 1e6 for time, _ in subset], [value for _, value in subset]
                n = len(x)
                # Independent closed-form OLS, rather than the engine's centered covariance.
                expected = (n * math.fsum(a * b for a, b in zip(x, y)) - math.fsum(x) * math.fsum(y)) / (n * math.fsum(a * a for a in x) - math.fsum(x) ** 2)
            close(r[name], expected)
        prior = row


def simulate(rows, policy, meta=None, check=False):
    dual = policy["mode"] == "DUAL"
    prior, start_a, start_b = None, None, None
    before_a = before_b = before_union = False
    n = raw_n = confirmed_n = episodes = a_starts = b_starts = 0
    exposure = 0.
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    attr_counts = dict(A_ONLY=0, B_ONLY=0, BOTH=0)
    pre_n = pre_raw = pre_confirm = support_n = support_raw = 0
    existing, fresh, first_path = False, None, None
    onset, end = (int(meta[k]) for k in ("onset_time_us", "fault_end_time_us")) if meta else (None, None)
    support_times = []
    for row in rows:
        r, t, ready = row["original"], row["time"], row["ready"]
        boundary = prior is None or row["segment"] != prior["segment"] or t - prior["time"] > 500000
        if boundary:
            start_a = start_b = None
            before_a = before_b = before_union = False
        elif ready and prior["ready"]:
            exposure += (t - prior["time"]) / 1e6
        raw_a = bool(ready and row["score"] > policy["threshold_high_v"])
        raw_b = False
        if dual and ready:
            name, _, flag = slope_names(policy["slope_window_seconds"])
            raw_b = bool(yes(r[flag]) and row["score"] > policy["threshold_low_v"] and float(r[name]) > policy["threshold_slope_v_per_s"])
        start_a = (t if start_a is None else start_a) if raw_a else None
        start_b = (t if start_b is None else start_b) if raw_b else None
        da, db = (t - start_a) / 1e6 if start_a is not None else 0., (t - start_b) / 1e6 if start_b is not None else 0.
        ca, cb = bool(raw_a and da >= .5), bool(raw_b and db >= policy["path_b_persistence_seconds"]) if dual else False
        ea, eb = ca and not before_a, cb and not before_b
        confirmed, raw = ca or cb, raw_a or raw_b
        episode = confirmed and not before_union
        attr = "BOTH" if ca and cb else "A_ONLY" if ca else "B_ONLY" if cb else "NONE"
        if check:
            verify_metrics(r, dict(path_a_raw=raw_a, path_b_raw=raw_b, path_a_confirmed=ca, path_b_confirmed=cb,
                                  path_a_duration_seconds=da, path_b_duration_seconds=db, path_a_episode_start=ea, path_b_episode_start=eb,
                                  anomaly_raw=raw, anomaly_confirmed=confirmed, confirmed_episode_start=episode,
                                  anomaly_duration_seconds=max(da, db), alarm_attribution=attr,
                                  first_confirmation_path=attr if episode else None,
                                  detector_state="NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw else "NORMAL"))
        if ready:
            n += 1
            raw_n += raw
            confirmed_n += confirmed
            episodes += episode
            a_starts += ea
            b_starts += eb
            if confirmed:
                attr_counts[attr] += 1
            if meta:
                label = row["support"]
                assert label == (onset <= t <= end)
                counts["tp" if label and confirmed else "fn" if label else "fp" if confirmed else "tn"] += 1
                if label:
                    support_times.append(t)
                    if support_n == 0:
                        existing = confirmed and not episode
                    support_n += 1
                    support_raw += raw
                    if episode and fresh is None:
                        fresh, first_path = t, attr
                if t < onset:
                    pre_n += 1
                    pre_raw += raw
                    pre_confirm += confirmed
        before_a, before_b, before_union, prior = ca, cb, confirmed, row
    result = dict(ready_samples=n, raw_alarm_samples=raw_n, confirmed_alarm_samples=confirmed_n,
                  raw_fpr=raw_n / n if n else None, confirmed_fpr=confirmed_n / n if n else None,
                  confirmed_episodes=episodes, observed_ready_seconds=exposure, confirmed_episodes_per_minute=60 * episodes / exposure if exposure else None,
                  path_a_episode_starts=a_starts, path_b_episode_starts=b_starts,
                  **{"confirmed_" + key.lower() + "_samples": value for key, value in attr_counts.items()})
    if meta:
        assert support_times[0] == onset and support_times[-1] == end
        detected = fresh is not None and not existing
        delay = (fresh - onset) / 1e6 if detected else None
        ratio = BASE["ratio"]
        result.update(counts, preexisting_alarm=existing, incident_detected=detected, delay_seconds=delay,
                      evaluation_status="PREEXISTING_ALARM" if existing else "DETECTED" if detected else "UNDETECTED",
                      detection_reason="CONFIRMED_ALARM_CARRIED_INTO_ONSET" if existing else "NEW_CONFIRMATION_WITHIN_SUPPORT" if detected else "NO_RAW_EXCEEDANCE_IN_SUPPORT" if not support_raw else "NO_NEW_CONFIRMATION_WITHIN_SUPPORT",
                      first_confirmation_time_us=fresh if detected else None, candidate_new_confirmation_time_us=fresh,
                      incident_first_path=first_path if detected else None, raw_fault_support_samples=support_raw,
                      evaluated_samples=n, fault_support_samples=support_n, pre_fault_samples=pre_n,
                      pre_fault_raw_alarm_samples=pre_raw, pre_fault_confirmed_alarm_samples=pre_confirm,
                      pre_fault_raw_fpr=ratio(pre_raw, pre_n), pre_fault_fpr=ratio(pre_confirm, pre_n),
                      precision=ratio(counts["tp"], counts["tp"] + counts["fp"]), recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
                      f1=ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
                      synthetic_sample_fpr=ratio(counts["fp"], counts["fp"] + counts["tn"]),
                      **{f"deadline_le_{d}s": bool(detected and delay <= d) for d in DEADLINES})
    return result


def health_fields(metrics):
    return dict(healthy_ready_samples=metrics["ready_samples"], healthy_confirmed_fpr=metrics["confirmed_fpr"],
                healthy_episodes_per_minute=metrics["confirmed_episodes_per_minute"],
                healthy_confirmed_alarm_samples=metrics["confirmed_alarm_samples"], healthy_confirmed_episodes=metrics["confirmed_episodes"],
                healthy_observed_ready_seconds=metrics["observed_ready_seconds"])


def policy_from(row):
    return dict(mode=row["mode"], **{k: float(row[k]) if row[k] != "" and row[k] is not None else None
                                    for k in (*PARAMS, "threshold_high_v", "threshold_low_v", "threshold_slope_v_per_s")})


def safe(candidate, reference):
    return all(candidate[k] is not None and reference[k] is not None and candidate[k] <= reference[k]
               for k in ("healthy_confirmed_fpr", "healthy_episodes_per_minute"))


def parameter_tuple(policy):
    return tuple(policy[k] for k in PARAMS)


def refined_stats(records):
    metrics = aggregate(records)
    a = aggregate([r for r in records if r["fault_family"] in ("increased_resistance", "sudden_drop") and r["severity"] == "severe"])
    g = aggregate([r for r in records if r["fault_family"] == "progressive" and r["observability"] in ("MEDIUM", "HIGH")])
    return dict(metrics, abrupt_severe_le_1s_rate=a["deadline_le_1s_rate"], abrupt_severe_applicable=a["applicable_scenarios"],
                abrupt_severe_successes=a["deadline_le_1s_successes"], gradual_observable_le_5s_rate=g["deadline_le_5s_rate"],
                gradual_observable_applicable=g["applicable_scenarios"], gradual_observable_successes=g["deadline_le_5s_successes"])


def refinement_rank(record):
    return (*[-record[k] if record[k] is not None else math.inf for k in RANK[:4]],
            *[record[k] if record[k] is not None else math.inf for k in RANK[4:]], *parameter_tuple(record))


def verify_synthetic(rows, meta, original):
    for row in rows:
        r = row["original"]
        b = original[r["source_record"]]
        for key in ("predicted_voltage_sag", "V_ref", "reference_ready", "Curr", "dI_dt"):
            assert r[key] == b[key], (meta["scenario_id"], key)
        if row["ready"]:
            close(r["voltage_sag_actual"], float(b["voltage_sag_actual"]) + float(r["added_sag_v"]))


def scenario_result(rows, meta, policy, *, check=False):
    if meta["fault_applicable"] == "False":
        assert not meta["scores_csv"]
        for key in ("tp", "fp", "tn", "fn", "delay_seconds", "confirmed_b_only_samples", "incident_first_path"):
            if key in meta:
                assert meta[key] == ""
        value = dict(applicable=False)
    else:
        value = simulate(rows, policy, meta, check=check)
        verify_metrics(meta, value)
        value["applicable"] = True
    value.update({k: meta[k] for k in ("fault_family", "severity", "observability")})
    return value


def main():
    summary = load(ROOT / "artifacts/sprint9/dual_result.json")
    output = Path(summary["output_directory"])
    manifest = load(output / "manifest.json")
    for name, digest in manifest["outputs"].items():
        path = (output / name).resolve()
        assert path.is_relative_to(output) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    freeze = load(output / "all_policies_frozen.json")
    for name, digest in freeze["files"].items():
        assert manifest["outputs"][name] == digest
    protocol = load(output / "experiment_protocol.json")
    windows = protocol["dual_config"]["slope_windows_seconds"]
    selections = load(output / "selected_per_fold.json")
    grid = table(output / "validation_healthy_grid.csv")
    refined = table(output / "validation_refined_candidates.csv")
    val_cases = table(output / "validation_fault_per_candidate.csv")
    val_manifest = table(output / "validation_fault_manifest.csv")
    outer_cases = table(output / "fault_per_scenario.csv")
    healthy = table(output / "healthy_active_per_fold.csv")
    anomaly = table(output / "anomaly_per_fold.csv")
    recomputed_outer, per_fold_health, per_fold_syn = defaultdict(list), {}, {}
    exported = replayed = slope_values = 0
    for entry in selections:
        fold, split = entry["fold_id"], entry["split"]
        train, val, test = (set(split[k]) for k in ("train", "validation", "test"))
        assert len(train) == 4 and not (train & val or train & test or val & test)
        folder = output / "folds" / fold
        validation = score_input(folder / "healthy_validation_scores.csv.gz", "validation")
        assert {r["original"]["NamaFile"] for r in validation} == val
        diagnostics(validation, windows)
        exported += len(validation)
        slope_values += sum(yes(r["original"][slope_names(w)[2]]) for r in validation for w in windows)
        reference = entry["reference"]
        reference_health = health_fields(simulate(validation, reference, check=True))
        replayed += len(validation)
        residuals = load(folder / "residual_thresholds.json")
        slope_payloads = load(folder / "slope_thresholds.json")
        for p, payload in residuals.items():
            ready = [r for r in validation if r["ready"]]
            scores = sorted(r["score"] for r in ready)
            rank = float(p) / 100 * (len(scores) - 1)
            lo, hi = math.floor(rank), math.ceil(rank)
            close(payload["threshold_v"], scores[lo] + (rank - lo) * (scores[hi] - scores[lo]))
            fingerprint = [{"NamaFile": r["original"]["NamaFile"], "Inst": int(r["original"]["Inst"]), "TimeUS": r["time"],
                            "source_record": int(r["original"]["source_record"]), "anomaly_score": r["score"]} for r in ready]
            assert payload["source_samples"] == len(ready) and set(payload["source_flights"]) == val
            assert payload["source_sha256"] == hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        slopes = {}
        for payload in slope_payloads:
            w, p = payload["window_seconds"], payload["percentile"]
            name, _, flag = slope_names(w)
            ready = [r for r in validation if r["ready"] and yes(r["original"][flag])]
            values = sorted(float(r["original"][name]) for r in ready)
            if values:
                rank = p / 100 * (len(values) - 1)
                lo, hi = math.floor(rank), math.ceil(rank)
                close(payload["threshold_v_per_s"], values[lo] + (rank - lo) * (values[hi] - values[lo]))
            else:
                assert payload["threshold_v_per_s"] is None
            fingerprint = [{"NamaFile": r["original"]["NamaFile"], "Inst": int(r["original"]["Inst"]), "TimeUS": r["time"],
                            "source_record": int(r["original"]["source_record"]), name: float(r["original"][name])} for r in ready]
            assert payload["source_samples"] == len(ready) and set(payload["source_flights"]) == val and payload["source_role"] == "validation"
            assert payload["source_sha256"] == hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            slopes[(w, p)] = payload["threshold_v_per_s"]
        path_as = load(folder / "path_a_selection.json")
        a_candidates = []
        for r in path_as["candidates"]:
            policy = dict(reference, high_percentile=r["high_percentile"], threshold_high_v=residuals[format(r["high_percentile"], "g")]["threshold_v"])
            computed = health_fields(simulate(validation, policy))
            for k, v in computed.items():
                close(r[k], v)
            replayed += len(validation)
            if safe(computed, reference_health):
                a_candidates.append(dict(computed, high_percentile=r["high_percentile"]))
        best_a = min(a_candidates, key=lambda r: (r["healthy_confirmed_fpr"], r["healthy_episodes_per_minute"], r["high_percentile"])) if a_candidates else None
        assert entry["selected_path_a_high_percentile"] == (best_a["high_percentile"] if best_a else None)
        candidates = [r for r in grid if r["fold_id"] == fold]
        assert len(candidates) == 144
        computed = {}
        eligible = []
        for r in candidates:
            assert r["source_role"] == "validation" and r["validation_flight"] in val
            p = policy_from(r)
            close(p["threshold_high_v"], residuals[format(p["high_percentile"], "g")]["threshold_v"])
            close(p["threshold_low_v"], residuals[format(p["low_percentile"], "g")]["threshold_v"])
            close(p["threshold_slope_v_per_s"], slopes[(p["slope_window_seconds"], p["slope_percentile"])])
            h = health_fields(simulate(validation, p)) if p["threshold_slope_v_per_s"] is not None else dict.fromkeys(reference_health)
            for key, value in h.items():
                close(r[key], value)
            replayed += len(validation) if p["threshold_slope_v_per_s"] is not None else 0
            good = safe(h, reference_health) and p["threshold_slope_v_per_s"] is not None and best_a is not None and p["high_percentile"] == best_a["high_percentile"]
            assert yes(r["healthy_eligible"]) == good
            computed[r["candidate_id"]] = dict(p, **h, records=[])
            if good:
                eligible.append((r["candidate_id"], computed[r["candidate_id"]]))
        eligible.sort(key=lambda pair: (pair[1]["healthy_confirmed_fpr"], pair[1]["healthy_episodes_per_minute"], parameter_tuple(pair[1])))
        shortlisted = eligible[:protocol["dual_config"]["refinement_max_candidates"]]
        assert entry["shortlisted_ids"] == [key for key, _ in shortlisted]
        for r in candidates:
            assert yes(r["shortlisted"]) == (r["candidate_id"] in entry["shortlisted_ids"])
            close(r["shortlist_rank"], next((i + 1 for i, (key, _) in enumerate(shortlisted) if key == r["candidate_id"]), None))
        base = {r["original"]["source_record"]: r["original"] for r in validation}
        meta_rows = [r for r in val_manifest if r["fold_id"] == fold]
        assert len(meta_rows) == 27
        for meta in meta_rows:
            rows = score_input(output / meta["scores_csv"], "validation", synthetic=True) if meta["scores_csv"] else []
            exported += len(rows)
            if rows:
                diagnostics(rows, windows)
                verify_synthetic(rows, meta, base)
                simulate(rows, reference, meta, check=True)
                replayed += len(rows)
            records = [r for r in val_cases if r["fold_id"] == fold and r["scenario_id"] == meta["scenario_id"]]
            assert {r["candidate_id"] for r in records} == set(entry["shortlisted_ids"])
            for r in records:
                assert r["source_role"] == "validation" and r["used_for_threshold"] == "False"
                c = computed[r["candidate_id"]]
                value = scenario_result(rows, r, c)
                c["records"].append(value)
                replayed += len(rows)
        refined_candidates = []
        for key, c in shortlisted:
            metrics = refined_stats(c["records"])
            c.update(metrics)
            record = next(r for r in refined if r["fold_id"] == fold and r["candidate_id"] == key)
            verify_metrics(record, metrics)
            if metrics["applicable_scenarios"]:
                refined_candidates.append((key, c))
        if refined_candidates:
            best_id, best = min(refined_candidates, key=lambda pair: refinement_rank(pair[1]))
            assert entry["selected_policy"]["mode"] == "DUAL" and entry["selected_policy"]["candidate_id"] == best_id
            assert parameter_tuple(entry["selected_policy"]) == parameter_tuple(best)
        else:
            assert entry["selected_policy"]["mode"] == "SINGLE_FALLBACK"
            close(entry["selected_policy"]["threshold_high_v"], reference["threshold_high_v"])
        for policy, payload in (("reference", reference), ("selected", entry["selected_policy"])):
            original = score_input(folder / policy / "healthy_test_scores.csv.gz", "test")
            assert {r["original"]["NamaFile"] for r in original} == test
            diagnostics(original, windows)
            h = simulate(original, payload, check=True)
            verify_metrics(next(r for r in healthy if r["fold_id"] == fold and r["policy"] == policy), h)
            per_fold_health[(fold, policy)] = h
            exported += len(original)
            replayed += len(original)
            base = {r["original"]["source_record"]: r["original"] for r in original}
            local = []
            for meta in [r for r in outer_cases if r["fold_id"] == fold and r["policy"] == policy]:
                assert meta["used_for_policy_selection"] == meta["used_for_threshold"] == "False" and meta["source_role"] == "test"
                rows = score_input(output / meta["scores_csv"], "test", synthetic=True) if meta["scores_csv"] else []
                exported += len(rows)
                if rows:
                    diagnostics(rows, windows)
                    verify_synthetic(rows, meta, base)
                value = scenario_result(rows, meta, payload, check=True)
                local.append(value)
                replayed += len(rows)
            per_fold_syn[(fold, policy)] = refined_stats(local)
            verify_metrics(next(r for r in anomaly if r["fold_id"] == fold and r["policy"] == policy), per_fold_syn[(fold, policy)])
            recomputed_outer[policy].extend(local)
    results = load(output / "results.json")
    for policy in ("reference", "selected"):
        computed = aggregate(recomputed_outer[policy])
        for k, v in computed.items():
            close(results["policies"][policy]["pooled"]["synthetic"][k], v)
        health = [per_fold_health[(entry["fold_id"], policy)] for entry in selections]
        pooled_h = results["policies"][policy]["pooled"]["healthy_active"]
        total_n = sum(r["ready_samples"] for r in health)
        total_duration = math.fsum(r["observed_ready_seconds"] for r in health)
        close(pooled_h["confirmed_fpr"], sum(r["confirmed_alarm_samples"] for r in health) / total_n)
        close(pooled_h["confirmed_episodes_per_minute"], sum(r["confirmed_episodes"] for r in health) * 60 / total_duration)
        for scope, data in (("healthy_active", per_fold_health), ("synthetic", per_fold_syn)):
            for metric, stats in results["policies"][policy]["macro_per_fold"][scope].items():
                values = [data[(e["fold_id"], policy)][metric] for e in selections if data[(e["fold_id"], policy)][metric] is not None]
                assert stats["n_defined"] == len(values)
                for key, value in dict(mean=mean(values) if values else None, median=median(values) if values else None,
                                       std=pstdev(values) if values else None, min=min(values) if values else None, max=max(values) if values else None).items():
                    close(stats[key], value)
    decision = load(output / "promotion_decision.json")
    gates = [("healthy_fpr", per_fold_health, "confirmed_fpr", "non_increase"),
             ("episodes_per_minute", per_fold_health, "confirmed_episodes_per_minute", "non_increase"),
             ("f1", per_fold_syn, "f1", "non_decrease"),
             *[(f"deadline_{d}s", per_fold_syn, f"deadline_le_{d}s_rate", "non_decrease") for d in DEADLINES],
             ("mean_incident_delay", per_fold_syn, "delay_mean_seconds", "strict_decrease")]
    all_pass = any(e["selected_policy"]["mode"] == "DUAL" for e in selections)
    for name, data, metric, comparison in gates:
        means = []
        defined = True
        for policy in ("reference", "selected"):
            values = [data[(e["fold_id"], policy)][metric] for e in selections]
            defined &= all(v is not None and math.isfinite(v) for v in values)
            means.append(mean(v for v in values if v is not None) if any(v is not None for v in values) else None)
        ref, sel = means
        passed = bool(defined and (sel <= ref if comparison == "non_increase" else sel >= ref if comparison == "non_decrease" else sel < ref))
        assert decision["checks"][name]["passed"] == passed and decision["checks"][name]["defined_all_folds"] == defined
        close(decision["checks"][name]["reference"], ref)
        close(decision["checks"][name]["selected"], sel)
        all_pass &= passed
    assert decision["gate_passed"] == all_pass
    assert decision["status"] == ("PROMOTED_RESEARCH_ONLY" if all_pass else "NOT_PROMOTED")
    source = load(ROOT / "artifacts/sprint6/cross_validation_result.json")
    assert summary["reference_pooled"] == dict(regression=source["regression_pooled"], healthy_active=source["healthy_active_pooled"], synthetic=source["synthetic_pooled"])
    failed = table(output / "failed_cases.csv")
    assert {(r["policy"], r["scenario_id"]) for r in failed} == {(r["policy"], r["scenario_id"]) for r in outer_cases if r["incident_detected"] != "True"}
    assert hashlib.sha256((ROOT / "data/raw/dataset_baterai.csv").read_bytes()).hexdigest() == summary["data_sha256"]
    report = dict(status="PASS", run_id=summary["run_id"], output_hashes_verified=len(manifest["outputs"]), pre_outer_hashes_verified=len(freeze["files"]),
                  candidate_pairs_verified=len(grid), refined_candidates_verified=len(refined), validation_candidate_scenarios_verified=len(val_cases),
                  outer_scenarios_verified=len(outer_cases), exported_score_rows_verified=exported, policy_row_decisions_replayed=replayed,
                  healthy_ready_slope_values_verified=slope_values, selected_fold_policies_independently_verified=len(selections),
                  promotion_status_independently_verified=decision["status"], reference_matches_sprint6=True,
                  project_code_imported=False, fitting_or_prediction_performed=False,
                  checks="hash/freeze/split, independent closed-form causal signed OLS/delta/gap/readiness, residual/slope healthy quantile fingerprints, separate A/B timers/OR union/attribution, A selection/144-pair screen/shortlist/refinement/ties/fallback, confusion/incident/NA/deadlines, pooled/macro and fixed promotion gate, complete failures/raw preservation")
    (ROOT / "artifacts/sprint9/independent_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
