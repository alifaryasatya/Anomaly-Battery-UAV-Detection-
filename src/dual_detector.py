"""Independent A/B persistence and union attribution; validation-only selection."""

from itertools import product
import hashlib
import math

import numpy as np

from .errors import DataError
from .persistence import TimePersistence
from .provenance import require_healthy_source
from .reporting import format_json
from .residual_diagnostics import slope_columns
from .single_path_tuning import RANK_FIELDS, healthy_eligible

PARAMETERS = ("high_percentile", "low_percentile", "slope_percentile", "slope_window_seconds", "path_b_persistence_seconds")
PATH_COLUMNS = ("path_a_raw", "path_b_raw", "path_a_duration_seconds", "path_b_duration_seconds",
                "path_a_confirmed", "path_b_confirmed", "path_a_episode_start", "path_b_episode_start",
                "alarm_attribution", "first_confirmation_path")


def parameter_tuple(row):
    return tuple(row[k] for k in PARAMETERS)


def candidate_grid(config):
    cfg = config.as_dict()["dual_path"]
    names = ("high_percentiles", "low_percentiles", "slope_percentiles", "slope_windows_seconds", "path_b_persistence_seconds")
    return [dict(zip(PARAMETERS, values), candidate_id=f"dual_{i:03d}") for i, values in enumerate(product(*(sorted(cfg[k]) for k in names)))]


def calibrate_slopes(rows, split, windows, percentiles):
    require_healthy_source(rows, "assumed_from_source")
    if {r["NamaFile"] for r in rows} != set(split.validation) or any(r["split_role"] != "validation" for r in rows):
        raise DataError("Slope calibration accepts assigned healthy validation only.")
    payloads = {}
    for w in windows:
        name, _, flag = slope_columns(w)
        ready = [r for r in rows if r["reference_ready"] and r[flag]]
        values = [r[name] for r in ready]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise DataError("Slope calibration requires finite ready slopes.")
        fingerprint = [{k: r[k] for k in ("NamaFile", "Inst", "TimeUS", "source_record", name)} for r in ready]
        for percentile in percentiles:
            payloads[(w, percentile)] = dict(window_seconds=w, percentile=percentile,
                threshold_v_per_s=float(np.percentile(values, percentile, method="linear")) if values else None,
                source_role="validation", source_flights=list(split.validation), source_samples=len(values),
                source_sha256=hashlib.sha256(format_json({"rows": fingerprint}).encode()).hexdigest(),
                definition="OLS signed residual V/s in (t-window,t], >=2 timestamps; reset gap/segment/unready",
                quantile_method="numpy.percentile_linear", comparison="slope > threshold")
    return payloads


def apply_dual(rows, policy, gap_seconds):
    dual = policy["mode"] == "DUAL"
    if policy["mode"] not in ("DUAL", "SINGLE_FALLBACK", "REFERENCE"):
        raise DataError("Unknown dual-path policy mode.")
    high = policy["threshold_high_v"]
    if not math.isfinite(high) or high < 0:
        raise DataError("Invalid Path A threshold.")
    if dual and (not math.isfinite(policy["threshold_low_v"]) or policy["threshold_low_v"] < 0
                 or policy["threshold_slope_v_per_s"] is None or not math.isfinite(policy["threshold_slope_v_per_s"])):
        raise DataError("Invalid Path B thresholds.")
    a = TimePersistence(.5, gap_seconds)
    b = TimePersistence(policy["path_b_persistence_seconds"] if dual else .5, gap_seconds)
    previous, result = {}, []
    for row in rows:
        ready = row["reference_ready"]
        score = row.get("anomaly_score")
        if type(ready) is not bool or (ready and (type(score) not in (float, int) or not math.isfinite(score) or score < 0)):
            raise DataError("Dual detector requires Boolean readiness and finite positive score.")
        raw_a = bool(ready and score > high)
        raw_b = False
        if dual and ready:
            slope, _, flag = slope_columns(policy["slope_window_seconds"])
            if row[flag] and (row[slope] is None or not math.isfinite(row[slope])):
                raise DataError("Ready Path B slope must be finite.")
            raw_b = bool(row[flag] and score > policy["threshold_low_v"] and row[slope] > policy["threshold_slope_v_per_s"])
        args = dict(flight=row["NamaFile"], instance=row["Inst"], segment_id=row["segment_id"], time_us=row["TimeUS"], ready=ready)
        da, db = a.consume(**args, raw=raw_a), b.consume(**args, raw=raw_b)
        ca, cb = da["anomaly_confirmed"], db["anomaly_confirmed"]
        key = row["NamaFile"], row["Inst"]
        before = previous.get(key)
        boundary = before is None or before[1] != row["segment_id"] or (row["TimeUS"] - before[0]) / 1e6 > gap_seconds
        confirmed = ca or cb
        episode = confirmed and (boundary or not before[2])
        attribution = "BOTH" if ca and cb else "A_ONLY" if ca else "B_ONLY" if cb else "NONE"
        result.append(dict(row, path_a_raw=raw_a, path_b_raw=raw_b, path_a_duration_seconds=da["anomaly_duration_seconds"],
                           path_b_duration_seconds=db["anomaly_duration_seconds"], path_a_confirmed=ca, path_b_confirmed=cb,
                           path_a_episode_start=da["confirmed_episode_start"], path_b_episode_start=db["confirmed_episode_start"],
                           alarm_attribution=attribution, first_confirmation_path=attribution if episode else None,
                           anomaly_raw=raw_a or raw_b, anomaly_confirmed=confirmed, confirmed_episode_start=episode,
                           anomaly_duration_seconds=max(da["anomaly_duration_seconds"], db["anomaly_duration_seconds"]),
                           detector_state="NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if raw_a or raw_b else "NORMAL"))
        previous[key] = (row["TimeUS"], row["segment_id"], confirmed)
    return result


def attribution_metrics(rows, metrics=None):
    result = {"confirmed_" + key.lower() + "_samples": sum(r["reference_ready"] and r["alarm_attribution"] == key for r in rows)
              for key in ("A_ONLY", "B_ONLY", "BOTH")}
    result.update(path_a_episode_starts=sum(r["path_a_episode_start"] for r in rows), path_b_episode_starts=sum(r["path_b_episode_start"] for r in rows))
    if metrics is not None and not metrics["fault_applicable"]:
        return dict.fromkeys((*result, "incident_first_path"))
    if metrics is not None:
        time = metrics["first_confirmation_time_us"]
        result["incident_first_path"] = next((r["first_confirmation_path"] for r in rows if r["TimeUS"] == time), None) if time is not None else None
    return result


def choose_path_a(rows, reference, validation_flight):
    if not rows or any(r.get("source_role") != "validation" or r.get("validation_flight") != validation_flight for r in rows):
        raise DataError("Path A selection requires assigned healthy validation.")
    good = [r for r in rows if healthy_eligible(r, reference)[0]]
    return min(good, key=lambda r: (r["healthy_confirmed_fpr"], r["healthy_episodes_per_minute"], r["high_percentile"])) if good else None


def shortlist(rows, reference, chosen_high, validation_flight, limit=3):
    if not 1 <= limit <= 3 or any(r.get("source_role") != "validation" or r.get("validation_flight") != validation_flight for r in rows):
        raise DataError("Shortlist requires validation-only metrics and fixed cap <=3.")
    good = []
    for r in rows:
        safe, reason = healthy_eligible(r, reference)
        if r["threshold_slope_v_per_s"] is None:
            safe, reason = False, "SLOPE_CALIBRATION_UNAVAILABLE"
        if chosen_high is None or r["high_percentile"] != chosen_high:
            safe, reason = False, "PATH_A_NOT_SELECTED"
        r.update(healthy_eligible=safe, eligibility_reason=reason, shortlisted=False, shortlist_rank=None, synthetic_refined=False)
        if safe:
            good.append(r)
    good.sort(key=lambda r: (r["healthy_confirmed_fpr"], r["healthy_episodes_per_minute"], parameter_tuple(r)))
    for i, r in enumerate(good[:limit]):
        r.update(shortlisted=True, shortlist_rank=i + 1)
    return good[:limit]


def choose_refined(rows, validation_flight):
    if any(r.get("source_role") != "validation" or r.get("validation_flight") != validation_flight
           or not r.get("healthy_eligible") or not r.get("shortlisted") or not r.get("synthetic_refined") for r in rows):
        raise DataError("Dual candidates require healthy screen AND completed synthetic validation refinement.")
    def key(r):
        return (*[-r[k] if r[k] is not None and math.isfinite(r[k]) else math.inf for k in RANK_FIELDS[:4]],
                *[r[k] if r[k] is not None and math.isfinite(r[k]) else math.inf for k in RANK_FIELDS[4:]], *parameter_tuple(r))
    return min(rows, key=key) if rows else None


def promotion_decision(policy_results, folds, dual_folds):
    """Predeclared macro paired gate; undefined in any fold cannot pass."""
    checks = {}
    comparisons = [("healthy_fpr", "healthy_active", "confirmed_fpr", "non_increase"),
                   ("episodes_per_minute", "healthy_active", "confirmed_episodes_per_minute", "non_increase"),
                   ("f1", "synthetic", "f1", "non_decrease"),
                   *[(f"deadline_{d}s", "synthetic", f"deadline_le_{d}s_rate", "non_decrease") for d in (1, 2, 5, 10, 30)],
                   ("mean_incident_delay", "synthetic", "delay_mean_seconds", "strict_decrease")]
    for name, scope, metric, mode in comparisons:
        stats = [policy_results[p]["macro_per_fold"][scope][metric] for p in ("reference", "selected")]
        defined = all(s["n_defined"] == folds and s["mean"] is not None and math.isfinite(s["mean"]) for s in stats)
        baseline, candidate = [s["mean"] for s in stats]
        passed = bool(defined and (candidate <= baseline if mode == "non_increase" else candidate >= baseline if mode == "non_decrease" else candidate < baseline))
        checks[name] = dict(reference=baseline, selected=candidate, mode=mode, defined_all_folds=defined, passed=passed)
    checks["refined_dual_candidate_exists"] = dict(dual_folds=dual_folds, passed=dual_folds > 0)
    passed = all(c["passed"] for c in checks.values())
    return dict(status="PROMOTED_RESEARCH_ONLY" if passed else "NOT_PROMOTED", gate_passed=passed,
                aggregation="arithmetic macro means over identical paired folds; all fold values required defined",
                checks=checks, operational_config_changed=False, operational_dual_path_enabled=False,
                interpretation="historical development research gate only; no hardware or deployment readiness")
