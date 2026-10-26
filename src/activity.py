"""Current-only activity: causal row flags and retrospective audit of valid runs."""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby, product
import math

import numpy as np


SENSITIVITY_CURRENTS_A = (0.5, 1.0, 2.0)
SENSITIVITY_DURATIONS_SECONDS = (0.5, 1.0, 2.0)


def classify_activity(rows: list[dict], *, min_current_a: float,
                      min_duration_seconds: float, min_active_time_percentage: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Return annotated copies, per-flight audit, and all candidate active runs.

    Input is cleaned and sorted by flight/instance/segment/time. Row state is
    causal: confirmation begins only after the minimum duration has elapsed.
    The flight audit counts full duration of qualifying runs retrospectively.
    These offline flight labels/exposure are not inference features.
    """
    if not (math.isfinite(min_current_a) and min_current_a >= 0
            and math.isfinite(min_duration_seconds) and min_duration_seconds > 0
            and math.isfinite(min_active_time_percentage) and 0 <= min_active_time_percentage <= 100):
        raise ValueError("Invalid activity current/duration/percentage parameters.")
    annotated = []
    runs = []
    by_flight = defaultdict(list)
    observed_us = defaultdict(int)
    for key, iterator in groupby(rows, key=lambda r: (r["NamaFile"], r["Inst"], r["segment_id"])):
        block = list(iterator)
        observed_us[key[0]] += block[-1]["TimeUS"] - block[0]["TimeUS"]
        run = []
        run_id = -1

        def finish_run():
            if run:
                duration = (run[-1]["TimeUS"] - run[0]["TimeUS"]) / 1e6
                runs.append({"NamaFile": key[0], "Inst": key[1], "segment_id": key[2],
                             "active_run_id": run_id, "start_time_us": run[0]["TimeUS"],
                             "end_time_us": run[-1]["TimeUS"], "samples": len(run),
                             "duration_seconds": duration, "is_valid": duration >= min_duration_seconds})

        for row in block:
            item = dict(row)
            active = item["Curr"] >= min_current_a
            if active:
                if not run:
                    run_id += 1
                run.append(item)
                elapsed = (item["TimeUS"] - run[0]["TimeUS"]) / 1e6
            else:
                finish_run()
                run = []
                elapsed = 0.0
            item.update(active_by_current=active, active_run_id=run_id if active else None,
                        active_duration_seconds=elapsed,
                        active_confirmed=active and elapsed >= min_duration_seconds)
            annotated.append(item)
            by_flight[key[0]].append(item)
        finish_run()

    runs_by_flight = defaultdict(list)
    for run in runs:
        runs_by_flight[run["NamaFile"]].append(run)
    summaries = []
    for flight, records in sorted(by_flight.items()):
        flight_runs = runs_by_flight[flight]
        valid = [r for r in flight_runs if r["is_valid"]]
        active_us = sum(r["end_time_us"] - r["start_time_us"] for r in valid)
        observed = observed_us[flight] / 1e6
        active_duration = active_us / 1e6
        percentage = 100 * active_us / observed_us[flight] if observed > 0 else None
        current = np.array([r["Curr"] for r in records], dtype=float)
        eligible = observed > 0 and bool(valid) and percentage >= min_active_time_percentage
        summaries.append({
            "NamaFile": flight, "samples": len(records),
            "min_current_a": min_current_a, "min_duration_seconds": min_duration_seconds,
            "min_active_time_percentage": min_active_time_percentage,
            "observed_duration_seconds": observed, "valid_active_duration_seconds": active_duration,
            "active_time_percentage": percentage, "active_samples": sum(r["active_by_current"] for r in records),
            "active_sample_percentage": 100 * sum(r["active_by_current"] for r in records) / len(records),
            "causal_confirmed_samples": sum(r["active_confirmed"] for r in records),
            "active_runs": len(flight_runs), "valid_active_runs": len(valid),
            "longest_valid_active_run_seconds": max((r["duration_seconds"] for r in valid), default=0.0),
            "current_min_a": float(current.min()), "current_mean_a": float(current.mean()),
            "current_std_a": float(current.std(ddof=0)), "current_p05_a": float(np.quantile(current, .05)),
            "current_median_a": float(np.median(current)), "current_p95_a": float(np.quantile(current, .95)),
            "current_max_a": float(current.max()),
            "v2_current_based_label": "ACTIVE_LOAD" if eligible else "LOW_LOAD_OR_IDLE",
            "activity_eligible": eligible,
            "activity_reason": ("VALID_ACTIVE_EXPOSURE" if eligible else "NO_OBSERVED_EXPOSURE" if observed == 0
                                else "NO_VALID_ACTIVE_RUN" if not valid else "BELOW_ACTIVE_TIME_PERCENTAGE"),
        })
    return annotated, summaries, runs


def sensitivity_report(rows: list[dict], min_active_time_percentage: float) -> tuple[list[dict], list[dict]]:
    """Frozen nine-point diagnostic grid; never selects a detection threshold."""
    per_flight, summary = [], []
    for current, duration in product(SENSITIVITY_CURRENTS_A, SENSITIVITY_DURATIONS_SECONDS):
        _, flights, _ = classify_activity(rows, min_current_a=current, min_duration_seconds=duration,
                                         min_active_time_percentage=min_active_time_percentage)
        per_flight.extend(flights)
        active = [r["NamaFile"] for r in flights if r["activity_eligible"]]
        summary.append({"min_current_a": current, "min_duration_seconds": duration,
                        "min_active_time_percentage": min_active_time_percentage,
                        "active_load_flights": len(active), "active_flight_ids": "|".join(active)})
    return per_flight, summary
