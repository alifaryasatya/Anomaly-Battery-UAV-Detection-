"""Positive residual, healthy-validation threshold and single-path decisions."""

import hashlib
import math

import numpy as np

from .cross_validation import FlightSplit
from .errors import DataError
from .persistence import TimePersistence
from .provenance import require_healthy_source
from .reporting import format_json


def residual_score(actual: float, predicted: float) -> tuple[float, float]:
    if not math.isfinite(actual) or not math.isfinite(predicted):
        raise DataError("Residual inputs must be finite.")
    residual = float(actual - predicted)
    if not math.isfinite(residual):
        raise DataError("Nonfinite residual.")
    return residual, max(0.0, residual)


def calibrate_threshold(rows: list[dict], split: FlightSplit, *, percentile: float = 99.0,
                        healthy_status: str = "assumed_from_source") -> dict:
    split.validate()
    require_healthy_source(rows, healthy_status)
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise DataError("Threshold percentile must lie in (0,100].")
    if {r["NamaFile"] for r in rows} != set(split.validation):
        raise DataError("Threshold source must contain exactly the healthy validation flights.")
    if any(r.get("split_role") != "validation" or r.get("reference_ready") is not True for r in rows):
        raise DataError("Threshold calibration accepts reference-ready validation rows only.")
    scores = [r["anomaly_score"] for r in rows]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in scores):
        raise DataError("Threshold scores must be finite nonnegative positive residuals.")
    fingerprint = [{key: r[key] for key in ("NamaFile", "Inst", "TimeUS", "source_record", "anomaly_score")}
                   for r in rows]
    return dict(method="healthy_validation_percentile", percentile=float(percentile),
                quantile_method="numpy.percentile_linear", threshold_v=float(np.percentile(scores, percentile, method="linear")),
                source_role="validation", source_flights=list(split.validation), source_samples=len(rows),
                source_sha256=hashlib.sha256(format_json({"rows": fingerprint}).encode()).hexdigest(),
                healthy_status=healthy_status, score_definition="max(0, actual_voltage_sag - predicted_voltage_sag)",
                comparison="score > threshold")


def apply_single_path(rows: list[dict], *, threshold_v: float, duration_seconds: float,
                      gap_threshold_seconds: float) -> list[dict]:
    if not math.isfinite(threshold_v) or threshold_v < 0:
        raise DataError("Single-path threshold must be finite and nonnegative.")
    timer = TimePersistence(duration_seconds, gap_threshold_seconds)
    result = []
    for row in rows:
        ready = row["reference_ready"]
        if type(ready) is not bool:
            raise DataError("reference_ready must be Boolean.")
        score = row.get("anomaly_score")
        if ready and (type(score) not in (float, int) or not math.isfinite(score) or score < 0):
            raise DataError("Ready detector rows require finite nonnegative scores.")
        decision = timer.consume(flight=row["NamaFile"], instance=row["Inst"], segment_id=row["segment_id"],
                                 time_us=row["TimeUS"], raw=bool(ready and score > threshold_v), ready=ready)
        result.append({**row, **decision})
    return result
