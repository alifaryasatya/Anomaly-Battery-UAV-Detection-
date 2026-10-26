"""Causal raw-telemetry replay of frozen single-path reference detectors."""

from decimal import InvalidOperation
import math
from time import perf_counter_ns

import pandas as pd

from .config import AppConfig
from .detectors import residual_score
from .errors import DataError
from .feature_sets import BATTERY_COLUMNS, validate_regression_predictors
from .features import CausalFeatureBuilder
from .persistence import TimePersistence
from .preprocessing import _parse
from .residual_diagnostics import ResidualDiagnostics

TIMING_COLUMNS = ("parse_ns", "features_ns", "prediction_ns", "diagnostics_detector_ns", "end_to_end_ns")


class StreamingDetector:
    """Raw sample in, causal decision out; no offline activity labels.

    Interleaved flights have isolated state. Multiple instances in one flight,
    duplicate or backward source identities are rejected, never buffered/sorted.
    Invalid numerics are dropped without advancing feature/persistence state.
    Threadpool limits belong around the caller's replay session, not each row.
    """

    def __init__(self, bundle, config: AppConfig, *, prediction_cache=None, use_cache=False, clock=perf_counter_ns):
        if bundle.get("engine") != "battery_v2" or bundle.get("format_version") != 1:
            raise DataError("Streaming replay requires a frozen battery_v2 reference bundle.")
        if bundle["config"] != config.as_dict():
            raise DataError("Streaming bundle configuration differs from the replay configuration.")
        settings = config.as_dict()["features"]
        names = list(validate_regression_predictors(bundle["predictors"], settings["operational_set"], settings["rolling_windows_seconds"]))
        if settings["operational_set"] != "battery_d" or list(bundle["model"].feature_names_in_) != names:
            raise DataError("Streaming model must preserve the ordered Battery-D predictor registry.")
        threshold = bundle["threshold"]
        if (threshold["source_role"] != "validation" or threshold["percentile"] != 99
                or threshold["persistence_seconds"] != .5 or config.as_dict()["dual_path"]["enabled"]):
            raise DataError("Replay validates frozen P99/0.5 s single-path baseline; no unpromoted emergency candidate.")
        self.model, self.names, self.threshold = bundle["model"], names, threshold["threshold_v"]
        self.features = CausalFeatureBuilder(config)
        self.diagnostics = ResidualDiagnostics(config.as_dict()["dual_path"]["slope_windows_seconds"], config.gap_threshold_seconds)
        self.timer = TimePersistence(.5, config.gap_threshold_seconds)
        self.instances, self.last_seen, self.ordinal = {}, {}, 1
        self.cache = prediction_cache if prediction_cache is not None else {}
        self.use_cache, self.clock = use_cache, clock

    def consume(self, raw):
        start = self.clock()
        if not isinstance(raw, dict) or not set(BATTERY_COLUMNS).issubset(raw) or set(raw) - set(BATTERY_COLUMNS) - {"source_record"}:
            raise DataError("Streaming input accepts only raw battery telemetry and optional source_record.")
        if not isinstance(raw["NamaFile"], str) or not raw["NamaFile"].strip():
            raise DataError("Streaming flight identity must be nonempty.")
        self.ordinal += 1
        record = raw.get("source_record", self.ordinal)
        if type(record) is not int or record < 2:
            raise DataError("Streaming source_record must be an integer >=2.")
        row = dict(NamaFile=raw["NamaFile"], source_record=record)
        errors = []
        for name in ("TimeUS", "Inst", "Volt", "Curr", "CurrTot"):
            value = raw[name]
            try:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError("INVALID_TYPE")
                row[name] = _parse(str(value), name in ("TimeUS", "Inst"))
            except (InvalidOperation, ValueError, OverflowError) as exc:
                errors.append(name + ":" + str(exc))
        if "Inst" in row:
            instance = self.instances.setdefault(row["NamaFile"], row["Inst"])
            if instance != row["Inst"]:
                raise DataError("Multiple battery instances in a streaming flight require explicit selection.")
        if "Inst" in row and "TimeUS" in row:
            key = row["NamaFile"], row["Inst"]
            if key in self.last_seen and row["TimeUS"] <= self.last_seen[key]:
                raise DataError("Streaming timestamps must strictly increase; duplicates/backward rows cannot be sorted online.")
            self.last_seen[key] = row["TimeUS"]
        parsed = self.clock()
        if errors:
            return dict(accepted=False, source_record=record, NamaFile=row["NamaFile"], drop_reason="|".join(errors),
                        prediction_mode="DROPPED", parse_ns=parsed - start, features_ns=0, prediction_ns=0,
                        diagnostics_detector_ns=0, end_to_end_ns=parsed - start)
        item = self.features.consume(row)
        featured = self.clock()
        ready = item["reference_ready"]
        prediction, mode = None, "NOT_READY"
        if ready:
            values = tuple(float(item[n]) for n in self.names)
            cache_key = self.model, values  # retains model identity, never shares predictions across estimators
            if self.use_cache and cache_key in self.cache:
                prediction, mode = self.cache[cache_key], "EXACT_INPUT_CACHE"
            else:
                frame = pd.DataFrame([values], columns=self.names, dtype=float)
                prediction = float(self.model.predict(frame)[0])
                if not math.isfinite(prediction):
                    raise DataError("Streaming model prediction is nonfinite.")
                self.cache[cache_key] = prediction
                mode = "UNCACHED"
        predicted = self.clock()
        residual, score = residual_score(item["voltage_sag_actual"], prediction) if ready else (None, None)
        item.update(predicted_voltage_sag=prediction, residual_signed_v=residual, anomaly_score=score)
        item = self.diagnostics.consume(item)
        item.update(self.timer.consume(flight=item["NamaFile"], instance=item["Inst"], segment_id=item["segment_id"],
                                       time_us=item["TimeUS"], raw=bool(ready and score > self.threshold), ready=ready))
        ended = self.clock()
        return dict(item, accepted=True, drop_reason="", prediction_mode=mode,
                    parse_ns=parsed - start, features_ns=featured - parsed, prediction_ns=predicted - featured,
                    diagnostics_detector_ns=ended - predicted, end_to_end_ns=ended - start)


def replay(config: AppConfig) -> dict:
    from .replay_experiment import run_replay
    return run_replay(config)
