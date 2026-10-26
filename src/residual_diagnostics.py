"""Causal signed-residual deltas and timestamp OLS; never regression predictors."""

from collections import deque
from dataclasses import dataclass, field
import math

from .errors import DataError
from .feature_sets import validate_windows, window_token


def slope_columns(window):
    token = window_token(window)
    return (f"residual_slope_{token}s", f"residual_slope_count_{token}s", f"residual_slope_ready_{token}s")


def diagnostic_columns(windows):
    return ("dI_dt", "residual_delta_v", "residual_delta_v_per_s", "residual_delta_ready",
            *(name for w in windows for name in slope_columns(w)))


@dataclass
class _State:
    last_time: int
    segment: int
    previous_residual: float | None = None
    history: dict = field(default_factory=dict)


class ResidualDiagnostics:
    def __init__(self, windows, gap_seconds):
        validate_windows(windows)
        if not math.isfinite(gap_seconds) or gap_seconds <= 0:
            raise DataError("Residual diagnostics require positive gap threshold.")
        self.windows, self.gap = list(windows), gap_seconds
        self.states = {}

    def consume(self, row):
        if not isinstance(row.get("NamaFile"), str) or not row["NamaFile"].strip():
            raise DataError("Residual diagnostic flight identity is required.")
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("Inst", "TimeUS", "segment_id")):
            raise DataError("Residual diagnostics require integer time/instance/segment.")
        if type(row.get("reference_ready")) is not bool:
            raise DataError("Residual readiness must be Boolean.")
        key, time = (row["NamaFile"], row["Inst"]), row["TimeUS"]
        before = self.states.get(key)
        if before and (time <= before.last_time or row["segment_id"] < before.segment):
            raise DataError("Residual timestamps/segments must increase within stream.")
        boundary = before is None or row["segment_id"] != before.segment or (time - before.last_time) / 1e6 > self.gap
        state = _State(time, row["segment_id"], history={w: deque() for w in self.windows}) if boundary else before
        ready = row["reference_ready"]
        residual = row.get("residual_signed_v")
        if ready and (type(residual) not in (int, float) or not math.isfinite(residual)):
            raise DataError("Ready diagnostics require finite signed residual.")
        delta_ready = ready and state.previous_residual is not None and not boundary
        delta = float(residual - state.previous_residual) if delta_ready else None
        result = dict(row, residual_delta_v=delta, residual_delta_v_per_s=delta / ((time - state.last_time) / 1e6) if delta_ready else None,
                      residual_delta_ready=delta_ready)
        for w, queue in state.history.items():
            name, count, flag = slope_columns(w)
            if not ready:
                queue.clear()
            else:
                queue.append((time, residual))
                while queue and queue[0][0] <= time - w * 1e6:
                    queue.popleft()
            valid = len(queue) >= 2
            slope = None
            if valid:
                x = [(t - time) / 1e6 for t, _ in queue]  # local origin avoids large absolute-time cancellation
                y = [r for _, r in queue]
                x_mean, y_mean = math.fsum(x) / len(x), math.fsum(y) / len(y)
                denominator = math.fsum((v - x_mean) ** 2 for v in x)
                valid = denominator > 0
                slope = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator if valid else None
            result.update({name: slope, count: len(queue), flag: valid})
        state.last_time, state.segment, state.previous_residual = time, row["segment_id"], residual if ready else None
        self.states[key] = state
        return result


def diagnose(rows, windows, gap_seconds):
    builder = ResidualDiagnostics(windows, gap_seconds)
    return [builder.consume(r) for r in rows]
