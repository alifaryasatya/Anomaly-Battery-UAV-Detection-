"""Timestamp persistence, independent per stream, reset on false/unready/gap."""

from dataclasses import dataclass
import math

from .errors import DataError


@dataclass
class _Run:
    last_time_us: int
    segment_id: int
    start_time_us: int | None = None
    confirmed: bool = False


class TimePersistence:
    def __init__(self, duration_seconds: float, gap_threshold_seconds: float):
        if (not math.isfinite(duration_seconds) or duration_seconds <= 0
                or not math.isfinite(gap_threshold_seconds) or gap_threshold_seconds <= 0):
            raise DataError("Persistence duration and gap threshold must be finite and positive.")
        self.duration_seconds = duration_seconds
        self.gap_threshold_seconds = gap_threshold_seconds
        self.states: dict[tuple[str, int], _Run] = {}

    def consume(self, *, flight: str, instance: int, segment_id: int, time_us: int,
                raw: bool, ready: bool = True) -> dict:
        if not isinstance(flight, str) or not flight.strip():
            raise DataError("Persistence requires a flight ID.")
        if any(type(v) is not int or v < 0 for v in (instance, segment_id, time_us)):
            raise DataError("Persistence instance, segment, time must be nonnegative integers.")
        if type(raw) is not bool or type(ready) is not bool:
            raise DataError("Persistence raw/readiness flags must be Boolean.")
        key = flight, instance
        before = self.states.get(key)
        if before and (time_us <= before.last_time_us or segment_id < before.segment_id):
            raise DataError("Persistence input must be ordered by time/segment within each stream.")
        boundary = (before is None or before.segment_id != segment_id
                    or (time_us - before.last_time_us) / 1e6 > self.gap_threshold_seconds)
        run = _Run(time_us, segment_id) if boundary else before
        previous_confirmed = run.confirmed
        effective_raw = raw and ready
        if effective_raw:
            if run.start_time_us is None:
                run.start_time_us = time_us
            duration = (time_us - run.start_time_us) / 1e6
            confirmed = duration >= self.duration_seconds
        else:
            run.start_time_us = None
            duration, confirmed = 0.0, False
        run.last_time_us, run.segment_id, run.confirmed = time_us, segment_id, confirmed
        self.states[key] = run
        return dict(anomaly_raw=effective_raw, anomaly_duration_seconds=duration, anomaly_confirmed=confirmed,
                    confirmed_episode_start=confirmed and not previous_confirmed,
                    detector_state="NOT_READY" if not ready else "WARNING" if confirmed else "SUSPECT" if effective_raw else "NORMAL")
