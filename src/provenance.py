"""Guards for healthy-source fitting/calibration, shared across model stages."""

from .errors import DataError

SYNTHETIC_MARKERS = frozenset(("is_synthetic", "synthetic", "scenario_id", "fault_family", "fault_support",
                               "fault_injected", "added_sag", "added_sag_v", "fault_label", "is_fault"))


def require_healthy_source(rows: list[dict], healthy_status: str) -> None:
    if healthy_status != "assumed_from_source":
        raise DataError("Fitting/calibration requires explicit healthy status assumed_from_source.")
    if not rows:
        raise DataError("No healthy rows available for fitting/calibration.")
    for row in rows:
        if SYNTHETIC_MARKERS.intersection(row):
            raise DataError("Synthetic/scenario rows cannot enter healthy fitting or threshold calibration.")
