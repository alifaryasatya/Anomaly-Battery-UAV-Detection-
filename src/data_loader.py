"""Read the combined battery CSV once, preserving source fields and row order."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

from .config import AppConfig
from .errors import DataError
from .feature_sets import BATTERY_COLUMNS


def load_battery_csv(path: Path) -> list[dict[str, str]]:
    """Validate CSV shape only. Numeric cleaning/segmentation belongs to Stage 2."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(BATTERY_COLUMNS):
                raise DataError(f"Expected battery-only columns {list(BATTERY_COLUMNS)}; got {reader.fieldnames}")
            rows = []
            for line, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise DataError(f"Malformed CSV record {line}: unexpected field count.")
                if not row["NamaFile"].strip():
                    raise DataError(f"CSV record {line}: NamaFile is empty.")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataError(f"Cannot read battery CSV {path}: {exc}") from exc
    if not rows:
        raise DataError("Battery CSV contains no telemetry rows.")
    return rows


def inspect_dataset(config: AppConfig) -> dict:
    rows = load_battery_csv(config.data_path)
    counts = Counter(row["NamaFile"] for row in rows)
    instances = defaultdict(set)
    for row in rows:
        instances[row["NamaFile"]].add(row["Inst"])
    return {
        "status": "OK", "command": "inspect", "project_root": str(config.root),
        "config_path": str(config.source), "data_path": str(config.data_path),
        "artifacts_root": str(config.artifacts_path),
        "data_sha256": hashlib.sha256(config.data_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "random_seed": config.seed, "gap_threshold_seconds": config.gap_threshold_seconds,
        "columns": list(BATTERY_COLUMNS), "rows": len(rows), "flights": len(counts),
        "per_flight": [{"NamaFile": flight, "samples": counts[flight],
                        "raw_instances": sorted(instances[flight]), "v2_load_label": "NOT_COMPUTED"}
                       for flight in sorted(counts)],
        "currtot_unit": config.as_dict()["data"]["currtot_unit"],
        "healthy_status": config.as_dict()["data"]["healthy_status"],
        "inspection_scope": "CSV schema, identity and counts; numerical audit/activity deferred to Stage 2",
        "model_training_performed": False,
    }
