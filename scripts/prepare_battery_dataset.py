"""Export battery-only historical data while preserving CSV field strings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"]
ACTIVE = {"00000066.BIN", "00000069.BIN", "00000070.BIN", "00000071.BIN", "00000073.BIN", "00000074.BIN"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    summary = args.summary.resolve()
    source_bytes = source.read_bytes()
    summary_bytes = summary.read_bytes()
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))
    if not set(COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError("Source schema is missing required battery columns.")
    rows = [{column: row[column] for column in COLUMNS} for row in reader]
    labels = {row["NamaFile"]: row["activity_label"] for row in csv.DictReader(io.StringIO(summary_bytes.decode("utf-8-sig")))}
    counts = Counter(row["NamaFile"] for row in rows)
    if {flight for flight, label in labels.items() if label == "ACTIVE_FLIGHT"} != ACTIVE:
        raise ValueError("Historical active labels differ from expected six; inspect source metadata.")
    if not ACTIVE.issubset(counts):
        raise ValueError("An expected historical active flight is missing.")
    outputs: dict[Path, bytes] = {
        ROOT / "data/raw/dataset_baterai.csv": csv_bytes(rows, COLUMNS)
    }
    for flight in sorted(ACTIVE):
        subset = [row for row in rows if row["NamaFile"] == flight]
        outputs[ROOT / "data/active_healthy_flights" / f"{Path(flight).stem}.csv"] = csv_bytes(subset, COLUMNS)
    inventory = [
        {"NamaFile": flight, "samples": str(counts[flight]), "historical_v1_label": labels.get(flight, "UNKNOWN"), "v2_current_based_label": "NOT_COMPUTED", "healthy_status": "ASSUMED_FROM_SOURCE"}
        for flight in sorted(counts)
    ]
    outputs[ROOT / "data/metadata/flight_inventory.csv"] = csv_bytes(inventory, list(inventory[0]))
    manifest_path = ROOT / "data/metadata/source_manifest.json"
    collisions = [path for path in [*outputs, manifest_path] if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing exports: {collisions}")
    for target, payload in outputs.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    # Compare source field strings, not floats after CSV re-serialization.
    with (ROOT / "data/raw/dataset_baterai.csv").open(encoding="utf-8", newline="") as handle:
        if list(csv.DictReader(handle)) != rows:
            raise RuntimeError("Combined export field preservation failed.")
    for flight in sorted(ACTIVE):
        with (ROOT / "data/active_healthy_flights" / f"{Path(flight).stem}.csv").open(encoding="utf-8", newline="") as handle:
            if list(csv.DictReader(handle)) != [row for row in rows if row["NamaFile"] == flight]:
                raise RuntimeError(f"Per-flight preservation failed: {flight}")
    source_unchanged = source.read_bytes() == source_bytes and summary.read_bytes() == summary_bytes
    if not source_unchanged:
        raise RuntimeError("An input changed during export.")
    manifest = {
        "source_dataset": source.name,
        "source_sha256_before": sha(source_bytes),
        "source_sha256_after": sha(source.read_bytes()),
        "historical_summary_sha256": sha(summary_bytes),
        "source_unchanged": source_unchanged,
        "rows": len(rows), "flights": len(counts),
        "columns": COLUMNS, "removed_column": "ThO",
        "field_strings_and_row_order_preserved": True,
        "historical_active_flights": sorted(ACTIVE),
        "note": "V1 activity labels are provenance only. Recompute V2 load labels from current.",
        "outputs": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(payload)} for path, payload in outputs.items()],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "flights": len(counts), "active_copies": len(ACTIVE), "source_unchanged": source_unchanged, "output_root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
