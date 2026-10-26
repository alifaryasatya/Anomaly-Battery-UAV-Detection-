"""Audit supplied V2 inputs and freeze the pre-model experiment contract.

Run with .venv/Scripts/python.exe scripts/run_sprint0.py from any directory.
No training, cleaning, activity classification, or access to V1 is performed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value: str | None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def diagnose(rows: list[dict[str, str]], gap_seconds: float) -> dict:
    """Summarize raw values without repairing or merging streams."""
    result = {"samples": len(rows)}
    result["missing_or_nonfinite"] = {
        column: sum(numeric(row[column]) is None for row in rows)
        for column in COLUMNS[1:]
    }
    result["missing_flight_ids"] = sum(not row["NamaFile"].strip() for row in rows)
    result["invalid_timestamp_rows"] = sum(
        numeric(row["TimeUS"]) is None
        or float(row["TimeUS"]) < 0
        or not float(row["TimeUS"]).is_integer()
        for row in rows
    )
    result["invalid_instance_rows"] = sum(
        numeric(row["Inst"]) is None
        or float(row["Inst"]) < 0
        or not float(row["Inst"]).is_integer()
        for row in rows
    )
    for column in ["Volt", "Curr", "CurrTot"]:
        values = [v for row in rows if (v := numeric(row[column])) is not None]
        result[f"{column}_min"] = min(values) if values else None
        result[f"{column}_max"] = max(values) if values else None
        result[f"{column}_negative_rows"] = sum(v < 0 for v in values)
    streams = defaultdict(list)
    for row in rows:
        if numeric(row["Inst"]) is not None:
            streams[float(row["Inst"])].append(row)
    result["instances"] = sorted(streams)
    result.update(duplicate_timestamps=0, backwards_timestamps_in_source_order=0,
                  currtot_decreases=0, gaps_over_threshold=0)
    result["gap_threshold_seconds"] = gap_seconds
    for stream in streams.values():
        timed = [(int(float(r["TimeUS"])), r) for r in stream
                 if numeric(r["TimeUS"]) is not None and float(r["TimeUS"]).is_integer()
                 and float(r["TimeUS"]) >= 0]
        result["duplicate_timestamps"] += sum(n - 1 for n in Counter(t for t, _ in timed).values())
        result["backwards_timestamps_in_source_order"] += sum(b[0] < a[0] for a, b in zip(timed, timed[1:]))
        timed.sort(key=lambda pair: pair[0])
        for (ta, a), (tb, b) in zip(timed, timed[1:]):
            result["gaps_over_threshold"] += (tb - ta) / 1e6 > gap_seconds
            av, bv = numeric(a["CurrTot"]), numeric(b["CurrTot"])
            if tb > ta and av is not None and bv is not None:
                result["currtot_decreases"] += bv < av
    return result


def frozen_write(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Frozen artifact differs: {path}. Create a new protocol version; do not overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    audit_dir = ROOT / "artifacts/audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["currtot_unit"] != "unverified":
        raise ValueError("CSV-only audit cannot verify CurrTot units.")
    if config["data"]["required_columns"] != COLUMNS:
        raise ValueError("Battery schema differs from Sprint 0 contract.")
    raw_path = ROOT / config["data"]["path"]
    manifest_path = ROOT / "data/metadata/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for output in manifest["outputs"]:
        path = ROOT / output["path"]
        actual = sha(path)
        checks.append({"path": output["path"], "expected_sha256": output["sha256"],
                       "actual_sha256": actual, "matches": actual == output["sha256"]})
    if not all(item["matches"] for item in checks):
        raise ValueError("Supplied file checksum mismatch; raw data must remain unchanged.")
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected schema: {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["NamaFile"]].append(row)
    if len(rows) != manifest["rows"] or len(groups) != manifest["flights"]:
        raise ValueError("Manifest row/flight counts mismatch.")
    with (ROOT / "data/metadata/flight_inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    if {r["NamaFile"]: int(r["samples"]) for r in inventory} != {k: len(v) for k, v in groups.items()}:
        raise ValueError("Historical inventory mismatch.")
    for flight in manifest["historical_active_flights"]:
        with (ROOT / "data/active_healthy_flights" / f"{Path(flight).stem}.csv").open(encoding="utf-8", newline="") as handle:
            if list(csv.DictReader(handle)) != groups[flight]:
                raise ValueError(f"Historical copy differs from combined raw input: {flight}")
    gap = config["time"]["max_gap_multiplier"] / config["time"]["nominal_frequency_hz"]
    labels = {r["NamaFile"]: r["historical_v1_label"] for r in inventory}
    flights = [{"NamaFile": flight, "historical_v1_label": labels[flight],
                "healthy_status": "ASSUMED_FROM_SOURCE", "v2_current_based_label": "NOT_COMPUTED",
                **diagnose(groups[flight], gap)} for flight in sorted(groups)]
    summary = {"rows": len(rows), "flights": len(groups), "columns": COLUMNS,
               "currtot_unit": "unverified", "unit_conversion_applied": False,
               "healthy_status": "ASSUMED_FROM_SOURCE", "checksums": checks,
               "historical_copies_equal_combined_subsets": True, "per_flight": flights}
    summary["totals"] = {key: sum(r[key] for r in flights) for key in [
        "invalid_timestamp_rows", "invalid_instance_rows", "duplicate_timestamps",
        "backwards_timestamps_in_source_order", "currtot_decreases", "gaps_over_threshold",
        "Volt_negative_rows", "Curr_negative_rows", "CurrTot_negative_rows", "missing_flight_ids"]}
    summary["totals"]["missing_or_nonfinite_cells"] = sum(sum(r["missing_or_nonfinite"].values()) for r in flights)
    summary["totals"]["multi_instance_flights"] = sum(len(r["instances"]) > 1 for r in flights)
    json_payload = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
    frozen_write(audit_dir / "data_audit.json", json_payload)
    stream = io.StringIO(newline="")
    fields = [key for key in flights[0] if key not in ["instances", "missing_or_nonfinite"]]
    fields += ["instances", "missing_or_nonfinite"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows([{**r, "instances": json.dumps(r["instances"]),
                       "missing_or_nonfinite": json.dumps(r["missing_or_nonfinite"])} for r in flights])
    frozen_write(audit_dir / "flight_diagnostics.csv", stream.getvalue().encode())
    frozen_write(audit_dir / "config_used.yaml", config_path.read_bytes())
    frozen_write(audit_dir / "experiment_protocol.md", (ROOT / "configs/experiment_protocol.md").read_bytes())
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, capture_output=True, text=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    frozen_write(ROOT / "requirements.lock.txt", freeze.replace("\r\n", "\n").encode())
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()))}
    frozen_write(audit_dir / "dependency_versions.json", (json.dumps(versions, indent=2) + "\n").encode())
    lines = ["# Sprint 0 — Audit data dan sumber", "", "Status: selesai; tidak ada model dilatih atau performa V2 diukur.", "",
             f"Input tunggal: `{config['data']['path']}`; **{len(rows):,} baris, {len(groups)} flight**.",
             f"Seluruh {len(checks)} checksum ekspor cocok dengan manifest. Enam salinan historis cocok persis dengan subset input dan tidak digabung sebagai sampel tambahan.", "",
             "## Unit, instance, dan asumsi", "",
             "`TimeUS` diperlakukan sebagai mikrodetik, `Volt` sebagai V dan `Curr` sebagai A sesuai kontrak CSV. Definisi firmware/log asli tidak disertakan; ini bukan verifikasi unit independen.",
             "`CurrTot` tetap **unverified**, tanpa konversi Ah/mAh. Monotonicity tidak membuktikan satuan. Battery-A menjadi pembanding tanpa CurrTot; Battery-B–E hanya boleh memakai nilai sumber dengan status unit tercatat. Penurunan CurrTot wajib ditinjau sebelum penggunaan fitur tersebut.",
             "Healthy adalah **ASSUMED_FROM_SOURCE**, bukan diagnosis fisik. Pack, jumlah sel, kapasitas, sensor, dan firmware tidak tersedia. Log asli dan sumber ekspor V1 tidak dibaca; hanya integritas paket V2 yang diverifikasi ulang.", "",
             "## Diagnosis per flight", "",
             "Duplicate dihitung dalam stream flight/instance. Penurunan CurrTot dihitung antar timestamp berbeda setelah pengurutan, sebagai kandidat reset/discontinuity, bukan diagnosis sebab reset.", "",
             "| Flight | Rows | Inst | Missing/nonfinite | Invalid time/instance | Duplicate time | Backward time | Curr < 0 | CurrTot < 0 | CurrTot turun | Gap > 0.5 s |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in flights:
        lines.append(f"| {r['NamaFile']} | {r['samples']} | {r['instances']} | {sum(r['missing_or_nonfinite'].values())} | {r['invalid_timestamp_rows']}/{r['invalid_instance_rows']} | {r['duplicate_timestamps']} | {r['backwards_timestamps_in_source_order']} | {r['Curr_negative_rows']} | {r['CurrTot_negative_rows']} | {r['currtot_decreases']} | {r['gaps_over_threshold']} |")
    lines += ["", "Detail rentang nilai, diagnosis dan checksum ada di `data_audit.json` dan `flight_diagnostics.csv`.", "",
              "## Keputusan sebelum implementasi", "",
              "- Input training nantinya hanya combined CSV; label historical active hanya provenance. Label active V2 belum dihitung (Tahap 2).",
              "- Multi-instance ditolak kecuali pilihan instance eksplisit. Sorting/cleaning dikerjakan pada salinan; raw tidak diubah.",
              "- Drop missing/nonfinite atau timestamp/instance invalid dengan laporan; tanpa imputasi. Duplicate timestamp ditolak untuk ditinjau. Arus negatif dilaporkan dan dipertahankan sebagai telemetry, tidak diubah ke absolut.",
              "- CurrTot menurun menjadi gate review sebelum Battery-B–E dilatih; tidak diubah atau dikonversi otomatis.",
              "- Protokol dan config dibekukan sebelum training; lihat `experiment_protocol.md`, `config_used.yaml`, dan `freeze_manifest.json`.",
              "- Hasil berikutnya disebut historical development cross-validation; tidak ada independent final holdout baru.", "",
              "## Batas interpretasi wajib", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    frozen_write(audit_dir / "data_audit.md", "\n".join(lines).encode("utf-8"))
    frozen_files = ["configs/config.yaml", "configs/experiment_protocol.md", "requirements.txt", "requirements.lock.txt", "tests/test_sprint0.py",
                    "scripts/run_sprint0.py", "data/metadata/source_manifest.json"]
    frozen_files += [item["path"] for item in manifest["outputs"]]
    frozen_files += [f"artifacts/audit/{name}" for name in ["config_used.yaml", "experiment_protocol.md", "data_audit.md", "data_audit.json", "flight_diagnostics.csv", "dependency_versions.json"]]
    # Planning Markdown is mutable for status updates; requirements live in the frozen protocol snapshot.
    frozen_write(audit_dir / "freeze_manifest.json", (json.dumps({"protocol_version": "sprint0-v1",
                 "stage": "pre_model_evaluation", "files": {p: sha(ROOT / p) for p in frozen_files}}, indent=2) + "\n").encode())
    print(json.dumps({"status": "SPRINT_0_COMPLETE", "rows": len(rows), "flights": len(groups),
                      "checksums_verified": len(checks), "currtot_unit": "unverified",
                      "model_training_performed": False, "artifacts": "artifacts/audit"}, indent=2))


if __name__ == "__main__":
    main()
