"""Independently verify every exported synthetic row without invoking a model.

Run after main.py inject-faults and recording its summary in
artifacts/sprint5/faults_result.json. Uses only Python's standard library.
"""

import csv
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def main():
    summary = json.loads((ROOT / "artifacts/sprint5/faults_result.json").read_text())
    output = Path(summary["output_directory"])
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, digest in manifest["outputs"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    assert hashlib.sha256((ROOT / "data/raw/dataset_baterai.csv").read_bytes()).hexdigest() == summary["data_sha256"]
    feature_dir = Path(summary["feature_directory"])
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    source_path = feature_dir / "battery_features.csv"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == feature_manifest["outputs"]["battery_features.csv"]
    source = {}
    for row in read_csv(source_path):
        source.setdefault(row["NamaFile"], {})[int(row["source_record"])] = row
    registry = json.loads((feature_dir / "feature_registry.json").read_text())
    predictor_names = registry["regression_sets"]["battery_e"]
    protocol = json.loads((output / "scenario_protocol.json").read_text())
    thresholds = protocol["config_faults"]["observability_max_sag_v"]
    records = read_csv(output / "fault_manifest.csv")
    assert len(records) == summary["scenarios"]
    checked_rows = checked_scenarios = not_applicable = 0
    for meta in records:
        if meta["fault_applicable"] != "True":
            assert meta["evaluation_status"] == meta["observability"] == "NOT_APPLICABLE"
            assert not meta["scenario_csv"] and not meta["max_added_sag_v"] and meta["reason"]
            not_applicable += 1
            continue
        originals = source[meta["NamaFile"]]
        onset, end = int(meta["onset_time_us"]), int(meta["fault_end_time_us"])
        assert Decimal(onset) >= Decimal(meta["requested_onset_time_us"])
        assert (end - onset) / 1e6 >= 1
        dose = float(meta["configured_dose"])
        added_values, currents, effect_times = [], [], []
        previous, first_post, seen_records = None, None, set()
        with gzip.open(ROOT / meta["scenario_csv"], "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                original = originals[int(row["source_record"])]
                seen_records.add(int(row["source_record"]))
                time = int(row["TimeUS"])
                inside = onset <= time <= end
                assert row["scenario_id"] == meta["scenario_id"] and row["is_synthetic"] == "True"
                assert (row["fault_support"] == "True") == inside and int(row["fault_label"]) == int(inside)
                for name in ("NamaFile", "Inst", "TimeUS", "reference_ready", "reference_status", "V_ref"):
                    assert row[name] == original[name], (meta["scenario_id"], name)
                for name in predictor_names:
                    close(row[name], original[name])
                close(row["Volt_healthy"], original["Volt"])
                added = 0.
                if inside:
                    assert row["reference_ready"] == "True" and row["segment_id"] == meta["selected_segment_id"]
                    elapsed = (time - onset) / 1e6
                    if meta["fault_family"] == "increased_resistance":
                        assert float(row["Curr"]) >= 1
                        added = float(row["Curr"]) * dose
                    elif meta["fault_family"] == "progressive":
                        assert float(row["Curr"]) >= 1
                        added = float(row["Curr"]) * dose * (time - onset) / (end - onset)
                    else:
                        ramp = float(meta["sudden_ramp_seconds"])
                        added = dose * min(1., elapsed / ramp) if ramp else dose
                    added_values.append(float(row["added_sag_v"]))
                    currents.append(float(row["Curr"]))
                    if added > 0:
                        effect_times.append(time)
                close(row["added_sag_v"], added)
                close(row["Volt"], float(original["Volt"]) - added)
                assert (row["fault_effect_present"] == "True") == (added > 0)
                phase = "FAULT_SUPPORT" if inside else "PRE_FAULT" if time < onset else "POST_FAULT_ORIGINAL"
                assert row["fault_phase"] == phase
                if row["reference_ready"] == "True":
                    close(row["voltage_sag_actual"], float(row["V_ref"]) - float(row["Volt"]))
                else:
                    assert row["voltage_sag_actual"] == ""
                derivative = 0. if previous is None or row["segment_id"] != previous["segment_id"] else (
                    (float(row["Volt"]) - float(previous["Volt"])) / ((time - int(previous["TimeUS"])) / 1e6))
                close(row["dV_dt"], derivative)
                if time > end and first_post is None:
                    first_post = time
                previous = row
                checked_rows += 1
        assert seen_records == set(originals)
        assert len(added_values) == int(meta["support_samples"])
        assert len(effect_times) == int(meta["effect_samples"])
        assert (int(meta["first_effect_time_us"]) if meta["first_effect_time_us"] else None) == (effect_times[0] if effect_times else None)
        assert (int(meta["post_fault_original_start_time_us"]) if meta["post_fault_original_start_time_us"] else None) == first_post
        close(meta["max_added_sag_v"], max(added_values))
        close(meta["mean_added_sag_v"], math.fsum(added_values) / len(added_values))
        close(meta["mean_current_after_fault_a"], math.fsum(currents) / len(currents))
        peak = max(added_values)
        label = "VERY_LOW" if peak < thresholds["very_low"] else "LOW" if peak < thresholds["low"] else "MEDIUM" if peak < thresholds["medium"] else "HIGH"
        assert meta["observability"] == label
        checked_scenarios += 1
    assert checked_rows == summary["synthetic_rows"]
    assert checked_scenarios == summary["applicable_scenarios"] and not_applicable == summary["not_applicable_scenarios"]
    evidence = dict(status="PASS", run_id=summary["run_id"], rows_verified=checked_rows,
                    applicable_scenarios_verified=checked_scenarios, not_applicable_metadata_verified=not_applicable,
                    artifact_hashes_verified=len(manifest["outputs"]),
                    checks=["per-row formulas and timestamp support", "all source rows/predictors/reference preserved",
                            "voltage/target/derivative including ending", "signal-effect vs support labels",
                            "metadata max/mean/current/onset/ending and observability", "NOT_APPLICABLE separated"])
    (ROOT / "artifacts/sprint5/scenario_oracle_verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
