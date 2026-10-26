"""Guard provenance and raw diagnostics before any model can be evaluated."""

import json

import pytest

from scripts.run_sprint0 import ROOT, diagnose, frozen_write, sha


def row(time, instance="0", total="1", current="0"):
    return dict(NamaFile="flight", TimeUS=str(time), Inst=instance,
                Volt="24", Curr=current, CurrTot=total)


def test_instances_never_create_cross_stream_duplicates_or_resets():
    result = diagnose([row(0, "0", "10"), row(0, "1", "1"),
                       row(100000, "0", "11"), row(100000, "1", "2")], 0.5)
    assert result["instances"] == [0, 1]
    assert result["duplicate_timestamps"] == 0
    assert result["currtot_decreases"] == 0


def test_raw_discontinuities_are_reported_without_repair():
    rows = [row(700000, total="2"), row(0, total="3", current="-1"),
            row(0, total="3"), row("nan", total="inf")]
    before = [r.copy() for r in rows]
    result = diagnose(rows, 0.5)
    assert result["duplicate_timestamps"] == 1
    assert result["backwards_timestamps_in_source_order"] == 1
    assert result["currtot_decreases"] == 1
    assert result["gaps_over_threshold"] == 1
    assert result["Curr_negative_rows"] == 1
    assert result["invalid_timestamp_rows"] == 1
    assert result["missing_or_nonfinite"]["CurrTot"] == 1
    assert rows == before


def test_gap_boundary_and_invalid_instance():
    result = diagnose([row(0), row(500000), row(1000001), row(-1, "-1")], 0.5)
    assert result["gaps_over_threshold"] == 1
    assert result["invalid_timestamp_rows"] == 1
    assert result["invalid_instance_rows"] == 1


def test_frozen_artifacts_refuse_changes(tmp_path):
    target = tmp_path / "frozen.yaml"
    frozen_write(target, b"seed: 42\n")
    frozen_write(target, b"seed: 42\n")
    with pytest.raises(RuntimeError, match="Frozen artifact differs"):
        frozen_write(target, b"seed: 43\n")
    assert target.read_bytes() == b"seed: 42\n"


def test_supplied_exports_still_match_original_manifest():
    manifest = json.loads((ROOT / "data/metadata/source_manifest.json").read_text())
    assert all(sha(ROOT / entry["path"]) == entry["sha256"] for entry in manifest["outputs"])
