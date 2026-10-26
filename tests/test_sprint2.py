"""Temporal/quality invariants, audit denominators and preprocessing integration."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import main
from src.activity import classify_activity, sensitivity_report
from src.config import PROJECT_ROOT, load_config
from src.errors import DataError
from src.preprocessing import clean_and_segment, preprocess


def row(time, current=2, *, flight="f", instance="0", voltage="24", total="0"):
    return dict(NamaFile=flight, TimeUS=str(time), Inst=str(instance), Volt=str(voltage),
                Curr=str(current), CurrTot=str(total))


def classify(source, *, current=1, duration=1, percentage=5):
    result = clean_and_segment(source, .5)
    assert result.errors == []
    return classify_activity(result.rows, min_current_a=current, min_duration_seconds=duration,
                             min_active_time_percentage=percentage)


def test_sort_drop_and_recompute_gaps_without_imputation():
    source = [row(600000), row(0), row(300000, voltage="nan"), row(900000, total="")]
    before = deepcopy(source)
    result = clean_and_segment(source, .5)
    assert not result.errors
    assert [r["TimeUS"] for r in result.rows] == [0, 600000]
    assert [r["segment_id"] for r in result.rows] == [0, 1]
    assert [r["dt_seconds"] for r in result.rows] == [0, 0]
    assert result.gaps[0]["observed_duration_seconds"] == 0
    assert result.gaps[0]["excluded_gap_seconds"] == .6
    assert len(result.dropped) == 2
    assert {e["reason"] for e in result.events} >= {"BACKWARD_IN_SOURCE_ORDER", "NONFINITE", "MISSING"}
    assert source == before


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("TimeUS", "-1", "INVALID_NONNEGATIVE_INTEGER"), ("TimeUS", ".5", "INVALID_NONNEGATIVE_INTEGER"),
    ("TimeUS", "NaN", "NONFINITE"), ("Inst", "-1", "INVALID_NONNEGATIVE_INTEGER"),
    ("Inst", "0.5", "INVALID_NONNEGATIVE_INTEGER"), ("Curr", "inf", "NONFINITE"),
    ("Volt", "bad", "NONNUMERIC"), ("CurrTot", "", "MISSING"), ("Curr", "1e1000", "NONFINITE"),
])
def test_invalid_numeric_rows_have_explicit_drop_reasons(field, value, reason):
    sample = row(0)
    sample[field] = value
    result = clean_and_segment([sample], .5)
    assert result.rows == []
    assert len(result.dropped) == 1
    assert result.dropped[0]["drop_reasons"] == f"{field}:{reason}"
    assert result.inventory[0]["dropped_samples"] == 1


def test_duplicate_identity_is_rejected_even_if_other_field_is_invalid():
    result = clean_and_segment([row(0), row("0.0", voltage="nan")], .5)
    assert result.rows == []
    assert any("Duplicate timestamp" in error for error in result.errors)
    assert any(e["reason"] == "DUPLICATE_TIMESTAMP" for e in result.events)


def test_multiple_instances_are_rejected_even_if_a_row_is_dropped():
    result = clean_and_segment([row(0, instance=0), row(0, instance=1, current="nan")], .5)
    assert result.rows == []
    assert any("Multiple battery instances" in error for error in result.errors)
    assert result.inventory[0]["instances"] == "0|1"


def test_flights_do_not_share_timestamp_or_temporal_state():
    result = clean_and_segment([row(0, flight="b"), row(400000, flight="a"), row(0, flight="a")], .5)
    assert not result.errors
    assert [(r["NamaFile"], r["dt_seconds"]) for r in result.rows] == [("a", 0), ("a", .4), ("b", 0)]


def test_gap_threshold_equality_and_segment_boundary_dt():
    result = clean_and_segment([row(0), row(500000), row(1000001)], .5)
    assert [r["segment_id"] for r in result.rows] == [0, 0, 1]
    assert [r["dt_seconds"] for r in result.rows] == [0, .5, 0]
    assert [r["gap_before_seconds"] for r in result.rows] == [0, 0, .500001]
    assert result.gaps[0]["gaps"] == 1
    assert result.gaps[0]["observed_duration_seconds"] == .5


def test_integer_timestamps_are_not_rounded_by_float_conversion():
    start = 2**53 + 1
    result = clean_and_segment([row(start), row(start + 100001)], .5)
    assert result.rows[0]["TimeUS"] == start
    assert result.rows[1]["dt_seconds"] == .100001


def test_negative_current_and_currtot_resets_are_reported_without_conversion():
    result = clean_and_segment([row(0, current=-2, total=4), row(100000, total=-1, voltage=-24)], .5)
    assert not result.errors
    assert result.rows[0]["Curr"] == -2
    assert result.rows[1]["CurrTot"] == -1
    assert result.rows[1]["Volt"] == -24
    assert result.gaps[0]["currtot_decreases"] == 1
    assert sum(e["reason"] == "NEGATIVE_VALUE" for e in result.events) == 3
    assert any(e["action"] == "RETAIN_REVIEW_BEFORE_BATTERY_B_TO_E" for e in result.events)


def test_current_and_duration_equality_use_actual_irregular_timestamps():
    annotated, flights, runs = classify([row(0, 1), row(400000, 1), row(900000, 1), row(1000000, 1)])
    assert [r["active_duration_seconds"] for r in annotated] == [0, .4, .9, 1]
    assert [r["active_confirmed"] for r in annotated] == [False, False, False, True]
    assert runs[0]["is_valid"]
    assert flights[0]["valid_active_duration_seconds"] == 1
    assert flights[0]["active_time_percentage"] == 100
    assert flights[0]["v2_current_based_label"] == "ACTIVE_LOAD"


def test_exposure_requires_both_active_endpoints_without_leading_or_trailing_time():
    _, flights, runs = classify([row(0, .9999), row(100000, 1), row(400000, 1), row(600000, 0)], duration=.3)
    assert flights[0]["observed_duration_seconds"] == .6
    assert flights[0]["valid_active_duration_seconds"] == .3
    assert flights[0]["active_time_percentage"] == 50
    assert runs[0]["start_time_us"] == 100000
    assert runs[0]["end_time_us"] == 400000


def test_short_runs_do_not_combine_across_gap_or_low_current():
    source = [row(0), row(400000), row(500000, 0), row(600000), row(1000000), row(2000000), row(2400000)]
    annotated, flights, runs = classify(source)
    assert len(runs) == 3
    assert not any(r["is_valid"] for r in runs)
    assert not any(r["active_confirmed"] for r in annotated)
    assert flights[0]["observed_duration_seconds"] == 1.4
    assert flights[0]["valid_active_duration_seconds"] == 0
    assert flights[0]["v2_current_based_label"] == "LOW_LOAD_OR_IDLE"


def test_flight_ratio_threshold_equality():
    _, flights, _ = classify([row(0), row(500000), row(1000000), row(1500000, 0), row(2000000, 0)], percentage=50)
    assert flights[0]["active_time_percentage"] == 50
    assert flights[0]["activity_eligible"]


def test_default_five_percent_boundary_uses_exact_microsecond_exposure():
    source = [row(t, 1 if t <= 1000000 else 0) for t in range(0, 20000001, 500000)]
    _, flights, _ = classify(source)
    assert flights[0]["active_time_percentage"] == 5
    assert flights[0]["activity_eligible"]


def test_zero_exposure_is_undefined_and_never_eligible():
    _, flights, _ = classify([row(0), row(10000000)], percentage=0)
    assert flights[0]["active_time_percentage"] is None
    assert not flights[0]["activity_eligible"]
    assert flights[0]["activity_reason"] == "NO_OBSERVED_EXPOSURE"


def test_future_samples_do_not_change_causal_rows():
    prefix = [row(0), row(400000), row(900000)]
    earlier = classify(prefix)[0]
    extended = classify(prefix + [row(1000000), row(1300000, current=30)])[0]
    assert earlier == extended[:len(prefix)]
    assert not any(r["active_confirmed"] for r in earlier)
    assert extended[3]["active_confirmed"]


def test_activity_depends_only_on_current_and_time_not_voltage_capacity_or_v1_label():
    source = [row(0), row(500000), row(1000000)]
    first = classify(source)[1:]
    for sample in source:
        sample.update(Volt="100", CurrTot="100000", historical_v1_label="IDLE_OR_GROUND")
    assert classify(source)[1:] == first


def test_sensitivity_is_the_frozen_nine_point_grid():
    clean = clean_and_segment([row(0, 1), row(500000, 1), row(1000000, 1)], .5)
    flights, summary = sensitivity_report(clean.rows, 5)
    assert len(flights) == len(summary) == 9
    assert {(r["min_current_a"], r["min_duration_seconds"]) for r in summary} == {
        (current, duration) for current in (.5, 1, 2) for duration in (.5, 1, 2)}
    assert all(r["active_load_flights"] == 0 for r in summary if r["min_current_a"] == 2 or r["min_duration_seconds"] == 2)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    values = yaml.safe_load((PROJECT_ROOT / "configs/config.yaml").read_text())
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")

    def prepare(source):
        with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
            writer.writeheader()
            writer.writerows(source)
        return load_config(root=root)
    return root, prepare


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pipeline_persists_zero_retained_flights_and_exclusions(workspace):
    root, prepare = workspace
    config = prepare([row(0, flight="empty", current="nan"), row(0, flight="single")])
    result = preprocess(config)
    output = Path(result["output_directory"])
    assert result["input_rows"] == 2 and result["retained_rows"] == 1 and result["dropped_rows"] == 1
    activity = read_csv(output / "battery_activity.csv")
    assert len(activity) == 2
    assert activity[0]["samples"] == "0" and activity[0]["active_time_percentage"] == ""
    assert activity[0]["activity_reason"] == "NO_RETAINED_SAMPLES"
    assert len(read_csv(output / "activity_sensitivity_per_flight.csv")) == 18
    assert result["activity_gate_passed"] is False
    assert not (root / "data/processed").exists()  # processed copies live in the run's artifact directory


@pytest.mark.parametrize("source", [[row(0), row(0)], [row(0, instance=0), row(0, instance=1)]])
def test_rejected_dataset_writes_diagnostics_without_processed_success(workspace, source):
    root, prepare = workspace
    config = prepare(source)
    raw = config.data_path.read_bytes()
    with pytest.raises(DataError, match="Dataset rejected"):
        preprocess(config)
    output = next((root / "artifacts/preprocessing").iterdir())
    assert json.loads((output / "summary.json").read_text())["status"] == "REJECTED"
    assert not (output / "battery_segmented.csv").exists()
    assert read_csv(output / "data_quality_events.csv")
    assert config.data_path.read_bytes() == raw


def test_historical_provenance_does_not_determine_activity(workspace):
    root, prepare = workspace
    config = prepare([row(0), row(500000), row(1000000)])
    metadata = root / "data/metadata"
    metadata.mkdir()
    (metadata / "flight_inventory.csv").write_text("NamaFile,historical_v1_label\nf,IDLE_OR_GROUND\n")
    result = preprocess(config)
    inventory = read_csv(Path(result["output_directory"]) / "flight_inventory.csv")
    assert inventory[0]["historical_v1_label"] == "IDLE_OR_GROUND"
    assert inventory[0]["v2_current_based_label"] == "ACTIVE_LOAD"


def test_idempotent_outputs_have_verified_hashes_and_refuse_corruption(workspace):
    _, prepare = workspace
    config = prepare([row(0), row(500000), row(1000000)])
    first = preprocess(config)
    output = Path(first["output_directory"])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert preprocess(config) == first
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(hashlib.sha256((output / p).read_bytes()).hexdigest() == digest for p, digest in manifest["outputs"].items())
    (output / "battery_activity.csv").write_text("corrupted")
    with pytest.raises(DataError, match="Existing output differs"):
        preprocess(config)
    assert (output / "battery_activity.csv").read_text() == "corrupted"


def test_cli_preprocess_success_and_rejection_are_machine_readable(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config = prepare([row(0), row(500000), row(1000000)])
    monkeypatch.setattr(main, "load_config", lambda path: config)
    assert main.main(["preprocess"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "OK"
    assert captured.err == ""
    config = prepare([row(0), row(0)])
    assert main.main(["preprocess"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "ERROR"


def test_real_cli_runs_from_another_cwd_and_preserves_raw(tmp_path):
    raw_path = PROJECT_ROOT / "data/raw/dataset_baterai.csv"
    raw = raw_path.read_bytes()
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "preprocess"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["input_rows"] == summary["retained_rows"] == 32338
    assert summary["dropped_rows"] == 0
    assert summary["flights"] == 17 and summary["gaps"] == 75 and summary["segments"] == 92
    assert not summary["model_training_performed"]
    output = Path(summary["output_directory"])
    assert len(read_csv(output / "activity_sensitivity_per_flight.csv")) == 153
    inventory = read_csv(output / "flight_inventory.csv")
    active = [r["NamaFile"] for r in inventory if r["activity_eligible"] == "True"]
    assert active == summary["active_load_flights"]
    assert raw_path.read_bytes() == raw
