"""Fault formulas, applicability/masks, reference preservation and export guards."""

from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import main
import src.models as models
from src.config import AppConfig, PROJECT_ROOT, load_config
from src.cross_validation import FlightSplit
from src.detectors import calibrate_threshold
from src.errors import DataError
from src.fault_injection import FAMILIES, generate_faults, inject_fault, observability
from src.feature_sets import FEATURE_SETS, regression_predictors
from src.features import build_features


@pytest.fixture
def config():
    return load_config()


def features(config, *, times=None, current=None, flight="f", base=0):
    times = times if times is not None else list(range(0, 10000001, 100000))
    source = []
    for i, time in enumerate(times):
        value = current(time) if current else 10 if 2500000 <= time <= 6500000 or 8100000 <= time <= 9100000 else 0
        source.append(dict(NamaFile=flight, Inst=0, TimeUS=base + time, Volt=24 - .01 * value - .001 * time / 1e6,
                           Curr=value, CurrTot=i * .01, source_record=i + 2))
    return build_features(source, config)


def scenario(rows, config, *, family="increased_resistance", severity="moderate", position=.5):
    return inject_fault(rows, config, family=family, severity=severity, start_position=position)


def at(rows, time):
    return next(r for r in rows if r["TimeUS"] == time)


def test_resistance_formula_and_full_selected_active_support(config):
    source = features(config)
    result = scenario(source, config)
    meta = result.metadata
    assert meta["segment_start_time_us"] == 2500000 and meta["segment_end_time_us"] == 6500000
    assert meta["onset_time_us"] == 4500000 and meta["fault_end_time_us"] == 6500000
    assert meta["fault_duration_seconds"] == 2 and meta["support_samples"] == 21
    assert meta["current_at_onset_a"] == meta["mean_current_after_fault_a"] == 10
    assert meta["max_added_sag_v"] == meta["mean_added_sag_v"] == .5
    assert meta["observability"] == "HIGH"
    for original, injected in zip(source, result.rows):
        inside = 4500000 <= original["TimeUS"] <= 6500000
        assert injected["fault_support"] == inside and injected["fault_label"] == int(inside)
        assert injected["added_sag_v"] == (.5 if inside else 0)
        assert injected["Volt"] == original["Volt"] - (.5 if inside else 0)
        assert injected["fault_effect_present"] == inside


def test_progressive_uses_aligned_onset_and_actual_time_not_row_fraction(config):
    times = list(range(0, 2000001, 200000)) + [2500000, 2800000, 3300000, 3700000, 4000000, 4500000, 4900000, 5200000, 5700000, 6000000]
    source = features(config, times=times, current=lambda t: 10 if t >= 2500000 else 0)
    result = scenario(source, config, family="progressive", severity="severe", position=.4)
    assert result.metadata["requested_onset_time_us"] == "3900000.0"
    assert result.metadata["onset_time_us"] == 4000000
    assert result.metadata["first_effect_time_us"] == 4500000
    for time, progress in [(4000000, 0), (4500000, .25), (4900000, .45), (5200000, .6), (5700000, .85), (6000000, 1)]:
        assert at(result.rows, time)["added_sag_v"] == pytest.approx(10 * .08 * progress)
    assert at(result.rows, 4000000)["fault_support"] and not at(result.rows, 4000000)["fault_effect_present"]
    assert result.metadata["effect_samples"] == result.metadata["support_samples"] - 1


def test_sudden_ramp_is_timestamp_based_and_allowed_on_low_current(config):
    source = features(config, current=lambda t: 0)
    result = scenario(source, config, family="sudden_drop")
    assert result.metadata["segment_start_time_us"] == 2100000
    assert result.metadata["requested_onset_time_us"] == "6050000.0"
    assert result.metadata["onset_time_us"] == 6100000
    assert result.metadata["first_effect_time_us"] == 6200000
    for time, added in [(6100000, 0), (6200000, .25), (6300000, .5), (6500000, 1), (10000000, 1)]:
        assert at(result.rows, time)["added_sag_v"] == pytest.approx(added)
    assert result.metadata["mean_current_after_fault_a"] == 0
    assert result.metadata["fault_applicable"] and result.metadata["evaluation_status"] == "NOT_EVALUATED"


def test_zero_ramp_is_instantaneous(config):
    cfg = config.as_dict()
    cfg["fault_injection"]["sudden_ramp_seconds"] = 0
    custom = AppConfig(config.source, config.root, cfg)
    result = scenario(features(custom, current=lambda t: 0), custom, family="sudden_drop")
    onset = result.metadata["onset_time_us"]
    assert at(result.rows, onset)["added_sag_v"] == 1
    assert result.metadata["first_effect_time_us"] == onset


def test_long_ramp_observability_uses_actual_signal_not_configured_drop(config):
    cfg = config.as_dict()
    cfg["fault_injection"]["sudden_ramp_seconds"] = 2
    custom = AppConfig(config.source, config.root, cfg)
    source = features(custom, times=list(range(0, 4600001, 100000)), current=lambda t: 0)
    result = scenario(source, custom, family="sudden_drop", severity="mild", position=.6)
    assert result.metadata["onset_time_us"] == 3600000
    assert result.metadata["fault_duration_seconds"] == 1
    assert result.metadata["configured_dose"] == .5
    assert result.metadata["max_added_sag_v"] == .25 and result.metadata["observability"] == "MEDIUM"


@pytest.mark.parametrize("family", ["increased_resistance", "progressive"])
def test_current_dependent_faults_do_not_apply_to_low_load_without_valid_run(config, family):
    source = features(config, current=lambda t: .1)
    result = scenario(source, config, family=family)
    assert not result.metadata["fault_applicable"]
    assert result.metadata["reason"] == "NO_VALID_ACTIVE_SEGMENT"
    assert result.metadata["evaluation_status"] == result.metadata["observability"] == "NOT_APPLICABLE"
    assert result.metadata["max_added_sag_v"] is None and result.metadata["mean_added_sag_v"] is None
    assert result.metadata["onset_time_us"] is None
    assert all(not r["fault_support"] and r["fault_phase"] == "NOT_APPLICABLE" for r in result.rows)
    assert [r["Volt"] for r in result.rows] == [r["Volt"] for r in source]


def test_exact_one_second_support_is_applicable_but_shorter_is_not(config):
    source = features(config, times=list(range(0, 6000001, 100000)), current=lambda t: 10 if 2500000 <= t <= 5000000 else 0)
    accepted = scenario(source, config, position=.6)
    rejected = scenario(source, config, position=.600001)
    assert accepted.metadata["fault_applicable"] and accepted.metadata["fault_duration_seconds"] == 1
    assert rejected.metadata["candidate_onset_time_us"] == 4100000
    assert rejected.metadata["post_onset_available_seconds"] == .9
    assert not rejected.metadata["fault_applicable"] and rejected.metadata["onset_time_us"] is None
    assert rejected.metadata["reason"] == "INSUFFICIENT_POST_ONSET_SUPPORT"
    assert all(r["added_sag_v"] == 0 for r in rejected.rows)


def test_onset_before_reference_does_not_silently_shift_or_retry(config):
    source = features(config, times=list(range(0, 4000001, 100000)), current=lambda t: 1)
    result = scenario(source, config, position=.4)
    assert result.metadata["candidate_onset_time_us"] == 1600000
    assert result.metadata["reason"] == "ONSET_NOT_REFERENCE_READY"
    assert not result.metadata["fault_applicable"]
    assert all(r["Volt"] == r["Volt_healthy"] for r in result.rows)


def test_gap_never_bridged_and_equal_duration_tie_chooses_earliest(config):
    times = list(range(0, 5500001, 100000)) + list(range(20000000, 23000001, 100000))
    source = features(config, times=times, current=lambda t: 10 if 2500000 <= t <= 5500000 else 20 if t >= 20000000 else 0)
    result = scenario(source, config)
    assert result.metadata["segment_start_time_us"] == 2500000
    assert result.metadata["selected_segment_id"] == 0
    assert result.metadata["fault_end_time_us"] == 5500000
    assert result.metadata["post_fault_original_start_time_us"] == 20000000
    assert all(not r["fault_support"] and r["Volt"] == r["Volt_healthy"] for r in result.rows if r["TimeUS"] >= 20000000)
    assert at(result.rows, 20000000)["dV_dt"] == 0


def test_sudden_ready_segment_tie_is_earliest_and_never_crosses_gap(config):
    times = list(range(0, 5100001, 100000)) + list(range(10000000, 13000001, 100000))
    result = scenario(features(config, times=times, current=lambda t: 0), config, family="sudden_drop")
    assert result.metadata["segment_start_time_us"] == 2100000
    assert result.metadata["fault_end_time_us"] == 5100000
    assert all(not r["fault_support"] for r in result.rows if r["TimeUS"] >= 10000000)


def test_low_current_interrupts_support_and_voltage_derivative_sees_return(config):
    source = features(config)
    result = scenario(source, config)
    before = at(result.rows, 6500000)
    after = at(result.rows, 6600000)
    assert after["fault_phase"] == "POST_FAULT_ORIGINAL" and not after["fault_support"]
    assert after["Volt"] == after["Volt_healthy"]
    assert after["dV_dt"] == pytest.approx((after["Volt"] - before["Volt"]) / .1)
    assert after["dV_dt"] == pytest.approx(after["dV_dt_healthy"] + 5)
    assert not at(result.rows, 8100000)["fault_support"]  # no injection into the next active run


@pytest.mark.parametrize("family", FAMILIES)
def test_input_reference_and_all_regression_predictors_remain_unchanged(config, family):
    source = features(config)
    original = deepcopy(source)
    result = scenario(source, config, family=family)
    assert source == original
    for old, new in zip(source, result.rows):
        for name in ("NamaFile", "Inst", "TimeUS", "Curr", "CurrTot", "V_ref", "reference_ready", "reference_status"):
            assert new[name] == old[name]
        if new["reference_ready"]:
            assert new["voltage_sag_actual"] == new["V_ref"] - new["Volt"]
            assert new["voltage_sag_actual"] == pytest.approx(old["voltage_sag_actual"] + new["added_sag_v"])
        for feature_set in FEATURE_SETS:
            assert [new[n] for n in regression_predictors(feature_set, [.5, 1.0])] == [old[n] for n in regression_predictors(feature_set, [.5, 1.0])]


@pytest.mark.parametrize(("value", "label"), [(0, "VERY_LOW"), (.049999, "VERY_LOW"), (.05, "LOW"), (.199999, "LOW"),
                                             (.2, "MEDIUM"), (.499999, "MEDIUM"), (.5, "HIGH"), (1, "HIGH")])
def test_observability_half_open_boundaries(config, value, label):
    assert observability(value, config.as_dict()["fault_injection"]["observability_max_sag_v"]) == label


def test_unavailable_truncated_and_empty_have_explicit_not_applicable_reasons(config):
    unavailable = scenario(features(config, current=lambda t: 10), config)
    short = scenario(features(config, times=[0, 100000]), config)
    empty = inject_fault([], config, family="sudden_drop", severity="severe", start_position=.5, flight_id="empty")
    assert unavailable.metadata["reason"] == "REFERENCE_UNAVAILABLE"
    assert short.metadata["reason"] == "WARMUP_INCOMPLETE"
    assert empty.metadata["reason"] == "NO_RETAINED_SAMPLES" and empty.rows == []


def test_absolute_microsecond_onset_does_not_round_through_float(config):
    base = 2**53 + 1
    result = scenario(features(config, base=base), config)
    assert result.metadata["onset_time_us"] == base + 4500000
    assert result.metadata["fault_end_time_us"] == base + 6500000


@pytest.mark.parametrize("position", [0, 1, -.1, float("nan"), True])
def test_invalid_start_positions_rejected(config, position):
    with pytest.raises(DataError):
        scenario(features(config), config, position=position)


def test_mixed_unordered_or_synthetic_sources_are_rejected(config):
    source = features(config)
    with pytest.raises(DataError, match="one flight"):
        scenario([source[0], {**source[1], "NamaFile": "other"}], config)
    with pytest.raises(DataError, match="instances"):
        scenario([source[0], {**source[1], "Inst": 1}], config)
    with pytest.raises(DataError, match="strictly increase"):
        scenario([source[1], source[0]], config)
    with pytest.raises(DataError, match="synthetic"):
        scenario(scenario(source, config).rows, config)


def test_synthetic_prefault_and_not_applicable_rows_cannot_fit_or_calibrate(config):
    for source in (features(config), features(config, current=lambda t: 0)):
        copied = scenario(source, config).rows
        assert copied[0]["is_synthetic"] and not copied[0]["fault_support"]
        split = FlightSplit(("f",), ("v",), ("t",))
        with pytest.raises(DataError, match="Synthetic"):
            models.fit_hgb([copied[0]], config, split)
        with pytest.raises(DataError, match="Synthetic"):
            calibrate_threshold([copied[0]], split)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    (root / "configs/config.yaml").write_text(yaml.safe_dump(load_config().as_dict()), encoding="utf-8")
    source = []
    for flight in ("active", "idle", "dropped"):
        for i in range(81):
            current = 10 if flight == "active" and 25 <= i <= 65 else 0
            source.append(dict(NamaFile=flight, Inst=0, TimeUS=i * 100000, Volt="nan" if flight == "dropped" else 24 - .01 * current,
                               Curr=current, CurrTot=i * .01))
    with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
        writer.writeheader()
        writer.writerows(source)
    return root, load_config(root=root)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pipeline_exports_metadata_for_every_case_and_only_applicable_csv(workspace, monkeypatch):
    root, config = workspace
    monkeypatch.setattr(models, "fit_hgb", lambda *a: pytest.fail("No model fit in Stage 5"))
    monkeypatch.setattr(models, "predict_hgb", lambda *a: pytest.fail("No model inference in Stage 5"))
    raw = config.data_path.read_bytes()
    summary = generate_faults(config)
    assert summary["scenarios"] == 81 and summary["applicable_scenarios"] == 36 and summary["not_applicable_scenarios"] == 45
    assert summary["scenario_files"] == 36 and summary["synthetic_rows"] == 36 * 81
    output = Path(summary["output_directory"])
    metadata = read_csv(output / "fault_manifest.csv")
    assert len(metadata) == len(read_csv(output / "fault_applicability.csv")) == len(read_csv(output / "fault_observability.csv")) == 81
    assert len(list(Path(summary["synthetic_directory"]).glob("*.csv.gz"))) == 36
    for record in metadata:
        if record["fault_applicable"] == "True":
            path = root / record["scenario_csv"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["scenario_sha256"]
            assert path.read_bytes()[4:8] == b"\x00\x00\x00\x00"  # deterministic gzip timestamp
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 81 and all(r["is_synthetic"] == "True" for r in rows)
            assert not any("anomaly_score" in r or "predicted_voltage_sag" in r for r in rows)
        else:
            assert record["scenario_csv"] == record["max_added_sag_v"] == ""
    assert config.data_path.read_bytes() == raw
    assert not summary["model_training_performed"] and not summary["model_prediction_performed"] and not summary["detector_evaluation_performed"]


def test_pipeline_repeatability_cache_hashes_and_corruption_guard(workspace, monkeypatch):
    root, config = workspace
    summary = generate_faults(config)
    output = Path(summary["output_directory"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p,h in manifest["outputs"].items())
    import src.fault_injection as faults
    monkeypatch.setattr(faults, "inject_fault", lambda *a, **kw: pytest.fail("Completed bank must be reused"))
    assert generate_faults(config) == summary
    target = next(Path(summary["synthetic_directory"]).glob("*.csv.gz"))
    target.write_bytes(b"corrupted")
    with pytest.raises(DataError, match="checksum mismatch"):
        generate_faults(config)
    assert target.read_bytes() == b"corrupted"


def test_cli_generation_is_explicit_and_machine_readable(workspace, monkeypatch, capsys):
    _, config = workspace
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["inject-faults"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["command"] == "inject-faults" and not captured.err


def test_real_scenario_bank_cli_from_other_cwd_preserves_pilot_and_raw(tmp_path):
    pilot_summary = json.loads((PROJECT_ROOT / "artifacts/sprint4/baseline_result.json").read_text())
    pilot = Path(pilot_summary["output_directory"])
    before = {name: hashlib.sha256((pilot / name).read_bytes()).hexdigest() for name in ("model_bundle.joblib", "threshold.json")}
    raw = (PROJECT_ROOT / "data/raw/dataset_baterai.csv").read_bytes()
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "inject-faults"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["scenarios"] == 17 * 27
    assert summary["applicable_scenarios"] + summary["not_applicable_scenarios"] == summary["scenarios"]
    output = Path(summary["output_directory"])
    applicability = read_csv(output / "fault_applicability.csv")
    for flight in {r["NamaFile"] for r in applicability}:
        assert sum(r["NamaFile"] == flight for r in applicability) == 27
    assert before == {name: hashlib.sha256((pilot / name).read_bytes()).hexdigest() for name in before}
    assert (PROJECT_ROOT / "data/raw/dataset_baterai.csv").read_bytes() == raw
