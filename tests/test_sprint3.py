"""Independent temporal oracles and end-to-end reference/feature acceptance."""

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys

import pytest
import yaml

import main
from src.config import AppConfig, PROJECT_ROOT, load_config
from src.errors import ConfigError, DataError
from src.feature_sets import feature_registry, regression_predictors, validate_regression_predictors
from src.features import CausalFeatureBuilder, build_features, feature_columns, generate_features
from src.preprocessing import clean_and_segment


@pytest.fixture
def config():
    return load_config()


def row(time, current=0, voltage=24, *, flight="f", instance=0, total=0):
    return dict(NamaFile=flight, TimeUS=time, Inst=instance, Volt=voltage, Curr=current, CurrTot=total)


def warmup():
    return [row(t, voltage=20 + i) for i, t in enumerate(range(0, 2000001, 400000))]


def test_reference_uses_inclusive_fixed_window_and_locks_strictly_after_it(config):
    result = build_features(warmup() + [row(2100000, voltage=100), row(2200000, voltage=200)], config)
    assert all(r["readiness"] == "NOT_READY" and r["V_ref"] is None for r in result[:6])
    assert result[5]["reference_low_current_samples"] == 6  # sample exactly at 2 s is accepted
    assert result[6]["reference_lock_time_us"] == 2100000
    assert result[6]["V_ref"] == result[7]["V_ref"] == 22.5
    assert result[6]["voltage_sag_actual"] == -77.5  # signed target, never clipped
    assert result[6]["reference_low_current_samples"] == 6


def test_low_current_equality_and_exact_minimum_sample_count(config):
    samples = warmup()
    samples[0]["Curr"] = 1.0
    samples[1]["Curr"] = 1.000001
    result = build_features(samples + [row(2100000, voltage=500)], config)
    assert result[-1]["reference_low_current_samples"] == 5
    assert result[-1]["reference_ready"]
    assert result[-1]["V_ref"] == median([20, 22, 23, 24, 25])


def test_unavailable_reference_never_falls_back_to_later_low_current(config):
    samples = warmup()
    samples[-1]["Curr"] = samples[-2]["Curr"] = 2
    result = build_features(samples + [row(2100000, voltage=100), row(2200000, voltage=200), row(2300000)], config)
    assert result[-1]["reference_low_current_samples"] == 4
    assert all(r["reference_status"] == "REFERENCE_UNAVAILABLE" for r in result[6:])
    assert all(r["V_ref"] is None and r["voltage_sag_actual"] is None for r in result)
    assert all(r["readiness"] == "NOT_READY" for r in result)


def test_truncated_flight_does_not_finalize_or_rescale_warmup(config):
    short = warmup()[:3]
    result = build_features(short, config)
    extended = build_features(warmup() + [row(2100000)], config)
    assert result == extended[:3]
    assert result[-1]["reference_status"] == "WARMUP"


def test_gap_during_warmup_does_not_extend_or_restart_window(config):
    samples = [row(t, voltage=20 + i) for i, t in enumerate([0, 100000, 1000000, 1100000, 1200000, 2000000])]
    result = build_features(samples + [row(2100000, voltage=100)], config)
    assert result[-1]["reference_ready"]
    assert result[-1]["reference_low_current_samples"] == 6
    assert result[-1]["V_ref"] == 22.5
    assert result[-1]["reference_lock_time_us"] == 2100000
    assert result[2]["derivative_ready"] is False


def test_gap_resets_temporal_features_but_keeps_locked_reference(config):
    result = build_features(warmup() + [row(2100000, 2, 23), row(5000000, 30, 10)], config)
    last = result[-1]
    assert last["V_ref"] == result[-2]["V_ref"] == 22.5
    assert last["reference_ready"]
    assert last["segment_id"] == 1 and last["dt_seconds"] == 0
    assert last["elapsed_time"] == 5 and last["segment_elapsed_seconds"] == 0
    assert last["dI_dt"] == last["dV_dt"] == 0 and not last["derivative_ready"]
    for token in ("0p5s", "1p0s"):
        assert last[f"Curr_roll_mean_{token}"] == 30
        assert last[f"Curr_roll_std_{token}"] == 0
        assert last[f"Curr_roll_count_{token}"] == 1
        assert last[f"Curr_roll_coverage_{token}"] == 0
        assert not last[f"Curr_roll_ready_{token}"]


def test_derivatives_use_actual_dt_and_zero_at_start(config):
    result = build_features([row(0, 1, 24), row(100000, 2, 23.9), row(400000, 6, 23.6)], config)
    assert [r["dI_dt"] for r in result] == pytest.approx([0, 10, 4 / .3])
    assert [r["dV_dt"] for r in result] == pytest.approx([0, -1, -1])
    assert [r["derivative_ready"] for r in result] == [False, True, True]
    assert result[0]["dt_seconds"] == 0


def test_rolling_left_endpoint_ddof_zero_and_partial_coverage(config):
    result = build_features([row(0, 0), row(100000, 2), row(400000, 4), row(500000, 6)], config)
    last = result[-1]
    assert last["Curr_roll_mean_0p5s"] == 4
    assert last["Curr_roll_std_0p5s"] == pytest.approx(math.sqrt(8 / 3))
    assert last["Curr_roll_count_0p5s"] == 3
    assert last["Curr_roll_coverage_0p5s"] == 1 and last["Curr_roll_ready_0p5s"]
    assert last["Curr_roll_mean_1p0s"] == 3
    assert last["Curr_roll_std_1p0s"] == pytest.approx(math.sqrt(5))
    assert last["Curr_roll_count_1p0s"] == 4
    assert last["Curr_roll_coverage_1p0s"] == .5 and not last["Curr_roll_ready_1p0s"]
    assert result[0]["Curr_roll_std_0p5s"] == 0


def test_irregular_rolling_matches_independent_timestamp_oracle(config):
    samples = [row(t, i * i - 3) for i, t in enumerate([0, 100001, 280000, 500000, 800002, 900001, 1200000, 2000000])]
    result = build_features(samples, config)
    for index, output in enumerate(result):
        for window, token in ((.5, "0p5s"), (1.0, "1p0s")):
            history = [r["Curr"] for r in result[:index + 1] if r["segment_id"] == output["segment_id"]
                       and 0 <= output["TimeUS"] - r["TimeUS"] < window * 1e6]
            expected_mean = sum(history) / len(history)
            expected_std = math.sqrt(sum((value - expected_mean) ** 2 for value in history) / len(history))
            assert output[f"Curr_roll_mean_{token}"] == pytest.approx(expected_mean)
            assert output[f"Curr_roll_std_{token}"] == pytest.approx(expected_std)
            assert output[f"Curr_roll_count_{token}"] == len(history)


def test_batch_equals_incremental_at_arbitrary_chunk_boundaries(config):
    samples = warmup() + [row(2100000, 10), row(2200000, 20), row(4000000, 30)]
    expected = build_features(samples, config)
    builder = CausalFeatureBuilder(config)
    incremental = []
    for chunk in (samples[:2], samples[2:6], [], samples[6:7], samples[7:]):
        incremental.extend(builder.consume(sample) for sample in chunk)
    assert incremental == expected


def test_future_perturbation_does_not_change_any_prior_output(config):
    samples = warmup() + [row(t, i, 24 - i * .2) for i, t in enumerate(range(2100000, 3000001, 100000))]
    original = build_features(samples, config)
    changed = deepcopy(samples)
    for sample in changed[8:]:
        sample.update(Curr=100, CurrTot=999, Volt=5)
    assert build_features(changed, config)[:8] == original[:8]
    assert build_features(samples[:4], config) == original[:4]
    assert build_features(samples[:8], config) == original[:8]


def test_post_lock_voltage_edit_preserves_reference_and_regression_predictors(config):
    samples = warmup() + [row(2100000, 2, 23), row(2200000, 3, 22), row(2300000, 4, 21)]
    original = build_features(samples, config)
    edited = deepcopy(samples)
    for sample in edited[7:]:
        sample["Volt"] -= 3
    modified = build_features(edited, config)
    assert [r["V_ref"] for r in original] == [r["V_ref"] for r in modified]
    assert modified[7]["voltage_sag_actual"] == original[7]["voltage_sag_actual"] + 3
    for feature_set in ("battery_a", "battery_b", "battery_c", "battery_d", "battery_e"):
        names = regression_predictors(feature_set, [.5, 1.0])
        assert [[r[name] for name in names] for r in original] == [[r[name] for name in names] for r in modified]


def test_all_voltage_values_can_change_without_entering_regression_features(config):
    samples = warmup() + [row(2100000, 5)]
    original = build_features(samples, config)
    changed = [{**r, "Volt": 10 + i * .1} for i, r in enumerate(samples)]
    modified = build_features(changed, config)
    names = regression_predictors("battery_e", [.5, 1.0])
    assert [[r[n] for n in names] for r in original] == [[r[n] for n in names] for r in modified]
    assert original[-1]["V_ref"] != modified[-1]["V_ref"]


def test_interleaved_flights_and_instances_have_independent_state(config):
    streams = [[{**r, "NamaFile": flight, "Inst": instance, "Curr": r["Curr"] + instance * .1}
                for r in warmup() + [row(2100000, 2, 23)]] for flight in ("a", "b") for instance in (0, 1)]
    builder = CausalFeatureBuilder(config)
    interleaved = [builder.consume(sample) for time_group in zip(*streams) for sample in time_group]
    expected = [output for stream in streams for output in build_features(stream, config)]
    key = lambda r: (r["NamaFile"], r["Inst"], r["TimeUS"])
    assert sorted(interleaved, key=key) == sorted(expected, key=key)
    assert len(builder.states) == 4


def test_large_absolute_timestamps_do_not_lose_microsecond_precision(config):
    base = 2**53 + 1
    result = build_features([row(base, 1), row(base + 100001, 2)], config)
    assert result[0]["elapsed_time"] == 0
    assert result[1]["elapsed_time"] == .100001
    assert result[1]["dI_dt"] == pytest.approx(1 / .100001)


@pytest.mark.parametrize(("field", "value"), [("TimeUS", -1), ("TimeUS", 1.5), ("Inst", True),
                                               ("Curr", float("nan")), ("Volt", float("inf")), ("CurrTot", "1")])
def test_incremental_rejects_unclean_input_before_creating_state(config, field, value):
    builder = CausalFeatureBuilder(config)
    sample = row(0)
    sample[field] = value
    with pytest.raises(DataError):
        builder.consume(sample)
    assert not builder.states


@pytest.mark.parametrize(("field", "value"), [("segment_id", 1), ("dt_seconds", .2), ("is_segment_start", True)])
def test_segment_contract_is_validated_without_mutating_prior_state(config, field, value):
    clean = clean_and_segment([{k: str(v) for k, v in r.items()} for r in [row(0), row(100000)]], .5)
    builder = CausalFeatureBuilder(config)
    builder.consume(clean.rows[0])
    with pytest.raises(DataError, match="disagrees"):
        builder.consume({**clean.rows[1], field: value})
    assert builder.consume(clean.rows[1])["dt_seconds"] == .1


@pytest.mark.parametrize("time", [0, -1])
def test_repeated_or_backward_timestamp_is_rejected(config, time):
    builder = CausalFeatureBuilder(config)
    builder.consume(row(100000))
    with pytest.raises(DataError, match="strictly increase"):
        builder.consume(row(100000 + time))
    assert builder.consume(row(200000))["dt_seconds"] == .1


def test_registry_contains_exact_feature_sets_and_roles():
    registry = feature_registry([.5, 1.0], "battery_d")
    sets = registry["regression_sets"]
    assert sets["battery_a"] == ["Curr"]
    assert sets["battery_b"] == ["Curr", "CurrTot"]
    assert sets["battery_c"] == ["Curr", "CurrTot", "dI_dt"]
    assert len(sets["battery_d"]) == 7
    assert sets["battery_e"] == sets["battery_d"] + ["elapsed_time"]
    assert registry["ablation_only_sets"] == ["battery_e"]
    assert registry["direct_baseline_predictors"] == ["voltage_sag_actual", "Curr", "CurrTot", "dV_dt", "dI_dt"]
    assert all(not any("coverage" in n or "ready" in n or "count" in n for n in names) for names in sets.values())
    assert len(feature_columns([.5, 1.0])) == len(set(feature_columns([.5, 1.0])))


@pytest.mark.parametrize("predictor", ["Volt", "V_ref", "voltage_sag_actual", "dV_dt", "residual", "anomaly_score",
                                      "Volt_x_Curr", "Inst", "TimeUS", "readiness", "dt_seconds", "elapsed_time"])
def test_central_regression_validation_rejects_forbidden_predictors(predictor):
    names = [*regression_predictors("battery_d", [.5, 1.0]), predictor]
    with pytest.raises(ConfigError, match="regression_predictors"):
        validate_regression_predictors(names, "battery_d", [.5, 1.0])


@pytest.mark.parametrize("names", [None, [], ["CurrTot", "Curr"]])
def test_registry_validation_rejects_wrong_type_empty_or_wrong_order(names):
    with pytest.raises(ConfigError):
        validate_regression_predictors(names, "battery_b", [.5, 1.0])


def test_custom_windows_are_timestamp_based_and_use_registry_names(config):
    values = config.as_dict()
    values["features"]["rolling_windows_seconds"] = [.25, .75]
    custom = AppConfig(config.source, config.root, values)
    result = build_features([row(0, 0), row(250000, 4)], custom)
    assert result[-1]["Curr_roll_mean_0p25s"] == 4
    assert result[-1]["Curr_roll_mean_0p75s"] == 2
    assert "Curr_roll_mean_0p25s" in regression_predictors("battery_d", [.25, .75])


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    (root / "configs/config.yaml").write_text(yaml.safe_dump(load_config().as_dict()), encoding="utf-8")

    def prepare(samples):
        with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
            writer.writeheader()
            writer.writerows(samples)
        return load_config(root=root)
    return root, prepare


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pipeline_keeps_unavailable_truncated_and_dropped_flights_visible(workspace):
    _, prepare = workspace
    samples = ([{**r, "NamaFile": "normal"} for r in warmup() + [row(2100000), row(2200000), row(4000000)]]
               + [row(0, 10, flight="unavailable"), row(3000000, 10, flight="unavailable"), row(3100000, 0, flight="unavailable")]
               + [row(0, flight="short"), row(100000, flight="short")]
               + [row(0, voltage="nan", flight="dropped")])
    summary = generate_features(prepare(samples))
    output = Path(summary["output_directory"])
    coverage = {r["NamaFile"]: r for r in read_csv(output / "reference_coverage.csv")}
    assert coverage["normal"]["reference_status"] == "READY"
    assert float(coverage["normal"]["ready_observed_duration_seconds"]) == .1  # 2.1->2.2 only; gap to 4 s excluded
    assert coverage["unavailable"]["reference_status"] == "REFERENCE_UNAVAILABLE"
    assert coverage["unavailable"]["V_ref"] == ""
    assert coverage["short"]["coverage_reason"] == "WARMUP_NOT_COMPLETED"
    assert coverage["dropped"]["reference_status"] == "NO_RETAINED_SAMPLES"
    assert coverage["dropped"]["sample_coverage_percentage"] == ""
    rows = read_csv(output / "battery_features.csv")
    assert all(r["voltage_sag_actual"] == "" for r in rows if r["reference_ready"] == "False")
    assert summary["reference_unavailable_flights"] == ["unavailable"]
    assert summary["warmup_incomplete_flights"] == ["short"]
    assert summary["no_retained_sample_flights"] == ["dropped"]
    assert not summary["reference_activity_gate_passed"]


def test_all_dropped_pipeline_still_writes_registry_and_coverage(workspace):
    _, prepare = workspace
    summary = generate_features(prepare([row(0, current="nan")]))
    assert summary["retained_rows"] == summary["ready_samples"] == 0
    output = Path(summary["output_directory"])
    assert read_csv(output / "battery_features.csv") == []
    assert json.loads((output / "feature_registry.json").read_text())["regression_sets"]["battery_a"] == ["Curr"]


def test_features_are_idempotent_and_existing_output_corruption_is_rejected(workspace):
    _, prepare = workspace
    config = prepare(warmup() + [row(2100000)])
    result = generate_features(config)
    output = Path(result["output_directory"])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert generate_features(config) == result
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(hashlib.sha256((output / name).read_bytes()).hexdigest() == digest for name, digest in manifest["outputs"].items())
    (output / "feature_registry.json").write_text("corrupted")
    with pytest.raises(DataError, match="Existing feature output differs"):
        generate_features(config)
    assert (output / "feature_registry.json").read_text() == "corrupted"


def test_cli_features_and_invalid_dataset_have_explicit_status(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config = prepare(warmup() + [row(2100000)])
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["features"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["command"] == "features" and not captured.err
    config = prepare([row(0), row(0)])
    assert main.main(["features"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["status"] == "ERROR" and not captured.out


def test_real_cli_features_from_other_cwd_matches_independent_reference_oracle(tmp_path):
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "features"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["retained_rows"] == 32338 and summary["flights"] == 17
    assert not summary["model_training_performed"]
    assert summary["reference_activity_gate_passed"]
    assert len(summary["active_reference_ready_flights"]) == 6
    features = read_csv(Path(summary["processed_csv"]))
    raw = read_csv(PROJECT_ROOT / "data/raw/dataset_baterai.csv")
    for flight in sorted({r["NamaFile"] for r in raw}):
        records = [r for r in raw if r["NamaFile"] == flight]
        start = min(int(r["TimeUS"]) for r in records)
        accepted = [float(r["Volt"]) for r in records if int(r["TimeUS"]) - start <= 2000000 and float(r["Curr"]) <= 1]
        assert len(accepted) >= 5
        expected = median(accepted)
        outputs = [r for r in features if r["NamaFile"] == flight]
        assert all(float(r["V_ref"]) == expected for r in outputs if r["reference_ready"] == "True")
        assert all(r["reference_ready"] == "False" for r in outputs if int(r["TimeUS"]) - start <= 2000000)
    assert sum(r["reference_ready"] == "True" for r in features) == summary["ready_samples"]
