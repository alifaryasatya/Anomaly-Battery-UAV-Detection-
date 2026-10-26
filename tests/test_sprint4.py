"""Healthy training/calibration isolation, timestamp timers and bundle acceptance."""

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

import main
import src.models as models
from src.config import AppConfig, PROJECT_ROOT, load_config
from src.cross_validation import FlightSplit, pilot_split
from src.detectors import apply_single_path, calibrate_threshold, residual_score
from src.errors import ConfigError, DataError
from src.feature_sets import regression_predictors
from src.persistence import TimePersistence
from src.pilot_reporting import alarm_metrics, regression_metrics


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def split():
    return FlightSplit(("train",), ("validation",), ("test",))


def feature_row(index, *, flight="train", current=2, target=.2):
    row = dict(NamaFile=flight, Inst=0, TimeUS=index * 100000, source_record=index + 2,
               segment_id=0, dt_seconds=.1 if index else 0, reference_ready=True, Curr=current,
               CurrTot=index * .1, dI_dt=0., voltage_sag_actual=target, Volt=24 - target, V_ref=24.,
               elapsed_time=index * .1)
    for token in ("0p5s", "1p0s"):
        row[f"Curr_roll_mean_{token}"] = float(current)
        row[f"Curr_roll_std_{token}"] = 0.
    return row


def validation_score(index, value):
    return {**feature_row(index, flight="validation"), "split_role": "validation", "anomaly_score": value}


def test_pilot_assignment_is_first_sorted_cyclic_fold_without_overlap():
    split = pilot_split(["d", "b", "a", "c"])
    assert split.test == ("a",) and split.validation == ("b",) and split.train == ("c", "d")
    assert pilot_split(["c", "a", "d", "b"]) == split
    assert split.as_dict() == {"train": ["c", "d"], "validation": ["b"], "test": ["a"]}


@pytest.mark.parametrize("split", [FlightSplit(("a",), ("a",), ("b",)), FlightSplit(("a",), ("b",), ("a",)),
                                   FlightSplit(("a",), ("b",), ("b",)), FlightSplit((), ("b",), ("c",)),
                                   FlightSplit(("a", "a"), ("b",), ("c",))])
def test_bad_split_is_rejected(split):
    with pytest.raises(DataError):
        split.validate()


@pytest.mark.parametrize("eligible", [[], ["a", "b"], ["a", "b", "b"]])
def test_pilot_needs_three_distinct_eligible_flights(eligible):
    with pytest.raises(DataError):
        pilot_split(eligible)


def test_model_explicitly_disables_internal_row_validation_and_has_seed(config):
    estimator = models.make_hgb(config)
    assert estimator.early_stopping is False
    assert estimator.random_state == 42
    assert estimator.max_iter == config.as_dict()["models"]["hist_gradient_boosting"]["max_iter"]


def test_fit_receives_only_allowlisted_predictors_all_train_rows_and_no_scaler(config, split, monkeypatch):
    captured = {}

    class Spy:
        def fit(self, frame, target):
            captured.update(frame=frame.copy(), target=target.copy())
            return self
    monkeypatch.setattr(models, "make_hgb", lambda cfg: Spy())
    rows = [feature_row(i, current=0 if i < 3 else 5, target=-.1 if i < 3 else .5) for i in range(8)]
    result = models.fit_hgb(rows, config, split)
    assert isinstance(result, Spy)
    assert list(captured["frame"].columns) == list(regression_predictors("battery_d", [.5, 1.0]))
    assert len(captured["frame"]) == 8
    assert list(captured["frame"]["Curr"]) == [0, 0, 0, 5, 5, 5, 5, 5]
    assert list(captured["target"]) == [-.1, -.1, -.1, .5, .5, .5, .5, .5]
    assert "Volt" not in captured["frame"] and "voltage_sag_actual" not in captured["frame"]


@pytest.mark.parametrize("flight", ["validation", "test", "unknown"])
def test_fit_rejects_nontraining_flights_before_model_fit(config, split, flight):
    with pytest.raises(DataError, match="training flights"):
        models.fit_hgb([feature_row(0), feature_row(1, flight=flight)], config, split)


@pytest.mark.parametrize("marker", ["is_synthetic", "fault_support", "scenario_id", "added_sag_v", "fault_family"])
def test_synthetic_scenario_rows_cannot_fit_or_calibrate(config, split, marker):
    with pytest.raises(DataError, match="Synthetic"):
        models.fit_hgb([{**feature_row(0), marker: False}], config, split)
    with pytest.raises(DataError, match="Synthetic"):
        calibrate_threshold([{**validation_score(0, .5), marker: False}], split)


def test_unready_or_nonfinite_target_cannot_fit(config, split):
    with pytest.raises(DataError, match="reference-ready"):
        models.fit_hgb([{**feature_row(0), "reference_ready": False}], config, split)
    with pytest.raises(DataError, match="finite voltage sag"):
        models.fit_hgb([feature_row(0, target=float("nan"))], config, split)


def test_predictor_matrix_rejects_leakage_and_nonfinite_values(config):
    bad = [*config.as_dict()["features"]["regression_predictors"], "Volt"]
    with pytest.raises(ConfigError):
        models.regression_frame([feature_row(0)], config, predictors=bad)
    with pytest.raises(DataError, match="finite numeric"):
        models.regression_frame([feature_row(0, current=float("nan"))], config)


def test_repeat_fit_is_deterministic_and_other_flights_do_not_affect_training(config, split):
    rows = [feature_row(i, current=i % 10, target=(i % 10) * .07) for i in range(60)]
    first = models.fit_hgb(rows, config, split)
    second = models.fit_hgb(deepcopy(rows), config, split)
    assert np.array_equal(models.predict_hgb(first, rows, config), models.predict_hgb(second, rows, config))
    assert first.n_iter_ == 200 and first.do_early_stopping_ is False


def test_residual_is_signed_and_score_is_positive_direction_only():
    assert residual_score(.2, .5) == pytest.approx((-.3, 0))
    assert residual_score(.5, .2) == pytest.approx((.3, .3))
    assert residual_score(.5, .5) == (0, 0)
    with pytest.raises(DataError):
        residual_score(float("nan"), 0)


def test_threshold_is_linear_p99_of_healthy_validation_and_tracks_provenance(split):
    rows = [validation_score(i, score) for i, score in enumerate([0., 0., 1., 3., 10.])]
    payload = calibrate_threshold(rows, split)
    assert payload["threshold_v"] == pytest.approx(9.72)
    assert payload["source_role"] == "validation" and payload["source_flights"] == ["validation"]
    assert payload["source_samples"] == 5 and len(payload["source_sha256"]) == 64
    assert payload["quantile_method"] == "numpy.percentile_linear"


@pytest.mark.parametrize("edit", [{"NamaFile": "train"}, {"NamaFile": "test"}, {"split_role": "test"},
                                  {"reference_ready": False}, {"anomaly_score": float("nan")}, {"anomaly_score": -1}])
def test_threshold_rejects_wrong_source_or_invalid_scores(split, edit):
    with pytest.raises(DataError):
        calibrate_threshold([{**validation_score(0, .5), **edit}], split)


def test_strict_threshold_equality_and_unready_rows_do_not_warn():
    rows = [{**validation_score(0, 1.), "reference_ready": True},
            {**validation_score(1, 100.), "reference_ready": False}, validation_score(2, 1.0001)]
    result = apply_single_path(rows, threshold_v=1., duration_seconds=.5, gap_threshold_seconds=.5)
    assert [r["anomaly_raw"] for r in result] == [False, False, True]
    assert [r["detector_state"] for r in result] == ["NORMAL", "NOT_READY", "SUSPECT"]
    assert not any(r["anomaly_confirmed"] for r in result)


def consume(timer, time, raw=True, *, flight="f", instance=0, segment=0, ready=True):
    return timer.consume(flight=flight, instance=instance, segment_id=segment, time_us=time, raw=raw, ready=ready)


def test_persistence_uses_actual_elapsed_with_equality_and_interruption():
    timer = TimePersistence(.5, .5)
    decisions = [consume(timer, t) for t in [0, 110000, 310000, 500000]]
    assert [r["anomaly_duration_seconds"] for r in decisions] == [0, .11, .31, .5]
    assert [r["anomaly_confirmed"] for r in decisions] == [False, False, False, True]
    assert consume(timer, 510000, False)["anomaly_duration_seconds"] == 0
    assert consume(timer, 600000)["anomaly_duration_seconds"] == 0
    consume(timer, 900000)
    assert consume(timer, 1100000)["confirmed_episode_start"]


def test_persistence_gap_and_segment_change_restart_timer():
    timer = TimePersistence(.5, .5)
    consume(timer, 0)
    consume(timer, 300000)
    assert consume(timer, 900000)["anomaly_duration_seconds"] == 0  # actual gap alone resets
    consume(timer, 1200000)
    assert consume(timer, 1400000)["anomaly_confirmed"]
    assert consume(timer, 1500000, segment=1)["anomaly_duration_seconds"] == 0
    assert not consume(timer, 1600000, segment=1)["anomaly_confirmed"]


def test_persistence_ready_reset_and_independent_flight_instance_streams():
    timer = TimePersistence(.5, .5)
    consume(timer, 0, flight="a")
    consume(timer, 0, flight="b")
    consume(timer, 0, flight="a", instance=1)
    assert consume(timer, 100000, flight="a", ready=False)["detector_state"] == "NOT_READY"
    assert consume(timer, 500000, flight="b")["anomaly_confirmed"]
    assert consume(timer, 500000, flight="a", instance=1)["anomaly_confirmed"]
    assert consume(timer, 500000, flight="a")["anomaly_duration_seconds"] == 0


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_persistence_duration_rejected(duration):
    with pytest.raises(DataError):
        TimePersistence(duration, .5)


def test_out_of_order_persistence_input_is_rejected_without_advancing_state():
    timer = TimePersistence(.5, .5)
    consume(timer, 100000)
    with pytest.raises(DataError, match="ordered"):
        consume(timer, 100000)
    assert consume(timer, 600000)["anomaly_confirmed"]


def test_false_alarm_exposure_and_episode_counter_exclude_gaps():
    timer = TimePersistence(.2, .5)
    rows = []
    for time in [0, 200000, 300000, 400000, 500000, 700000, 1300000, 1500000]:
        result = consume(timer, time, raw=time != 400000)
        rows.append(dict(NamaFile="f", Inst=0, segment_id=0, TimeUS=time, reference_ready=True, **result))
    report = alarm_metrics(rows, .5)
    assert report["confirmed_episodes"] == 3
    assert report["confirmed_alarm_samples"] == 4
    assert report["observed_ready_seconds"] == .9
    assert report["confirmed_episodes_per_minute"] == 200


def test_regression_null_denominators_are_not_fabricated_zeros():
    assert regression_metrics([])["rmse_v"] is None
    rows = [{**feature_row(i, target=1.), "predicted_voltage_sag": .5} for i in range(3)]
    report = regression_metrics(rows)
    assert report["mae_v"] == report["rmse_v"] == .5 and report["r2"] is None


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    values = load_config().as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] = 10
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    source = []
    for flight in ("a", "b", "c"):
        for i in range(61):
            current = 0 if i <= 25 or i >= 50 else 2 + i % 5
            source.append(dict(NamaFile=flight, Inst=0, TimeUS=i * 100000, Curr=current, CurrTot=i * .01,
                               Volt=24 - .05 * current - .0005 * i))
    with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
        writer.writeheader()
        writer.writerows(source)
    return root, load_config(root=root)


def test_pilot_protocol_precedes_fit_and_reserved_test_is_never_scored(workspace, monkeypatch):
    root, config = workspace
    fit, score = models.fit_hgb, models.score_rows
    seen = []

    def guarded_fit(rows, config, split):
        protocol = next((root / "artifacts/baseline").glob("*/pilot_protocol.json"))
        assert json.loads(protocol.read_text())["split"] == {"train": ["c"], "validation": ["b"], "test": ["a"]}
        assert {r["NamaFile"] for r in rows} == {"c"}
        assert len(rows) == 40  # all post-warm-up training rows, including idle/low current
        assert any(r["Curr"] == 0 for r in rows)
        return fit(rows, config, split)

    def guarded_score(model, rows, config, role):
        seen.extend((role, r["NamaFile"]) for r in rows)
        assert not any(r["NamaFile"] == "a" for r in rows)
        return score(model, rows, config, role)

    monkeypatch.setattr(models, "fit_hgb", guarded_fit)
    monkeypatch.setattr(models, "score_rows", guarded_score)
    summary = models.baseline(config)
    assert summary["split"] == {"train": ["c"], "validation": ["b"], "test": ["a"]}
    assert not summary["reserved_test_evaluated"]
    assert {role for role, flight in seen} == {"train", "validation"}


def test_bundle_reload_and_completed_run_cache_and_corruption_guard(workspace, monkeypatch):
    _, config = workspace
    summary = models.baseline(config)
    output = Path(summary["output_directory"])
    bundle = models.load_model_bundle(output / "model_bundle.joblib")
    feature_summary = models.generate_features(config)
    rows, _, _, _ = models._feature_input(feature_summary, config)
    train = [r for r in rows if r["NamaFile"] == "c" and r["reference_ready"]]
    original = models.fit_hgb(train, config, FlightSplit(("c",), ("b",), ("a",)))
    assert np.array_equal(models.predict_hgb(original, train, config), models.predict_hgb(bundle["model"], train, config))
    assert bundle["threshold"]["source_flights"] == ["b"]
    assert bundle["input_schema_unit_policy"]["units"]["CurrTot"] == "unverified"
    assert bundle["training_provenance"]["scaler"] is None
    assert bundle["model"].early_stopping is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(hashlib.sha256((output / p).read_bytes()).hexdigest() == h for p, h in manifest["outputs"].items())
    monkeypatch.setattr(models, "fit_hgb", lambda *args: pytest.fail("Completed run must not refit"))
    assert models.baseline(config) == summary
    (output / "threshold.json").write_text("corrupted")
    with pytest.raises(DataError, match="checksum mismatch"):
        models.baseline(config)
    assert (output / "threshold.json").read_text() == "corrupted"


def test_currtot_reset_blocks_battery_d_fit(workspace, monkeypatch):
    _, config = workspace
    generate = models.generate_features

    def changed(config):
        return {**generate(config), "currtot_predictor_review_required": True}
    monkeypatch.setattr(models, "generate_features", changed)
    monkeypatch.setattr(models, "fit_hgb", lambda *args: pytest.fail("Reset gate must stop before fit"))
    with pytest.raises(DataError, match="CurrTot decreases"):
        models.baseline(config)


def test_pilot_rejects_parameter_drift_before_training(config):
    changed = config.as_dict()
    changed["persistence"]["reference_duration_seconds"] = .2
    custom = AppConfig(config.source, config.root, changed)
    with pytest.raises(ConfigError, match="reference pilot"):
        models.baseline(custom)


def test_cli_baseline_status(workspace, monkeypatch, capsys):
    _, config = workspace
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["baseline"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["command"] == "baseline" and not captured.err


def test_real_cli_baseline_from_other_cwd_matches_saved_score_oracles(tmp_path):
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "baseline"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["training_samples"] == 15742 and summary["validation_samples"] == 1622
    assert summary["split"]["test"] == ["00000066.BIN"] and not summary["reserved_test_evaluated"]
    output = Path(summary["output_directory"])
    with (output / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["NamaFile"] for r in rows} == set(summary["split"]["train"] + summary["split"]["validation"])
    validation = [r for r in rows if r["split_role"] == "validation" and r["reference_ready"] == "True"]
    scores = np.array([max(0., float(r["voltage_sag_actual"]) - float(r["predicted_voltage_sag"])) for r in validation])
    assert np.array_equal(scores, np.array([float(r["anomaly_score"]) for r in validation]))
    assert np.percentile(scores, 99, method="linear") == summary["threshold_v"]
    assert sum(r["anomaly_raw"] == "True" for r in validation) == summary["metrics_by_role"]["validation"]["healthy_alarm"]["raw_alarm_samples"]
    bundle = models.load_model_bundle(output / "model_bundle.joblib")
    assert bundle["training_provenance"]["low_current_training_samples"] > 0
    assert bundle["training_provenance"]["fitted_iterations"] == 200
