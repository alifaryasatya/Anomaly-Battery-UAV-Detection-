"""Frozen-model reuse, domain provenance, honest exposure and readiness tests."""

from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import main
import src.low_load as low
from src.config import AppConfig, load_config
from src.errors import DataError
from src.loafo import run_loafo


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def frozen_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("low_load")
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    values = load_config().as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] = 3
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    rows = []
    for flight in ("a", "b", "c", "idle", "truncated"):
        for i in range(11 if flight == "truncated" else 81):
            current = 10 + i % 3 if flight in ("a", "b", "c") and 25 <= i <= 65 else .1
            rows.append(dict(NamaFile=flight, TimeUS=i * 100000, Inst=0, Volt=24 - .02 * current - i * .0001,
                             Curr=current, CurrTot=i * .01))
    with (root / "data/raw/dataset_baterai.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    config = load_config(root=root)
    source = run_loafo(config)
    pointer = root / "artifacts/sprint6/cross_validation_result.json"
    pointer.parent.mkdir()
    pointer.write_bytes(low.json_bytes(source))
    return config, source


@pytest.fixture(scope="module")
def completed(frozen_workspace):
    config, source = frozen_workspace
    import src.models as models
    import src.detectors as detectors
    from sklearn.ensemble import HistGradientBoostingRegressor
    original = low.score_low_load

    def guard(rows, bundle, cfg, ranges):
        run = next((cfg.artifacts_path / "low_load").iterdir())
        assert (run / "pre_evaluation_freeze.json").exists()
        frozen = json.loads((run / "protocol.json").read_text())
        assert len(frozen["frozen_sources"]) == 3
        assert frozen["primary_fold"] == "fold_00"
        assert rows[0]["NamaFile"] in ("idle", "truncated")
        return original(rows, bundle, cfg, ranges)

    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*a, **kw):
            pytest.fail("low-load evaluation attempted fit/calibration")
        patch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
        patch.setattr(models, "fit_hgb", forbidden)
        patch.setattr(detectors, "calibrate_threshold", forbidden)
        patch.setattr(low, "score_low_load", guard)
        result = low.run_low_load(config)
    return config, source, result


def test_reuses_frozen_models_without_fitting_or_calibrating_and_counts_unique_exposure(completed):
    config, source, result = completed
    assert result["baseline_run_id"] == source["run_id"]
    assert result["unique_low_load_flights"] == 2
    assert result["evaluated_model_flight_pairs"] == 6
    assert result["primary_fold"] == "fold_00"
    assert result["primary_low_load"]["ready_samples"] == 60
    assert result["primary_low_load"]["observed_ready_seconds"] == 5.9
    assert result["all_low_load_unseen_by_model"] and result["all_low_load_unseen_by_detector"]
    assert not any(result[k] for k in ("model_training_performed", "threshold_calibration_performed", "threshold_changed", "policy_selection_performed", "synthetic_evaluation_performed"))
    assert "f1" not in result["primary_low_load"]
    assert low.verify_manifest(Path(result["output_directory"]), config)["run_id"] == result["run_id"]


def test_unready_flight_remains_visible_with_null_rates_and_no_predictions(completed):
    _, _, result = completed
    run = Path(result["output_directory"])
    records = [r for r in table(run / "per_model_flight.csv") if r["NamaFile"] == "truncated"]
    assert len(records) == 3
    assert all(r["evaluation_status"] == "NO_READY_EXPOSURE" and r["ready_samples"] == "0" for r in records)
    assert all(r["confirmed_fpr"] == r["confirmed_episodes_per_minute"] == "" for r in records)
    with gzip.open(run / "scores/fold_00/truncated.csv.gz", "rt", encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 11
    assert all(r["predicted_voltage_sag"] == r["anomaly_score"] == "" and r["detector_state"] == "NOT_READY" for r in scores)


def test_cache_is_read_only_and_corruption_is_rejected(completed, monkeypatch):
    config, _, result = completed
    run = Path(result["output_directory"])
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(low, "score_low_load", lambda *a: pytest.fail("cached low-load rescored"))
    assert low.run_low_load(config) == result
    assert before == {p: p.stat().st_mtime_ns for p in before}
    target = run / "per_model_flight.csv"
    original = target.read_bytes()
    try:
        target.write_bytes(b"corrupt")
        with pytest.raises(DataError, match="checksum"):
            low.run_low_load(config)
    finally:
        target.write_bytes(original)


def test_source_corruption_and_missing_pointer_fail_without_refitting(frozen_workspace, tmp_path):
    config, source = frozen_workspace
    path = Path(source["output_directory"]) / "folds/fold_00/threshold.json"
    original = path.read_bytes()
    try:
        path.write_bytes(b"corrupt")
        with pytest.raises(DataError, match="checksum"):
            low.load_sources(config)
    finally:
        path.write_bytes(original)
    pointer = config.artifacts_path / "sprint6/cross_validation_result.json"
    data = pointer.read_bytes()
    try:
        pointer.unlink()
        with pytest.raises(DataError, match="no automatic refitting"):
            low.load_sources(config)
    finally:
        pointer.write_bytes(data)


def test_training_range_fingerprint_rejects_changed_predictors(frozen_workspace):
    config, _ = frozen_workspace
    _, _, rows, _, _, bundles = low.load_sources(config)
    bundle = bundles[0][1]
    changed = deepcopy(rows)
    next(r for r in changed if r["NamaFile"] in bundle["split"]["train"] and r["reference_ready"])["Curr"] += 1
    with pytest.raises(DataError, match="Training domain"):
        low.training_ranges(changed, bundle)


@pytest.mark.parametrize("flight,unseen_model,unseen_detector", [("train", False, False), ("validation", True, False), ("idle", True, True)])
def test_scoring_flags_follow_source_membership_not_the_low_load_label(monkeypatch, flight, unseen_model, unseen_detector):
    cfg = load_config()
    bundle = dict(model=object(), fold_id="fold_00", split=dict(train=["train"], validation=["validation"], test=["test"]),
                  threshold=dict(threshold_v=.1, persistence_seconds=.5))
    rows = [dict(NamaFile=flight, Inst=0, segment_id=0, TimeUS=i * 100000, Curr=.1,
                 reference_ready=True, voltage_sag_actual=.2) for i in range(7)]
    monkeypatch.setattr(low, "predict_hgb", lambda *a: np.zeros(7))
    decisions = low.score_low_load(rows, bundle, cfg, {"Curr": dict(min=0, max=1)})
    assert all(r["unseen_by_model"] == unseen_model and r["unseen_by_detector"] == unseen_detector for r in decisions)
    assert all(not r["used_for_fit"] and not r["used_for_threshold"] for r in decisions)
    assert decisions[5]["confirmed_episode_start"] and decisions[6]["anomaly_confirmed"]


def test_synthetic_rows_are_rejected_before_prediction(monkeypatch):
    monkeypatch.setattr(low, "predict_hgb", lambda *a: pytest.fail("synthetic inference"))
    with pytest.raises(DataError, match="Synthetic"):
        low.score_low_load([dict(is_synthetic=True)], {}, load_config(), {})


def test_range_boundaries_are_inclusive_and_not_a_detection_rule():
    ranges = dict(Curr=dict(min=0, max=1), CurrTot=dict(min=10, max=20))
    assert not low.outside(dict(Curr=0, CurrTot=20), ranges)
    assert low.outside(dict(Curr=0, CurrTot=20.01), ranges)
    assert low.stats([])["p95"] is None


def test_pool_uses_counts_and_keeps_undefined_when_no_exposure():
    a = dict(ready_samples=1, raw_alarm_samples=1, confirmed_alarm_samples=1, confirmed_episodes=1, observed_ready_seconds=.1)
    b = dict(ready_samples=9, raw_alarm_samples=0, confirmed_alarm_samples=0, confirmed_episodes=0, observed_ready_seconds=.9)
    assert low.pool([a, b])["confirmed_fpr"] == .1
    empty = low.pool([])
    assert empty["confirmed_fpr"] is None and empty["confirmed_episodes_per_minute"] is None


def test_cli_routes_implemented_low_load_and_preserves_model_source(completed, monkeypatch, capsys):
    config, source, result = completed
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.COMMANDS["evaluate-low-load"].implemented
    assert main.main(["evaluate-low-load"]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert low.sha(config.data_path) == source["data_sha256"]


def test_gap_restarts_confirmation_and_is_excluded_from_monitored_duration(monkeypatch):
    config = load_config()
    bundle = dict(model=object(), fold_id="fold_00", split=dict(train=["a"], validation=["b"], test=["c"]),
                  threshold=dict(threshold_v=.1, persistence_seconds=.5))
    rows = [dict(NamaFile="idle", Inst=0, segment_id=int(t >= 1500000), TimeUS=t,
                 Curr=.1, reference_ready=True, voltage_sag_actual=.2) for t in (0, 500000, 1500000, 2000000)]
    monkeypatch.setattr(low, "predict_hgb", lambda *a: np.zeros(4))
    scored = low.score_low_load(rows, bundle, config, {"Curr": dict(min=0, max=1)})
    assert [r["anomaly_confirmed"] for r in scored] == [False, True, False, True]
    metrics = low.alarm_metrics(scored, config.gap_threshold_seconds)
    assert metrics["observed_ready_seconds"] == 1
    assert metrics["confirmed_episodes"] == 2 and metrics["confirmed_episodes_per_minute"] == 120


def test_training_or_calibration_overlap_cannot_be_reported_as_unseen_cohort(frozen_workspace, monkeypatch):
    config, _ = frozen_workspace
    source = list(low.load_sources(config))
    source[3] = deepcopy(source[3])
    next(c for c in source[3] if c["NamaFile"] == "a")["v2_current_based_label"] = "LOW_LOAD_OR_IDLE"
    monkeypatch.setattr(low, "load_sources", lambda _: source)
    monkeypatch.setattr(low, "score_low_load", lambda *a: pytest.fail("overlap was scored"))
    with pytest.raises(DataError, match="overlaps"):
        low.run_low_load(config)


def test_different_runtime_model_config_is_rejected(frozen_workspace):
    config, _ = frozen_workspace
    values = config.as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] += 1
    with pytest.raises(DataError, match="data/config/features"):
        low.load_sources(AppConfig(config.source, config.root, values))


def test_domain_and_response_exports_cover_whole_flights_and_original_predictors(completed):
    _, _, result = completed
    run = Path(result["output_directory"])
    profiles = table(run / "flight_domain_profiles.csv")
    assert len(profiles) == 5 * 9
    assert {r["feature"] for r in profiles} == {"Curr", "CurrTot", "dI_dt", "Curr_roll_mean_0p5s", "Curr_roll_std_0p5s", "Curr_roll_mean_1p0s", "Curr_roll_std_1p0s", "Volt", "voltage_sag_actual"}
    ranges = table(run / "outside_training_ranges.csv")
    assert len(ranges) == 3 * 2 * 7
    assert all(0 <= float(r["outside_fraction"]) <= 1 for r in ranges if r["outside_fraction"])
    assert len(table(run / "response_distributions.csv")) == 3 * 2 * 3
