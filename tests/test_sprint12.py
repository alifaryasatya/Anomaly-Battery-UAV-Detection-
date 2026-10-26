"""Causal streaming, raw validation, exact-cache isolation and frozen replay."""

from copy import deepcopy
import csv
from itertools import count
from pathlib import Path

import numpy as np
import pytest
import yaml

import main
from src.config import load_config
from src.errors import DataError
from src.features import build_features
from src.loafo import run_loafo
from src.low_load import run_low_load
from src.models import score_rows
from src.detectors import apply_single_path
from src.replay import StreamingDetector, TIMING_COLUMNS
import src.replay_experiment as experiment
from src.research_sources import encoded, load, sha


class ConstantModel:
    def __init__(self, names, value=0.):
        self.feature_names_in_ = names
        self.value, self.calls = value, 0

    def predict(self, frame):
        self.calls += 1
        return np.full(len(frame), self.value)


def bundle(config, value=0.):
    return dict(engine="battery_v2", format_version=1, model=ConstantModel(config.as_dict()["features"]["regression_predictors"], value),
                config=config.as_dict(), predictors=config.as_dict()["features"]["regression_predictors"],
                threshold=dict(source_role="validation", percentile=99, threshold_v=.1, persistence_seconds=.5))


def raw(i, **updates):
    return dict(NamaFile="f", TimeUS=i * 100000, Inst=0, Volt=24 if i < 30 else 23.5, Curr=.1, CurrTot=i * .01, **updates)


def test_raw_to_warning_matches_batch_and_does_not_double_persistence():
    config = load_config()
    b = bundle(config)
    inputs = [raw(i) for i in range(41)]
    expected = apply_single_path(score_rows(b["model"], build_features(inputs, config), config, "test"),
                                 threshold_v=.1, duration_seconds=.5, gap_threshold_seconds=.5)
    detector = StreamingDetector(b, config)
    actual = [detector.consume(r) for r in inputs]
    maxima = {}
    for a, e in zip(actual, expected):
        experiment.compare_row(a, e, ["V_ref", "dI_dt", "dV_dt", "reference_ready", *experiment.DECISION_COLUMNS], maxima)
    first = next(r for r in actual if r["confirmed_episode_start"])
    assert first["TimeUS"] == 3500000
    assert (first["TimeUS"] - 3000000) / 1e6 == .5  # persistence already in signal delay
    assert actual[20]["detector_state"] == "NOT_READY" and actual[21]["detector_state"] == "NORMAL"


def test_interleaved_flights_and_prefix_have_no_future_dependency():
    config = load_config()
    b = bundle(config)
    first = StreamingDetector(b, config)
    prefix = [first.consume(raw(i)) for i in range(25)]
    second = StreamingDetector(b, config)
    observed = []
    for i in range(25):
        observed.append(second.consume(raw(i)))
        other = raw(i)
        other.update(NamaFile="other", Volt=50)
        second.consume(other)
    for a, e in zip(observed, prefix):
        experiment.compare_row(a, e, ["V_ref", "dV_dt", "reference_ready", *experiment.DECISION_COLUMNS], {})
    for i in range(25, 40):
        later = raw(i)
        later["Volt"] = 5
        second.consume(later)
    assert observed[-1]["V_ref"] == 24


def test_gap_resets_rolling_slope_and_timer_but_keeps_reference():
    config = load_config()
    detector = StreamingDetector(bundle(config), config)
    original = [detector.consume(raw(i)) for i in range(36)]
    assert original[-1]["anomaly_confirmed"]
    after = detector.consume(raw(50))
    assert after["V_ref"] == 24 and after["segment_id"] == 1
    assert after["dI_dt"] == after["dV_dt"] == after["anomaly_duration_seconds"] == 0
    assert not after["anomaly_confirmed"] and after["detector_state"] == "SUSPECT"
    assert after["residual_slope_count_0p5s"] == 1 and not after["residual_slope_ready_0p5s"]
    for i in range(51, 56):
        warning = detector.consume(raw(i))
    assert warning["confirmed_episode_start"]


def test_invalid_numeric_drop_does_not_fill_or_advance_reference():
    config = load_config()
    detector = StreamingDetector(bundle(config), config)
    a = raw(0)
    assert detector.consume(a)["accepted"]
    bad = raw(1)
    bad["Volt"] = "nan"
    assert not detector.consume(bad)["accepted"]
    good = detector.consume(raw(2))
    assert good["dt_seconds"] == .2 and good["reference_low_current_samples"] == 2
    with pytest.raises(DataError, match="strictly increase"):
        detector.consume(bad)  # dropped payload still participates in identity audit


@pytest.mark.parametrize("mutation", [("TimeUS", 0), ("Inst", 1)])
def test_duplicate_and_unselected_instances_rejected(mutation):
    config = load_config()
    detector = StreamingDetector(bundle(config), config)
    detector.consume(raw(0))
    bad = raw(1)
    bad[mutation[0]] = mutation[1]
    with pytest.raises(DataError):
        detector.consume(bad)


def test_reference_unavailable_and_truncated_warmup_never_predict():
    config = load_config()
    b = bundle(config)
    detector = StreamingDetector(b, config)
    for i in range(35):
        row = raw(i)
        row["Curr"] = 5
        result = detector.consume(row)
    assert result["reference_status"] == "REFERENCE_UNAVAILABLE" and result["predicted_voltage_sag"] is None
    assert b["model"].calls == 0
    detector = StreamingDetector(b, config)
    for i in range(11):
        result = detector.consume(raw(i))
    assert result["reference_status"] == "WARMUP" and b["model"].calls == 0


def test_exact_cache_never_shares_across_models_or_reuses_alarm_state():
    config = load_config()
    b, other = bundle(config), bundle(config, .4)
    cache = {}
    first = StreamingDetector(b, config, prediction_cache=cache)
    expected = [first.consume(raw(i)) for i in range(41)]
    calls = b["model"].calls
    second = StreamingDetector(b, config, prediction_cache=cache, use_cache=True)
    values = []
    for i in range(41):
        r = raw(i)
        if i >= 30:
            r["Volt"] = 24  # current predictors identical, residual/decisions must change
        values.append(second.consume(r))
    assert b["model"].calls == calls
    assert values[-1]["prediction_mode"] == "EXACT_INPUT_CACHE" and not values[-1]["anomaly_confirmed"]
    assert expected[-1]["anomaly_confirmed"]
    independent = StreamingDetector(other, config, prediction_cache=cache, use_cache=True)
    for i in range(23):
        result = independent.consume(raw(i))
    assert other["model"].calls == 2 and result["prediction_mode"] == "UNCACHED"


def test_monotonic_phase_measurements_sum_and_ignore_telemetry_gap():
    config = load_config()
    ticks = count(100, 10)
    detector = StreamingDetector(bundle(config), config, clock=lambda: next(ticks))
    before = detector.consume(raw(0))
    after = detector.consume(raw(10000))
    assert after["elapsed_time"] == 1000
    assert before["end_to_end_ns"] == after["end_to_end_ns"] == 40
    assert sum(after[n] for n in TIMING_COLUMNS[:-1]) == after["end_to_end_ns"]


@pytest.mark.parametrize("extra", ["anomaly_score", "segment_id", "fault_support", "voltage_sag_actual"])
def test_processed_or_fault_fields_are_not_streaming_inputs(extra):
    config = load_config()
    r = raw(0)
    r[extra] = 1
    with pytest.raises(DataError, match="only raw battery"):
        StreamingDetector(bundle(config), config).consume(r)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("replay")
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    values = load_config().as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] = 3
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    (root / "configs/experiment_protocol.md").write_text("Test-only frozen replay baseline.\n", encoding="utf-8")
    rows = []
    for flight in ("a", "b", "c", "idle", "truncated"):
        for i in range(11 if flight == "truncated" else 81):
            current = 10 + i % 3 if flight in ("a", "b", "c") and 30 <= i <= 40 else .1
            rows.append(dict(NamaFile=flight, TimeUS=i * 100000, Inst=0,
                             Volt=24 - .02 * current - i * .0001, Curr=current, CurrTot=i * .01))
    with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    cfg = load_config(root=root)
    audit = cfg.artifacts_path / "audit"
    audit.mkdir(parents=True)
    names = ("configs/config.yaml", "configs/experiment_protocol.md", "data/raw/dataset_baterai.csv")
    (audit / "freeze_manifest.json").write_bytes(encoded(dict(files={n: sha(root / n) for n in names})))
    baseline = run_loafo(cfg)
    pointer = cfg.artifacts_path / "sprint6/cross_validation_result.json"
    pointer.parent.mkdir()
    pointer.write_bytes(encoded(baseline))
    low = run_low_load(cfg)
    pointer = cfg.artifacts_path / "sprint10/low_load_result.json"
    pointer.parent.mkdir()
    pointer.write_bytes(encoded(low))
    from sklearn.ensemble import HistGradientBoostingRegressor
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*a, **kw):
            pytest.fail("replay attempted fitting/calibration")
        patch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
        import src.detectors as detectors
        patch.setattr(detectors, "calibrate_threshold", forbidden)
        result = experiment.run_replay(cfg)
    return cfg, result, baseline


def test_full_replay_matches_frozen_baseline_and_preserves_na(completed):
    cfg, result, baseline = completed
    assert result["batch_stream_equivalence"] == "PASS"
    assert result["healthy_active"] == pytest.approx(baseline["healthy_active_pooled"])
    assert result["synthetic"] == baseline["synthetic_pooled"]
    assert result["synthetic"]["not_applicable_scenarios"] == 54
    assert result["replay_jobs"] == 5 + 27
    assert result["prediction_mode_counts"]["EXACT_INPUT_CACHE"] > 0
    assert result["prediction_mode_counts"]["UNCACHED"] == 240  # four complete ready flights
    assert not result["model_training_performed"] and not result["threshold_changed"]
    assert not result["emergency_candidate_replayed"] and not result["paced_replay_performed"]


def test_cached_timing_evidence_is_not_remeasured(completed, monkeypatch):
    cfg, result, _ = completed
    run = Path(result["output_directory"])
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(StreamingDetector, "consume", lambda *a: pytest.fail("cache remeasured"))
    assert experiment.run_replay(cfg) == result
    assert all((sha(p), p.stat().st_mtime_ns) == v for p, v in before.items())


def test_corrupt_replay_or_source_is_rejected(completed):
    cfg, result, baseline = completed
    for target in (Path(result["output_directory"]) / "latency_summary.csv", Path(baseline["output_directory"]) / "folds/fold_00/threshold.json"):
        data = target.read_bytes()
        try:
            target.write_bytes(b"corrupt")
            with pytest.raises(DataError, match="checksum"):
                experiment.run_replay(cfg)
        finally:
            target.write_bytes(data)


def test_cli_replay_is_implemented_and_reuses_frozen_run(completed, monkeypatch, capsys):
    import json
    cfg, result, _ = completed
    monkeypatch.setattr(main, "load_config", lambda _: cfg)
    assert main.COMMANDS["replay"].implemented
    assert main.main(["replay"]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_equivalence_and_undefined_latency_checks_are_strict():
    assert experiment.latency_stats([], 100)["p99_ms"] is None
    with pytest.raises(DataError, match="Replay/batch mismatch"):
        experiment.compare_row(dict(x=False), dict(x=None), ["x"], {})
    with pytest.raises(DataError, match="Replay/batch mismatch"):
        experiment.compare_row(dict(x=.6), dict(x=.5), ["x"], {})
