"""Sprint 7 estimator isolation, fixed policies and paired experiment contracts."""

from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

import main
import src.comparison as comparison
from src.config import AppConfig, load_config
from src.cross_validation import FlightSplit
from src.errors import ConfigError, DataError
from src.experimental_models import Candidate, candidates, fit_candidate, frame, load_candidate_bundle, score_candidate
from src.features import generate_features
from src.models import _feature_input


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    values = load_config().as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] = 5
    values["models"]["random_forest"].update(n_estimators=5, n_jobs=1)
    values["models"]["isolation_forest"].update(n_estimators=5, max_samples=32)
    values["models"]["one_class_svm"]["max_train_samples"] = 25
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")

    def prepare(short=False):
        rows = []
        for n, flight in enumerate(("a", "b", "c")):
            for i in range(81):
                active = 30 <= i <= 40 if short else 25 <= i <= 65
                current = 10 + i % 3 if active else 0
                rows.append(dict(NamaFile=flight, Inst=0, TimeUS=i * 100000, Volt=24 - .02 * current - i * .0001 - n * .001,
                                 Curr=current, CurrTot=i * .01 + n * .1))
        with (root / "data/raw/dataset_baterai.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
            writer.writeheader()
            writer.writerows(rows)
        cfg = load_config(root=root)
        feature_summary = generate_features(cfg)
        features, _, _, _ = _feature_input(feature_summary, cfg)
        return cfg, features
    return root, prepare


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_predeclared_candidate_matrix_and_exact_feature_roles():
    config = load_config()
    assert len(candidates("ablation")) == 10 and len(candidates("compare-models")) == 4
    for model in ("hgb", "rf"):
        for name in ("a", "b", "c", "d", "e"):
            spec = Candidate(model, "battery_" + name)
            predictors = spec.predictors(config)
            assert not set(predictors) & {"Volt", "V_ref", "voltage_sag_actual", "dV_dt", "TimeUS", "Inst"}
            assert ("elapsed_time" in predictors) == (name == "e")
            assert ("CurrTot" in predictors) == (name != "a")
    assert Candidate("if", "direct").predictors(config) == config.as_dict()["features"]["baseline_predictors"]
    with pytest.raises(DataError):
        candidates("tune")


@pytest.mark.parametrize("model", ["hgb", "rf", "if", "ocsvm"])
def test_fitted_score_direction_and_regression_voltage_leakage(workspace, model):
    _, prepare = workspace
    config, rows = prepare()
    spec = Candidate(model, "battery_d" if model in ("hgb", "rf") else "direct")
    split = FlightSplit(("a",), ("b",), ("c",))
    train = [r for r in rows if r["NamaFile"] == "a" and r["reference_ready"]]
    bundle = fit_candidate(train, config, split, spec)
    validation = [r for r in rows if r["NamaFile"] == "b"]
    scored = score_candidate(bundle, validation, config, "validation")
    ready = [r for r in validation if r["reference_ready"]]
    scores = [r["anomaly_score"] for r in scored if r["reference_ready"]]
    x = frame(ready, config, spec)
    if spec.regression:
        expected = np.maximum(0, np.array([r["voltage_sag_actual"] for r in ready]) - bundle["model"].predict(x))
        changed = deepcopy(validation)
        for row in changed:
            if row["reference_ready"]:
                row["voltage_sag_actual"] += .8
                row["Volt"] -= .8
                row["dV_dt"] += 100
        predicted = score_candidate(bundle, changed, config, "validation", healthy_cache=scored)
        assert [r["predicted_voltage_sag"] for r in predicted] == [r["predicted_voltage_sag"] for r in scored]
        for before, after in zip(scored, predicted):
            if before["reference_ready"]:
                assert after["residual_signed_v"] == pytest.approx(before["residual_signed_v"] + .8)
        changed[-1]["Curr"] += 1
        with pytest.raises(DataError, match="predictors changed"):
            score_candidate(bundle, changed, config, "validation", healthy_cache=scored)
    else:
        transformed = bundle["scaler"].transform(x) if bundle["scaler"] else x
        expected = -bundle["model"].decision_function(transformed)
        assert all(r["predicted_voltage_sag"] is None and r["residual_signed_v"] is None for r in scored)
        assert bundle["score_unit"] != "V"
        changed = deepcopy(validation)
        for row in changed:
            if row["reference_ready"]:
                row["voltage_sag_actual"] += 20
                row["dV_dt"] += 50
        altered = score_candidate(bundle, changed, config, "validation")
        assert any(a["anomaly_score"] != b["anomaly_score"] for a, b in zip(scored, altered) if a["reference_ready"])
        with pytest.raises(DataError, match="aligned regression"):
            score_candidate(bundle, validation, config, "validation", healthy_cache=scored)
    assert scores == pytest.approx(expected)
    assert all(r["anomaly_score"] is None for r in scored if not r["reference_ready"])


def test_ocsvm_scaler_and_capped_sample_identities_are_train_only(workspace):
    _, prepare = workspace
    config, rows = prepare()
    train = [r for r in rows if r["NamaFile"] == "a" and r["reference_ready"]]
    spec, split = Candidate("ocsvm", "direct"), FlightSplit(("a",), ("b",), ("c",))
    bundle = fit_candidate(train, config, split, spec)
    provenance = bundle["training_provenance"]
    indices = np.sort(np.random.default_rng(42).choice(len(train), 25, replace=False))
    assert provenance["fit_sample_indices"] == indices.tolist()
    assert provenance["model_fit_samples"] == 25 and provenance["scaler_fit_samples"] == len(train)
    assert {r["NamaFile"] for r in provenance["fit_sample_identities"]} == {"a"}
    assert bundle["scaler"].mean_ == pytest.approx(frame(train, config, spec).mean().to_numpy())
    assert bundle["model"].support_.max() < 25
    mutated = deepcopy(rows)
    for r in mutated:
        if r["NamaFile"] != "a":
            r["Curr"] += 1000
    again = fit_candidate([r for r in mutated if r["NamaFile"] == "a" and r["reference_ready"]], config, split, spec)
    assert again["training_provenance"] == provenance
    np.testing.assert_array_equal(again["scaler"].mean_, bundle["scaler"].mean_)
    np.testing.assert_array_equal(again["model"].support_vectors_, bundle["model"].support_vectors_)


@pytest.mark.parametrize("bad", ["synthetic", "validation", "unready", "wrong_flight"])
def test_candidate_fitting_rejects_data_leakage(workspace, bad):
    _, prepare = workspace
    config, rows = prepare()
    train = [r for r in rows if r["NamaFile"] == "a" and r["reference_ready"]]
    if bad == "synthetic":
        train[0]["is_synthetic"] = False
    elif bad == "validation":
        train[0]["split_role"] = "validation"
    elif bad == "unready":
        train[0]["reference_ready"] = False
    else:
        train[0]["NamaFile"] = "c"
    with pytest.raises(DataError):
        fit_candidate(train, config, FlightSplit(("a",), ("b",), ("c",)), Candidate("rf", "battery_a"))


def test_signed_direct_threshold_is_not_clipped_and_preserves_timestamp_persistence():
    config = load_config()
    split = FlightSplit(("a",), ("b",), ("c",))
    rows = [dict(NamaFile="b", Inst=0, source_record=i, segment_id=0, TimeUS=i * 100000, reference_ready=True,
                 split_role="validation", anomaly_score=-4 + i * .01) for i in range(20)]
    bundle = dict(config=config.as_dict(), regression=False, score_definition="-decision_function", score_unit="model_score_not_volts")
    threshold = comparison.calibrate(rows, split, bundle)
    assert threshold["threshold"] == pytest.approx(-3.8119) and threshold["threshold_v"] is None
    decisions = comparison.decide([{**r, "anomaly_score": -3.} for r in rows], -3.8, config)
    assert not decisions[4]["anomaly_confirmed"] and decisions[5]["confirmed_episode_start"]
    assert decisions[5]["anomaly_duration_seconds"] == .5
    with pytest.raises(DataError):
        comparison.calibrate([{**r, "split_role": "test"} for r in rows], split, bundle)
    with pytest.raises(DataError):
        comparison.decide([{**rows[0], "anomaly_score": float("nan")}], -3.8, config)


@pytest.mark.parametrize("mode", ["ablation", "compare-models"])
def test_all_candidate_fold_models_frozen_before_outer_and_complete_pairing(workspace, monkeypatch, mode):
    root, prepare = workspace
    config, _ = prepare()
    score = comparison.score_candidate
    seen = []
    number = len(candidates(mode))

    def guard(bundle, rows, config, role, **kwargs):
        if role == "test":
            run = next((root / "artifacts" / ("ablation" if mode == "ablation" else "model_comparison")).iterdir())
            assert len(list(run.glob("candidates/*/*/model_bundle.joblib"))) == number * 3
            frozen = json.loads((run / "all_policies_frozen.json").read_text())
            assert frozen["candidate_folds"] == number * 3
            seen.append(bundle["candidate"])
        return score(bundle, rows, config, role, **kwargs)

    monkeypatch.setattr(comparison, "score_candidate", guard)
    result = comparison.run_comparison(config, mode)
    assert result["candidate_folds"] == number * 3 and result["scenario_evaluations"] == number * 81
    assert result["selected_winner"] is None and result["operational_config_changed"] is False
    run = Path(result["output_directory"])
    scenarios = read_csv(run / "fault_per_scenario.csv")
    pairs = {}
    for record in scenarios:
        signature = tuple(record[k] for k in ("fault_applicable", "onset_time_us", "fault_end_time_us", "support_samples", "observability", "frozen_v_ref"))
        assert pairs.setdefault(record["scenario_id"], signature) == signature
    assert len(pairs) == 81
    assert len(read_csv(run / "regression_per_fold.csv")) == (number if mode == "ablation" else 2) * 3
    results = json.loads((run / "results.json").read_text())
    for path in run.glob("candidates/*/*/model_bundle.joblib"):
        bundle = load_candidate_bundle(path)
        assert bundle["training_provenance"]["synthetic_rows"] == 0
        if not bundle["regression"]:
            assert results["candidates"][bundle["candidate"]]["pooled"]["regression"] is None
    monkeypatch.setattr(comparison, "fit_candidate", lambda *a: pytest.fail("cached experiment refitted"))
    assert comparison.run_comparison(config, mode) == result
    (run / "summary.csv").write_text("corrupted")
    with pytest.raises(DataError, match="checksum"):
        comparison.run_comparison(config, mode)


def test_not_applicable_never_scored_and_metrics_null(workspace, monkeypatch):
    _, prepare = workspace
    config, _ = prepare(short=True)
    score = comparison.score_candidate

    def guard(bundle, rows, config, role, **kwargs):
        if "is_synthetic" in rows[0]:
            assert rows[0]["fault_applicable"] and rows[0]["fault_family"] == "sudden_drop"
        return score(bundle, rows, config, role, **kwargs)

    monkeypatch.setattr(comparison, "score_candidate", guard)
    result = comparison.run_comparison(config, "compare-models")
    records = read_csv(Path(result["output_directory"]) / "fault_per_scenario.csv")
    rejected = [r for r in records if r["fault_applicable"] == "False"]
    assert len(rejected) == 4 * 54
    assert all(r["tp"] == r["fp"] == r["scores_csv"] == r["incident_detected"] == r["deadline_le_5s"] == "" for r in rejected)


def test_bundle_reload_rejects_changed_schema_and_role_mismatch(workspace, tmp_path):
    _, prepare = workspace
    config, rows = prepare()
    split = FlightSplit(("a",), ("b",), ("c",))
    train = [r for r in rows if r["NamaFile"] == "a" and r["reference_ready"]]
    bundle = fit_candidate(train, config, split, Candidate("rf", "battery_a"))
    val = [r for r in rows if r["NamaFile"] == "b"]
    bundle["threshold"] = comparison.calibrate(score_candidate(bundle, val, config, "validation"), split, bundle)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    restored = load_candidate_bundle(path)
    assert score_candidate(restored, val, config, "validation") == score_candidate(bundle, val, config, "validation")
    with pytest.raises(DataError, match="role"):
        score_candidate(bundle, val, config, "test")
    bundle["predictors"].append("Volt")
    joblib.dump(bundle, path)
    with pytest.raises(DataError, match="schema"):
        load_candidate_bundle(path)


def test_sprint7_cli_handlers_and_fixed_policy_rejection(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config, _ = prepare()
    for command in ("ablation", "compare-models"):
        assert main.COMMANDS[command].implemented
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["compare-models"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 4
    values = config.as_dict()
    values["threshold"]["reference_percentile"] = 95
    with pytest.raises(ConfigError, match="no tuning"):
        comparison.run_comparison(AppConfig(config.source, config.root, values), "ablation")
