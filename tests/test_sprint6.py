"""LOAFO isolation, incident/denominator edge cases and complete-run verification."""

from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import main
import src.loafo as loafo
from src.config import AppConfig, PROJECT_ROOT, load_config
from src.cross_validation import loafo_splits
from src.errors import ConfigError, DataError
from src.evaluation import aggregate_scenarios, count_metrics, deadline_name, distribution, scenario_metrics
from src.models import load_model_bundle

DEADLINES = [1., 2., 5., 10., 30.]


def decision(time, *, support=False, confirmed=False, episode=False, raw=None, ready=True, segment=0):
    return dict(NamaFile="f", Inst=0, segment_id=segment, TimeUS=time, reference_ready=ready,
                fault_support=support, anomaly_raw=confirmed if raw is None else raw,
                anomaly_confirmed=confirmed, confirmed_episode_start=episode)


def metadata(applicable=True):
    return dict(fault_applicable=applicable, onset_time_us=1000000 if applicable else None,
                fault_end_time_us=2000000 if applicable else None)


def clean_detection():
    return [decision(0), decision(500000), decision(1000000, support=True, raw=True),
            decision(1500000, support=True, confirmed=True, episode=True),
            decision(2000000, support=True, confirmed=True), decision(2500000, confirmed=True)]


def test_every_flight_tests_once_validates_once_and_cyclic_wrap_is_deterministic():
    folds = loafo_splits(["d", "a", "c", "b"])
    assert [s.test[0] for s in folds] == ["a", "b", "c", "d"]
    assert [s.validation[0] for s in folds] == ["b", "c", "d", "a"]
    for split in folds:
        assert len(split.train) == 2
        assert not set(split.train) & set(split.validation + split.test)
        assert not set(split.validation) & set(split.test)
    assert folds == loafo_splits(["b", "d", "a", "c"])


@pytest.mark.parametrize("flights", [[], ["a", "b"], ["a", "b", "b"]])
def test_insufficient_or_duplicate_groups_cannot_run_loafo(flights):
    with pytest.raises(DataError):
        loafo_splits(flights)


def test_conventional_confusion_and_incident_delay_use_distinct_denominators():
    result = scenario_metrics(clean_detection(), metadata(), DEADLINES, .5)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (2, 1, 2, 1)
    assert result["precision"] == result["recall"] == result["f1"] == pytest.approx(2 / 3)
    assert result["synthetic_sample_fpr"] == pytest.approx(1 / 3)
    assert result["pre_fault_samples"] == 2 and result["pre_fault_fpr"] == 0
    assert result["evaluation_status"] == "DETECTED" and result["incident_detected"]
    assert result["first_confirmation_time_us"] == 1500000 and result["delay_seconds"] == .5
    assert result["confirmed_episodes"] == 1 and result["observed_ready_seconds"] == 2.5
    assert all(result[deadline_name(d)] for d in DEADLINES)


def test_undetected_is_a_failure_at_every_deadline_with_null_delay():
    rows = [decision(t, support=1000000 <= t <= 2000000) for t in range(0, 2500001, 500000)]
    result = scenario_metrics(rows, metadata(), DEADLINES, .5)
    assert result["evaluation_status"] == "UNDETECTED" and not result["incident_detected"]
    assert result["delay_seconds"] is None and result["first_confirmation_time_us"] is None
    assert result["recall"] == result["f1"] == 0 and result["precision"] is None
    assert result["detection_reason"] == "NO_RAW_EXCEEDANCE_IN_SUPPORT"
    assert not any(result[deadline_name(d)] for d in DEADLINES)


def test_preexisting_alarm_cannot_be_zero_delay_or_later_incident_success():
    rows = [decision(0), decision(500000, confirmed=True, episode=True),
            decision(1000000, support=True, confirmed=True), decision(1500000, support=True),
            decision(2000000, support=True, confirmed=True, episode=True), decision(2500000)]
    result = scenario_metrics(rows, metadata(), DEADLINES, .5)
    assert result["preexisting_alarm"] and result["evaluation_status"] == "PREEXISTING_ALARM"
    assert result["tp"] == 2  # conventional sample metrics are still reported
    assert result["candidate_new_confirmation_time_us"] == 2000000
    assert result["first_confirmation_time_us"] is None and result["delay_seconds"] is None
    assert not result["incident_detected"] and not any(result[deadline_name(d)] for d in DEADLINES)


def test_genuine_new_confirmation_at_onset_is_not_mislabelled_preexisting():
    rows = [decision(0), decision(500000, raw=True), decision(1000000, support=True, confirmed=True, episode=True),
            decision(1500000, support=True, confirmed=True), decision(2000000, support=True, confirmed=True)]
    result = scenario_metrics(rows, metadata(), DEADLINES, .5)
    assert not result["preexisting_alarm"] and result["incident_detected"]
    assert result["delay_seconds"] == 0


def test_post_support_confirmation_does_not_detect_the_incident():
    rows = [decision(0), decision(500000), decision(1000000, support=True), decision(1500000, support=True),
            decision(2000000, support=True, raw=True), decision(2500000, confirmed=True, episode=True)]
    result = scenario_metrics(rows, metadata(), DEADLINES, .5)
    assert not result["incident_detected"] and result["fp"] == 1
    assert result["delay_seconds"] is None


def test_not_applicable_never_produces_confusion_counts_or_deadline_failures():
    result = scenario_metrics([], metadata(False), DEADLINES, .5)
    assert result["evaluation_status"] == "NOT_APPLICABLE"
    assert all(result[key] is None for key in ("tp", "fp", "tn", "fn", "f1", "incident_detected", "delay_seconds"))
    assert all(result[deadline_name(d)] is None for d in DEADLINES)
    aggregate = aggregate_scenarios([result], DEADLINES)
    assert aggregate["applicable_scenarios"] == 0 and aggregate["not_applicable_scenarios"] == 1
    assert aggregate["incident_recall"] is None and aggregate["deadline_le_1s_rate"] is None


def test_applicable_with_bad_support_metadata_is_rejected():
    with pytest.raises(DataError, match="support"):
        scenario_metrics([decision(0)], metadata(), DEADLINES, .5)


def test_aggregation_keeps_misses_contamination_and_applicable_denominator():
    detected = scenario_metrics(clean_detection(), metadata(), DEADLINES, .5)
    missed_rows = [decision(t, support=1000000 <= t <= 2000000) for t in range(0, 2500001, 500000)]
    missed = scenario_metrics(missed_rows, metadata(), DEADLINES, .5)
    contaminated = deepcopy(detected)
    contaminated.update(incident_detected=False, preexisting_alarm=True, evaluation_status="PREEXISTING_ALARM", delay_seconds=None)
    contaminated.update({deadline_name(d): False for d in DEADLINES})
    result = aggregate_scenarios([detected, missed, contaminated, scenario_metrics([], metadata(False), DEADLINES, .5)], DEADLINES)
    assert result["scenarios"] == 4 and result["applicable_scenarios"] == 3 and result["not_applicable_scenarios"] == 1
    assert result["detected_scenarios"] == 1 and result["undetected_scenarios"] == 2
    assert result["preexisting_alarm_scenarios"] == result["undetected_without_preexisting"] == 1
    assert result["incident_recall"] == result["deadline_le_1s_rate"] == pytest.approx(1 / 3)
    assert result["delay_detected_n"] == 1 and result["delay_mean_seconds"] == result["delay_median_seconds"] == .5
    assert result["delay_std_seconds"] == 0
    assert result["tp"] == detected["tp"] + missed["tp"] + contaminated["tp"]


def test_distribution_is_population_std_and_records_defined_values():
    result = distribution([1, 3, None])
    assert result == dict(n_defined=2, mean=2, std=1, median=2, min=1, max=3)
    assert distribution([None])["n_defined"] == 0 and distribution([None])["mean"] is None


def test_pooled_ratios_are_counts_not_macro_mean():
    assert count_metrics(1, 0, 1, 0)["f1"] == 1
    assert count_metrics(0, 9, 1, 9)["f1"] == 0
    assert count_metrics(1, 9, 2, 9)["f1"] == .1
    assert count_metrics(0, 0, 0, 0)["f1"] is None


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    cfg = load_config().as_dict()
    cfg["models"]["hist_gradient_boosting"]["max_iter"] = 8
    (root / "configs/config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def prepare(short_active=False):
        records = []
        for flight in ("a", "b", "c"):
            for index in range(81):
                active = (30 <= index <= 40) if short_active else (25 <= index <= 65)
                current = 10 + index % 3 if active else 0
                records.append(dict(NamaFile=flight, Inst=0, TimeUS=index * 100000, Volt=24 - .02 * current - index * .0001,
                                    Curr=current, CurrTot=index * .01))
        with (root / "data/raw/dataset_baterai.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
            writer.writeheader()
            writer.writerows(records)
        return load_config(root=root)
    return root, prepare


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_fold_policies_and_models_are_saved_before_any_outer_prediction(workspace, monkeypatch):
    root, prepare = workspace
    config = prepare()
    fit, score = loafo.fit_hgb, loafo.score_rows
    assigned = {}

    def guard_fit(rows, config, split):
        assert {r["NamaFile"] for r in rows} == set(split.train)
        assert all(r["reference_ready"] and "is_synthetic" not in r for r in rows)
        assert any(r["Curr"] == 0 for r in rows)
        model = fit(rows, config, split)
        assigned[id(model)] = split
        return model

    def guard_score(model, rows, config, role):
        split = assigned[id(model)]
        assert {r["NamaFile"] for r in rows} == set(split.validation if role == "validation" else split.test)
        if role == "test":
            run = next((root / "artifacts/cross_validation").iterdir())
            assert (run / "all_fold_policies_frozen.json").is_file()
            assert len(list(run.glob("folds/*/selected_policy.json"))) == 3
            assert len(list(run.glob("folds/*/model_bundle.joblib"))) == 3
        return score(model, rows, config, role)

    monkeypatch.setattr(loafo, "fit_hgb", guard_fit)
    monkeypatch.setattr(loafo, "score_rows", guard_score)
    result = loafo.run_loafo(config)
    assert result["folds"] == 3 and result["all_policies_frozen_before_outer"]
    assert result["synthetic_pooled"]["scenarios"] == result["synthetic_pooled"]["applicable_scenarios"] == 81
    run = Path(result["output_directory"])
    assert len(read_csv(run / "regression_per_fold.csv")) == 3
    assert len(read_csv(run / "fold_assignments.csv")) == 9
    assert len(read_csv(run / "fault_per_scenario.csv")) == 81
    for path in run.glob("folds/*/model_bundle.joblib"):
        bundle = load_model_bundle(path)
        assert bundle["model"].early_stopping is False and bundle["training_provenance"]["synthetic_rows"] == 0
        assert set(bundle["split"]["train"]).isdisjoint(bundle["split"]["test"] + bundle["split"]["validation"])


def test_not_applicable_scenarios_are_saved_without_scoring_or_counts(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare(short_active=True)
    score = loafo.score_rows
    seen = []

    def guard(model, rows, config, role):
        if rows and "is_synthetic" in rows[0]:
            assert rows[0]["fault_applicable"] and rows[0]["fault_family"] == "sudden_drop"
            seen.append(rows[0]["scenario_id"])
        return score(model, rows, config, role)
    monkeypatch.setattr(loafo, "score_rows", guard)
    result = loafo.run_loafo(config)
    assert len(seen) == result["synthetic_pooled"]["applicable_scenarios"] == 27
    assert result["synthetic_pooled"]["not_applicable_scenarios"] == 54
    records = read_csv(Path(result["output_directory"]) / "fault_per_scenario.csv")
    rejected = [r for r in records if r["fault_applicable"] == "False"]
    assert len(rejected) == 54
    assert all(r["tp"] == r["fp"] == r["tn"] == r["fn"] == r["delay_seconds"] == r["scores_csv"] == "" for r in rejected)
    assert all(r["evaluation_status"] == "NOT_APPLICABLE" for r in rejected)


def test_complete_run_is_reused_hashes_validate_and_corruption_is_not_overwritten(workspace, monkeypatch):
    root, prepare = workspace
    config = prepare()
    result = loafo.run_loafo(config)
    run = Path(result["output_directory"])
    manifest = json.loads((run / "manifest.json").read_text())
    assert all(hashlib.sha256((run / p).read_bytes()).hexdigest() == h for p, h in manifest["outputs"].items())
    monkeypatch.setattr(loafo, "fit_hgb", lambda *a: pytest.fail("Completed run must not refit"))
    assert loafo.run_loafo(config) == result
    (run / "fault_per_scenario.csv").write_text("corrupted")
    with pytest.raises(DataError, match="checksum mismatch"):
        loafo.run_loafo(config)
    assert (run / "fault_per_scenario.csv").read_text() == "corrupted"


def test_baseline_policy_drift_is_rejected_before_test_evaluation():
    config = load_config()
    cfg = config.as_dict()
    cfg["threshold"]["reference_percentile"] = 95
    with pytest.raises(ConfigError, match="reference baseline"):
        loafo.run_loafo(AppConfig(config.source, config.root, cfg))


def test_cli_cross_validation_success(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config = prepare()
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["cross-validate"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["command"] == "cross-validate" and not captured.err


def test_real_cli_loafo_from_other_cwd_and_pooled_count_oracles(tmp_path):
    raw = (PROJECT_ROOT / "data/raw/dataset_baterai.csv").read_bytes()
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "cross-validate"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["folds"] == 6 and summary["synthetic_pooled"]["scenarios"] == 162
    run = Path(summary["output_directory"])
    assert len(read_csv(run / "regression_per_fold.csv")) == len(read_csv(run / "anomaly_per_fold.csv")) == 6
    scenarios = read_csv(run / "fault_per_scenario.csv")
    applicable = [r for r in scenarios if r["fault_applicable"] == "True"]
    detected = [r for r in applicable if r["incident_detected"] == "True"]
    assert len(applicable) == summary["synthetic_pooled"]["applicable_scenarios"]
    assert len(detected) == summary["synthetic_pooled"]["detected_scenarios"]
    assert sum(int(r["tp"]) for r in applicable) == summary["synthetic_pooled"]["tp"]
    assert sum(int(r["fp"]) for r in applicable) == summary["synthetic_pooled"]["fp"]
    assert sum(r["deadline_le_1s"] == "True" for r in applicable) / len(applicable) == summary["synthetic_pooled"]["deadline_le_1s_rate"]
    assert all(r["delay_seconds"] == "" and r["incident_detected"] == "False" for r in applicable if r["preexisting_alarm"] == "True")
    healthy = read_csv(run / "healthy_active_per_fold.csv")
    assert sum(int(r["ready_samples"]) for r in healthy) == summary["healthy_active_pooled"]["ready_samples"] == 26870
    assert sum(int(r["confirmed_alarm_samples"]) for r in healthy) == summary["healthy_active_pooled"]["confirmed_alarm_samples"]
    assert (PROJECT_ROOT / "data/raw/dataset_baterai.csv").read_bytes() == raw
