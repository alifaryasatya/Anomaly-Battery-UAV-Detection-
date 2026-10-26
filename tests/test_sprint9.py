"""Causal residual history, independent timers, validation shortlist and gate."""

from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path

import pytest
import yaml

import main
import src.dual_path_experiment as experiment
from src.config import AppConfig, load_config
from src.cross_validation import FlightSplit
from src.dual_detector import (apply_dual, attribution_metrics, candidate_grid, calibrate_slopes, choose_path_a,
                               choose_refined, promotion_decision, shortlist)
from src.errors import ConfigError, DataError
from src.evaluation import scenario_metrics
from src.experimental_models import load_candidate_bundle
from src.feature_sets import validate_regression_predictors
from src.residual_diagnostics import ResidualDiagnostics, diagnose, slope_columns


def row(time, residual=0., *, flight="v", segment=0, ready=True):
    return dict(NamaFile=flight, Inst=0, TimeUS=time, source_record=time // 100000, segment_id=segment,
                reference_ready=ready, residual_signed_v=residual if ready else None,
                anomaly_score=max(0., residual) if ready else None, split_role="validation")


def policy(**kwargs):
    p = dict(mode="DUAL", threshold_high_v=1., threshold_low_v=.1, threshold_slope_v_per_s=1.,
             slope_window_seconds=.3, path_b_persistence_seconds=.3)
    p.update(kwargs)
    return p


def scored(time, score, slope, *, support=False, segment=0, slope_ready=True):
    r = row(time, score, segment=segment)
    name, count, flag = slope_columns(.3)
    r.update({name: slope if slope_ready else None, count: 2 if slope_ready else 1, flag: slope_ready,
              "fault_support": support})
    return r


def test_irregular_large_timestamps_and_signed_ols_have_independent_affine_oracle():
    origin = 10 ** 15
    times = [0, 50000, 170000, 300000, 410000]
    rows = [row(origin + t, -5 + 2.75 * t / 1e6) for t in times]
    result = diagnose(rows, [.3, .5, 1.], .5)
    assert not result[0]["residual_delta_ready"]
    assert result[0]["residual_delta_v"] is None
    for r in result[1:]:
        assert r["residual_delta_v_per_s"] == pytest.approx(2.75)
        for w in (.3, .5, 1.):
            assert r[slope_columns(w)[0]] == pytest.approx(2.75)
    # t=0 is exactly left boundary at .3 s and must be excluded.
    assert result[3][slope_columns(.3)[1]] == 3
    assert all(r["anomaly_score"] == 0 for r in result)  # slope uses signed residual, not clipped score


def test_residual_gap_segment_unready_reset_and_prefix_causality():
    rows = [row(0, 1), row(100000, 2), row(700001, 3, segment=1), row(800001, 4, segment=1),
            row(900001, ready=False, segment=1), row(1000001, 7, segment=1), row(1100001, 9, segment=1)]
    result = diagnose(rows, [.3], .5)
    flag = slope_columns(.3)[2]
    assert [r[flag] for r in result] == [False, True, False, True, False, False, True]
    assert result[2]["residual_delta_v"] is None and result[5]["residual_delta_v"] is None
    future = deepcopy(rows)
    future[-1]["residual_signed_v"] = 999
    assert diagnose(future, [.3], .5)[:-1] == result[:-1]
    builder = ResidualDiagnostics([.3], .5)
    assert [builder.consume(r) for r in rows[:4]] == result[:4]


def test_residual_history_is_per_stream_and_unregistered_for_regression():
    result = diagnose([row(0, 0, flight="a"), row(0, 99, flight="b"), row(100000, 1, flight="a")], [.3], .5)
    assert result[-1][slope_columns(.3)[0]] == pytest.approx(10.)
    with pytest.raises(ConfigError):
        validate_regression_predictors(["Curr", "residual_delta_v"], "battery_a", [.5, 1.])
    builder = ResidualDiagnostics([.3], .5)
    builder.consume(row(10))
    with pytest.raises(DataError, match="increase"):
        builder.consume(row(10))
    with pytest.raises(DataError):
        diagnose([row(0, float("nan"))], [.3], .5)


def test_slope_quantile_is_ready_healthy_validation_only_and_signed():
    split = FlightSplit(("t",), ("v",), ("o",))
    rows = diagnose([row(i * 100000, -i) for i in range(5)], [.3], .5)
    payload = calibrate_slopes(rows, split, [.3], [95.])[(.3, 95.)]
    assert payload["source_samples"] == 4 and payload["threshold_v_per_s"] == pytest.approx(-10.)
    for changed in ([dict(r, is_synthetic=False) for r in rows], [dict(r, split_role="test") for r in rows], [dict(r, NamaFile="o") for r in rows]):
        with pytest.raises(DataError):
            calibrate_slopes(changed, split, [.3], [95.])
    unavailable = calibrate_slopes(rows[:1], split, [.3], [95.])[(.3, 95.)]
    assert unavailable["source_samples"] == 0 and unavailable["threshold_v_per_s"] is None


def test_alternating_raw_paths_never_accumulate_shared_persistence():
    rows = [scored(i * 100000, 2 if i % 2 == 0 else .5, 0 if i % 2 == 0 else 2) for i in range(20)]
    result = apply_dual(rows, policy(), .5)
    assert all(r["anomaly_raw"] for r in result)
    assert not any(r["anomaly_confirmed"] for r in result)


def test_union_attribution_switch_has_one_episode_and_gap_restarts_timers():
    rows = [scored(i * 100000, 2 if i <= 6 else .5, 2 if i >= 3 else 0) for i in range(11)]
    rows.extend([scored(1700000, .5, 2, segment=1), scored(2000000, .5, 2, segment=1)])
    result = apply_dual(rows, policy(), .5)
    assert result[5]["alarm_attribution"] == "A_ONLY"
    assert result[6]["alarm_attribution"] == "BOTH" and result[6]["path_b_episode_start"]
    assert result[7]["alarm_attribution"] == "B_ONLY" and not result[7]["confirmed_episode_start"]
    assert sum(r["confirmed_episode_start"] for r in result[:11]) == 1
    assert result[11]["anomaly_confirmed"] is False and result[12]["confirmed_episode_start"]
    assert result[12]["first_confirmation_path"] == "B_ONLY"


def test_path_b_unready_slope_is_false_and_new_confirmation_at_onset_is_valid():
    rows = [scored(i * 100000, .5, 2, support=i >= 3, slope_ready=i != 5) for i in range(8)]
    # B begins at t=0 and confirms at t=.3, exactly at fault onset.
    result = apply_dual(rows, policy(threshold_high_v=10.), .5)
    metrics = scenario_metrics(result, dict(fault_applicable=True, onset_time_us=300000, fault_end_time_us=700000), [1., 2., 5., 10., 30.], .5)
    assert metrics["incident_detected"] and metrics["delay_seconds"] == 0
    assert attribution_metrics(result, metrics)["incident_first_path"] == "B_ONLY"
    assert not result[5]["path_b_raw"] and not result[5]["anomaly_confirmed"]


def test_carried_union_alarm_prevents_credit_for_new_other_path_and_na_path_counts_null():
    rows = [scored(i * 100000, 2 if i <= 6 else .5, 2 if i >= 3 else 0, support=i >= 6) for i in range(11)]
    result = apply_dual(rows, policy(), .5)
    metrics = scenario_metrics(result, dict(fault_applicable=True, onset_time_us=600000, fault_end_time_us=1000000), [1., 2., 5., 10., 30.], .5)
    assert metrics["preexisting_alarm"] and not metrics["incident_detected"] and metrics["tp"] > 0
    assert attribution_metrics(result, metrics)["incident_first_path"] is None
    na = scenario_metrics([], dict(fault_applicable=False), [1., 2., 5., 10., 30.], .5)
    assert all(v is None for v in attribution_metrics([], na).values())


def healthy_candidate(i, high=99.5):
    return dict(candidate_id=f"d{i}", high_percentile=high, low_percentile=95. + i / 10, slope_percentile=95.,
                slope_window_seconds=.3, path_b_persistence_seconds=.3, threshold_slope_v_per_s=.1,
                source_role="validation", validation_flight="v", healthy_confirmed_fpr=0., healthy_episodes_per_minute=0.)


def test_path_a_then_b_shortlist_is_deterministic_capped_and_validation_only():
    ref = dict(healthy_confirmed_fpr=.1, healthy_episodes_per_minute=.2)
    a = [dict(healthy_candidate(0, high=p), healthy_confirmed_fpr=.01) for p in (99.5, 99.)]
    assert choose_path_a(a, ref, "v")["high_percentile"] == 99.
    rows = [healthy_candidate(i) for i in (4, 2, 3, 0, 1)] + [healthy_candidate(99, high=99.)]
    short = shortlist(rows, ref, 99.5, "v")
    assert [r["candidate_id"] for r in short] == ["d0", "d1", "d2"]
    assert rows[-1]["eligibility_reason"] == "PATH_A_NOT_SELECTED"
    assert len(candidate_grid(load_config())) == 144
    with pytest.raises(DataError):
        shortlist([dict(rows[0], source_role="test")], ref, 99.5, "v")
    with pytest.raises(DataError):
        shortlist(rows, ref, 99.5, "v", 4)
    missing = [dict(healthy_candidate(0), threshold_slope_v_per_s=None)]
    assert shortlist(missing, ref, 99.5, "v") == []


def test_refinement_required_before_selection_and_frozen_lexicographic_tie():
    r = dict(healthy_candidate(0), healthy_eligible=True, shortlisted=True, synthetic_refined=False,
             abrupt_severe_le_1s_rate=.5, gradual_observable_le_5s_rate=.5, incident_recall=.9, f1=.9,
             delay_median_seconds=.4, delay_mean_seconds=.5)
    with pytest.raises(DataError, match="completed"):
        choose_refined([r], "v")
    r["synthetic_refined"] = True
    better = dict(r, candidate_id="better", abrupt_severe_le_1s_rate=.6, f1=0., delay_mean_seconds=100.)
    assert choose_refined([r, better], "v")["candidate_id"] == "better"
    undefined = dict(r, candidate_id="undefined", abrupt_severe_le_1s_rate=None)
    zero = dict(r, candidate_id="zero", abrupt_severe_le_1s_rate=0.)
    assert choose_refined([undefined, zero], "v")["candidate_id"] == "zero"
    assert choose_refined([], "v") is None


def gate_fixture():
    h = {k: dict(n_defined=3, mean=.1) for k in ("confirmed_fpr", "confirmed_episodes_per_minute")}
    syn = {k: dict(n_defined=3, mean=.8) for k in ("f1", *[f"deadline_le_{d}s_rate" for d in (1, 2, 5, 10, 30)])}
    syn["delay_mean_seconds"] = dict(n_defined=3, mean=2.)
    data = dict(reference=dict(macro_per_fold=dict(healthy_active=h, synthetic=syn)))
    data["selected"] = deepcopy(data["reference"])
    data["selected"]["macro_per_fold"]["synthetic"]["delay_mean_seconds"]["mean"] = 1.
    return data


@pytest.mark.parametrize("metric", ["confirmed_fpr", "confirmed_episodes_per_minute", "f1", "deadline_le_1s_rate", "deadline_le_30s_rate", "delay_mean_seconds"])
def test_every_promotion_rule_can_independently_fail(metric):
    data = gate_fixture()
    assert promotion_decision(data, 3, 1)["gate_passed"]
    scope = "healthy_active" if metric.startswith("confirmed") else "synthetic"
    data["selected"]["macro_per_fold"][scope][metric]["mean"] = .2 if scope == "healthy_active" else 2. if metric == "delay_mean_seconds" else .7
    assert promotion_decision(data, 3, 1)["status"] == "NOT_PROMOTED"


def test_promotion_undefined_fold_or_no_dual_fails_and_uses_macro_not_pooled():
    data = gate_fixture()
    assert promotion_decision(data, 3, 0)["status"] == "NOT_PROMOTED"
    data["selected"]["macro_per_fold"]["synthetic"]["f1"]["n_defined"] = 2
    data["selected"]["pooled"] = dict(synthetic=dict(f1=1.))
    assert promotion_decision(data, 3, 1)["status"] == "NOT_PROMOTED"


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    cfg = load_config().as_dict()
    cfg["models"]["hist_gradient_boosting"]["max_iter"] = 5
    (root / "configs/config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def prepare(short=False):
        rows = []
        for n, flight in enumerate(("a", "b", "c")):
            for i in range(81):
                active = 30 <= i <= 40 if short else 25 <= i <= 65
                curr = 10 + i % 3 if active else 0
                rows.append(dict(NamaFile=flight, Inst=0, TimeUS=i * 100000, Volt=24 - .02 * curr - i * .0001 - n * .001,
                                 Curr=curr, CurrTot=i * .01 + n * .1))
        with (root / "data/raw/dataset_baterai.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot"])
            writer.writeheader()
            writer.writerows(rows)
        return load_config(root=root)
    return root, prepare


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_all_fold_freeze_quantile_provenance_shortlist_budget_bundle_and_cache(workspace, monkeypatch):
    root, prepare = workspace
    config = prepare()
    score, slopes = experiment.score_candidate, experiment.calibrate_slopes

    def guard_score(bundle, rows, cfg, role, **kwargs):
        if role == "test":
            run = next((root / "artifacts/dual_path").iterdir())
            assert len(json.loads((run / "selected_per_fold.json").read_text())) == 3
            assert (run / "all_policies_frozen.json").exists()
            assert len(list(run.glob("folds/*/model_bundle.joblib"))) == 3
        return score(bundle, rows, cfg, role, **kwargs)

    def guard_slopes(rows, split, windows, percentiles):
        assert all("is_synthetic" not in r and r["split_role"] == "validation" for r in rows)
        return slopes(rows, split, windows, percentiles)

    monkeypatch.setattr(experiment, "score_candidate", guard_score)
    monkeypatch.setattr(experiment, "calibrate_slopes", guard_slopes)
    result = experiment.run_dual_experiment(config)
    run = Path(result["output_directory"])
    assert result["candidate_pairs_evaluated"] == 3 * 144
    assert result["shortlisted_refined_candidates"] <= 9
    assert result["outer_scenario_evaluations"] == 162
    grid = table(run / "validation_healthy_grid.csv")
    assert sum(r["shortlisted"] == "True" for r in grid) <= 9
    cases = table(run / "validation_fault_per_candidate.csv")
    assert all(r["used_for_threshold"] == "False" and r["source_role"] == "validation" for r in cases)
    assert len(cases) <= 9 * 27
    for path in run.glob("folds/*/model_bundle.joblib"):
        bundle = load_candidate_bundle(path)
        assert bundle["experiment_stage"] == 9 and bundle["model"].early_stopping is False
    monkeypatch.setattr(experiment, "fit_candidate", lambda *a: pytest.fail("cached dual refitted"))
    assert experiment.run_dual_experiment(config) == result
    (run / "selected_per_fold.json").write_text("corrupted")
    with pytest.raises(DataError, match="checksum"):
        experiment.run_dual_experiment(config)


def test_fallback_without_refinement_is_reference_and_not_promoted(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare()
    short = experiment.shortlist

    def no_refinement(*args, **kwargs):
        short(*args, **kwargs)
        return []

    monkeypatch.setattr(experiment, "shortlist", no_refinement)
    result = experiment.run_dual_experiment(config)
    assert result["single_fallback_folds"] == 3 and result["dual_selected_folds"] == 0
    assert result["promotion_status"] == "NOT_PROMOTED"
    assert result["reference_pooled"] == result["selected_pooled"]
    assert result["validation_candidate_scenario_evaluations"] == 0


def test_na_skips_inference_and_keeps_path_metrics_null(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare(short=True)
    score = experiment.score_candidate

    def guard(bundle, rows, cfg, role, **kwargs):
        if "is_synthetic" in rows[0]:
            assert rows[0]["fault_applicable"] and rows[0]["fault_family"] == "sudden_drop"
        return score(bundle, rows, cfg, role, **kwargs)

    monkeypatch.setattr(experiment, "score_candidate", guard)
    result = experiment.run_dual_experiment(config)
    records = table(Path(result["output_directory"]) / "fault_per_scenario.csv")
    na = [r for r in records if r["fault_applicable"] == "False"]
    assert len(na) == 3 * 2 * 18
    assert all(r["tp"] == r["fp"] == r["scores_csv"] == r["confirmed_b_only_samples"] == r["incident_first_path"] == "" for r in na)


def test_outer_perturbation_cannot_change_any_dual_selection(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare()
    first = experiment.run_dual_experiment(config)
    chosen = json.loads((Path(first["output_directory"]) / "selected_per_fold.json").read_text())
    values = config.as_dict()
    values["artifacts"]["root"] = "artifacts_perturbed"
    score = experiment.score_candidate

    def changed_outer(bundle, rows, cfg, role, **kwargs):
        out = score(bundle, rows, cfg, role, **kwargs)
        if role == "test":
            for r in out:
                if r["reference_ready"]:
                    r["anomaly_score"] = 100.
                    r["residual_signed_v"] = 100.
        return out

    monkeypatch.setattr(experiment, "score_candidate", changed_outer)
    second = experiment.run_dual_experiment(AppConfig(config.source, config.root, values))
    assert chosen == json.loads((Path(second["output_directory"]) / "selected_per_fold.json").read_text())
    assert first["selected_pooled"]["healthy_active"] != second["selected_pooled"]["healthy_active"]


def test_both_cli_handlers_share_the_same_frozen_run_and_policy_drift_rejected(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config = prepare()
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.main(["tune-dual-path"]) == 0
    tuned = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(experiment, "fit_candidate", lambda *a: pytest.fail("comparison refitted"))
    assert main.main(["compare-dual-path"]) == 0
    compared = json.loads(capsys.readouterr().out)
    assert compared["run_id"] == tuned["run_id"] and compared["comparison_from_frozen_experiment"]
    values = config.as_dict()
    values["dual_path"]["refinement_max_candidates"] = 4
    with pytest.raises(ConfigError, match="shortlist"):
        experiment.run_dual_experiment(AppConfig(config.source, config.root, values))
