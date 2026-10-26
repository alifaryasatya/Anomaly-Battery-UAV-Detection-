"""Validation-only tuning isolation, fixed lexicographic selection and fallback."""

from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path

import pytest
import yaml

import main
import src.single_path_tuning as tuning
from src.config import AppConfig, load_config
from src.errors import ConfigError, DataError
from src.experimental_models import load_candidate_bundle


def candidate(p=99., d=.5, **kwargs):
    return dict(candidate_id=tuning.policy_id(p, d), percentile=p, persistence_seconds=d,
                source_role="validation", validation_flight="v", used_for_policy_selection=True,
                healthy_confirmed_fpr=0., healthy_episodes_per_minute=0., abrupt_severe_le_1s_rate=.5,
                gradual_observable_le_5s_rate=.5, incident_recall=.8, f1=.8,
                delay_median_seconds=.5, delay_mean_seconds=1., **kwargs)


@pytest.mark.parametrize("field", tuning.RANK_FIELDS)
def test_lexicographic_ranking_obeys_order_before_lower_priority_metrics(field):
    ref = candidate()
    other = candidate(99.5, 1.)
    index = tuning.RANK_FIELDS.index(field)
    other[field] = ref[field] + (.1 if index < 4 else -.1)
    for lower in tuning.RANK_FIELDS[index + 1:]:
        other[lower] = 0. if lower in tuning.RANK_FIELDS[:4] else 100.
    assert tuning.select_policy([ref, other], "v")["candidate_id"] == other["candidate_id"]


def test_ties_use_ascending_percentile_then_persistence_and_are_order_independent():
    rows = [candidate(), candidate(95., .3), candidate(95., .2), candidate(97.5, .2)]
    assert tuning.select_policy(deepcopy(rows), "v")["candidate_id"] == "p95_d0p2"
    assert tuning.select_policy(list(reversed(deepcopy(rows))), "v")["candidate_id"] == "p95_d0p2"


def test_undefined_rate_is_worse_than_defined_zero_and_undefined_delay_is_infinite():
    ref, other = candidate(), candidate(99.5, 1.)
    ref["abrupt_severe_le_1s_rate"] = None
    other["abrupt_severe_le_1s_rate"] = 0.
    assert tuning.select_policy([ref, other], "v")["candidate_id"] == other["candidate_id"]
    ref, other = candidate(), candidate(99.5, 1.)
    ref["delay_median_seconds"] = None
    other["delay_median_seconds"] = 999.
    assert tuning.select_policy([ref, other], "v")["candidate_id"] == other["candidate_id"]


@pytest.mark.parametrize("field", ["healthy_confirmed_fpr", "healthy_episodes_per_minute"])
def test_each_healthy_gate_can_independently_reject_a_high_detection_candidate(field):
    ref, other = candidate(), candidate(95., .2)
    other["abrupt_severe_le_1s_rate"] = 1.
    other[field] = .001
    chosen = tuning.select_policy([ref, other], "v")
    assert chosen["candidate_id"] == ref["candidate_id"]
    assert not other["healthy_eligible"] and "EXCEEDS_REFERENCE" in other["eligibility_reason"]


def test_undefined_healthy_gate_falls_back_to_reference_with_explicit_reason():
    ref, other = candidate(), candidate(95., .2)
    ref["healthy_confirmed_fpr"] = None
    chosen = tuning.select_policy([ref, other], "v")
    assert chosen["candidate_id"] == ref["candidate_id"]
    assert chosen["eligible_candidates"] == 0 and chosen["selection_reason"] == "NO_ELIGIBLE_CANDIDATE_REFERENCE_RETAINED"


@pytest.mark.parametrize("kind", ["test", "wrong_flight", "not_selection", "duplicate", "missing_reference"])
def test_selection_rejects_outer_roles_wrong_provenance_and_incomplete_reference(kind):
    rows = [candidate(), candidate(95., .2)]
    if kind == "test":
        rows[1]["source_role"] = "test"
    elif kind == "wrong_flight":
        rows[1]["validation_flight"] = "outer"
    elif kind == "not_selection":
        rows[1]["used_for_policy_selection"] = False
    elif kind == "duplicate":
        rows.append(candidate())
    else:
        rows = rows[1:]
    with pytest.raises(DataError):
        tuning.select_policy(rows, "v")


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
                current = 10 + i % 3 if active else 0
                rows.append(dict(NamaFile=flight, Inst=0, TimeUS=i * 100000, Volt=24 - .02 * current - i * .0001 - n * .001,
                                 Curr=current, CurrTot=i * .01 + n * .1))
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


def test_full_budget_quantiles_are_healthy_only_all_selected_before_any_outer(workspace, monkeypatch):
    root, prepare = workspace
    config = prepare()
    score, calibrate, inject, select = tuning.score_candidate, tuning.calibrate_threshold, tuning.inject_fault, tuning.select_policy
    counts = dict(quantiles=0, validation_injections=0, test_injections=0, selections=0)

    def guard_score(bundle, rows, cfg, role, **kwargs):
        if role == "test":
            run = next((root / "artifacts/tuning").iterdir())
            frozen = json.loads((run / "all_policies_frozen.json").read_text())
            assert frozen["folds"] == counts["selections"] == 3
            assert len(list(run.glob("folds/*/model_bundle.joblib"))) == 3
            assert len(json.loads((run / "selected_per_fold.json").read_text())) == 3
        return score(bundle, rows, cfg, role, **kwargs)

    def guard_calibrate(rows, split, **kwargs):
        assert {r["NamaFile"] for r in rows} == set(split.validation)
        assert all("is_synthetic" not in r and r["used_for_threshold"] and r["split_role"] == "validation" for r in rows)
        counts["quantiles"] += 1
        return calibrate(rows, split, **kwargs)

    def guard_select(rows, val):
        assert len(rows) == 20 and all(r["source_role"] == "validation" for r in rows)
        counts["selections"] += 1
        return select(rows, val)

    monkeypatch.setattr(tuning, "score_candidate", guard_score)
    monkeypatch.setattr(tuning, "calibrate_threshold", guard_calibrate)
    monkeypatch.setattr(tuning, "select_policy", guard_select)
    result = tuning.run_tuning(config)
    assert counts["quantiles"] == 12 and result["candidates_evaluated"] == 60
    assert result["validation_scenario_evaluations"] == 3 * 20 * 27
    assert result["outer_scenario_evaluations"] == 3 * 2 * 27
    assert result["outer_used_for_selection"] is False
    run = Path(result["output_directory"])
    grid = table(run / "threshold_persistence_grid.csv")
    assert sum(r["selected"] == "True" for r in grid) == 3
    for path in run.glob("folds/*/model_bundle.joblib"):
        bundle = load_candidate_bundle(path)
        assert bundle["experiment_stage"] == 8 and bundle["model"].early_stopping is False
    path = next(run.glob("folds/*/validation_scenario_scores/*.gz"))
    synthetic = table(path)
    assert all(r["used_for_threshold"] == "False" for r in synthetic)
    assert all(r["used_for_policy_selection"] == r["reference_ready"] for r in synthetic)
    outer = table(next(run.glob("folds/*/selected/scenario_scores/*.gz")))
    assert all(r["used_for_policy_selection"] == r["used_for_threshold"] == "False" for r in outer)
    assert result["reference_pooled"]["regression"] == result["selected_pooled"]["regression"]
    monkeypatch.setattr(tuning, "fit_candidate", lambda *a: pytest.fail("cache refitted"))
    assert tuning.run_tuning(config) == result
    (run / "selected_per_fold.json").write_text("corrupted")
    with pytest.raises(DataError, match="checksum"):
        tuning.run_tuning(config)


def test_not_applicable_validation_and_outer_have_no_score_or_confusion(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare(short=True)
    score = tuning.score_candidate

    def guard(bundle, rows, cfg, role, **kwargs):
        if "is_synthetic" in rows[0]:
            assert rows[0]["fault_applicable"] and rows[0]["fault_family"] == "sudden_drop"
        return score(bundle, rows, cfg, role, **kwargs)

    monkeypatch.setattr(tuning, "score_candidate", guard)
    result = tuning.run_tuning(config)
    run = Path(result["output_directory"])
    for name, expected in (("validation_fault_per_candidate.csv", 3 * 20 * 18), ("fault_per_scenario.csv", 3 * 2 * 18)):
        rejected = [r for r in table(run / name) if r["fault_applicable"] == "False"]
        assert len(rejected) == expected
        assert all(r["tp"] == r["fp"] == r["incident_detected"] == r["delay_seconds"] == r["scores_csv"] == "" for r in rejected)


def test_outer_result_perturbation_cannot_change_selected_policies(workspace, monkeypatch):
    _, prepare = workspace
    config = prepare()
    first = tuning.run_tuning(config)
    chosen = json.loads((Path(first["output_directory"]) / "selected_per_fold.json").read_text())
    values = config.as_dict()
    values["artifacts"]["root"] = "artifacts_perturbed"
    score = tuning.score_candidate

    def changed_outer(bundle, rows, cfg, role, **kwargs):
        result = score(bundle, rows, cfg, role, **kwargs)
        if role == "test":
            for r in result:
                if r["reference_ready"]:
                    r["anomaly_score"] = 100.
                    r["residual_signed_v"] = 100.
        return result

    monkeypatch.setattr(tuning, "score_candidate", changed_outer)
    second = tuning.run_tuning(AppConfig(config.source, config.root, values))
    again = json.loads((Path(second["output_directory"]) / "selected_per_fold.json").read_text())
    assert chosen == again
    assert first["selected_pooled"]["healthy_active"] != second["selected_pooled"]["healthy_active"]


def test_cli_and_config_policy_drift_guard(workspace, monkeypatch, capsys):
    _, prepare = workspace
    config = prepare()
    monkeypatch.setattr(main, "load_config", lambda _: config)
    assert main.COMMANDS["tune"].implemented and main.main(["tune"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates_evaluated"] == 60
    values = config.as_dict()
    values["threshold"]["candidates"] = [95., 99.]
    with pytest.raises(ConfigError, match="4x5"):
        tuning.run_tuning(AppConfig(config.source, config.root, values))
