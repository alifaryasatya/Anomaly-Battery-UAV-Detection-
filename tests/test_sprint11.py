"""Real cold pipeline, immutable reuse, report reconciliation and failure gates."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import yaml

import main
import src.research as research
import src.research_sources as sources
from src.config import load_config
from src.errors import DataError


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("research")
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    values = load_config().as_dict()
    values["models"]["hist_gradient_boosting"]["max_iter"] = 3
    values["models"]["random_forest"].update(n_estimators=3, n_jobs=1)
    values["models"]["isolation_forest"].update(n_estimators=3, max_samples=32)
    values["models"]["one_class_svm"]["max_train_samples"] = 32
    (root / "configs/config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    (root / "configs/experiment_protocol.md").write_text("Test-only fixed validation-before-outer protocol.\n", encoding="utf-8")
    rows = []
    for flight in ("a", "b", "c", "idle", "truncated"):
        for i in range(11 if flight == "truncated" else 81):
            current = 10 + i % 3 if flight in ("a", "b", "c") and 25 <= i <= 65 else .1
            rows.append(dict(NamaFile=flight, TimeUS=i * 100000, Inst=0,
                             Volt=24 - .02 * current - i * .0001, Curr=current, CurrTot=i * .01))
    with (root / "data/raw/dataset_baterai.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    frozen_names = ("configs/config.yaml", "configs/experiment_protocol.md", "data/raw/dataset_baterai.csv")
    audit = root / "artifacts/audit"
    audit.mkdir(parents=True)
    (audit / "freeze_manifest.json").write_bytes(sources.encoded(dict(files={n: sources.sha(root / n) for n in frozen_names})))
    return load_config(root=root)


@pytest.fixture(scope="module")
def completed(workspace):
    config = workspace
    seen = []
    build = sources.build_stage

    def guard(command, cfg, resolved):
        expected = [s[0] for s in sources.STAGES]
        assert command == expected[len(seen)]
        assert list((cfg.artifacts_path / "research_evaluation/plans").glob("*/orchestration_plan.json"))
        if command == "evaluate-low-load":
            assert "cross-validate" in resolved
            assert not (cfg.artifacts_path / "sprint6/cross_validation_result.json").exists()
        seen.append(command)
        return build(command, cfg, resolved)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sources, "build_stage", guard)
        result = research.run_research(config)
    assert seen == [s[0] for s in sources.STAGES]
    return config, result


def resolved_sources(config):
    catalog = next((config.artifacts_path / "research_evaluation/sources").glob("*.json"))
    choices = sources.discover(config, catalog)
    return sources.resolve_all(config, sources.audit_inputs(config), choices)[0]


def test_real_cold_pipeline_builds_all_stages_and_indexes_completed_sources(completed):
    cfg, result = completed
    assert result["command"] == "research-evaluation" and result["status"] == "OK"
    assert result["source_stages"] == 10
    assert result["source_stages_built_at_creation"] == [s[0] for s in sources.STAGES]
    assert set(result["training_stages_built_at_creation"]) == sources.MODEL_STAGES
    assert result["model_bundle_references"] == 52  # pilot + 3*(reference + 10 + 4 + tuning + dual)
    assert result["synthetic_series"] == 19
    assert result["selected_new_winner"] is None and not result["operational_config_changed"]
    assert not result["replay_evaluated"] and not result["independent_final_holdout"]
    run = Path(result["output_directory"])
    assert sources.verify_outputs(run, cfg)["run_id"] == result["run_id"]
    assert (run / "research_report.md").exists()
    assert len(list((run / "plots").glob("*.png"))) == 5
    assert not (cfg.artifacts_path / "sprint6").exists()  # explicit source injection, no overwritten sprint pointers


def test_repeat_uses_catalog_without_engine_calls_or_writes(completed, monkeypatch):
    cfg, result = completed
    files = [p for p in cfg.artifacts_path.rglob("*") if p.is_file()]
    before = {p: p.stat().st_mtime_ns for p in files}
    monkeypatch.setattr(sources, "build_stage", lambda *a: pytest.fail("completed sources were rebuilt"))
    assert research.run_research(cfg) == result
    assert all(p.stat().st_mtime_ns == value for p, value in before.items())


def test_cli_routes_research_and_reports_current_completed_run(completed, monkeypatch, capsys):
    cfg, result = completed
    monkeypatch.setattr(main, "load_config", lambda _: cfg)
    monkeypatch.setattr(sources, "build_stage", lambda *a: pytest.fail("CLI refit"))
    assert main.COMMANDS["research-evaluation"].implemented
    assert main.main(["research-evaluation"]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_corrupted_source_fails_before_any_missing_stage_build(completed, monkeypatch):
    cfg, _ = completed
    resolved = resolved_sources(cfg)
    choices = {k: v["summary"] for k, v in resolved.items()}
    choices["preprocess"] = None
    target = Path(choices["tune"]["output_directory"]) / "selected_per_fold.json"
    data = target.read_bytes()
    try:
        target.write_bytes(b"corrupt")
        monkeypatch.setattr(sources, "build_stage", lambda *a: pytest.fail("built before source preflight"))
        with pytest.raises(DataError, match="checksum"):
            sources.resolve_all(cfg, sources.audit_inputs(cfg), choices)
    finally:
        target.write_bytes(data)


def test_corrupted_consolidated_output_is_not_silently_replaced(completed):
    cfg, result = completed
    target = Path(result["output_directory"]) / "synthetic_group_metrics.csv"
    data = target.read_bytes()
    try:
        target.write_bytes(b"corrupt")
        with pytest.raises(DataError, match="checksum"):
            research.run_research(cfg)
    finally:
        target.write_bytes(data)


def test_frozen_raw_change_stops_before_pipeline(completed, monkeypatch):
    cfg, _ = completed
    data = cfg.data_path.read_bytes()
    try:
        cfg.data_path.write_bytes(data + b"\n")
        monkeypatch.setattr(research, "resolve_all", lambda *a: pytest.fail("used changed raw"))
        with pytest.raises(DataError, match="Frozen audit input checksum"):
            research.run_research(cfg)
    finally:
        cfg.data_path.write_bytes(data)


def test_missing_pre_model_audit_contract_is_explicit_error(workspace):
    cfg = workspace
    target = cfg.artifacts_path / "audit/freeze_manifest.json"
    data = target.read_bytes()
    try:
        target.unlink()
        with pytest.raises(DataError, match="Sprint 0"):
            sources.audit_inputs(cfg)
    finally:
        target.write_bytes(data)


def test_catalog_must_cover_all_known_stages(completed, tmp_path):
    cfg, _ = completed
    path = tmp_path / "incomplete.json"
    path.write_bytes(sources.encoded({"preprocess": {}}))
    with pytest.raises(DataError, match="catalog"):
        sources.discover(cfg, path)


def test_feature_lineage_and_outer_selection_drift_are_rejected(completed):
    cfg, _ = completed
    resolved = resolved_sources(cfg)
    altered = deepcopy(resolved)
    altered["tune"]["summary"]["feature_manifest_sha256"] = "wrong"
    with pytest.raises(DataError, match="feature lineage"):
        research.validate_chain(altered)
    altered_summary = deepcopy(resolved["tune"]["summary"])
    altered_summary["outer_used_for_selection"] = True
    with pytest.raises(DataError, match="pointer/verified summary"):
        sources.verify_source(next(s for s in sources.STAGES if s[0] == "tune"), altered_summary, cfg, sources.audit_inputs(cfg))


def test_low_load_reference_and_promotion_gate_cannot_be_relabelled(completed):
    cfg, _ = completed
    resolved = resolved_sources(cfg)
    altered = deepcopy(resolved)
    altered["evaluate-low-load"]["summary"]["baseline_run_id"] = "other-model"
    with pytest.raises(DataError, match="Low-load source"):
        research.validate_chain(altered)
    altered = deepcopy(resolved)
    gate = altered["tune-dual-path"]["content"]["promotion"]
    gate["gate_passed"] = not gate["gate_passed"]
    with pytest.raises(DataError, match="promotion does not agree"):
        research.validate_chain(altered)


def test_grouped_totals_reconcile_and_na_never_becomes_missed_fault(completed):
    cfg, result = completed
    run = Path(result["output_directory"])
    groups = research.table(run / "synthetic_group_metrics.csv")
    overall = next(r for r in groups if r["experiment"] == "cross-validate" and r["dimension"] == "overall")
    assert int(overall["scenarios"]) == 81
    assert int(overall["applicable_scenarios"]) + int(overall["not_applicable_scenarios"]) == 81
    by_family = [r for r in groups if r["experiment"] == "cross-validate" and r["dimension"] == "family"]
    assert sum(int(r["applicable_scenarios"]) for r in by_family) == int(overall["applicable_scenarios"])
    # Every experimental series is kept separate, even identical reference policies.
    assert len(research.table(run / "synthetic_series.csv")) == 19
    registry = research.table(run / "model_bundle_registry.csv")
    direct = [r for r in registry if r["candidate"] in ("ocsvm_direct", "if_direct")]
    assert direct and all("voltage_sag_actual" in r["predictors"] and "dV_dt" in r["predictors"] for r in direct)


def test_grouping_rejects_changed_or_duplicated_source_case(completed, monkeypatch):
    cfg, _ = completed
    resolved = resolved_sources(cfg)
    read = research.table

    def duplicate(path):
        rows = read(path)
        return rows + [rows[0]]

    monkeypatch.setattr(research, "table", duplicate)
    with pytest.raises(DataError, match="Duplicate scenario"):
        research.consolidate_groups(resolved, cfg.as_dict()["evaluation"]["deadlines_seconds"])


def test_metric_comparison_preserves_undefined_and_catches_wrong_denominator():
    research.same_metrics(dict(rate=None, n=3), dict(rate=None, n=3), "fixture")
    with pytest.raises(DataError, match="metrics disagree"):
        research.same_metrics(dict(rate=0), dict(rate=None), "fixture")
    with pytest.raises(DataError, match="metrics disagree"):
        research.same_metrics(dict(rate=.5), dict(rate=.25), "fixture")


def test_report_preserves_limits_and_source_decisions(completed):
    _, result = completed
    text = (Path(result["output_directory"]) / "research_report.md").read_text(encoding="utf-8")
    for phrase in (result["dual_promotion"], "CurrTot remains unverified", "not a new final holdout", "Sprint 12", "Sprint 13"):
        # The report describes historical development explicitly.
        if phrase == "not a new final holdout":
            assert "not proof of a globally untouched prospective final holdout" in text
        else:
            assert phrase in text
    assert "Synthetic faults represent controlled battery-related electrical anomalies" in text
    assert "does not diagnose the underlying internal physical failure mechanism" in text


def test_historical_preprocess_pointer_follows_explicit_feature_parent(completed):
    cfg, _ = completed
    resolved = resolved_sources(cfg)
    paths = [cfg.artifacts_path / "sprint2/preprocess_result.json", cfg.artifacts_path / "sprint3/features_result.json"]
    try:
        old = dict(resolved["preprocess"]["summary"], run_id="older-sprint2-snapshot")
        for path, value in zip(paths, (old, resolved["features"]["summary"])):
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(sources.encoded(value))
        choices = sources.discover(cfg, cfg.artifacts_path / "absent_catalog.json")
        assert choices["preprocess"] == resolved["preprocess"]["summary"]
        assert sources.load(paths[0])["run_id"] == "older-sprint2-snapshot"
    finally:
        for path in paths:
            if path.exists():
                path.unlink()


@pytest.mark.parametrize("pointer", [None, [], {}, {"run_id": "missing-data"}])
def test_invalid_source_pointer_is_explicit_data_error(workspace, pointer):
    with pytest.raises(DataError, match="Invalid research source pointer"):
        sources.verify_source(sources.STAGES[0], pointer, workspace, {})
