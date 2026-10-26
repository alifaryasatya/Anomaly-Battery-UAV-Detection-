"""Sprint 1 acceptance: root paths, validated config, RNGs, CLI and V1 isolation."""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import yaml

from main import COMMANDS
from src.config import PROJECT_ROOT, load_config, project_path, seed_everything
from src.data_loader import load_battery_csv
from src.errors import ConfigError, DataError


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "V2"
    (root / "configs").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    values = yaml.safe_load((PROJECT_ROOT / "configs/config.yaml").read_text())
    (root / "data/raw/dataset_baterai.csv").write_text(
        "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\nflight,0,0,24,0,0\n", encoding="utf-8")
    return root, values


def config_file(workspace, change=None):
    root, original = workspace
    values = deepcopy(original)
    if change:
        change(values)
    path = root / "configs/config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def set_field(values, field, value):
    parts = field.split(".")
    for part in parts[:-1]:
        values = values[part]
    values[parts[-1]] = value


@pytest.mark.parametrize(("field", "value"), [
    ("random_seed", True), ("random_seed", -1), ("random_seed", 2**32),
    ("time.nominal_frequency_hz", 0), ("time.max_gap_multiplier", float("inf")),
    ("time.max_gap_multiplier", float("nan")), ("activity.min_duration_seconds", -1),
    ("activity.min_active_time_percentage", 101), ("evaluation.minimum_active_flights", 2),
    ("threshold.reference_percentile", 101), ("persistence.candidates_seconds", [0.2, -1]),
    ("threshold.candidates", []), ("features.rolling_windows_seconds", [1, 1.0]),
    ("models.random_forest.n_jobs", 0), ("models.one_class_svm.nu", 2),
    ("models.hist_gradient_boosting.max_leaf_nodes", 1),
    ("dual_path.enabled", "false"), ("data.currtot_unit", "mAh"),
    ("data.required_columns", ["Volt", "Curr", "ThO"]),
    ("fault_injection.start_positions", [1.0]),
    ("fault_injection.active_load_only_for_current_dependent_faults", False),
    ("fault_injection.observability_max_sag_v.low", 0.01),
    ("fault_injection.resistance_ohm.mild", 1.0),
    ("promotion.max_mean_healthy_fpr_increase", 0.1),
    ("replay.require_batch_equivalence", False),
])
def test_invalid_config_is_rejected_with_field(workspace, field, value):
    path = config_file(workspace, lambda data: set_field(data, field, value))
    with pytest.raises(ConfigError, match=field.split(".")[0]):
        load_config(path, root=workspace[0])


@pytest.mark.parametrize("bad_feature", ["Volt", "voltage_sag_actual", "dV_dt", "residual",
                                          "anomaly_score", "TimeUS", "ThO", "V_ref", "Volt_x_Curr"])
def test_regression_allowlist_rejects_leakage(workspace, bad_feature):
    path = config_file(workspace, lambda data: data["features"]["regression_predictors"].append(bad_feature))
    with pytest.raises(ConfigError, match="regression_predictors"):
        load_config(path, root=workspace[0])


@pytest.mark.parametrize("field", ["data.path", "artifacts.root"])
def test_paths_cannot_escape_v2(workspace, field):
    path = config_file(workspace, lambda data: set_field(data, field, "../V1/engine"))
    with pytest.raises(ConfigError, match="inside V2"):
        load_config(path, root=workspace[0])


def test_invalid_path_is_a_config_error():
    with pytest.raises(ConfigError, match="Invalid project path"):
        project_path("invalid\x00.csv")


def test_paths_rooted_in_project_and_config_copies_are_independent(workspace, monkeypatch, tmp_path):
    root, _ = workspace
    config_file(workspace)
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/config.yaml", root=root)
    assert config.data_path == root / "data/raw/dataset_baterai.csv"
    assert config.artifacts_path == root / "artifacts"
    assert config.gap_threshold_seconds == 0.5
    exported = config.as_dict()
    exported["random_seed"] = 900
    assert config.seed == 42
    assert not config.artifacts_path.exists()


@pytest.mark.parametrize("edit", [lambda d: d.pop("time"),
                                  lambda d: d["time"].update(nominal_frequncy_hz=10),
                                  lambda d: d.update(unexpected=True)])
def test_missing_and_unknown_keys_are_errors(workspace, edit):
    path = config_file(workspace, edit)
    with pytest.raises(ConfigError, match="keys"):
        load_config(path, root=workspace[0])


def test_duplicate_yaml_keys_are_not_silently_accepted(workspace):
    path = config_file(workspace)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("random_seed: 99\n")
    with pytest.raises(ConfigError, match="Duplicate config key: random_seed"):
        load_config(path, root=workspace[0])


@pytest.mark.parametrize("payload", ["[invalid yaml", "null", "- list", "42"])
def test_bad_yaml_becomes_config_error(workspace, payload):
    path = config_file(workspace)
    path.write_text(payload)
    with pytest.raises(ConfigError):
        load_config(path, root=workspace[0])


def test_missing_config_and_missing_data_are_explicit(workspace):
    root, _ = workspace
    with pytest.raises(ConfigError, match="Cannot read config"):
        load_config("missing.yaml", root=root)
    path = config_file(workspace, lambda d: set_field(d, "data.path", "missing.csv"))
    with pytest.raises(ConfigError, match="CSV file does not exist"):
        load_config(path, root=root)


def test_seeding_repeats_python_numpy_and_explicit_generator():
    def sample(seed):
        generator = seed_everything(seed)
        return random.random(), np.random.random(), generator.random()
    assert sample(42) == sample(42)
    assert sample(42) != sample(43)


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), *args],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=30)


def test_inspect_from_different_working_directory_is_repeatable_and_read_only(tmp_path):
    raw = PROJECT_ROOT / "data/raw/dataset_baterai.csv"
    before_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    first = run_cli("inspect", cwd=tmp_path)
    second = run_cli("inspect", cwd=PROJECT_ROOT)
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    payload = json.loads(first.stdout)
    assert payload["rows"] == 32338
    assert payload["flights"] == 17
    assert payload["random_seed"] == 42
    assert payload["data_path"] == str(raw)
    assert not payload["model_training_performed"]
    assert {r["v2_load_label"] for r in payload["per_flight"]} == {"NOT_COMPUTED"}
    assert before_hash == hashlib.sha256(raw.read_bytes()).hexdigest() == payload["data_sha256"]


@pytest.mark.parametrize("position", ["before", "after"])
def test_cli_accepts_config_on_either_side_of_command(position, tmp_path):
    args = ["--config", "configs/config.yaml"]
    result = run_cli(*(args + ["inspect"] if position == "before" else ["inspect"] + args), cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", [name for name, spec in COMMANDS.items() if not spec.implemented] or [None])
def test_future_commands_report_unavailable_without_success(command, tmp_path):
    if command is None:
        result = run_cli("--help", cwd=tmp_path)
        assert result.returncode == 0 and "not implemented;" not in result.stdout
        return
    result = run_cli(command, cwd=tmp_path)
    assert result.returncode == 3
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "NOT_IMPLEMENTED"
    assert payload["command"] == command
    assert payload["required_stage"] == COMMANDS[command].stage


def test_cli_errors_are_nonzero_and_machine_readable():
    result = run_cli("inspect", "--config", "missing.yaml")
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["status"] == "ERROR"
    assert "Traceback" not in result.stderr


def test_help_lists_all_commands_and_unknown_command_fails():
    result = run_cli("--help")
    assert result.returncode == 0
    assert all(command in result.stdout for command in COMMANDS)
    result = run_cli("train-v1")
    assert result.returncode == 2


@pytest.mark.parametrize("payload", [
    "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot,ThO\nf,0,0,24,0,0,0\n",
    "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\nf,0,0,24,0\n",
    "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\nf,0,0,24,0,0,extra\n",
    "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\n",
    "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\n,0,0,24,0,0\n",
])
def test_loader_rejects_ambiguous_or_nonbattery_csv(tmp_path, payload):
    path = tmp_path / "bad.csv"
    path.write_text(payload)
    with pytest.raises(DataError):
        load_battery_csv(path)


def test_loader_preserves_field_strings_and_source_order(tmp_path):
    path = tmp_path / "source.csv"
    payload = "NamaFile,TimeUS,Inst,Volt,Curr,CurrTot\nf,200,0,24.00,-0.10,1.00\nf,100,0,nan,0,0\n"
    path.write_text(payload)
    loaded = load_battery_csv(path)
    assert [r["TimeUS"] for r in loaded] == ["200", "100"]
    assert loaded[0]["Volt"] == "24.00"
    assert loaded[0]["Curr"] == "-0.10"
    assert loaded[1]["Volt"] == "nan"  # numerical cleaning is a separate Stage 2 operation
    assert path.read_text() == payload


def test_modules_are_self_contained_and_frozen_sprint0_remains_intact():
    # Check resolved imports, not just naming conventions: V2 must not import sibling V1 code.
    for name in ("config", "data_loader", "preprocessing", "activity", "feature_sets", "features",
                 "models", "detectors", "persistence", "fault_injection", "evaluation",
                 "cross_validation", "tuning", "reporting", "replay"):
        module = importlib.import_module(f"src.{name}")
        assert Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT / "src")
    repo_parent = PROJECT_ROOT.parent
    external_local = [module.__file__ for module in list(sys.modules.values())
                      if getattr(module, "__file__", None)
                      and Path(module.__file__).resolve().is_relative_to(repo_parent)
                      and not Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT)]
    assert external_local == []
    manifest = json.loads((PROJECT_ROOT / "artifacts/audit/freeze_manifest.json").read_text())
    assert all(hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() == digest
               for p, digest in manifest["files"].items())
