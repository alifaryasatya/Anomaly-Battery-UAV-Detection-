"""Resolve immutable historical results or build missing research stages.

Historical results retain their original code/version provenance. A current
engine code change never silently refits or replaces a verified historical run.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import yaml

from .config import project_path
from .data_loader import inspect_dataset
from .errors import DataError
from .reporting import format_json

# Order is fixed independently of all outcome metrics. Every modelling engine
# freezes ALL its candidates/folds before its own first outer evaluation.
STAGES = (
    ("preprocess", "preprocessing", "sprint2/preprocess_result.json", "summary.json"),
    ("features", "features", "sprint3/features_result.json", "summary.json"),
    ("baseline", "baseline", "sprint4/baseline_result.json", "summary.json"),
    ("inject-faults", "fault_injection", "sprint5/faults_result.json", "summary.json"),
    ("cross-validate", "cross_validation", "sprint6/cross_validation_result.json", "results.json"),
    ("ablation", "ablation", "sprint7/ablation_result.json", "results.json"),
    ("compare-models", "model_comparison", "sprint7/comparison_result.json", "results.json"),
    ("tune", "tuning", "sprint8/tuning_result.json", "results.json"),
    ("tune-dual-path", "dual_path", "sprint9/dual_result.json", "results.json"),
    ("evaluate-low-load", "low_load", "sprint10/low_load_result.json", "summary.json"),
)
MODEL_STAGES = frozenset(("baseline", "cross-validate", "ablation", "compare-models", "tune", "tune-dual-path"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise DataError(f"Invalid research artifact JSON: {path}") from exc


def encoded(value):
    return (format_json(value) + "\n").encode()


def versions():
    return dict(python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "PyYAML", "matplotlib", "joblib")})


def audit_inputs(config):
    frozen_path = config.artifacts_path / "audit/freeze_manifest.json"
    if not frozen_path.exists():
        raise DataError("Research evaluation requires the completed Sprint 0 audit/freeze contract.")
    frozen = load(frozen_path)
    if not isinstance(frozen, dict) or not isinstance(frozen.get("files"), dict):
        raise DataError("Invalid Sprint 0 frozen audit file registry.")
    files = frozen["files"]
    required = (config.source, config.data_path, config.root / "configs/experiment_protocol.md")
    for path in required:
        name = path.relative_to(config.root).as_posix()
        if name not in files:
            raise DataError(f"Research input is absent from the frozen audit contract: {name}")
    for name, digest in files.items():
        if sha(project_path(name, config.root)) != digest:
            raise DataError(f"Frozen audit input checksum mismatch: {name}")
    return dict(status="PASS", freeze_manifest_sha256=sha(frozen_path), frozen_files=len(files),
                inspection=inspect_dataset(config), current_dependency_versions=versions(),
                experiment_protocol_sha256=sha(config.root / "configs/experiment_protocol.md"))


def verify_outputs(directory, config, *, root_relative=False):
    manifest = load(directory / "manifest.json")
    for name, digest in manifest["outputs"].items():
        target = project_path((config.root if root_relative else directory) / name, config.root)
        if sha(target) != digest:
            raise DataError(f"Research source checksum mismatch: {target}")
    return manifest


def verify_source(stage, summary, config, audit):
    command, category, _, summary_name = stage
    if not isinstance(summary, dict) or any(k not in summary for k in ("output_directory", "run_id", "data_sha256")):
        raise DataError(f"Invalid research source pointer: {command}")
    directory = project_path(summary["output_directory"], config.root)
    if not directory.is_relative_to(config.artifacts_path / category):
        raise DataError(f"Research source is outside its stage artifact directory: {command}")
    manifest = verify_outputs(directory, config, root_relative=command == "inject-faults")
    content = load(directory / summary_name)
    actual = content["run_summary"] if summary_name == "results.json" else content
    if actual != summary or actual.get("status") != "OK" or actual.get("command") != command:
        raise DataError(f"Research pointer/verified summary mismatch: {command}")
    if manifest["run_id"] != summary["run_id"] or directory.name != summary["run_id"]:
        raise DataError(f"Research source run identity mismatch: {command}")
    if summary["data_sha256"] != audit["inspection"]["data_sha256"]:
        raise DataError(f"Research source raw data mismatch: {command}")
    if yaml.safe_load((directory / "config_used.yaml").read_text(encoding="utf-8")) != config.as_dict():
        raise DataError(f"Research source configuration mismatch: {command}")
    freeze_count = 0
    if command in MODEL_STAGES - {"baseline"}:
        if not summary.get("all_policies_frozen_before_outer") or summary.get("outer_used_for_selection", False):
            raise DataError(f"Research source lacks validation-before-outer isolation: {command}")
        freeze_name = "all_fold_policies_frozen.json" if command == "cross-validate" else "all_policies_frozen.json"
        freeze = load(directory / freeze_name)
        if not freeze["status"].startswith("FROZEN_BEFORE") or not freeze["files"]:
            raise DataError(f"Research source lacks a completed policy freeze: {command}")
        for name, digest in freeze["files"].items():
            if manifest["outputs"].get(name) != digest:
                raise DataError(f"Research pre-outer freeze disagrees with final manifest: {command}/{name}")
        freeze_count = len(freeze["files"])
    if command in ("ablation", "compare-models") and summary.get("selected_winner") is not None:
        raise DataError("Comparison results cannot select a new research winner from outer scores.")
    if summary.get("operational_config_changed", False) or summary.get("operational_dual_path_enabled", False):
        raise DataError("Research results must not change the frozen operational configuration.")
    reference = dict(command=command, run_id=summary["run_id"], output_directory=str(directory),
                     manifest_sha256=sha(directory / "manifest.json"), summary_sha256=sha(directory / summary_name),
                     verified_files=len(manifest["outputs"]), pre_outer_frozen_files=freeze_count,
                     model_bundles=sum(name.endswith("model_bundle.joblib") for name in manifest["outputs"]))
    return dict(summary=summary, reference=reference, manifest=manifest, content=content)


def build_stage(command, config, resolved):
    # Lazy imports keep orchestration imports independent of the engines.
    from .preprocessing import preprocess
    from .features import generate_features
    from .models import baseline
    from .fault_injection import generate_faults
    from .loafo import run_loafo
    from .comparison import run_comparison
    from .single_path_tuning import run_tuning
    from .dual_path_experiment import run_dual_experiment
    from .low_load import run_low_load
    engines = {"preprocess": preprocess, "features": generate_features, "baseline": baseline,
               "inject-faults": generate_faults, "cross-validate": run_loafo,
               "ablation": lambda c: run_comparison(c, "ablation"),
               "compare-models": lambda c: run_comparison(c, "compare-models"), "tune": run_tuning,
               "tune-dual-path": run_dual_experiment,
               "evaluate-low-load": lambda c: run_low_load(c, baseline_summary=resolved["cross-validate"]["summary"])}
    return engines[command](config)


def discover(config, catalog_path):
    catalog = load(catalog_path) if catalog_path.exists() else None
    if catalog is not None and (not isinstance(catalog, dict) or set(catalog) != {s[0] for s in STAGES}):
        raise DataError("Research source catalog is incomplete or has unknown stages.")
    choices = {}
    for stage in STAGES:
        command, _, pointer_name, _ = stage
        if catalog is not None:
            choices[command] = catalog[command]
        else:
            pointer = project_path(config.artifacts_path / pointer_name, config.root)
            choices[command] = load(pointer) if pointer.exists() else None
    if catalog is None and choices["features"] is not None:
        # Sprint 3 can have regenerated preprocessing after a code change in
        # Sprint 2. Follow the feature run's explicit parent, rather than mixing
        # the older Sprint 2 snapshot with later features. Preserve both runs.
        feature = choices["features"]
        parent = project_path(feature["preprocessing_directory"], config.root)
        if not parent.is_relative_to(config.artifacts_path / "preprocessing"):
            raise DataError("Feature preprocessing parent leaves the preprocessing artifact directory.")
        choices["preprocess"] = load(parent / "summary.json")
    return choices


def resolve_all(config, audit, choices):
    # Check every supplied result before allowing any missing stage to build.
    resolved = {s[0]: verify_source(s, choices[s[0]], config, audit) for s in STAGES if choices[s[0]] is not None}
    trace = []
    for stage in STAGES:
        command = stage[0]
        summary = choices[command]
        action = "VERIFIED_REUSE" if summary is not None else "BUILT_MISSING_STAGE"
        if summary is None:
            summary = build_stage(command, config, resolved)
            resolved[command] = verify_source(stage, summary, config, audit)
        trace.append(dict(order=len(trace) + 1, command=command, action=action, run_id=summary["run_id"],
                          training_stage_built=action == "BUILT_MISSING_STAGE" and command in MODEL_STAGES))
    return resolved, trace
