"""Strict YAML configuration, project-root paths, and deterministic RNG setup."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

from .errors import ConfigError
from .feature_sets import BATTERY_COLUMNS, DIRECT_BASELINE_PREDICTORS, validate_regression_predictors

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently taking the last value."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigError("Config keys must be strings.")
        if key in result:
            raise ConfigError(f"Duplicate config key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _choice(*values):
    return lambda value: any(type(value) is type(expected) and value == expected for expected in values)


def _number(low=0, high=math.inf, *, positive=False, integer=False):
    def valid(value):
        if type(value) not in ((int,) if integer else (int, float)):
            return False
        try:
            return math.isfinite(value) and low <= value <= high and (not positive or value > 0)
        except OverflowError:
            return False
    return valid


def _list_of(check):
    return lambda value: (isinstance(value, list) and bool(value)
                          and all(check(item) for item in value)
                          and len(set(value)) == len(value))


POSITIVE = _number(positive=True)
POSITIVE_INT = _number(positive=True, integer=True)
PERCENTILE = _number(0, 100, positive=True)
TEXT = lambda value: isinstance(value, str) and bool(value.strip())
BOOL = lambda value: type(value) is bool
SEVERITIES = {key: POSITIVE for key in ("mild", "moderate", "severe")}

# Every supplied field is checked. Unknown keys are errors (including misspellings).
SCHEMA = {
    "version": _choice(2), "random_seed": _number(0, 2**32 - 1, integer=True),
    "data": {"path": TEXT, "required_columns": _choice(list(BATTERY_COLUMNS)),
             "healthy_status": _choice("assumed_from_source"), "currtot_unit": _choice("unverified"),
             "instance_policy": _choice("reject_multiple_instances_without_explicit_selection")},
    "time": {"nominal_frequency_hz": POSITIVE, "max_gap_multiplier": POSITIVE},
    "activity": {"min_current_a": _number(), "min_duration_seconds": POSITIVE,
                 "min_active_time_percentage": _number(0, 100),
                 "duration_basis": _choice("observed_valid_intervals")},
    "reference_voltage": {"strategy": _choice("causal_fixed_warmup"), "warmup_seconds": POSITIVE,
                          "low_current_max_a": _number(), "min_low_current_samples": POSITIVE_INT,
                          "unavailable_policy": _choice("mark_not_ready_and_report_coverage")},
    "features": {"operational_set": _choice("battery_a", "battery_b", "battery_c", "battery_d", "battery_e"),
                 "rolling_windows_seconds": _list_of(POSITIVE),
                 "regression_predictors": _list_of(TEXT), "baseline_predictors": _list_of(TEXT)},
    "operational_detector": {"model": _choice("HistGradientBoostingRegressor"),
                             "method": _choice("positive_residual")},
    "models": {
        "hist_gradient_boosting": {"max_iter": POSITIVE_INT, "learning_rate": POSITIVE,
                                   "max_leaf_nodes": _number(2, integer=True), "l2_regularization": _number()},
        "random_forest": {"n_estimators": POSITIVE_INT, "min_samples_leaf": POSITIVE_INT,
                          "n_jobs": lambda v: type(v) is int and v != 0},
        "isolation_forest": {"n_estimators": POSITIVE_INT, "max_samples": POSITIVE_INT,
                             "contamination": lambda v: v == "auto" or _number(0, 0.5, positive=True)(v)},
        "one_class_svm": {"kernel": _choice("rbf"), "nu": _number(0, 1, positive=True),
                          "gamma": lambda v: v in ("scale", "auto") or POSITIVE(v),
                          "max_train_samples": POSITIVE_INT}},
    "evaluation": {"primary": _choice("leave_one_active_load_flight_out"),
                   "validation_selection": _choice("cyclic_next_complete_flight"),
                   "minimum_active_flights": _number(3, integer=True),
                   "tuning_scope": _choice("development_only"), "deadlines_seconds": _list_of(POSITIVE)},
    "threshold": {"method": _choice("healthy_validation_percentile"),
                  "reference_percentile": PERCENTILE, "candidates": _list_of(PERCENTILE)},
    "persistence": {"mode": _choice("time"), "reference_duration_seconds": POSITIVE,
                    "candidates_seconds": _list_of(POSITIVE)},
    "dual_path": {"enabled": BOOL, "slope_windows_seconds": _list_of(POSITIVE),
                  "high_percentiles": _list_of(PERCENTILE), "low_percentiles": _list_of(PERCENTILE),
                  "slope_percentiles": _list_of(PERCENTILE), "path_b_persistence_seconds": _list_of(POSITIVE),
                  "refinement_max_candidates": POSITIVE_INT},
    "fault_injection": {"active_load_only_for_current_dependent_faults": _choice(True),
                        "start_positions": _list_of(lambda v: _number(0, 1, positive=True)(v) and v < 1),
                        "resistance_ohm": SEVERITIES, "sudden_drop_v": SEVERITIES,
                        "sudden_ramp_seconds": _number(), "progressive_max_delta_r_ohm": SEVERITIES,
                        "observability_max_sag_v": {"very_low": POSITIVE, "low": POSITIVE, "medium": POSITIVE}},
    "promotion": {"max_mean_healthy_fpr_increase": _choice(0.0),
                  "max_mean_episodes_per_minute_increase": _choice(0.0),
                  "require_f1_non_decrease": _choice(True), "require_deadline_non_decrease": _choice(True),
                  "require_mean_delay_improvement": _choice(True)},
    "research_targets": {"abrupt_severe_rate_le_1s": _number(0, 1),
                         "gradual_observable_rate_le_5s": _number(0, 1),
                         "status": _choice("aspirational_not_measured")},
    "artifacts": {"root": TEXT},
    "replay": {"enabled": BOOL, "nominal_rate_hz": POSITIVE, "require_batch_equivalence": _choice(True)},
}


def _validate(value: Any, schema: Any, field: str = "config") -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{field}: expected a mapping.")
        missing, unknown = set(schema) - set(value), set(value) - set(schema)
        if missing or unknown:
            raise ConfigError(f"{field}: missing keys={sorted(missing)}, unknown keys={sorted(unknown)}")
        for key, validator in schema.items():
            _validate(value[key], validator, f"{field}.{key}")
    elif not schema(value):
        raise ConfigError(f"{field}: invalid value {value!r}.")


def project_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve against V2, never CWD; reject escape to V1 or other directories."""
    root = root.resolve()
    try:
        if "\x00" in str(value):
            raise ValueError("embedded null character")
        path = Path(value)
        resolved = (path if path.is_absolute() else root / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"Invalid project path {value!r}: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise ConfigError(f"Path must stay inside V2 root {root}: {value}")
    return resolved


@dataclass(frozen=True)
class AppConfig:
    source: Path
    root: Path
    _values: dict

    def as_dict(self) -> dict:
        return deepcopy(self._values)

    @property
    def seed(self) -> int:
        return self._values["random_seed"]

    @property
    def data_path(self) -> Path:
        return project_path(self._values["data"]["path"], self.root)

    @property
    def artifacts_path(self) -> Path:
        return project_path(self._values["artifacts"]["root"], self.root)

    @property
    def gap_threshold_seconds(self) -> float:
        return self._values["time"]["max_gap_multiplier"] / self._values["time"]["nominal_frequency_hz"]


def load_config(path: str | Path = "configs/config.yaml", *, root: Path = PROJECT_ROOT) -> AppConfig:
    root = root.resolve()
    source = project_path(path, root)
    try:
        values = yaml.load(source.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config {source}: {exc}") from exc
    _validate(values, SCHEMA)
    features = values["features"]
    validate_regression_predictors(features["regression_predictors"], features["operational_set"],
                                   features["rolling_windows_seconds"])
    if tuple(features["baseline_predictors"]) != DIRECT_BASELINE_PREDICTORS:
        raise ConfigError(f"config.features.baseline_predictors: expected {list(DIRECT_BASELINE_PREDICTORS)}")
    obs = values["fault_injection"]["observability_max_sag_v"]
    if not obs["very_low"] < obs["low"] < obs["medium"]:
        raise ConfigError("config.fault_injection.observability_max_sag_v: thresholds must increase.")
    for key in ("resistance_ohm", "sudden_drop_v", "progressive_max_delta_r_ohm"):
        levels = values["fault_injection"][key]
        if not levels["mild"] < levels["moderate"] < levels["severe"]:
            raise ConfigError(f"config.fault_injection.{key}: severities must increase.")
    config = AppConfig(source, root, deepcopy(values))
    if not math.isfinite(config.gap_threshold_seconds) or config.gap_threshold_seconds <= 0:
        raise ConfigError("config.time: derived gap threshold must be finite and positive.")
    if not config.data_path.is_file():
        raise ConfigError(f"config.data.path: CSV file does not exist: {config.data_path}")
    if config.artifacts_path == root or config.artifacts_path.is_relative_to(root / "data"):
        raise ConfigError("config.artifacts.root: use a dedicated artifact directory outside data/.")
    if config.artifacts_path.exists() and not config.artifacts_path.is_dir():
        raise ConfigError("config.artifacts.root: expected a directory path.")
    return config


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python/NumPy and return the explicit RNG for deterministic sampling.

    Future sklearn estimators must also receive random_state=config.seed; seeding
    does not promise bitwise equivalence across library versions or hardware.
    """
    if not _number(0, 2**32 - 1, integer=True)(seed):
        raise ConfigError("random_seed: expected integer in [0, 2**32-1].")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
