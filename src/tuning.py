"""Stages 8/9: validation-only parameter search; freeze before outer evaluation."""

from .config import AppConfig
from .errors import StageNotImplementedError


def tune(config: AppConfig) -> dict:
    from .single_path_tuning import run_tuning
    return run_tuning(config)


def tune_dual_path(config: AppConfig) -> dict:
    from .dual_path_experiment import run_dual_experiment
    return run_dual_experiment(config)
