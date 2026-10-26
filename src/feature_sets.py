"""Single allowlist for regression predictors, direct baselines and feature roles."""

import math

from .errors import ConfigError

BATTERY_COLUMNS = ("NamaFile", "TimeUS", "Inst", "Volt", "Curr", "CurrTot")
DIRECT_BASELINE_PREDICTORS = ("voltage_sag_actual", "Curr", "CurrTot", "dV_dt", "dI_dt")
FEATURE_SETS = ("battery_a", "battery_b", "battery_c", "battery_d", "battery_e")


def window_token(window: float) -> str:
    return str(float(window)).replace(".", "p")


def validate_windows(windows: list[float]) -> None:
    if (not windows or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in windows)
            or len(set(windows)) != len(windows)):
        raise ConfigError("features.rolling_windows_seconds: expected unique positive finite seconds.")


def regression_predictors(feature_set: str, windows: list[float]) -> tuple[str, ...]:
    """Allowlist predictors instead of attempting to blacklist every voltage alias."""
    validate_windows(windows)
    if feature_set not in FEATURE_SETS:
        raise ConfigError(f"features.operational_set: unknown feature set {feature_set!r}")
    a = ("Curr",)
    b = (*a, "CurrTot")
    c = (*b, "dI_dt")
    rolling = tuple(f"Curr_roll_{stat}_{window_token(window)}s"
                    for window in windows for stat in ("mean", "std"))
    d = (*c, *rolling)
    return {"battery_a": a, "battery_b": b, "battery_c": c,
            "battery_d": d, "battery_e": (*d, "elapsed_time")}[feature_set]


def validate_regression_predictors(predictors: list[str] | tuple[str, ...], feature_set: str,
                                   windows: list[float]) -> tuple[str, ...]:
    expected = regression_predictors(feature_set, windows)
    if not isinstance(predictors, (list, tuple)) or tuple(predictors) != expected:
        raise ConfigError(f"features.regression_predictors: expected {list(expected)}; "
                          "actual voltage, sag, residual, identifiers, readiness and all unregistered predictors are forbidden.")
    return expected


def feature_registry(windows: list[float], operational_set: str) -> dict:
    sets = {name: list(regression_predictors(name, windows)) for name in FEATURE_SETS}
    if operational_set not in sets:
        raise ConfigError(f"features.operational_set: unknown feature set {operational_set!r}")
    return {
        "version": 1, "operational_set": operational_set, "regression_sets": sets,
        "ablation_only_sets": ["battery_e"], "direct_baseline_predictors": list(DIRECT_BASELINE_PREDICTORS),
        "target": "voltage_sag_actual", "target_unit": "V", "target_definition": "V_ref - Volt",
        "units": {"Curr": "A", "CurrTot": "unverified_source_units", "dI_dt": "A/s",
                  "dV_dt": "V/s", "elapsed_time": "s"},
        "currtot_policy": "No conversion; review any reported decrease before Battery-B–E fitting.",
        "excluded_from_regression": ["Volt", "V_ref", "voltage_sag_actual", "dV_dt", "residual",
                                     "anomaly_score", "Volt_x_Curr", "NamaFile", "Inst", "TimeUS",
                                     "dt_seconds", "segment_id", "source_record", "readiness",
                                     "reference_ready", "reference_status", "derivative_ready",
                                     "rolling count/coverage/readiness", "offline activity/flight labels"],
        "elapsed_time_policy": "Seconds since first accepted stream sample; Battery-E ablation only.",
        "rolling": [{"window_seconds": w, "interval": "(t-window, t]", "ddof": 0,
                     "mean_column": f"Curr_roll_mean_{window_token(w)}s",
                     "std_column": f"Curr_roll_std_{window_token(w)}s", "unit": "A",
                     "count_column": f"Curr_roll_count_{window_token(w)}s",
                     "coverage_column": f"Curr_roll_coverage_{window_token(w)}s",
                     "ready_column": f"Curr_roll_ready_{window_token(w)}s"} for w in windows],
        "coverage_policy": "min(segment elapsed / window, 1); flags and counts are diagnostics, never automatic predictors.",
    }
