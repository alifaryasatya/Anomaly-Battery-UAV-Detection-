"""Scenario/aggregate metrics and later comparison orchestration entry points."""

import math
from statistics import mean, median, pstdev

from .config import AppConfig
from .errors import DataError
from .pilot_reporting import alarm_metrics


def deadline_name(seconds: float) -> str:
    return "deadline_le_" + format(seconds, "g").replace(".", "p") + "s"


def ratio(numerator: int | float, denominator: int | float):
    return numerator / denominator if denominator > 0 else None


def count_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn), synthetic_sample_fpr=ratio(fp, fp + tn))


def distribution(values: list[float | None]) -> dict:
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return dict(n_defined=0, mean=None, std=None, median=None, min=None, max=None)
    return dict(n_defined=len(finite), mean=mean(finite), std=pstdev(finite), median=median(finite),
                min=min(finite), max=max(finite))


def scenario_metrics(rows: list[dict], metadata: dict, deadlines: list[float], gap_seconds: float) -> dict:
    """Conventional sample metrics and incident metrics have distinct numerators.

    A confirmed alarm carried into onset contaminates the entire incident: even
    a later fresh alarm cannot turn that scenario into an incident success.
    """
    result = dict(fault_applicable=metadata["fault_applicable"], evaluation_status="NOT_APPLICABLE",
                  detection_reason="NOT_APPLICABLE", raw_fault_support_samples=None,
                  preexisting_alarm=None, incident_detected=None, first_confirmation_time_us=None,
                  candidate_new_confirmation_time_us=None, delay_seconds=None, evaluated_samples=None,
                  fault_support_samples=None, pre_fault_samples=None, pre_fault_raw_alarm_samples=None,
                  pre_fault_confirmed_alarm_samples=None, pre_fault_raw_fpr=None, pre_fault_fpr=None,
                  confirmed_episodes=None, observed_ready_seconds=None, confirmed_episodes_per_minute=None,
                  **{key: None for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "synthetic_sample_fpr")},
                  **{deadline_name(d): None for d in deadlines})
    if not metadata["fault_applicable"]:
        return result  # no inference/counts/denominator for NOT_APPLICABLE
    onset, end = metadata["onset_time_us"], metadata["fault_end_time_us"]
    if type(onset) is not int or type(end) is not int or end < onset:
        raise DataError("Applicable scenario lacks valid onset/end timestamps.")
    ready = [r for r in rows if r["reference_ready"]]
    support = [r for r in ready if r["fault_support"]]
    if not support or support[0]["TimeUS"] != onset or support[-1]["TimeUS"] != end:
        raise DataError("Scenario support and declared onset/end disagree.")
    if any(not onset <= r["TimeUS"] <= end for r in support):
        raise DataError("Fault support lies outside declared bounds.")
    preexisting = bool(support[0]["anomaly_confirmed"] and not support[0]["confirmed_episode_start"])
    fresh = next((r for r in support if r["anomaly_confirmed"] and r["confirmed_episode_start"]), None)
    detected = fresh is not None and not preexisting
    delay = (fresh["TimeUS"] - onset) / 1e6 if detected else None
    tp = sum(r["fault_support"] and r["anomaly_confirmed"] for r in ready)
    fp = sum(not r["fault_support"] and r["anomaly_confirmed"] for r in ready)
    tn = sum(not r["fault_support"] and not r["anomaly_confirmed"] for r in ready)
    fn = sum(r["fault_support"] and not r["anomaly_confirmed"] for r in ready)
    pre = [r for r in ready if r["TimeUS"] < onset]
    raw_pre = sum(r["anomaly_raw"] for r in pre)
    confirmed_pre = sum(r["anomaly_confirmed"] for r in pre)
    alarm = alarm_metrics(rows, gap_seconds)
    result.update(evaluation_status="PREEXISTING_ALARM" if preexisting else "DETECTED" if detected else "UNDETECTED",
                  detection_reason="CONFIRMED_ALARM_CARRIED_INTO_ONSET" if preexisting else "NEW_CONFIRMATION_WITHIN_SUPPORT" if detected
                  else "NO_RAW_EXCEEDANCE_IN_SUPPORT" if not any(r["anomaly_raw"] for r in support) else "NO_NEW_CONFIRMATION_WITHIN_SUPPORT",
                  raw_fault_support_samples=sum(r["anomaly_raw"] for r in support),
                  preexisting_alarm=preexisting, incident_detected=detected,
                  first_confirmation_time_us=fresh["TimeUS"] if detected else None,
                  candidate_new_confirmation_time_us=fresh["TimeUS"] if fresh else None, delay_seconds=delay,
                  evaluated_samples=len(ready), fault_support_samples=len(support),
                  pre_fault_samples=len(pre), pre_fault_raw_alarm_samples=raw_pre,
                  pre_fault_confirmed_alarm_samples=confirmed_pre,
                  pre_fault_raw_fpr=ratio(raw_pre, len(pre)), pre_fault_fpr=ratio(confirmed_pre, len(pre)),
                  confirmed_episodes=alarm["confirmed_episodes"], observed_ready_seconds=alarm["observed_ready_seconds"],
                  confirmed_episodes_per_minute=alarm["confirmed_episodes_per_minute"], **count_metrics(tp, fp, tn, fn),
                  **{deadline_name(d): bool(detected and delay <= d) for d in deadlines})
    return result


def aggregate_scenarios(records: list[dict], deadlines: list[float]) -> dict:
    applicable = [r for r in records if r["fault_applicable"]]
    detected = [r for r in applicable if r["incident_detected"]]
    contaminated = sum(r["preexisting_alarm"] for r in applicable)
    counts = {key: sum(r[key] for r in applicable) for key in ("tp", "fp", "tn", "fn")}
    delay = distribution([r["delay_seconds"] for r in detected])
    pre_count = sum(r["pre_fault_samples"] for r in applicable)
    pre_alarms = sum(r["pre_fault_confirmed_alarm_samples"] for r in applicable)
    result = dict(scenarios=len(records), applicable_scenarios=len(applicable), not_applicable_scenarios=len(records) - len(applicable),
                  detected_scenarios=len(detected), undetected_scenarios=len(applicable) - len(detected),
                  preexisting_alarm_scenarios=contaminated,
                  undetected_without_preexisting=len(applicable) - len(detected) - contaminated,
                  incident_recall=ratio(len(detected), len(applicable)),
                  **count_metrics(**counts), delay_detected_n=delay["n_defined"], delay_mean_seconds=delay["mean"],
                  delay_median_seconds=delay["median"], delay_std_seconds=delay["std"],
                  pre_fault_samples=pre_count, pre_fault_confirmed_alarm_samples=pre_alarms,
                  pre_fault_fpr=ratio(pre_alarms, pre_count))
    for deadline in deadlines:
        key = deadline_name(deadline)
        hits = sum(r[key] for r in applicable)
        result[key + "_successes"] = hits
        result[key + "_rate"] = ratio(hits, len(applicable))
    return result


def ablation(config: AppConfig) -> dict:
    from .comparison import run_comparison
    return run_comparison(config, "ablation")


def compare_models(config: AppConfig) -> dict:
    from .comparison import run_comparison
    return run_comparison(config, "compare-models")


def compare_dual_path(config: AppConfig) -> dict:
    from .dual_path_experiment import run_dual_experiment
    return dict(run_dual_experiment(config), command="compare-dual-path", comparison_from_frozen_experiment=True)


def evaluate_low_load(config: AppConfig) -> dict:
    from .low_load import run_low_load
    return run_low_load(config)


def research_evaluation(config: AppConfig) -> dict:
    from .research import run_research
    return run_research(config)
