"""Pilot-only regression/healthy alarm diagnostics; no outer or fault metrics."""

from collections import defaultdict
import io
import math

import numpy as np


def regression_metrics(rows: list[dict]) -> dict:
    ready = [r for r in rows if r["reference_ready"]]
    if not ready:
        return dict(samples=0, mae_v=None, rmse_v=None, r2=None)
    actual = np.array([r["voltage_sag_actual"] for r in ready], dtype=float)
    prediction = np.array([r["predicted_voltage_sag"] for r in ready], dtype=float)
    residual = actual - prediction
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    return dict(samples=len(ready), mae_v=float(np.mean(np.abs(residual))),
                rmse_v=float(np.sqrt(np.mean(residual ** 2))),
                r2=1 - float(np.sum(residual ** 2)) / total_variance if len(ready) >= 2 and total_variance > 0 else None)


def alarm_metrics(rows: list[dict], gap_threshold_seconds: float) -> dict:
    ready = [r for r in rows if r["reference_ready"]]
    raw = sum(r["anomaly_raw"] for r in ready)
    confirmed = sum(r["anomaly_confirmed"] for r in ready)
    episodes = sum(r["confirmed_episode_start"] for r in ready)
    previous, observed_us = {}, 0
    for row in rows:
        key = row["NamaFile"], row["Inst"]
        before = previous.get(key)
        if before and before["reference_ready"] and row["reference_ready"] and before["segment_id"] == row["segment_id"]:
            interval = row["TimeUS"] - before["TimeUS"]
            if 0 < interval / 1e6 <= gap_threshold_seconds:
                observed_us += interval
        previous[key] = row
    duration = observed_us / 1e6
    return dict(ready_samples=len(ready), raw_alarm_samples=raw, confirmed_alarm_samples=confirmed,
                raw_fpr=raw / len(ready) if ready else None, confirmed_fpr=confirmed / len(ready) if ready else None,
                confirmed_episodes=episodes, observed_ready_seconds=duration,
                confirmed_episodes_per_minute=episodes * 60 / duration if duration > 0 else None)


def pilot_tables(rows: list[dict], gap_threshold_seconds: float) -> tuple[list[dict], list[dict], dict]:
    groups, roles = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[(row["split_role"], row["NamaFile"])].append(row)
        roles[row["split_role"]].append(row)
    regression, alarms = [], []
    for (role, flight), records in sorted(groups.items()):
        labels = dict(split_role=role, NamaFile=flight, unseen_by_model=role != "train", used_for_threshold=role == "validation")
        regression.append({**labels, **regression_metrics(records)})
        alarms.append({**labels, **alarm_metrics(records, gap_threshold_seconds)})
    by_role = {role: {"regression": regression_metrics(records), "healthy_alarm": alarm_metrics(records, gap_threshold_seconds),
                      "interpretation": "in_sample_fit_diagnostic" if role == "train" else "healthy_validation_calibration_diagnostic"}
               for role, records in sorted(roles.items())}
    return regression, alarms, by_role


def pilot_report(summary: dict, training: dict, regression: list[dict], alarms: list[dict]) -> str:
    lines = ["# Sprint 4 — single-path HGB reference pilot", "",
             f"Run `{summary['run_id']}`. Scope: historical development pilot; no independent final holdout claim.",
             f"Train flights: {', '.join(summary['split']['train'])}; validation: {', '.join(summary['split']['validation'])}; reserved test: {', '.join(summary['split']['test'])}.",
             "Pilot assignment/parameters were saved before fit. Reserved test has no model prediction or performance metric here. Other low-load flights are excluded from this pilot; their evaluation remains a later stage.", "",
             f"HGB fit used {training['ready_training_samples']} healthy-assumed reference-ready rows, including {training['low_current_training_samples']} low-current rows. No synthetic/scenario rows or scaler. Fixed seed {summary['random_seed']}; early stopping disabled to avoid a random row validation split; OpenMP limited to one thread.",
             f"Threshold: {summary['threshold_v']:.12g} V, percentile 99 (linear) of healthy validation positive residuals. Score=max(0, actual sag - predicted sag); raw iff score > threshold. Persistence fixed at 0.5 s from first raw timestamp; false, unready, flight/instance/segment boundary or gap resets it.", "",
             "## Regression diagnostics", "", "| Role | Flight | Ready rows | MAE V | RMSE V | R2 |", "|---|---|---:|---:|---:|---:|"]
    for r in regression:
        r2 = f"{r['r2']:.6f}" if r["r2"] is not None else "undefined"
        mae = f"{r['mae_v']:.6f}" if r["mae_v"] is not None else "undefined"
        rmse = f"{r['rmse_v']:.6f}" if r["rmse_v"] is not None else "undefined"
        lines.append(f"| {r['split_role']} | {r['NamaFile']} | {r['samples']} | {mae} | {rmse} | {r2} |")
    lines += ["", "## Healthy alarm diagnostics", "", "| Role | Flight | Raw FPR | Confirmed FPR | Episodes | Ready observed s | Episodes/min |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in alarms:
        def display(value):
            return f"{value:.6f}" if value is not None else "undefined"
        lines.append(f"| {r['split_role']} | {r['NamaFile']} | {display(r['raw_fpr'])} | {display(r['confirmed_fpr'])} | {r['confirmed_episodes']} | {r['observed_ready_seconds']:.6f} | {display(r['confirmed_episodes_per_minute'])} |")
    lines += ["", "FPR denominator is reference-ready healthy-assumed samples. Observed duration includes only intervals with both endpoints ready in the same continuous segment, excluding gaps/warm-up. Confirmed episode starts are counted after persistence; a new confirmed run after a gap is a new episode. Undefined rates/R2 remain null/blank, never fabricated zeros.",
              "Training metrics are in-sample and must not be presented as unseen FPR. Validation is unseen by the regressor but used for threshold calibration, so these are calibration diagnostics rather than independent final performance. No threshold/persistence/model candidate is changed using these results.", "",
              "## Limits and next stages", "", "CurrTot remains unverified in source units. Healthy status is assumed; original firmware/pack metadata and real battery fault ground truth are absent. Model bundle includes fitted HGB, ordered feature registry, schema/unit/reference policies, split, seed, exact parameters, threshold provenance, dependency versions and data/artifact hashes.",
              "Synthetic applicability/observability is Stage 5; full LOAFO with fault/held-out test metrics is Stage 6. Recall, F1, delay/deadline rates, final detector gate, promotion and end-to-end replay have not been measured. Status remains RESEARCH_PILOT_ONLY; no hardware readiness claim.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def validation_plot(rows: list[dict], threshold_v: float) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ready = [r for r in rows if r["reference_ready"]]
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    blocks = defaultdict(list)
    for row in ready:
        blocks[(row["NamaFile"], row["Inst"], row["segment_id"])].append(row)
    for index, block in enumerate(blocks.values()):
        x = [r["elapsed_time"] for r in block]
        axes[0].plot(x, [r["voltage_sag_actual"] for r in block], color="#2457a6", label="Actual sag" if index == 0 else None)
        axes[0].plot(x, [r["predicted_voltage_sag"] for r in block], color="#d17a15", alpha=.85, label="HGB prediction" if index == 0 else None)
        axes[1].plot(x, [r["residual_signed_v"] for r in block], color="#247c5c", label="Signed residual" if index == 0 else None)
        axes[2].plot(x, [r["Curr"] for r in block], color="#6651a5")
    axes[1].axhline(threshold_v, color="#b52d35", linestyle="--", label="Validation P99 threshold")
    warnings = [r for r in ready if r["anomaly_confirmed"]]
    if warnings:
        axes[1].scatter([r["elapsed_time"] for r in warnings], [r["residual_signed_v"] for r in warnings],
                         color="#b52d35", s=12, label="Confirmed warning", zorder=5)
    axes[0].set_title("Healthy validation calibration diagnostic — " + ", ".join(sorted({r["NamaFile"] for r in rows})))
    for axis, label in zip(axes, ("Voltage sag (V)", "Residual (V)", "Current (A)")):
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[2].set_xlabel("Telemetry elapsed time (s); gaps excluded from exposure")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, metadata={"Software": "UAV Battery V2"})
    plt.close(figure)
    return buffer.getvalue()
