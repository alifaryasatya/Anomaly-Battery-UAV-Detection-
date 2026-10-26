"""Historical development cross-validation tables, plots and factual report."""

from collections import Counter
import io

import numpy as np

from .evaluation import aggregate_scenarios, deadline_name


def _fmt(value, digits=6):
    return f"{value:.{digits}f}" if value is not None else "undefined"


def cv_report(results, regression, healthy, anomaly, scenarios) -> str:
    summary = results["run_summary"]
    pooled = results["pooled"]
    syn = pooled["synthetic"]
    lines = ["# Sprint 6 — HGB Battery-D LOAFO baseline", "",
             "Scope: **historical development cross-validation**. These historical flights were previously examined in V1; this is not a new independent final holdout.",
             f"Run `{summary['run_id']}`; {summary['folds']} outer folds; active/reference-ready flight IDs: {', '.join(summary['active_flights'])}.", "",
             "## Frozen procedure", "",
             "Sorted flight i is outer test; cyclic next flight is validation; remaining whole flights train. All models/thresholds were fit, calibrated and saved before ANY outer prediction. HGB uses Battery-D, fixed config/seed, no scaler, no synthetic fitting and no internal random row early-stopping split. Healthy training includes low-current periods; reference-unready rows are excluded. Percentile 99 linear healthy-validation threshold and 0.5 s persistence are unchanged across the reference experiment. No candidate is revised after outer results.",
             "Outer healthy active results below are unseen by their evaluating regressor and unused for that fold's calibration. Validation calibration diagnostics remain in a separate CSV. Low-load evaluation is deferred to Stage 10 and is not inferred from these results.", "",
             "## Outer results per fold", "",
             "| Fold | Test | RMSE V | Healthy confirmed FPR | Healthy episodes/min | Sample F1 | Incident detected/applicable | Contaminated | Median detected delay s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for reg, alarm, fault in zip(regression, healthy, anomaly):
        lines.append(f"| {reg['fold_id']} | {reg['test_flight']} | {_fmt(reg['rmse_v'])} | {_fmt(alarm['confirmed_fpr'])} | {_fmt(alarm['confirmed_episodes_per_minute'])} | {_fmt(fault['f1'])} | {fault['detected_scenarios']}/{fault['applicable_scenarios']} | {fault['preexisting_alarm_scenarios']} | {_fmt(fault['delay_median_seconds'])} |")
    reg, alarm = pooled["regression"], pooled["healthy_active"]
    lines += ["", "## Pooled counts and macro statistics", "",
              f"Untouched outer healthy: {alarm['ready_samples']} ready samples, {alarm['confirmed_alarm_samples']} confirmed alarm samples, FPR {_fmt(alarm['confirmed_fpr'])}; {alarm['confirmed_episodes']} episodes / {_fmt(alarm['observed_ready_seconds'])} s = {_fmt(alarm['confirmed_episodes_per_minute'])} episodes/min. These healthy counts include each outer flight once, without scenario replication.",
              f"Pooled regression: MAE {_fmt(reg['mae_v'])} V, RMSE {_fmt(reg['rmse_v'])} V, R2 {_fmt(reg['r2'])}. Macro per-fold RMSE mean {_fmt(results['macro_per_fold']['regression']['rmse_v']['mean'])}, population std {_fmt(results['macro_per_fold']['regression']['rmse_v']['std'])} V.",
              f"Synthetic cases: {syn['scenarios']} planned, {syn['applicable_scenarios']} applicable, {syn['not_applicable_scenarios']} NOT_APPLICABLE. Incident detected {syn['detected_scenarios']}; undetected applicable {syn['undetected_scenarios']} (including {syn['preexisting_alarm_scenarios']} PREEXISTING_ALARM). Incident recall {_fmt(syn['incident_recall'])}.",
              f"Conventional synthetic samples: TP={syn['tp']}, FP={syn['fp']}, TN={syn['tn']}, FN={syn['fn']}; precision {_fmt(syn['precision'])}, recall {_fmt(syn['recall'])}, F1 {_fmt(syn['f1'])}, sample FPR {_fmt(syn['synthetic_sample_fpr'])}. Macro per-fold F1 mean {_fmt(results['macro_per_fold']['synthetic']['f1']['mean'])}. These expanded sample counts intentionally repeat source flights across distinct synthetic scenarios and are separate from untouched healthy FPR.",
              f"Delay on incident-detected cases only: n={syn['delay_detected_n']}, mean {_fmt(syn['delay_mean_seconds'])} s, median {_fmt(syn['delay_median_seconds'])} s, population std {_fmt(syn['delay_std_seconds'])} s. Undetected applicable cases remain failures in all deadline denominators.", "",
              "| Deadline | Incident successes / applicable | Rate |", "|---|---:|---:|"]
    for key, value in syn.items():
        if key.startswith("deadline_le_") and key.endswith("_rate"):
            base = key[:-5]
            lines.append(f"| {base} | {syn[base + '_successes']}/{syn['applicable_scenarios']} | {_fmt(value)} |")
    lines += ["", "All scalar macro metrics include mean/std/median/min/max and n_defined in summary.csv/results.json. Pooled precision/recall/F1/FPR/deadlines are recomputed from counts, not averages of fold rates. Pooled delays use all incident-detected scenarios; macro delay statistics use defined per-fold delays. Undefined values remain null/blank with their denominators.", "",
              "## Research targets (fixed before outer evaluation)", ""]
    for name, check in results["research_target_checks"].items():
        lines.append(f"- {name}: rate {_fmt(check['rate'])}, target {_fmt(check['target'])}, applicable={check['applicable']}, status={check['status']}.")
    lines += ["", "Abrupt/severe is increased-resistance or sudden-drop, severe level, <=1 s. Observable gradual is progressive with MEDIUM/HIGH added-sag observability, <=5 s. These development target checks do not authorize deployment or hardware switching.", "",
              "## Denominators, contamination and failed cases", "",
              "Only reference-ready rows enter conventional confusion counts. NOT_APPLICABLE has no confusion counts or incident/recall/deadline eligibility and remains visible with its reason. Applicable undetected and PREEXISTING_ALARM stay in every deadline denominator as failures. Support includes aligned onset (possibly zero effect for ramp/progressive); onset and first-effect timestamps are both retained from the frozen scenario definition.",
              "A confirmed warning carried into onset (onset confirmed without a new episode start) contaminates the entire incident. It is never an incident success, even if another episode begins later. Conventional TP samples are retained separately. A genuinely new confirmation exactly at onset can have delay zero. Detection after support ends does not count. Delay already includes persistence; it is not added again.",
              "Healthy/pre-fault exposure uses observed intervals with both endpoints ready in the same stream/segment, excluding gaps and warm-up. Episode starts after gaps are new. Sample FPR and episodes/minute are distinct metrics. failed_cases.csv includes every undetected, contaminated and NOT_APPLICABLE case; no failure is dropped from the full fault table.", "",
              "## Files and limits", "",
              "Root CSVs: fold_assignments, regression_per_fold, healthy_active_per_fold, anomaly_per_fold, fault_per_scenario, failed_cases, validation_calibration_per_fold and summary. Each fold saves model bundle, training provenance, threshold/selected policy, calibration scores, untouched test scores and applicable scenario scores (deterministic CSV.gz). all_fold_policies_frozen.json records pre-outer hashes; manifest.json verifies the complete run. results.json contains pooled counts/macro distributions and research target checks. Plot examples include a fixed severe progressive trace and the first unsuccessful applicable case when one exists; they are diagnostic only.",
              "Healthy labels are assumed, CurrTot remains unverified in source units, no real internal battery fault ground truth exists, and flight count/domain are limited. No feature/model selection, tuning, dual-path promotion, low-load false-alarm validation or end-to-end replay is claimed here. Status remains RESEARCH_BASELINE_ONLY; subsequent outer-informed refinement must be labelled exploratory/development selection.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _png(plt, figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, metadata={"Software": "UAV Battery V2"})
    plt.close(figure)
    return buffer.getvalue()


def fold_plot(regression, healthy, anomaly) -> bytes:
    plt = _plotting()
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    labels = [r["test_flight"].replace(".BIN", "")[-2:] for r in regression]
    panels = [(regression, "rmse_v", "Outer RMSE (V)", 1), (anomaly, "f1", "Synthetic sample F1", 1),
              (anomaly, "incident_recall", "Incident recall", 1), (healthy, "confirmed_fpr", "Healthy confirmed FPR (%)", 100),
              (anomaly, "delay_median_seconds", "Median incident-detected delay (s)", 1),
              (anomaly, "deadline_le_5s_rate", "Applicable scenarios detected <=5 s", 1)]
    for axis, (records, key, title, scale) in zip(axes.flat, panels):
        values = [r.get(key) for r in records]
        axis.bar(labels, [v * scale if v is not None else np.nan for v in values], color="#346c9c")
        axis.set_title(title)
        axis.set_xlabel("Outer test flight")
        axis.grid(axis="y", alpha=.2)
        if key == "delay_median_seconds":
            for x, record in enumerate(records):
                axis.text(x, 0, f"u={record['undetected_scenarios']}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Historical development LOAFO — HGB Battery-D, validation P99, persistence 0.5 s")
    return _png(plt, figure)


def deadline_plot(scenarios, deadlines) -> bytes:
    plt = _plotting()
    labels, grid, counts = [], [], []
    for family in ("increased_resistance", "progressive", "sudden_drop"):
        for severity in ("mild", "moderate", "severe"):
            selected = [r for r in scenarios if r["fault_family"] == family and r["severity"] == severity]
            result = aggregate_scenarios(selected, deadlines)
            labels.append(f"{family} / {severity}")
            grid.append([result[deadline_name(d) + "_rate"] for d in deadlines])
            counts.append([(result[deadline_name(d) + "_successes"], result["applicable_scenarios"]) for d in deadlines])
    matrix = np.array([[v if v is not None else np.nan for v in row] for row in grid])
    figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    chart = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(deadlines)), [f"<={d:g} s" for d in deadlines])
    axis.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(counts):
        for j, (success, denominator) in enumerate(row):
            axis.text(j, i, f"{success}/{denominator}" if denominator else "N/A", ha="center", va="center",
                      color="white" if denominator and success / denominator > .6 else "black")
    axis.set_title("Incident deadline rates — all applicable cases; undetected/contaminated fail")
    figure.colorbar(chart, ax=axis, label="Detected within deadline / applicable")
    return _png(plt, figure)


def fault_plot(rows, metadata) -> bytes:
    plt = _plotting()
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    groups = {}
    for row in rows:
        if row["reference_ready"]:
            groups.setdefault(row["segment_id"], []).append(row)
    for index, block in enumerate(groups.values()):
        time = [r["elapsed_time"] for r in block]
        axes[0].plot(time, [r["voltage_sag_actual"] for r in block], color="#285d98", label="Injected actual sag" if index == 0 else None)
        axes[0].plot(time, [r["predicted_voltage_sag"] for r in block], color="#cb8024", label="HGB prediction" if index == 0 else None)
        axes[1].plot(time, [r["anomaly_score"] for r in block], color="#267253", label="Positive residual" if index == 0 else None)
        axes[2].plot(time, [r["Curr"] for r in block], color="#6c58a2")
    start = rows[0]["TimeUS"]
    onset, end = (metadata["onset_time_us"] - start) / 1e6, (metadata["fault_end_time_us"] - start) / 1e6
    axes[1].axhline(metadata["threshold_v"], color="#b22f38", linestyle="--", label="Frozen validation threshold")
    warnings = [r for r in rows if r["anomaly_confirmed"]]
    axes[1].scatter([r["elapsed_time"] for r in warnings], [r["anomaly_score"] for r in warnings], color="#b22f38", s=8, label="Confirmed")
    for axis, ylabel in zip(axes, ("Voltage sag (V)", "Score (V)", "Current (A)")):
        axis.axvspan(onset, end, color="#d8ad48", alpha=.15)
        axis.axvline(onset, color="#555555", linestyle=":")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[2].set_xlabel("Telemetry elapsed time (s); shaded = fault support")
    axes[0].set_title(f"{metadata['fold_id']} / {metadata['test_flight']} / {metadata['fault_family']} {metadata['severity']} {metadata['start_position']:.0%}\n{metadata['evaluation_status']}; delay={_fmt(metadata['delay_seconds'], 3)} s")
    return _png(plt, figure)
