"""Factual Sprint 7 comparison report, without performance-based promotion."""

import io


def comparison_report(result, scenarios):
    summary = result["run_summary"]
    lines = [f"# Sprint 7 — {summary['command']}", "", "Scope: **historical development comparison**. Outer results were already examined in Sprint 6. Feature/model choice from these results is development selection, not an independent final estimate.",
             f"Run `{summary['run_id']}`; {summary['candidates']} predeclared candidates × {summary['folds']} identical folds.", "",
             "## Fixed experiment", "",
             "All candidate-fold models and healthy-validation P99 policies were fitted, saved and frozen before ANY outer prediction in this experiment. Fixed persistence 0.5 s, seed 42, whole healthy-ready training flights including idle, no synthetic training. No parameter search, winner selection, operational config change or detector promotion is performed. Battery-E elapsed time is ablation only. CurrTot remains unverified, without conversion.",
             "HGB/RF use only exact registered Battery-A–E predictors and positive sag residual in V; actual voltage behavior is forbidden in their inputs. IF/OCSVM use the separate direct registry [voltage_sag_actual, Curr, CurrTot, dV_dt, dI_dt]. Their score is -decision_function, higher is more abnormal, may be negative, and is not in volts. Scores/threshold magnitudes across model families are not comparable. Direct models do not predict voltage; regression metrics are not applicable and omitted, never invented.",
             "Only OCSVM has StandardScaler, fitted on all healthy-ready training rows; SVM fit uses at most 2500 uniformly sampled training rows (configured cap), without replacement, sorted indices with seed 42 and saved source identities. IF uses its configured internal max_samples. RF/HGB fit all assigned ready rows. HGB internal random-row early stopping is disabled.",
             "All candidates share source features, six eligible flights, cyclic splits, readiness, fault definitions and applicable denominators. Synthetic regression predictions are reused only after exact registered-input/identity/readiness equality; direct baselines rescore modified voltage/sag/dV_dt. Persistence starts anew for every full-flight copy.", "",
             "## Pooled outer results", "",
             "| Candidate | RMSE V | Healthy FPR % | Episodes/min | Incident detected/applicable | Contaminated | Sample F1 | Median detected delay s | Detected <=5 s/applicable |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: f"{v:.6f}" if v is not None else "N/A"
    for name, record in result["candidates"].items():
        pool = record["pooled"]
        reg, health, syn = pool["regression"], pool["healthy_active"], pool["synthetic"]
        lines.append(f"| {name} | {fmt(reg['rmse_v'] if reg else None)} | {fmt(health['confirmed_fpr'] * 100)} | {fmt(health['confirmed_episodes_per_minute'])} | {syn['detected_scenarios']}/{syn['applicable_scenarios']} | {syn['preexisting_alarm_scenarios']} | {fmt(syn['f1'])} | {fmt(syn['delay_median_seconds'])} (n={syn['delay_detected_n']}) | {syn['deadline_le_5s_successes']}/{syn['applicable_scenarios']} |")
    lines += ["", "## Aggregation and failures", "",
              "Untouched healthy counts include each test flight once per candidate. Synthetic counts repeat source flights for different scenarios and are reported separately. Macro per-fold mean/population std/median/min/max/n_defined and pooled counts/exposure are in summary.csv/results.json. Pooled F1/rates are computed from combined counts, not averages of fold rates. Regression metrics are included only for HGB/RF. Paired fold deltas versus HGB Battery-D are in paired_per_fold.csv; no automatic significance or winner claim is made.",
              "Only reference-ready samples enter conventional TP/FP/TN/FN. A warning carried into onset contaminates the entire incident even if a fresh episode occurs later; conventional TPs remain separately counted. New confirmation exactly at onset can have zero delay; post-support detection does not count. Undetected and contaminated applicable cases fail every deadline; delay is null. NOT_APPLICABLE has no inference, confusion or denominator eligibility and remains visible. Delay statistics are detected-only with n. Pre-fault alarms/FPR, observability and all failure reasons remain in fault_per_scenario.csv and failed_cases.csv.",
              "Healthy FPR and episodes/minute must be assessed alongside recall/F1 and deadlines. No candidate is accepted by improving recall alone; no denominator, threshold or scope is changed based on outer results. Severe abrupt and progressive observability subsets can be inspected in the full scenario table; future tuning must remain validation-only and be labelled development work.", "",
              "## Reproducibility and limits", "",
              "Each candidate/fold stores model + optional scaler bundle, exact features/parameters, training provenance/sample indices/identities, threshold source/fingerprint, selected policy and validation/test/scenario scores. all_policies_frozen.json records pre-outer hashes. manifest.json verifies every output before cache reuse. Root includes assignments, reference coverage, calibration diagnostics, regression/healthy/anomaly tables, scenario/failure tables, paired deltas, summary, results and plot.",
              "These are healthy-assumed historical flights and controlled synthetic faults, not measured battery failures. Six flights and incomplete pack/firmware metadata limit inference. CurrTot units remain unverified. Low-load false alarms, threshold/persistence tuning, dual-path, end-to-end replay and deployment readiness are outside Sprint 7. Status RESEARCH_COMPARISON_ONLY; fixed Battery-D operational config and prior baseline artifacts are preserved.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def comparison_plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = result["candidates"]
    labels = list(records)
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(labels) * 1.1), 9), constrained_layout=True)
    for ax, metric, scope, factor, title in zip(axes.flat,
            ("confirmed_fpr", "confirmed_episodes_per_minute", "f1", "deadline_le_5s_rate"),
            ("healthy_active", "healthy_active", "synthetic", "synthetic"), (100, 1, 1, 1),
            ("Untouched healthy confirmed FPR (%)", "Untouched healthy episodes/minute", "Synthetic sample F1 (pooled counts)", "Incident detected <=5 s / all applicable")):
        values = [records[name]["pooled"][scope][metric] for name in labels]
        ax.bar(range(len(labels)), [v * factor if v is not None else 0 for v in values], color=["#34699a" if s.startswith("hgb") else "#d18239" if s.startswith("rf") else "#4b8c68" for s in labels])
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for i, v in enumerate(values):
            if v is None:
                ax.text(i, 0, "N/A", ha="center")
    fig.suptitle("Sprint 7 historical development comparison — fixed P99, persistence 0.5 s; no winner promotion")
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=140)
    plt.close(fig)
    return out.getvalue()
