"""Consolidated factual research report and scientific summary figures."""

import io


def make_plots(result, grouped):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = {}

    def save(fig, name):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, metadata={"Software": "UAV Battery V2"})
        plt.close(fig)
        payloads["plots/" + name + ".png"] = buffer.getvalue()

    overview = result["model_comparison"]["overview"]
    names = list(overview)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].bar(names, [100 * overview[n]["healthy_confirmed_fpr"] for n in names], color="#527a9c")
    axes[0].set_ylabel("Healthy confirmed FPR (%)")
    axes[0].set_title("Fixed model comparison: historical development, no automatic winner")
    axes[1].bar(names, [overview[n]["sample_f1"] for n in names], color="#3b8a70")
    axes[1].set_ylabel("Synthetic sample F1")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    save(fig, "model_comparison")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    names = list(result["ablation"]["overview"])
    x = list(range(len(names)))
    for ax, key, scale, label in ((axes[0], "healthy_confirmed_fpr", 100, "Healthy confirmed FPR (%)"),
                                   (axes[1], "sample_f1", 1, "Synthetic sample F1")):
        ax.bar(x, [result["ablation"]["overview"][n][key] * scale for n in names],
               color=["#527a9c" if n.startswith("hgb") else "#c67b40" for n in names])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Feature ablation: HGB blue, RF orange; Battery-E is ablation only")
    axes[1].set_xticks(x, [n.replace("_battery_", "-") for n in names], rotation=30)
    axes[1].set_ylim(0, 1)
    save(fig, "feature_ablation")

    deadlines = (1, 2, 5, 10, 30)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for policy, color, label in (("reference_pooled", "#527a9c", "Frozen reference"), ("selected_pooled", "#c67b40", "Selected dual candidate")):
        metrics = result["dual_path"][policy]["synthetic"]
        rates = [metrics["deadline_le_" + str(d) + "s_rate"] for d in deadlines]
        ax.plot(range(len(deadlines)), [100 * v if v is not None else float("nan") for v in rates], "o-", color=color, label=label)
    ax.set_xticks(range(len(deadlines)), [str(d) for d in deadlines])
    ax.set_xlabel("Confirmation deadline after onset (s)")
    ax.set_ylabel("Success / all applicable incidents (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Dual-path outcome: {result['promotion_decision']['status']}\nSame paired support/denominator; misses remain in deadline rates")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "paired_dual_deadlines")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    low = result["low_load_per_model"]
    axes[0].bar([r["fold_id"] for r in low], [100 * r["confirmed_fpr"] if r["confirmed_fpr"] is not None else float("nan") for r in low], color="#3b8a70")
    axes[1].bar([r["fold_id"] for r in low], [r["confirmed_episodes_per_minute"] if r["confirmed_episodes_per_minute"] is not None else float("nan") for r in low], color="#c67b40")
    axes[0].set_ylabel("Confirmed sample FPR (%)")
    axes[1].set_ylabel("Episodes / ready minute")
    axes[0].set_title(f"{result['low_load']['unique_low_load_flights']} unique low-load flights, repeated model sensitivity\nPrimary = {result['low_load']['primary_fold']}; exposure is not pooled across models")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    save(fig, "low_load_sensitivity")

    groups = [r for r in grouped if r["experiment"] == "cross-validate" and r["dimension"] == "observability"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [f"{r['group']}\napplicable n={r['applicable_scenarios']}" for r in groups]
    ax.bar(labels, [100 * r["incident_recall"] if r["incident_recall"] is not None else float("nan") for r in groups], color="#527a9c")
    ax.set_ylabel("Incident detected / applicable (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Frozen baseline by synthetic observability\nNOT_APPLICABLE excluded from recall; undefined remains missing")
    ax.grid(axis="y", alpha=.2)
    save(fig, "baseline_observability")
    return payloads


def research_report(summary, result, checks):
    def fmt(value, scale=1):
        return "undefined" if value is None else f"{value * scale:.6f}"

    baseline = result["baseline"]
    healthy = baseline["healthy_active_pooled"]
    fault = baseline["synthetic_pooled"]
    reg = baseline["regression_pooled"]
    pre, features = result["preprocessing"], result["features"]
    lines = ["# Consolidated research evaluation - Sprint 11", "",
             f"Run `{summary['run_id']}`. Scope: `{summary['evaluation_scope']}`. Status: `RESEARCH_CONSOLIDATION_ONLY`.", "",
             "## Main findings", "",
             f"Frozen HGB Battery-D/P99/0.5 s baseline: healthy active FPR {fmt(healthy['confirmed_fpr'], 100)}%, {healthy['confirmed_episodes']} episodes over {fmt(healthy['observed_ready_seconds'])} ready seconds ({fmt(healthy['confirmed_episodes_per_minute'])}/minute). Synthetic incidents detected {fault['detected_scenarios']}/{fault['applicable_scenarios']}, sample F1 {fmt(fault['f1'])}; detected-only median delay {fmt(fault['delay_median_seconds'])} s, n={fault['delay_detected_n']}.", "",
             f"Dual-path result remains **{summary['dual_promotion']}**. The orchestration does not select a winner from these outer results or change the operational configuration. Completion of this software/reporting stage does not imply that research targets, hardware readiness or the detector promotion gate passed.", "",
             "## Dataset, audit and causal processing", "",
             f"Raw {result['data_audit']['rows']} rows / {result['data_audit']['flights']} flights. Preprocessing retains {pre['retained_rows']}, drops {pre['dropped_rows']}, forms {pre['segments']} segments with {pre['gaps']} gaps. Current-based labels: {pre['active_load_flights']} ACTIVE_LOAD and {pre['low_load_or_idle_flights']} LOW_LOAD_OR_IDLE. Observed duration {fmt(pre['observed_duration_seconds'])} s; excluded gaps {fmt(pre['excluded_gap_seconds'])} s. Features: {features['ready_samples']} ready, {features['warmup_samples']} warm-up, {features['reference_unavailable_samples']} reference unavailable samples.", "",
             "Source status healthy is assumed_from_source. CurrTot remains unverified source units without conversion. Raw battery CSV contains only flight/time/instance/Volt/Curr/CurrTot; no throttle or V1 engine/model/threshold is used. Fixed causal warm-up locks V_ref or remains unavailable; actual-dt derivatives and trailing rolling features reset at gaps. Warm-up/readiness/IDs/actual voltage/sag/residual are excluded from regression predictors. Battery-E elapsed_time is ablation only.", "",
             "## Orchestration and source provenance", "",
             f"{summary['source_stages']} source stages, {summary['verified_source_files']} verified outputs and {summary['model_bundle_references']} saved-model references. Source run IDs, manifest/summary hashes, pre-outer freeze counts and model bundle hashes are in source_inventory.json/csv and model_bundle_registry.csv. Model weights and training/calibration provenance remain in their immutable source directories; the orchestrator does not copy or refit a historical model merely because current code changed. Current dependency versions and source configurations/versions remain separately traceable.", "",
             f"Stages built at creation: {summary['source_stages_built_at_creation']}; training stages built at creation: {summary['training_stages_built_at_creation']}. Existing results are verified before reuse. Missing stages use their own fixed engine budgets; corrupt/incompatible results fail without silent rebuild. A completed source catalog supports reproducible repetition without overwriting historical sprint pointers.", "",
             "The sequence is audit -> segmentation/reference/features -> pilot/fault bank -> reference LOAFO -> fixed ablation/comparison -> validation tuning/dual-path -> low-load -> consolidated grouping/plots/report. Within each modelling experiment, ALL candidate-fold policies/models and selection inputs are frozen before that experiment's first outer evaluation. These are sequential historical development experiments; this report is a retrospective consolidation, not proof of a globally untouched prospective final holdout. Outer outcomes never revise source candidates/policies in this stage.", "",
             "## Reference baseline", "",
             f"Regression pooled: MAE {fmt(reg['mae_v'])} V; RMSE {fmt(reg['rmse_v'])} V; R2 {fmt(reg['r2'])}. Healthy raw alarms {healthy['raw_alarm_samples']}/{healthy['ready_samples']}; confirmed {healthy['confirmed_alarm_samples']}/{healthy['ready_samples']}. Healthy FPR uses ready samples; episode rate uses actual monitored ready intervals, excluding warm-up/gaps and requiring both endpoints ready in one stream/segment.", "",
             f"Synthetic planned {fault['scenarios']}; applicable {fault['applicable_scenarios']}; NOT_APPLICABLE {fault['not_applicable_scenarios']}; undetected {fault['undetected_scenarios']}; preexisting-alarm contaminated {fault['preexisting_alarm_scenarios']}. Incident success requires a fresh confirmation inside fault support. A carried warning at onset is not an incident success. Deadline denominators retain every applicable miss/contamination; NOT_APPLICABLE has null metrics and is excluded. Conventional sample confusion/F1 is distinct from incident recall; repeated synthetic samples are not unique healthy exposure.", "",
             "| Deadline s | Successes | Applicable denominator | Rate % |", "|---:|---:|---:|---:|"]
    for d in (1, 2, 5, 10, 30):
        key = f"deadline_le_{d}s"
        lines.append(f"| {d} | {fault[key + '_successes']} | {fault['applicable_scenarios']} | {fmt(fault[key + '_rate'], 100)} |")
    lines += ["", "Research target checks are preserved from the original baseline:", ""]
    for name, c in baseline["research_target_checks"].items():
        lines.append(f"- {name}: {c['status']}; measured {fmt(c['rate'], 100)}%, target {fmt(c['target'], 100)}%, applicable n={c['applicable']}.")
    lines += ["", "## Fixed ablation and model comparison", "",
              "HGB/RF Battery-A/B/C/D/E and voltage-behavior IF/OCSVM use matched flight folds/readiness/fault definitions. HGB/RF score is positive sag residual in V; direct IF/OCSVM use signed -decision_function in model units. Only OCSVM has a healthy-training-fitted scaler and deterministic capped train sample. Exact ordered predictors are recorded per bundle. No automatic winner is selected; Battery-E remains ablation only.", "",
              "| Experiment | Candidate | Healthy FPR % | Episodes/min | Incident detected/applicable | Sample F1 | Regression RMSE V |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key, label in (("ablation", "ablation"), ("model_comparison", "comparison")):
        for name, m in result[key]["overview"].items():
            lines.append(f"| {label} | {name} | {fmt(m['healthy_confirmed_fpr'], 100)} | {fmt(m['healthy_episodes_per_minute'])} | {m['incident_detected']}/{m['applicable']} | {fmt(m['sample_f1'])} | {fmt(m['regression_rmse_v'])} |")
    lines += ["", "## Validation-only tuning and paired dual-path", "",
              "Threshold quantiles use only healthy assigned validation; synthetic validation is secondary refinement and never calibrates quantiles or enters model fitting. Per-fold selected policies, exact source run/freeze records and unchanged reference results are retained. Outer-selected rollbacks, winners or threshold increases are not performed.", "",
              "| Experiment / policy | Healthy FPR % | Episodes/min | Detected/applicable | F1 | Mean detected delay s | Median s | Detected n |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("tuning", "single-path tuning"), ("dual_path", "dual-path")):
        for policy in ("reference", "selected"):
            m = result[key][policy + "_pooled"]
            h, f = m["healthy_active"], m["synthetic"]
            lines.append(f"| {label} / {policy} | {fmt(h['confirmed_fpr'], 100)} | {fmt(h['confirmed_episodes_per_minute'])} | {f['detected_scenarios']}/{f['applicable_scenarios']} | {fmt(f['f1'])} | {fmt(f['delay_mean_seconds'])} | {fmt(f['delay_median_seconds'])} | {f['delay_detected_n']} |")
    lines += ["", "The dual-path promotion gate uses predeclared paired arithmetic macro means with all fold values defined, not the pooled values in the table. Different detected n makes detected-only delay comparisons conditional on different subsets; new late detections do not erase deadline failures.", "",
              "| Gate | Reference macro | Selected macro | Passed |", "|---|---:|---:|---|"]
    for name, c in result["promotion_decision"]["checks"].items():
        lines.append(f"| {name} | {fmt(c.get('reference'))} | {fmt(c.get('selected'))} | {c['passed']} |")
    lines += ["", "## Healthy low-load domain", "",
              f"{result['low_load']['unique_low_load_flights']} unique LOW_LOAD_OR_IDLE flights are explicitly unseen by the evaluating regressor and detector calibration. Low current does not prove ground operation. Primary fold {result['low_load']['primary_fold']} was fixed before prediction; other models are sensitivity on the same cohort, with no cross-model exposure pooling or winner selection.", "",
              "| Fold | Ready samples | Ready seconds | Raw alarms | Confirmed alarms | Confirmed FPR % | Episodes | Episodes/min |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in result["low_load_per_model"]:
        lines.append(f"| {m['fold_id']} | {int(m['ready_samples'])} | {fmt(m['observed_ready_seconds'])} | {int(m['raw_alarm_samples'])} | {int(m['confirmed_alarm_samples'])} | {fmt(m['confirmed_fpr'], 100)} | {int(m['confirmed_episodes'])} | {fmt(m['confirmed_episodes_per_minute'])} |")
    lines += ["", "The low-load source report preserves initial voltage, marginal input ranges, response/residual distributions, per-flight membership and actual false-alarm Curr/dI_dt/residual context. A primary no-alarm context is explicitly distinguished from a sensitivity false-alarm illustration added after results. Range overlap/exceedance is descriptive, not an extra alarm or a physical-cause diagnosis. Thresholds remain frozen; low-load has no synthetic recall/F1/delay claims.", "",
              "## Consolidated groups, failures and checks", "",
              f"synthetic_group_metrics.csv has {summary['grouped_metric_rows']} rows across {summary['synthetic_series']} separate experimental series, grouped by overall, flight, family, severity, onset position and observability. Each series total is recomputed from source scenario CSV and reconciled to its JSON summary. Do not pool alternative candidates/policies as independent incidents. failed_or_not_applicable_cases.csv retains {summary['failed_or_not_applicable_cases']} missed/contaminated/NA case records, including explicit reasons and null delays. Missing results are not fabricated zeros.", "",
              "| Check | Status | Detail |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['status']} | {c['detail']} |")
    lines += ["", "## Source runs", "", "| Stage | Run | Verified outputs | Model bundles | Frozen pre-outer files |", "|---|---|---:|---:|---:|"]
    for source in result["source_runs"]:
        lines.append(f"| {source['command']} | {source['run_id']} | {source['verified_files']} | {source['model_bundles']} | {source['pre_outer_frozen_files']} |")
    lines += ["", "## Scientific limits and remaining work", "",
              "All seventeen flights are historical development data and have been examined in prior research. Healthy labels are assumed, CurrTot units unverified, and original firmware/pack/BIN metadata and independently labelled real faults are absent. Synthetic applicability and observability are controlled experimental properties, not measured internal battery failures. No independent final holdout, deployment validation, emergency-readiness claim or physical internal-failure diagnosis follows from this package.", "",
              "Sprint 12 still needs streaming replay/equivalence and latency checks. Sprint 13 final audit is not completed by this report. Baseline software may be replayed without claiming emergency targets or promoting the failed dual-path candidate. No further candidate refinement is justified as independent final validation on these same outer flights.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)
