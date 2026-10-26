"""Factual validation selection and paired outer tuning diagnostics."""

import io


def fmt(value):
    return f"{value:.6f}" if value is not None else "undefined"


def tuning_report(result, selected, paired):
    summary = result["run_summary"]
    lines = ["# Sprint 8 — validation-only threshold/persistence tuning", "",
             "Scope: **historical development validation selection**. Outer flights were examined in prior sprints. This is not a new independent final holdout. No feature/model/hyperparameter selection is performed here: HGB Battery-D remains fixed.",
             f"Run `{summary['run_id']}`; {summary['folds']} folds, 20 predefined policies per fold ({summary['candidates_evaluated']} candidate evaluations).", "",
             "## Predeclared selection", "",
             "Grid: four configured percentiles × five configured persistence durations, recorded exactly in experiment_protocol.json and the grid CSV. Thresholds are linear percentiles of assigned healthy validation scores only. The same 27 fixed synthetic validation copies are evaluated for every policy, including policies rejected by healthy screening, so the budget does not adapt to results. Synthetic rows never calibrate quantiles or fit models.",
             "Reject a candidate if confirmed healthy-validation sample FPR OR episodes/minute exceeds the P99/0.5 s reference. An undefined healthy gate is ineligible. Rank eligible policies lexicographically: descending abrupt/severe increased-resistance or sudden-drop <=1 s rate; progressive MEDIUM/HIGH <=5 s rate; incident recall; sample F1; then ascending detected median delay, mean delay; final tie ascending percentile, persistence. Undefined rates/F1 rank below a defined zero; undefined delays are +infinity. Retain reference if no eligible policy. There is no weighted score or outer-based tie-break.",
             "All folds complete training, validation search, selected policy/model saving and freeze of validation inputs before ANY outer prediction. One fixed HGB-D is trained per fold on whole healthy-ready training flights including idle; seed 42, no scaler, internal early stopping off. Synthetic prediction reuse requires exact registered predictor/identity/readiness equality. Outer evaluation compares reference and selected policies on the same model/flight/scenarios; it never revises selection.", "",
             "## Selected validation policies", "",
             "| Fold | Test | Validation | Percentile | Persistence s | Threshold V | Eligible/20 | Validation FPR | Validation episodes/min | Reason |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for entry in selected:
        r = entry["selection"]
        lines.append(f"| {entry['fold_id']} | {entry['split']['test'][0]} | {entry['split']['validation'][0]} | {r['percentile']:g} | {r['persistence_seconds']:g} | {fmt(r['threshold_v'])} | {r['eligible_candidates']}/20 | {fmt(r['healthy_confirmed_fpr'])} | {fmt(r['healthy_episodes_per_minute'])} | {r['selection_reason']} |")
    lines += ["", "Threshold magnitudes are per-fold validation calibration, not universal battery limits. threshold_persistence_grid.csv retains all health gates/rejection reasons, ranking fields/denominators, eligible rank and selected flag. validation_fault_per_candidate.csv retains every candidate × scenario result/reason, including failed/contaminated/NA cases. selected_per_fold.json records selection rationale and full split/threshold provenance.", "",
              "## Paired outer results (diagnostic only)", "",
              "| Policy | RMSE V | Healthy confirmed FPR | Healthy episodes/min | Detected/applicable | Pre-existing | Sample F1 | Median detected delay s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, r in result["policies"].items():
        p = r["pooled"]
        h, s = p["healthy_active"], p["synthetic"]
        lines.append(f"| {policy} | {fmt(p['regression']['rmse_v'])} | {fmt(h['confirmed_fpr'])} | {fmt(h['confirmed_episodes_per_minute'])} | {s['detected_scenarios']}/{s['applicable_scenarios']} | {s['preexisting_alarm_scenarios']} | {fmt(s['f1'])} | {fmt(s['delay_median_seconds'])} (n={s['delay_detected_n']}) |")
    lines += ["", "| Deadline | Reference successes/applicable | Selected successes/applicable |", "|---|---:|---:|"]
    ref, tuned = (result["policies"][p]["pooled"]["synthetic"] for p in ("reference", "selected"))
    for d in (1, 2, 5, 10, 30):
        key = f"deadline_le_{d}s_successes"
        lines.append(f"| <= {d} s | {ref[key]}/{ref['applicable_scenarios']} | {tuned[key]}/{tuned['applicable_scenarios']} |")
    lines += ["", "Healthy-validation safety does not guarantee lower outer false alarms. Any outer deterioration remains visible and does not trigger reselection, rollback, candidate refinement or operational promotion. Raw sample FPR, confirmed sample FPR and episodes/minute are different quantities; the screening gate uses confirmed FPR plus confirmed episodes/minute. Healthy original flight exposure is counted once per policy; synthetic copies repeat source flights and are reported separately.", "",
              "## Denominators and interpretation", "",
              "Only reference-ready samples enter confusion counts. NOT_APPLICABLE has null metrics and no inference/recall/deadline eligibility. Applicable undetected and PREEXISTING_ALARM remain failures in every deadline denominator. A confirmed alarm carried into onset contaminates the whole incident even if a fresh episode appears later; conventional TP samples remain separate. A new confirmation exactly at onset may have zero delay. Post-support confirmation is not detection. Delay includes persistence and is detected-only with n; missing delay is never treated as zero. Coverage/gaps/reference/observability/support definitions are unchanged from the baseline.",
              "summary.csv/results.json separate pooled counts/exposure from macro per-fold mean/population std/median/min/max/n_defined. Both policies use exactly the same fitted regressor, so regression error is identical. paired_per_fold.csv gives selected-minus-reference deltas; per-fold and full failure tables preserve heterogeneity. Validation candidate metrics are selection diagnostics, not outer performance estimates.", "",
              "## Artifacts and limits", "",
              "Each fold saves one HGB-D bundle (compatible with the existing experimental bundle loader), selected/reference policies, threshold candidates and training provenance. Healthy validation and 27 applicable synthetic validation base-score exports carry reference decisions; all other grid decisions are reproducible from their scores/timestamps and the grid parameters. This avoids storing 20 duplicate predictor/prediction matrices. Synthetic validation used_for_threshold is false and used_for_policy_selection true; outer used_for_policy_selection is false. No synthetic sample contributes to quantile fingerprints.",
              "all_policies_frozen.json covers all pre-outer selection inputs/outputs and models; manifest.json verifies the full run before cache reuse. Outer reference/selected decision CSV.gz, root assignments/coverage/regression/healthy/anomaly/scenario/failed/paired/summary tables, results and plots are saved. Entire validation budget and failed candidate cases are reviewable, not only the selected point.",
              "Healthy labels are assumed; CurrTot remains unverified without conversion. No measured internal battery failure ground truth or new final holdout exists. Prior baselines/config/protocol remain intact. Status RESEARCH_TUNED_CANDIDATE_ONLY, no operational config change or deployment. Low-load evaluation, dual-path and replay remain later work. Any further outer-informed refinement must be called development/exploratory.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def tuning_plots(grid, healthy, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = sorted({r["fold_id"] for r in grid})
    fig, axes = plt.subplots(2, (len(folds) + 1) // 2, figsize=(15, 8), squeeze=False, constrained_layout=True)
    for ax, fold in zip(axes.flat, folds):
        rows = [r for r in grid if r["fold_id"] == fold]
        good, bad = [r for r in rows if r["healthy_eligible"]], [r for r in rows if not r["healthy_eligible"]]
        for subset, color, label in ((bad, "#c5c5c5", "rejected"), (good, "#3477a0", "eligible")):
            ax.scatter([r["healthy_confirmed_fpr"] * 100 for r in subset if r["healthy_confirmed_fpr"] is not None],
                       [r["healthy_episodes_per_minute"] for r in subset if r["healthy_confirmed_fpr"] is not None], c=color, label=label)
        ref = next(r for r in rows if r["percentile"] == 99 and r["persistence_seconds"] == .5)
        chosen = next(r for r in rows if r["selected"])
        for r, marker, color, label in ((ref, "x", "black", "reference"), (chosen, "*", "#d17a22", "selected")):
            if r["healthy_confirmed_fpr"] is not None:
                ax.scatter(r["healthy_confirmed_fpr"] * 100, r["healthy_episodes_per_minute"], marker=marker, s=110, c=color, label=label)
        ax.set_title(f"{fold}: P{chosen['percentile']:g}, {chosen['persistence_seconds']:g} s")
        ax.set_xlabel("Validation confirmed sample FPR (%)")
        ax.set_ylabel("Validation episodes/minute")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(folds):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Validation-only screening — fixed 20-policy budget per fold (overlapping points possible)")
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=140)
    plt.close(fig)
    images = {"validation_tradeoff.png": out.getvalue()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, field, scale, title in zip(axes, ("confirmed_fpr", "confirmed_episodes_per_minute"), (100, 1),
                                      ("Outer confirmed sample FPR (%)", "Outer confirmed episodes/minute")):
        x = list(range(len(folds)))
        for policy, shift, color in (("reference", -.2, "#3477a0"), ("selected", .2, "#d17a22")):
            values = [next(r[field] for r in healthy if r["fold_id"] == fold and r["policy"] == policy) for fold in folds]
            ax.bar([i + shift for i in x], [v * scale for v in values], width=.4, label=policy, color=color)
        ax.set_xticks(x, folds, rotation=30)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Paired outer diagnostics — selected policies stay frozen even if outer metrics worsen")
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=140)
    plt.close(fig)
    images["outer_false_alarm_tradeoff.png"] = out.getvalue()
    return images
