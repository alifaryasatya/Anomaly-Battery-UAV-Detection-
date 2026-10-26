"""Factual residual/dual-path reporting with explicit research promotion gate."""

import io

from .residual_diagnostics import slope_columns


def fmt(v):
    return f"{v:.6f}" if v is not None else "undefined"


def promotion_report(decision):
    lines = ["# Dual-path promotion decision", "", f"Status: **{decision['status']}**", "",
             "Gate aggregation is predeclared: arithmetic macro means across identical paired folds. Every fold must define each metric. Pooled metrics remain separate diagnostics; the gate never switches aggregation after results.", "",
             "| Check | Reference macro mean | Selected macro mean | Rule | Defined all folds | Pass |", "|---|---:|---:|---|---|---|"]
    for name, c in decision["checks"].items():
        if name == "refined_dual_candidate_exists":
            lines.append(f"| {name} | N/A | {c['dual_folds']} dual folds | at least one | True | {c['passed']} |")
        else:
            lines.append(f"| {name} | {fmt(c['reference'])} | {fmt(c['selected'])} | {c['mode']} | {c['defined_all_folds']} | {c['passed']} |")
    lines += ["", "No outer reselection or denominator change is permitted. NOT_PROMOTED retains the fixed single-path operational baseline. Even a research gate pass does not enable dual-path in config or establish hardware/emergency readiness. Operational config and dual_path.enabled remain unchanged.", ""]
    return "\n".join(lines)


def dual_report(result, selections, contexts):
    s = result["run_summary"]
    lines = ["# Sprint 9 — causal residual diagnostics and dual-path", "",
             "Scope: **historical development dual-path selection**. Historical outer flights were inspected in prior sprints; this is not an independent final holdout. Residual features diagnose electrical response patterns, not internal battery failure mechanisms.",
             f"Run `{s['run_id']}`; {s['folds']} folds, {s['candidate_pairs_evaluated']} healthy candidate-pair evaluations, {s['shortlisted_refined_candidates']} shortlisted pairs, {s['validation_candidate_scenario_evaluations']} synthetic validation case evaluations.", "",
             "## Frozen procedure", "",
             "Reference is fixed HGB Battery-D P99/0.5 s. The unpromoted Sprint 8 candidate is not silently substituted. Same healthy-ready whole-flight training, seed/config, no scaler/synthetic fitting and no random-row early stopping. Signed residual deltas use actual dt. OLS slopes use local timestamp origin in trailing (t-window,t], at least two unique timestamps; windows reset on gap/segment/unready. Diagnostics never enter regression predictors.",
             "Path A high percentile [99,99.5], persistence 0.5 s. Select A from healthy validation alone by ascending FPR, episodes/minute, high percentile. Audit the full 144-pair grid (2 high ×3 low ×4 slope percentile ×3 slope windows ×2 B persistence); only pairs using chosen A may enter shortlist. Slope thresholds use ready assigned healthy validation slopes only, linear percentiles. Missing slope calibration is ineligible.",
             "Healthy union FPR AND episodes/minute must not exceed fixed reference. Shortlist at most three pairs sorted by FPR, episodes/minute, ascending full parameter tuple. Only shortlisted pairs undergo all 27 fixed synthetic validation cases; at least one applicable case is needed to complete refinement. No unrefined pair can win. Rank refined pairs by frozen severe-abrupt deadline, observable-progressive deadline, incident recall, F1, detected median/mean delay and ascending full parameter tuple. Undefined rates are below defined zero; missing delay is +infinity. If no safe refined dual candidate, use fixed single-path reference.",
             "All fold models, thresholds, selected/fallback policies and validation inputs are frozen before ANY outer prediction. No outer score changes selection. Timers A/B are independent; confirmed warning is OR. Alternating raw paths cannot accumulate a shared persistence timer. Union episode starts only on union false->true or gap; switching confirmed attribution does not start a new union episode. anomaly_duration_seconds is max(A,B raw elapsed), a diagnostic, not a combined-raw timer.", "",
             "## Selected/fallback policies", "",
             "| Fold | Test | Validation | Mode | High percentile | Low percentile | Slope percentile | Window s | B persistence s | Reason |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for e in selections:
        p = e["selected_policy"]
        lines.append(f"| {e['fold_id']} | {e['split']['test'][0]} | {e['split']['validation'][0]} | {p['mode']} | {p['high_percentile']} | {p['low_percentile']} | {p['slope_percentile']} | {p['slope_window_seconds']} | {p['path_b_persistence_seconds']} | {e['selection_reason']} |")
    lines += ["", "## Paired outer pooled diagnostics", "",
              "| Policy | Healthy FPR | Healthy episodes/min | Incident detected/applicable | Pre-existing | Sample F1 | Detected mean delay s | Detected median delay s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, r in result["policies"].items():
        h, a = r["pooled"]["healthy_active"], r["pooled"]["synthetic"]
        lines.append(f"| {policy} | {fmt(h['confirmed_fpr'])} | {fmt(h['confirmed_episodes_per_minute'])} | {a['detected_scenarios']}/{a['applicable_scenarios']} | {a['preexisting_alarm_scenarios']} | {fmt(a['f1'])} | {fmt(a['delay_mean_seconds'])} (n={a['delay_detected_n']}) | {fmt(a['delay_median_seconds'])} |")
    lines += ["", "Promotion status: **" + result["promotion"]["status"] + "**. See promotion_decision.md/json for every paired macro gate, defined denominator and pass/fail. Raw/confirmed healthy sample counts, observed exposure, conventional synthetic confusion and all deadline numerators/denominators are in CSV/JSON; summary separates macro distributions from pooled metrics. Neither aggregate nor parameter is changed after inspection.", "",
              "## Attribution and baseline transient context", "",
              "Exports include A/B raw flags, individual durations/confirmations/episode starts, sample attribution A_ONLY/B_ONLY/BOTH/NONE, and first_confirmation_path on new union starts. Scenario incident_first_path is only defined for an uncontaminated new confirmation within fault support; carried warnings never receive incident credit. A/B overlap samples count once in union confusion and union episodes.",
              f"baseline_validation_episode_context.csv records {len(contexts)} healthy-assumed validation baseline warning episodes with numeric current range, max abs dI_dt and residual range. This is development context, with no threshold-driven transient label, causal explanation or physical fault diagnosis. No current-context variable changes detector decisions. Fixed first-fold validation and severe progressive middle-position outer traces show residual, slope, paths and current; they are diagnostic examples, not selected for favorable performance.", "",
              "Only ready samples enter confusion. NOT_APPLICABLE has no inference/count/deadline eligibility, including path counts null. Applicable undetected and carried PREEXISTING_ALARM remain failures in every deadline denominator. Delay is detected-only with n and includes persistence; post-support confirmation does not count; a new confirmation exactly at onset can have zero delay. Source ready coverage, faults, observability, support and gap boundaries match the reference. Source flights repeat across synthetic cases; they are not independent samples. Failed cases remain fully visible.", "",
              "## Reproducibility and limits", "",
              "Per fold: HGB bundle including dual policy and training provenance, healthy residual/slope quantile source fingerprints, A selection, chosen/fallback policy, healthy/synthetic validation diagnostic base exports with reference decisions. Full healthy grid includes excluded A values/reasons; synthetic refinement table lists only shortlisted cases. All pre-outer files are hash-frozen. Outer reference/selected exports contain diagnostics and attribution. Root includes coverage/assignments, regression/healthy/anomaly/scenario/failed/paired/summary tables, results, gate decision, report/plots and complete manifest. tune-dual-path performs this frozen experiment; compare-dual-path verifies/reuses the same run rather than refitting a second model family.",
              "Config dual_path.enabled remains false. Any research promotion is not operational activation, independent final performance or hardware switching readiness. Healthy labels are assumed, CurrTot remains unverified without conversion, real internal fault ground truth and pack/firmware metadata are absent. Low-load evaluation and replay are later sprints. Further outer-informed refinements remain development/exploratory.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def dual_plots(healthy, selections, validation_example, example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = {}

    def save(fig, name):
        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=140)
        plt.close(fig)
        images[name] = out.getvalue()

    folds = [r["fold_id"] for r in selections]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, key, factor, title in zip(axes, ("confirmed_fpr", "confirmed_episodes_per_minute"), (100, 1),
                                     ("Outer confirmed sample FPR (%)", "Outer confirmed episodes/minute")):
        for p, shift, color in (("reference", -.2, "#3477a0"), ("selected", .2, "#d17a22")):
            values = [next(r[key] for r in healthy if r["fold_id"] == f and r["policy"] == p) for f in folds]
            ax.bar([i + shift for i in range(len(folds))], [v * factor for v in values], width=.4, color=color, label=p)
        ax.set_xticks(range(len(folds)), folds, rotation=30)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("Frozen dual-path versus fixed single-path — historical development")
    save(fig, "paired_false_alarms.png")
    for name, data in (("validation_residual_context.png", validation_example), ("outer_progressive_example.png", example)):
        if data is None:
            continue
        rows, policy, reference = data
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
        x = [r["elapsed_time"] for r in rows]

        def series(key):
            values, before = [], None
            for r in rows:
                if before is not None and r["segment_id"] != before:
                    values.append(float("nan"))
                else:
                    values.append(r[key] if r[key] is not None else float("nan"))
                before = r["segment_id"]
            return values

        axes[0].plot(x, series("residual_signed_v"), label="Signed residual V", lw=.8)
        axes[0].axhline(reference["threshold_high_v"], color="black", ls="--", label="Reference P99")
        axes[0].axhline(policy["threshold_high_v"], color="#a33", ls=":", label="Selected A")
        axes[0].set_ylabel("Residual (V)")
        w = policy["slope_window_seconds"] if policy["mode"] == "DUAL" else .5
        column = slope_columns(w)[0]
        axes[1].plot(x, series(column), label=f"Causal OLS slope {w:g} s", lw=.8)
        if policy["mode"] == "DUAL":
            axes[1].axhline(policy["threshold_slope_v_per_s"], color="#a33", ls="--", label="Healthy validation slope threshold")
        axes[1].set_ylabel("Slope (V/s)")
        axes[2].step(x, [int(r["path_a_confirmed"]) for r in rows], where="post", label="A confirmed")
        axes[2].step(x, [int(r["path_b_confirmed"]) * .8 for r in rows], where="post", label="B confirmed (0.8 level)")
        axes[2].set_ylabel("Path warning")
        axes[3].plot(x, series("Curr"), label="Current A", lw=.8)
        axes[3].set_ylabel("Current (A)")
        axes[3].set_xlabel("Telemetry elapsed time (s); gaps break residual/slope/current")
        for ax in axes:
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f"{rows[0]['split_role']} / {rows[0]['NamaFile']} / {policy['mode']} — fixed diagnostic example")
        save(fig, name)
    return images
