"""Software-only replay evidence, host latency and signal-delay figures."""

import io


def replay_plots(summary, faults, latency):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    payloads = {}

    def save(figure, name):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=140, metadata={"Software": "UAV Battery V2"})
        plt.close(figure)
        payloads["plots/" + name + ".png"] = buffer.getvalue()

    selected = [r for r in latency if r["phase"] == "end_to_end_ns"]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for key, offset, color in (("p50_ms", -.24, "#527a9c"), ("p95_ms", 0, "#3b8a70"), ("p99_ms", .24, "#c67b40")):
        axis.bar([i + offset for i in range(len(selected))], [r[key] for r in selected], width=.23, label=key.removesuffix("_ms"), color=color)
    axis.set_xticks(range(len(selected)), [r["group"].replace("/", "\n") for r in selected], rotation=15, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("In-process sample latency (ms, log scale)")
    budget = selected[0]["nominal_budget_ms"] if selected else 100.
    axis.axhline(budget, color="#b33d41", linestyle="--", label=f"Nominal interval ({budget:g} ms)")
    axis.set_title("Host replay latency: uncached inference separated from exact-input reuse\nParsing + causal features + prediction + diagnostics/detector; excludes disk and model loading")
    axis.legend(ncol=4)
    axis.grid(axis="y", alpha=.2)
    save(figure, "host_latency")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    delays = [r["delay_seconds"] for r in faults if r["incident_detected"]]
    axes[0].hist(delays, bins=20, color="#527a9c")
    axes[0].set_xlabel("Telemetry onset to new confirmed warning (s)")
    axes[0].set_ylabel("Detected incidents")
    axes[0].set_title(f"Detected-only signal delay, n={len(delays)}\nPersistence already included")
    metrics = summary["synthetic"]
    deadlines = (1, 2, 5, 10, 30)
    axes[1].bar([str(d) for d in deadlines], [100 * metrics[f"deadline_le_{d}s_rate"] if metrics[f"deadline_le_{d}s_rate"] is not None else float("nan") for d in deadlines], color="#3b8a70")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Confirmation deadline (s)")
    axes[1].set_ylabel("Success / all applicable incidents (%)")
    axes[1].set_title(f"Denominator={metrics['applicable_scenarios']}; misses retained\nReference software replay, not emergency promotion")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    save(figure, "signal_delay_and_deadlines")
    return payloads


def replay_report(summary, protocol, latency):
    def fmt(v):
        return "undefined" if v is None else f"{v:.6f}"

    f = summary["synthetic"]
    lines = ["# Sprint 12 - frozen baseline streaming replay", "",
             f"Run `{summary['run_id']}`; source reference `{summary['baseline_run_id']}`. Status **BASELINE_SOFTWARE_REPLAY_ONLY**.", "",
             "## Scope and frozen inputs", "",
             "This explicitly invoked command validates software despite replay.enabled=false in the frozen deployment configuration; it does not enable a device or change that configuration. The operational baseline stays HGB Battery-D, healthy-validation P99, 0.5 s timestamp persistence. The dual-path candidate remains NOT_PROMOTED and is not replayed as an emergency candidate. No fitting, calibration, threshold increase or policy selection occurs.", "",
             "When Sprint 11 exists, replay uses its pinned baseline and low-load source set after manifest/lineage verification. Otherwise explicit completed baseline/low-load results are required, without implicit training. Original raw/config/protocol/features, saved weights, validation thresholds and batch score exports are hashed in replay_protocol.json before replay begins.", "",
             "## Incremental state and equivalence", "",
             f"{summary['replay_jobs']} replay jobs / {summary['replayed_rows']} samples; **{summary['batch_stream_equivalence']}** across {summary['comparison_columns']} feature/diagnostic/decision columns. Every active flight uses its held-out fold; eleven low-load flights use the already frozen primary fold. Each synthetic case starts a fresh stream, reference warm-up, rolling history, residual slope and persistence timer. NOT_APPLICABLE is skipped, never inferred or assigned fabricated metrics.", "",
             "The streaming API accepts only raw battery fields plus optional source_record. It parses numeric fields causally, drops invalid numerics without imputation/state advance, rejects duplicate/backward timestamps and unselected multi-instance flights, and isolates interleaved flights. Future length, activity labels, processed features, synthetic support masks and fault metadata are not inputs. Fixed warm-up locks V_ref or stays unavailable, including truncated warm-up. Gaps reset rolling/derivative/slope/persistence, retain locked reference and preserve the original telemetry clock. States remain NOT_READY/NORMAL/SUSPECT/WARNING.", "",
             "Healthy feature outputs are compared with the frozen batch feature artifact. Synthetic voltage-derived sag/derivative are reconstructed from frozen injected voltage and actual dt, while unchanged input predictors remain matched to their source rows. One-sample HGB predictions, positive residual and exact decisions/timers match saved batch scores; signed residual OLS/delta match batch causal diagnostics. Numeric tolerance is absolute/relative 1e-10; categorical/Boolean/null states match exactly. No mismatching replay is marked complete.", "",
             "All original healthy rows perform uncached single-sample prediction and populate a model-specific exact ordered-input cache. Synthetic copies may reuse only an identical model object and exact current predictor tuple already observed in earlier healthy replay. Only prediction values are cached; feature/reference/residual/slope/timer/decision state is always recomputed. This cannot provide future voltage or alarm decisions. Uncached and cache latencies are reported separately; cached synthetic speed is not a claim about uncached inference.", "",
             "## Monotonic host timing", "",
             "time.perf_counter_ns measures parsing, causal feature processing, prediction, diagnostics/detector and their total in-process latency per input sample. The total equals the sum of phases. Disk reading/writing, model loading and CSV serialization are excluded; first model calls are retained, not silently warmed away. This is accelerated timestamp-driven replay without pacing/sleep. Nominal 10 Hz provides a 100 ms reporting interval, not an imposed clock or measured hardware deadline.", "",
             "| Sample group | Phase | Samples | p50 ms | p95 ms | p99 ms | Max ms | Over nominal interval |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in latency:
        lines.append(f"| {r['group']} | {r['phase']} | {r['samples']} | {fmt(r['p50_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['p99_ms'])} | {fmt(r['max_ms'])} | {r['over_nominal_budget_samples']} |")
    lines += ["", "Host latency depends on this environment and load; percentile values are measurements from this run, not cross-machine reproducibility guarantees. Repeating an unchanged completed command verifies/reuses the recorded latency artifact instead of silently replacing timing measurements. A deliberately new code/protocol version creates a new run.", "",
              "## Healthy metrics and signal warning delay", "",
              "Healthy FPR/exposure/episode counts remain identical to the frozen sources. Ready exposure uses actual intervals with both endpoints ready in one segment and excludes warm-up/gaps. No repeated-model exposure is pooled.", "",
              "| Cohort | Ready samples | Confirmed alarms | FPR | Episodes | Ready s | Episodes/min |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("healthy_active", "healthy_low_load"):
        h = summary[key]
        lines.append(f"| {key} | {h['ready_samples']} | {h['confirmed_alarm_samples']} | {fmt(h['confirmed_fpr'])} | {h['confirmed_episodes']} | {fmt(h['observed_ready_seconds'])} | {fmt(h['confirmed_episodes_per_minute'])} |")
    lines += ["", f"Synthetic: {f['scenarios']} planned, {f['applicable_scenarios']} applicable, {f['not_applicable_scenarios']} NOT_APPLICABLE, {f['detected_scenarios']} detected, {f['undetected_scenarios']} undetected and {f['preexisting_alarm_scenarios']} contaminated. Detected-only mean {fmt(f['delay_mean_seconds'])} s, median {fmt(f['delay_median_seconds'])} s, n={f['delay_detected_n']}. Incident/sample counts and all deadlines match the frozen batch reference.", "",
              "Signal end-to-end warning delay is actual telemetry onset to the first fresh confirmed sample inside fault support. Persistence is already inside this delay and must not be added again. Carried confirmed alarms are PREEXISTING_ALARM, not zero-delay successes. Every applicable miss/contamination stays in deadline denominators; NA does not. Per-case exports separately record processing latency at the confirmation sample and raw-run start/persistence elapsed; the accelerated host execution clock is not substituted for telemetry signal time or added to it as a hardware latency claim.", "",
              "## Limits and remaining work", "",
              "Original healthy labels are assumed_from_source, CurrTot units remain unverified and all flights remain historical development, not a fresh independent holdout. Failure of the baseline progressive research target and NOT_PROMOTED dual-path are preserved. This stage establishes software equivalence and host processing measurements; it does not establish emergency switching, real flight fault ground truth, hardware readiness or operational promotion. Paced replay is optional and was not performed. Sprint 13 final audit remains outstanding.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)
