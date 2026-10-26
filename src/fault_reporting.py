"""Generation-only scenario report and voltage traces; no detector metrics."""

from collections import defaultdict
import io


def fault_report(summary: dict, counts: list[dict]) -> str:
    lines = ["# Sprint 5 — synthetic fault generation", "",
             f"Run `{summary['run_id']}`. {summary['flights']} flights; {summary['scenarios']} planned scenarios, {summary['applicable_scenarios']} applicable, {summary['not_applicable_scenarios']} NOT_APPLICABLE.",
             "This is an applicability/observability audit, not detector evaluation. No model is trained or called; pilot model/threshold are unchanged.", "",
             "Current-dependent families select the longest valid active run (tie earliest timestamp). The aligned onset must already be reference-ready; no silent shift or retry on a different run. Sudden drop selects the longest continuous ready segment and can apply on low-load flights. Remaining support must span at least 1 s. Requested onset is fractional timestamp (Decimal); actual onset is the first sample at/after it.",
             "Resistance adds Curr*delta_R. Progressive adds Curr*delta_R_max*clipped((t-onset)/(end-onset)). Sudden drop adds configured voltage times a timestamp-based ramp (default 0.4 s). Support includes the aligned onset even if progressive/ramp effect is initially zero; first_effect_time_us records the first nonzero change separately. No row-index timing or throttle is used.",
             "Only copies are changed. V_ref/readiness and all regression predictors remain copied from healthy features. Volt is lowered by added sag; target is recomputed from the preserved reference and modified Volt. dV_dt is recomputed, including return to original telemetry after support ends, and resets at gaps. Curr/TimeUS/Inst/CurrTot are never changed.",
             "Fault end is the last supported sample, inclusive; next samples use original telemetry with no synthetic recovery ramp. Row fault_support/fault_label identify support, fault_effect_present identifies nonzero perturbation, and fault_phase distinguishes PRE_FAULT/FAULT_SUPPORT/POST_FAULT_ORIGINAL. Every copied row has is_synthetic and scenario_id so fitting/calibration guards reject the whole scenario, including pre-fault rows.", "",
             "## Applicability and observability", "", "| Family | Severity | Planned | Applicable | NOT_APPLICABLE | VERY_LOW | LOW | MEDIUM | HIGH |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in counts:
        lines.append(f"| {r['fault_family']} | {r['severity']} | {r['scenarios']} | {r['applicable']} | {r['not_applicable']} | {r['VERY_LOW']} | {r['LOW']} | {r['MEDIUM']} | {r['HIGH']} |")
    lines += ["", "Observability uses actual maximum added sag on support: <0.05 VERY_LOW; [0.05,0.20) LOW; [0.20,0.50) MEDIUM; >=0.50 V HIGH (config values). It describes synthetic signal magnitude, not a battery diagnosis. NOT_APPLICABLE stays separate with reason and null signal statistics; it does not enter confusion/recall denominators. Applicable evaluation_status is NOT_EVALUATED until Stage 6.",
              f"NOT_APPLICABLE reasons: {summary['not_applicable_reasons']}.", "",
              "## Files and scope", "", "fault_manifest.csv records every planned scenario, parameters, selection/onset/end/reference/current/support/ending metadata and gzip CSV path/hash. Applicability and observability tables have one row for every planned scenario. Applicable scenarios are full-flight CSV.gz copies in data/synthetic/<run_id>/, compressed with deterministic mtime=0. NOT_APPLICABLE has no scenario CSV; its diagnostics remain in all tables. All output hashes are in manifest.json.",
              "Examples use the second sorted eligible active flight (pilot validation when the default six flights exist) and first sorted ready low-load flight, severe severity and median configured start position. Example voltage traces show injected telemetry only, with no predictions, thresholds or detection claims.",
              "The scenario bank spans source flights for later fold-specific use; it is never appended to healthy fitting data. Reserved-test scenarios may be prepared here but no regressor/detector is run on them. No recall/F1/delay/deadline or final holdout result is claimed. Original healthy labels remain assumed and CurrTot unit unverified.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)


def fault_examples(scenarios) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(11, 2.9 * len(scenarios)), constrained_layout=True, squeeze=False)
    for axis, scenario in zip(axes[:, 0], scenarios):
        meta = scenario.metadata
        start = scenario.rows[0]["TimeUS"]
        blocks = defaultdict(list)
        for row in scenario.rows:
            blocks[row["segment_id"]].append(row)
        for index, block in enumerate(blocks.values()):
            time = [(r["TimeUS"] - start) / 1e6 for r in block]
            axis.plot(time, [r["Volt_healthy"] for r in block], color="#255c9e", label="Original voltage" if index == 0 else None)
            axis.plot(time, [r["Volt"] for r in block], color="#c3433d", label="Synthetic voltage" if index == 0 else None)
        onset, end = (meta["onset_time_us"] - start) / 1e6, (meta["fault_end_time_us"] - start) / 1e6
        axis.axvspan(onset, end, color="#e5b041", alpha=.15, label="Fault support")
        axis.axvline(onset, color="#555555", linestyle="--", label="Aligned onset")
        axis.set_title(f"{meta['NamaFile']} — {meta['fault_family']}, {meta['severity']}, position {meta['start_position']:.0%}")
        axis.set_ylabel("Pack voltage (V)")
        axis.set_xlabel("Telemetry elapsed time (s); gaps shown as breaks")
        axis.grid(alpha=.2)
        axis.legend(loc="lower left", ncols=2, fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, metadata={"Software": "UAV Battery V2"})
    plt.close(figure)
    return buffer.getvalue()
