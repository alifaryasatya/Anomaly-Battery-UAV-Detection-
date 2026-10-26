"""Descriptive figures and factual healthy-assumed low-load report."""

from collections import defaultdict
import io


def plots(by_flight, records, profiles, primary, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = {}

    def save(figure, name):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=140, metadata={"Software": "UAV Battery V2"})
        plt.close(figure)
        payloads["plots/" + name + ".png"] = buffer.getvalue()

    example = None
    for flag in ("anomaly_confirmed", "anomaly_raw", "reference_ready"):
        example = next((f for f in sorted(by_flight) if any(r[flag] for r in by_flight[f])), None)
        if example:
            break
    if example:
        rows = by_flight[example]
        blocks = defaultdict(list)
        for row in rows:
            if row["reference_ready"]:
                blocks[row["segment_id"]].append(row)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for i, block in enumerate(blocks.values()):
            x = [r["elapsed_time"] for r in block]
            for key, label, color in (("voltage_sag_actual", "Actual sag", "#2457a6"),
                                      ("predicted_voltage_sag", "Predicted sag", "#c3761c")):
                axes[0].plot(x, [r[key] for r in block], color=color, label=label if i == 0 else None)
            axes[1].plot(x, [r["residual_signed_v"] for r in block], color="#237753", label="Signed residual" if i == 0 else None)
            axes[2].plot(x, [r["Curr"] for r in block], color="#6651a5")
            axes[3].plot(x, [r["dI_dt"] for r in block], color="#547a80")
        confirmed = [r for r in rows if r["anomaly_confirmed"]]
        axes[1].axhline(threshold, color="#bc303c", linestyle="--", label="Frozen validation P99")
        if confirmed:
            axes[1].scatter([r["elapsed_time"] for r in confirmed], [r["residual_signed_v"] for r in confirmed],
                            color="#bc303c", s=10, label="Confirmed false alarm", zorder=3)
        status = "Confirmed false-alarm example" if confirmed else "Raw alarm only" if any(r["anomaly_raw"] for r in rows) else "No alarm: ready-sample context"
        axes[0].set_title(f"Healthy-assumed LOW_LOAD_OR_IDLE: {example}, {primary}\n{status}; descriptive context only")
        for ax, label in zip(axes, ("Voltage sag (V)", "Residual (V)", "Current (A)", "dI/dt (A/s)")):
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        axes[-1].set_xlabel("Telemetry elapsed time (s); gaps and warm-up excluded from exposure")
        save(fig, "example_false_alarm_context")

    flights = sorted(by_flight)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    x = list(range(len(flights)))
    for fold in sorted({r["fold_id"] for r in records}):
        selected = [next(r for r in records if r["fold_id"] == fold and r["NamaFile"] == f) for f in flights]
        for ax, key, scale in ((axes[0], "confirmed_fpr", 100), (axes[1], "confirmed_episodes_per_minute", 1)):
            ax.plot(x, [r[key] * scale if r[key] is not None else float("nan") for r in selected],
                    "o-", linewidth=2 if fold == primary else 1, alpha=1 if fold == primary else .55,
                    label=fold + (" (primary)" if fold == primary else ""))
    axes[0].set_title(f"Same {len(flights)} low-load flight(s) under frozen reference models; sensitivity only")
    axes[0].set_ylabel("Confirmed false-alarm samples (%)")
    axes[1].set_ylabel("Confirmed episodes / ready minute")
    axes[1].set_xticks(x, [f.removesuffix(".BIN") for f in flights], rotation=35)
    axes[0].legend(ncol=3)
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "per_flight_false_alarms")

    voltage = [r for r in profiles if r["feature"] == "Volt"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for i, row in enumerate(voltage):
        color = "#b16035" if row["v2_current_based_label"] == "LOW_LOAD_OR_IDLE" else "#2457a6"
        if row["min"] is not None:
            axes[0].vlines(i, row["min"], row["max"], color=color, linewidth=3)
            axes[0].scatter(i, row["initial_voltage_v"], color=color, marker="x", s=45)
        sag = next(r for r in profiles if r["NamaFile"] == row["NamaFile"] and r["feature"] == "voltage_sag_actual")
        if sag["min"] is not None:
            axes[1].vlines(i, sag["min"], sag["max"], color=color, alpha=.5)
            axes[1].vlines(i, sag["p05"], sag["p95"], color=color, linewidth=5)
            axes[1].scatter(i, sag["median"], color="black", s=12)
    axes[0].set_title("Flight domain context: blue ACTIVE_LOAD, brown LOW_LOAD_OR_IDLE\nVoltage range with initial-voltage cross; sag range, P05-P95 and median")
    axes[0].set_ylabel("Accepted voltage (V)")
    axes[1].set_ylabel("Reference-ready actual sag (V)")
    axes[1].set_xticks(range(len(voltage)), [r["NamaFile"].removesuffix(".BIN") for r in voltage], rotation=45)
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "voltage_response_domains")
    return payloads, example


def report(summary, records, folds, profiles):
    p = summary["primary_low_load"]

    def number(value, scale=1):
        return "undefined" if value is None else f"{value * scale:.6f}"

    lines = ["# Sprint 10 - low-load false alarms and domain diagnostics", "",
             f"Run `{summary['run_id']}`; frozen Sprint 6 reference `{summary['baseline_run_id']}`.", "",
             "Historical development evaluation, healthy status assumed_from_source. LOW_LOAD_OR_IDLE is a current-based label; low current is not proof of being on the ground. No measured fault labels or independent final holdout are available.", "",
             "## Frozen evaluation design", "",
             f"Primary model `{summary['primary_fold']}` is the first sorted frozen reference fold, fixed before low-load prediction. All {summary['frozen_models']} frozen HGB Battery-D models are evaluated separately as sensitivity analyses on the same {summary['unique_low_load_flights']} unique flights. No best-fold selection, ensemble, additional training, threshold calibration, synthetic evaluation or low-load-driven policy selection occurs. Sprint 9 NOT_PROMOTED remains in force; operational dual-path stays disabled.", "",
             "Models and per-fold thresholds are loaded from checksum-verified Sprint 6 bundles. Each threshold remains the healthy-validation P99 with 0.5 s persistence. protocol.json and pre_evaluation_freeze.json record all source hashes/policies and training_domain_ranges.json before prediction.", "",
             f"All low-load flights unseen by regressor: **{summary['all_low_load_unseen_by_model']}**; unseen by detector fitting/calibration: **{summary['all_low_load_unseen_by_detector']}**. Per-flight source membership and unseen flags are explicit. Training-flight alarms must not be called unseen FPR. These historical flights have been examined in prior research, so model-unseen does not imply a new independent final holdout.", "",
             "## Primary healthy-assumed results", "",
             f"Ready samples {p['ready_samples']}; raw alarms {p['raw_alarm_samples']} ({number(p['raw_fpr'], 100)}%); confirmed alarms {p['confirmed_alarm_samples']} ({number(p['confirmed_fpr'], 100)}%). Confirmed episodes {p['confirmed_episodes']} across {p['unique_flights']} unique flights; observed ready duration {number(p['observed_ready_seconds'])} s; {number(p['confirmed_episodes_per_minute'])} episodes/minute.", "",
             "FPR denominator is reference-ready healthy-assumed samples, not flight count. Monitored duration sums actual intervals with both endpoints ready in the same stream/segment, excluding warm-up, unavailable reference and gaps. A fresh warning after a gap is a new episode. Undefined rates remain null/blank. No synthetic recall, F1, detection delay or fault-diagnosis claim is made for this cohort.", "",
             "| Flight | Unseen model / detector | Ready | Raw | Confirmed | Confirmed FPR % | Episodes | Ready seconds | Episodes/min |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["primary"]:
            lines.append(f"| {r['NamaFile']} | {r['unseen_by_model']} / {r['unseen_by_detector']} | {r['ready_samples']} | {r['raw_alarm_samples']} | {r['confirmed_alarm_samples']} | {number(r['confirmed_fpr'], 100)} | {r['confirmed_episodes']} | {number(r['observed_ready_seconds'])} | {number(r['confirmed_episodes_per_minute'])} |")
    lines += ["", "## Model sensitivity, repeated cohort", "",
              "Each row below evaluates the same cohort. Do not add their exposures, samples or flight counts as independent observations; this is not a confidence interval.", "",
              "| Frozen fold | Raw FPR % | Confirmed FPR % | Episodes | Ready seconds | Episodes/min |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in folds:
        lines.append(f"| {r['fold_id']} | {number(r['raw_fpr'], 100)} | {number(r['confirmed_fpr'], 100)} | {r['confirmed_episodes']} | {number(r['observed_ready_seconds'])} | {number(r['confirmed_episodes_per_minute'])} |")
    voltages = [r for r in profiles if r["feature"] == "Volt"]
    initial = {}
    for label in ("ACTIVE_LOAD", "LOW_LOAD_OR_IDLE"):
        values = [r["initial_voltage_v"] for r in voltages if r["v2_current_based_label"] == label and r["initial_voltage_v"] is not None]
        initial[label] = f"{min(values):.6f}-{max(values):.6f} V" if values else "undefined"
    primary_records = [r for r in records if r["primary"]]
    outside_n = sum(r["outside_any_training_range_samples"] for r in primary_records)
    ready_n = sum(r["ready_samples"] for r in primary_records)
    extra = summary["sensitivity_false_alarm_example"]
    lines += ["", "## Domain diagnostics and limits", "",
              f"Initial accepted voltage range: ACTIVE_LOAD {initial['ACTIVE_LOAD']}; LOW_LOAD_OR_IDLE {initial['LOW_LOAD_OR_IDLE']}. For the primary model, {outside_n}/{ready_n} ready low-load rows lie outside at least one original training marginal predictor range. These comparisons do not establish a cause for alarms or their absence.", "",
              "flight_domain_profiles.csv reports every flight's initial accepted voltage, locked V_ref, raw voltage range and ready predictor/actual-sag distributions (n, mean, population std, median, min/max, P05/P95). Initial voltage is descriptive and is not an additional regression feature. response_distributions.csv reports predicted sag, signed residual and positive score per model/low-load flight.", "",
              "outside_training_ranges.csv compares each ready low-load predictor with that model's original healthy-ready training min/max; per_model_flight.csv also counts rows outside any marginal range. Training row identities/targets/predictors match the frozen training fingerprint. Range overlap does not establish a matching joint distribution or physiological health; a range exceedance is descriptive, not a new alarm or physical fault label. CurrTot remains in unverified source units without conversion.", "",
              f"Primary context plot: `{summary['example_flight']}`, status `{summary['example_status']}`, selected by the predeclared first-confirmed/first-raw/first-ready rule. A no-alarm plot is not an example of a false alarm. Lines break at gaps. Curr, dI/dt, actual/predicted voltage sag and signed residual provide numerical context only. episode_context.csv lists every confirmed episode and its current/derivative/residual ranges; no physical cause is inferred.", "",
              (f"Additional actual false-alarm plot: `{extra['NamaFile']}` / `{extra['fold_id']}` in plots/sensitivity_false_alarm_context.png. After the initial diagnostic run showed no primary alarm, this illustration was added using the first sorted sensitivity fold/flight with a confirmed alarm. This is a descriptive example selection after results, not a model/policy selection; primary allocation, all thresholds and reported cohort stay unchanged." if extra else "No sensitivity-model confirmed false alarm exists to illustrate."), "",
              "Observed alarms or input/response differences do not justify raising thresholds to hide domain mismatch. Collect representative independently labelled healthy operating conditions and battery/firmware/pack metadata before a new versioned development/validation study. This stage does not establish hardware or emergency-response readiness. Operational config and frozen earlier artifacts remain unchanged.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    return "\n".join(lines)
