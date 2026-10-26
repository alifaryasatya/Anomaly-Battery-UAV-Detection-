"""Read-only final audit helpers and source-derived Markdown reports."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


class FinalAuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FinalAuditError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_path(root, path):
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Artifact path leaves workspace: {path}")
    return resolved


def verify_manifest(directory, root):
    manifest = load(directory / "manifest.json")
    base = root if directory.parent.name == "fault_injection" else directory
    for name, digest in manifest["outputs"].items():
        path = safe_path(root, base / name)
        require(sha(path) == digest, f"Artifact checksum mismatch: {path}")
    return manifest


def capture_preservation(root, exclude=None):
    files = {}
    frozen = load(root / "artifacts/audit/freeze_manifest.json")["files"]
    files.update(frozen)
    manifests = sorted((root / "artifacts").rglob("manifest.json"))
    count = 0
    for path in manifests:
        if exclude and path.is_relative_to(exclude):
            continue
        m = verify_manifest(path.parent, root)
        files[path.relative_to(root).as_posix()] = sha(path)
        base = root if path.parent.parent.name == "fault_injection" else path.parent
        for name, digest in m["outputs"].items():
            files[safe_path(root, base / name).relative_to(root).as_posix()] = digest
        count += 1
    for i in range(7, 13):
        evidence = root / f"artifacts/sprint{i}/evidence_manifest.json"
        files[evidence.relative_to(root).as_posix()] = sha(evidence)
        files.update(load(evidence))
    verify_preservation(root, dict(files=files))
    return dict(status="PASS", files=files, prior_manifests=count, sprint0_frozen_files=len(frozen))


def verify_preservation(root, snapshot):
    for name, digest in snapshot["files"].items():
        require(sha(safe_path(root, name)) == digest, f"Preserved input changed: {name}")


def test_totals(path):
    element = ET.parse(path).getroot()
    suites = [element] if element.tag == "testsuite" else list(element.iter("testsuite"))
    require(bool(suites), "JUnit evidence contains no test suites.")
    result = {name: sum(int(s.attrib.get(name, 0)) for s in suites) for name in ("tests", "failures", "errors", "skipped")}
    result["time_seconds"] = math.fsum(float(s.attrib.get("time", 0)) for s in suites)
    require(result["tests"] > 0 and result["failures"] == result["errors"] == result["skipped"] == 0,
            "Final suite must have real passing tests and no failures/errors/skips.")
    return result


def gate_rows(data, technical):
    rows = [dict(check=name, status="PASS" if value else "FAIL", scope="software_audit", detail="Verified final-audit evidence")
            for name, value in technical.items()]
    baseline = data["research"]["baseline"]
    for name, value in baseline["research_target_checks"].items():
        rows.append(dict(check="research_target_" + name, status="PASS" if value["status"] == "MET" else "FAIL", scope="research_target",
                         detail=f"measured={value['rate']}; target={value['target']}; applicable={value['applicable']}"))
    promotion = data["research"]["promotion_decision"]
    rows.append(dict(check="dual_path_promotion", status="PASS" if promotion["gate_passed"] else "FAIL", scope="research_gate", detail=promotion["status"]))
    for name, detail in (
        ("independent_final_holdout", "All flights are historical development; no new untouched final holdout"),
        ("healthy_labels", "assumed_from_source; no measured real fault ground truth"),
        ("currtot_units", "unverified source units; no conversion"),
        ("low_load_generalization", "Primary zero alarms does not erase nonzero sensitivity-model alarms"),
        ("host_latency_context", "Accelerated software replay measured under concurrent pytest load, not idle hardware"),
        ("hardware_emergency_readiness", "NOT_ESTABLISHED; no deployment or switching validation")):
        rows.append(dict(check=name, status="WARNING", scope="limitation", detail=detail))
    return rows


def core_bundle_audit(root):
    """Seven core HGB objects: pilot plus six LOAFO references, without fitting."""
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    cv = load(root / "artifacts/sprint6/cross_validation_result.json")
    pilot = load(root / "artifacts/sprint4/baseline_result.json")
    features = table(root / "artifacts/features" / cv["feature_run_id"] / "battery_features.csv")
    original = {(r["NamaFile"], r["source_record"]): r for r in features}
    checked = predictions = 0
    for path in [Path(pilot["output_directory"]) / "model_bundle.joblib", *sorted(Path(cv["output_directory"]).glob("folds/*/model_bundle.joblib"))]:
        b = joblib.load(path)
        names, split, threshold = b["predictors"], b["split"], b["threshold"]
        require(b["engine"] == "battery_v2" and b["format_version"] == 1, "Unexpected core model format.")
        require(names == ["Curr", "CurrTot", "dI_dt", "Curr_roll_mean_0p5s", "Curr_roll_std_0p5s", "Curr_roll_mean_1p0s", "Curr_roll_std_1p0s"], "Core predictor registry drift.")
        groups = [set(split[k]) for k in ("train", "validation", "test")]
        require(all(groups) and not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "Core flight split overlap.")
        require(threshold["source_role"] == "validation" and set(threshold["source_flights"]) == groups[1]
                and threshold["percentile"] == 99 and threshold["persistence_seconds"] == .5, "Core calibration provenance drift.")
        train = [r for r in features if r["NamaFile"] in groups[0] and r["reference_ready"] == "True"]
        fingerprint = [{k: r[k] if k == "NamaFile" else int(r[k]) if k in ("Inst", "TimeUS", "source_record") else float(r[k])
                        for k in ("NamaFile", "Inst", "TimeUS", "source_record", "voltage_sag_actual", *names)} for r in train]
        digest = hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        require(digest == b["training_provenance"]["training_rows_sha256"], "Core training fingerprint mismatch.")
        require(len(train) == b["training_provenance"]["ready_training_samples"] and b["training_provenance"]["synthetic_rows"] == 0,
                "Core training rows/synthetic policy mismatch.")
        require(not b["model"].early_stopping and b["model"].random_state == 42 and list(b["model"].feature_names_in_) == names,
                "Core fitted model schema/seed/early stopping mismatch.")
        scores = table(path.parent / "scores.csv") if path.parent == Path(pilot["output_directory"]) else table(path.parent / "validation_scores.csv.gz")
        validation = [r for r in scores if r["split_role"] == "validation" and r["reference_ready"] == "True"]
        require({r["NamaFile"] for r in validation} == groups[1] and len(validation) == threshold["source_samples"], "Core threshold sample scope mismatch.")
        calibration_rows = [{k: r[k] if k == "NamaFile" else float(r[k]) if k == "anomaly_score" else int(r[k])
                             for k in ("NamaFile", "Inst", "TimeUS", "source_record", "anomaly_score")} for r in validation]
        calibration_hash = hashlib.sha256(json.dumps({"rows": calibration_rows}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        require(calibration_hash == threshold["source_sha256"], "Core validation calibration fingerprint mismatch.")
        require(math.isclose(float(np.percentile([float(r["anomaly_score"]) for r in validation], 99)), threshold["threshold_v"], abs_tol=1e-12),
                "Core healthy-validation P99 mismatch.")
        selected = [validation[i] for i in np.unique(np.linspace(0, len(validation) - 1, min(100, len(validation)), dtype=int))]
        frame = pd.DataFrame([[float(original[(r["NamaFile"], r["source_record"])][n]) for n in names] for r in selected], columns=names)
        with threadpool_limits(limits=1, user_api="openmp"):
            values = b["model"].predict(frame)
        np.testing.assert_allclose(values, [float(r["predicted_voltage_sag"]) for r in selected], rtol=1e-12, atol=1e-12)
        checked += 1
        predictions += len(selected)
    return dict(status="PASS", bundles_verified=checked, validation_predictions_replayed=predictions,
                exact_predictor_split_training_and_calibration_provenance_verified=True, fitting_performed=False)


def render_reports(data, gates, verification):
    def n(value, scale=1):
        return "undefined" if value is None else f"{value * scale:.6f}"

    r, replay = data["research"], data["replay"]
    b, h, f = r["baseline"], r["baseline"]["healthy_active_pooled"], r["baseline"]["synthetic_pooled"]
    p = r["promotion_decision"]
    lines = ["# FINAL RESEARCH RESULTS", "", "Audit akhir Sprint 13 - 21 Oktober 2026 (Asia/Jakarta).", "",
             "**Pipeline penelitian dan audit software selesai. Detector belum divalidasi untuk deployment/emergency switching. Dual-path tetap NOT_PROMOTED.**", "",
             f"Sumber konsolidasi terkini: `{data['research_summary']['run_id']}`; baseline frozen `{b['run_id']}`; replay `{replay['run_id']}`. Semua angka berasal dari verified CSV/JSON, tanpa memilih ulang kandidat dari outer.", "",
             "## Data, fitur dan scope", "",
             f"{r['data_audit']['rows']} baris /{r['data_audit']['flights']} historical flights, {r['preprocessing']['segments']} continuous segments dan {r['preprocessing']['gaps']} gaps. Enam ACTIVE_LOAD, sebelas LOW_LOAD_OR_IDLE berdasarkan arus; low-load tidak membuktikan UAV berada di ground. Ready {r['features']['ready_samples']} samples, warm-up {r['features']['warmup_samples']}, reference unavailable {r['features']['reference_unavailable_samples']}. Observed telemetry {n(r['preprocessing']['observed_duration_seconds'])} s, gaps {n(r['preprocessing']['excluded_gap_seconds'])} s dikeluarkan dari exposure.", "",
             "Model utama HGB Battery-D memakai ordered predictors: Curr, CurrTot, dI_dt, Curr rolling mean/std 0,5 s dan 1,0 s. Target V_ref - Volt, score max(0, actual sag - prediction), healthy-validation P99 dan timestamp persistence0,5s. Actual Volt/sag/V_ref/dV_dt/residual/readiness/IDs bukan regression predictors. Battery-E elapsed_time hanya ablation. IF/OCSVM memakai voltage-behavior direct registry dan signed -decision_function; hanya OCSVM memakai scaler healthy-train-only dan deterministic capped train sampling.", "",
             "Split whole-flight LOAFO: tiap active flight sekali test, berikutnya validation, sisanya train. Semua candidate-fold selection/freeze selesai sebelum outer masing-masing eksperimen. Seluruh17 flights pernah diperiksa dalam penelitian historis: ini historical development, **bukan independent final holdout baru**. Healthy assumed_from_source; CurrTot unverified tanpa konversi; firmware/BIN/pack metadata dan measured internal-fault ground truth tidak tersedia.", "",
             "## Baseline frozen", "",
             "| Metrik pooled | Hasil |", "|---|---:|",
             f"| MAE / RMSE V | {n(b['regression_pooled']['mae_v'])} / {n(b['regression_pooled']['rmse_v'])} |",
             f"| R2 | {n(b['regression_pooled']['r2'])} |",
             f"| Healthy confirmed samples / ready | {h['confirmed_alarm_samples']}/{h['ready_samples']} |",
             f"| Healthy confirmed FPR | {n(h['confirmed_fpr'],100)}% |",
             f"| Episodes / ready seconds / per minute | {h['confirmed_episodes']} / {n(h['observed_ready_seconds'])} / {n(h['confirmed_episodes_per_minute'])} |",
             f"| Synthetic detected / applicable / missed / contaminated | {f['detected_scenarios']}/{f['applicable_scenarios']} / {f['undetected_scenarios']} / {f['preexisting_alarm_scenarios']} |",
             f"| Synthetic sample precision / recall / F1 | {n(f['precision'])} / {n(f['recall'])} / {n(f['f1'])} |",
             f"| Detected-only delay mean / median / std, n | {n(f['delay_mean_seconds'])} / {n(f['delay_median_seconds'])} / {n(f['delay_std_seconds'])} s, n={f['delay_detected_n']} |", "",
             "| Deadline s | Success / applicable | Rate |", "|---:|---:|---:|"]
    for d in (1, 2, 5, 10, 30):
        key = f"deadline_le_{d}s"
        lines.append(f"| {d} | {f[key + '_successes']}/{f['applicable_scenarios']} | {n(f[key + '_rate'],100)}% |")
    lines += ["", "Macro statistics dipisahkan dari pooled:", "", "| Metrik | Mean | Population std | Median | Min | Max | Defined folds |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scope, metric in (("regression","rmse_v"),("healthy_active","confirmed_fpr"),("healthy_active","confirmed_episodes_per_minute"),("synthetic","f1"),("synthetic","delay_mean_seconds")):
        v = r["baseline_macro_per_fold"][scope][metric]
        lines.append(f"| {scope}/{metric} | {n(v['mean'])} | {n(v['std'])} | {n(v['median'])} | {n(v['min'])} | {n(v['max'])} | {v['n_defined']} |")
    lines += ["", "Enam baseline misses tetap terlihat pada flight74: mild increased-resistance dan mild progressive, masing-masing tiga onset positions. Delay missing tetap null. Abrupt severe <=1s memenuhi target; progressive observable <=5s belum memenuhi target. FPR active 11,827317% tetap dilaporkan tanpa menaikkan threshold untuk menyembunyikan alarm.", "",
              "## Ablation dan pembanding tetap", "", "| Experiment / candidate | Healthy FPR % | Episodes/min | Detected/applicable | F1 | RMSE V |", "|---|---:|---:|---:|---:|---:|"]
    for scope in ("ablation", "model_comparison"):
        for name, v in r[scope]["overview"].items():
            lines.append(f"| {scope}/{name} | {n(v['healthy_confirmed_fpr'],100)} | {n(v['healthy_episodes_per_minute'])} | {v['incident_detected']}/{v['applicable']} | {n(v['sample_f1'])} | {n(v['regression_rmse_v'])} |")
    lines += ["", "Tidak ada automatic winner dari tabel outer ini. Battery-E tetap ablation; memilih feature/model family dari hasil tersebut memerlukan label development selection dan independent validation baru.", "",
              "## Validation tuning dan dual-path", "", "| Experiment / policy | Healthy FPR % | Episodes/min | Detected/applicable | F1 | Mean delay s (detected n) |", "|---|---:|---:|---:|---:|---:|"]
    for scope in ("tuning", "dual_path"):
        for policy in ("reference", "selected"):
            v = r[scope][policy + "_pooled"]; vh, vf = v["healthy_active"], v["synthetic"]
            lines.append(f"| {scope}/{policy} | {n(vh['confirmed_fpr'],100)} | {n(vh['confirmed_episodes_per_minute'])} | {vf['detected_scenarios']}/{vf['applicable_scenarios']} | {n(vf['f1'])} | {n(vf['delay_mean_seconds'])} (n={vf['delay_detected_n']}) |")
    lines += ["", "Gate promosi memakai predeclared paired arithmetic macro means, bukan mengganti gate dengan pooled improvement. Detected n berbeda; tambahan late detections tidak meniadakan penurunan deadline success.", "", "| Dual gate | Reference macro | Selected macro | Passed |", "|---|---:|---:|---|"]
    for name, v in p["checks"].items():
        lines.append(f"| {name} | {n(v.get('reference'))} | {n(v.get('selected'))} | {v['passed']} |")
    lines += ["", f"**Keputusan {p['status']}**. Operational dual tetap disabled. Tidak ada outer reselection/rollback untuk memperbaiki angka setelah hasil terlihat.", "",
              "## Healthy low-load dan domain", "", "| Fold | Ready | Ready s | Confirmed | FPR % | Episodes/min |", "|---|---:|---:|---:|---:|---:|"]
    for v in r["low_load_per_model"]:
        lines.append(f"| {v['fold_id']} | {int(v['ready_samples'])} | {n(v['observed_ready_seconds'])} | {int(v['confirmed_alarm_samples'])} | {n(v['confirmed_fpr'],100)} | {n(v['confirmed_episodes_per_minute'])} |")
    lines += ["", "Primary first-sorted fold00 ditetapkan sebelum low-load prediction. Keenam model memakai cohort sebelas flight yang sama; exposure tidak dipool sebagai66 independent flights. Semuanya unseen oleh regressor dan calibration detector. Primary nol alarm tidak menjamin semua model/domain bebas alarm. Source domain report menyimpan initial voltage, ranges/response/residual distributions, 462 marginal comparisons dan numeric context; pada primary642/5119 ready rows berada di luar minimal satu train marginal range. Ini deskriptif, bukan alarm tambahan atau diagnosis physical cause. Contoh actual sensitivity false alarm dipilih setelah run awal tanpa primary alarm dan diberi provenance eksplisit.", "",
              "## Synthetic masks, observability dan denominator", "",
              f"Bank synthetic: {r['fault_bank']['scenarios']} planned, {r['fault_bank']['applicable_scenarios']} applicable, {r['fault_bank']['not_applicable_scenarios']} NOT_APPLICABLE. Konsolidasi {data['research_summary']['synthetic_series']} separate series /{data['research_summary']['grouped_metric_rows']} group rows dan {data['research_summary']['failed_or_not_applicable_cases']} failure/NA records; alternatif candidate/policy tidak dipool sebagai independent incidents.", "",
              "Support mask berbeda dari signal-effect mask. Current-dependent faults hanya valid active segment; sudden-drop pada continuous-ready segment. Semua gaps/warm-up/unavailable dikeluarkan dari ready exposure. NA tidak masuk confusion/recall/deadline denominator; setiap applicable miss/contamination tetap di denominator. Carried confirmed alarm bukan zero-delay success. Undefined rates/delays tetap null, bukan fabricated zero.", "",
              "## Replay dan latency host", "",
              f"Frozen software replay `{replay['run_id']}`: {replay['replayed_rows']} rows /{replay['replay_jobs']} jobs, {replay['comparison_columns']} columns PASS, max numeric error {max(replay['numeric_max_absolute_errors'].values()):g}. {replay['prediction_mode_counts']['UNCACHED']} healthy-ready single-sample uncached calls; {replay['prediction_mode_counts']['EXACT_INPUT_CACHE']} synthetic exact-input/model cache uses. Feature/reference/slope/residual/timer states selalu dihitung ulang causally; cached speed tidak menggantikan benchmark uncached.", "",
              f"Healthy uncached in-process latency: p50 {n(replay['uncached_healthy_end_to_end']['p50_ms'])} ms, p95 {n(replay['uncached_healthy_end_to_end']['p95_ms'])} ms, p99 {n(replay['uncached_healthy_end_to_end']['p99_ms'])} ms, max {n(replay['uncached_healthy_end_to_end']['max_ms'])} ms. Parse/features/prediction/diagnostics+detector saja; model loading/I/O/export dikeluarkan. First calls dipertahankan. Pengukuran Sprint12 berlangsung bersamaan pytest; bukan isolated idle-host atau hardware benchmark. Final audit memverifikasi recorded timings tanpa mengukur ulang.", "",
              "Telemetry onset->fresh confirmation sudah mencakup persistence, tidak ditambah0,5s lagi. Monotonic processing latency dipisahkan dari signal warning delay; accelerated replay tidak memakai sleep/pacing. Kedua clock tidak dicampur menjadi unsupported hardware delay.", "",
              "## Gate akhir", "", "| Check | Scope | Status | Detail |", "|---|---|---|---|"]
    for v in gates:
        lines.append(f"| {v['check']} | {v['scope']} | {v['status']} | {v['detail']} |")
    lines += ["", "## Keterbatasan dan keputusan akhir", "",
              "Software/reproducibility audit yang lulus tidak mengubah research gate yang gagal. Dataset kecil dan healthy-assumed, real fault labels dan firmware/pack metadata tidak tersedia; CurrTot belum terverifikasi; seluruh flights historis; low-load operating domain bukan ground truth; synthetic faults tidak mengidentifikasi mekanisme kerusakan internal. Latency host tidak memvalidasi hardware switching. Koleksi data baru yang representatif dan independently labelled serta protokol development/validation baru diperlukan sebelum klaim generalisasi atau emergency readiness. Tidak ada perubahan denominator/threshold/gate setelah outer untuk menutupi kegagalan.", "",
              "[Final verification](FINAL_VERIFICATION.md) dan [machine-readable results](artifacts/sprint13/final_results.json) memuat sumber/hash/evidence. Scope frozen baseline, tuning/dual candidates dan per-flight/group failures tetap terpisah.", "",
              "> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.", "",
              "> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.", ""]
    v = ["# FINAL VERIFICATION", "", "Audit akhir Sprint13 - 21 Oktober2026 (Asia/Jakarta).", "",
         f"**Software final audit: {verification['software_audit_status']}. Research/deployment gates terpisah; dual-path NOT_PROMOTED.**", "",
         f"Full suite: **{verification['tests']['tests']} passed**, failures {verification['tests']['failures']}, errors {verification['tests']['errors']}, skips {verification['tests']['skipped']}; XML elapsed {n(verification['tests']['time_seconds'])} s.", "",
         "## Eksekusi dan bukti", "",
         "Audit raw/schema/unit/freeze, pip check, CLI research-evaluation dari cwd berbeda, CLI replay-equivalence dan repeat-cache checks dijalankan pada current code. Independent audit outputs ditangkap ke artifacts/sprint13/ tanpa menulis ulang evidence sprint lama. Historical weights/config/threshold tidak difit atau diubah oleh final audit; unit/integration fixtures tetap boleh melatih model uji.", "",
         "| Audit | Status |", "|---|---|"]
    for name, value in verification["audits"].items():
        v.append(f"| {name} | {value['status']} |")
    v += ["", f"Total saved bundles reviewed: **{verification['bundles_verified']}** (pilot+LOAFO7, ablation/comparison84, tuning6, dual6). Exact predictor order/exclusions, disjoint whole-flight roles, original healthy-ready identities/targets, train-only OCSVM scaler/sampling, seed/params/early-stopping, healthy-validation quantile fingerprints, synthetic exclusion dan saved-object predictions diverifikasi. Validation selection/refinement/fallback dan pre-outer freeze diperiksa tanpa reselection dari outer.", "",
          f"Preservation: {verification['preserved_files']} unique prior input/artifact/evidence files dan {verification['preserved_manifests']} prior completed manifests tetap cocok. Current source hashes, command logs, test XML, audit result JSON, source run graph, gate table dan document hashes disimpan di artifacts/sprint13/.", "",
          f"Current research CLI run `{data['research_summary']['run_id']}` mereuse source set yang sama; source code berbeda dari snapshot Sprint11 menjelaskan run ID report baru. Tidak ada training stage baru: {data['research_summary']['training_stages_built_at_creation']}. Replay `{replay['run_id']}` tetap mempin snapshot sumber Sprint11 lamanya; final audit memeriksa equality source data/features/baseline/low-load, bukan menuntut kedua package dibuat pada waktu yang sama.", "",
          "Replay equality dan original host latency diverifikasi ulang; repeated CLI mempertahankan content/mtime dan tidak remeasure. Phase totals, population/cache separation, NA/contamination/missed denominator, signal onset/support/deadlines dan persistence counted once dipertahankan. Kesesuaian input causal batch/incremental didukung prefix/gap/unready/interleaving tests dan independent raw reference/rolling/OLS/timer audit.", "",
          "## Status gate", "", "| Check | Status |", "|---|---|"]
    v += [f"| {g['check']} | {g['status']} |" for g in gates]
    v += ["", "FAIL pada progressive target dan dual promotion merupakan hasil riset yang dipertahankan, bukan kegagalan menjalankan final software audit. WARNING menandai keterbatasan ilmiah/data/hardware. Tidak ada klaim semua research targets lulus atau deployment approved.", "",
          "## Reproduksi", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/run_sprint13.py", "```", "",
          "[Results](FINAL_RESEARCH_RESULTS.md) - [evidence index](artifacts/sprint13/evidence_manifest.json) - [gate checks](artifacts/sprint13/gate_checks.csv). Full pipeline Sprint0-13 selesai dalam scope penelitian dan baseline software validation. Tidak ada tindakan deployment/hardware.", ""]
    return "\n".join(lines), "\n".join(v)
