"""Saved-object/schema/dual-policy/source identity audit, without project imports."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    summary = load(ROOT / "artifacts/sprint9/dual_result.json")
    output = Path(summary["output_directory"])
    source = table(ROOT / "artifacts/features" / summary["feature_run_id"] / "battery_features.csv")
    originals = {(r["NamaFile"], r["source_record"]): r for r in source}
    selected = load(output / "selected_per_fold.json")
    names = load(output / "feature_registry.json")["regression_sets"]["battery_d"]
    assert len(names) == 7 and not set(names) & {"Volt", "V_ref", "voltage_sag_actual", "dV_dt", "elapsed_time", "TimeUS", "residual_delta_v"}
    checked = replayed = train_count = 0
    for entry in selected:
        folder = output / "folds" / entry["fold_id"]
        bundle = joblib.load(folder / "model_bundle.joblib")
        assert bundle["experiment_stage"] == 9 and bundle["candidate"] == "hgb_battery_d"
        assert bundle["dual_policy"] == entry["selected_policy"] and bundle["split"] == entry["split"]
        assert bundle["scaler"] is None and bundle["currtot_unit"] == "unverified" and not bundle["unit_conversion_applied"]
        assert bundle["config"]["dual_path"]["enabled"] is False
        model = bundle["model"]
        assert model.random_state == 42 and model.early_stopping is False and model.n_iter_ == 200
        assert list(model.feature_names_in_) == bundle["predictors"] == names
        training = [r for flight in entry["split"]["train"] for r in source if r["NamaFile"] == flight and r["reference_ready"] == "True"]
        ids = [{k: r[k] if k == "NamaFile" else int(r[k]) for k in ("NamaFile", "Inst", "TimeUS", "source_record")} for r in training]
        provenance = bundle["training_provenance"]
        assert ids == provenance["fit_sample_identities"] and provenance["fit_sample_indices"] == list(range(len(training)))
        assert provenance["ready_training_samples"] == provenance["model_fit_samples"] == len(training)
        assert provenance["synthetic_rows"] == provenance["scaler_fit_samples"] == 0
        assert any(float(r["Curr"]) == 0 for r in training)
        digest = hashlib.sha256(json.dumps({"rows": ids}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert digest == provenance["training_identity_sha256"]
        train_count += len(training)
        residuals = load(folder / "residual_thresholds.json")
        policy = entry["selected_policy"]
        assert bundle["threshold"]["threshold_v"] == policy["threshold_high_v"] == residuals[format(policy["high_percentile"], "g")]["threshold_v"]
        assert set(bundle["threshold"]["source_flights"]) == set(entry["split"]["validation"])
        if policy["mode"] == "DUAL":
            slopes = load(folder / "slope_thresholds.json")
            payload = next(s for s in slopes if s["window_seconds"] == policy["slope_window_seconds"] and s["percentile"] == policy["slope_percentile"])
            assert payload["threshold_v_per_s"] == policy["threshold_slope_v_per_s"]
            assert policy["threshold_low_v"] == residuals[format(policy["low_percentile"], "g")]["threshold_v"]
            assert policy["synthetic_refined"] and policy["shortlisted"] and policy["healthy_eligible"]
        paths = [folder / "healthy_validation_scores.csv.gz", folder / "reference/healthy_test_scores.csv.gz", folder / "selected/healthy_test_scores.csv.gz",
                 next((folder / "validation_scenario_scores").glob("*.gz")), next((folder / "selected/scenario_scores").glob("*.gz"))]
        for path in paths:
            rows = [r for r in table(path) if r["reference_ready"] == "True"]
            indices = np.unique(np.linspace(0, len(rows) - 1, min(100, len(rows)), dtype=int))
            samples = [rows[i] for i in indices]
            x = pd.DataFrame([[float(originals[(r["NamaFile"], r["source_record"])][k]) for k in names] for r in samples], columns=names)
            with threadpool_limits(limits=1, user_api="openmp"):
                predicted = model.predict(x)
            np.testing.assert_allclose(predicted, [float(r["predicted_voltage_sag"]) for r in samples], rtol=1e-12, atol=1e-12)
            replayed += len(samples)
        checked += 1
    result = dict(status="PASS", bundles_verified=checked, assigned_ready_training_identities_verified=train_count,
                  saved_object_prediction_replays=replayed, project_code_imported=False, fit_performed=False,
                  checks="fixed HGB-D feature exclusions/seed/iterations/no early stopping/scaler, full healthy-ready train identity/indices, saved dual policy/threshold/slope source consistency, operational enabled false, stratified validation/healthy/synthetic prediction replay")
    (ROOT / "artifacts/sprint9/bundle_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
