"""Independent saved-object audit: train-only scaler/sampling and score replay.

Uses sklearn objects via joblib, without project imports, fitting or modification.
"""

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
    summaries = [load(ROOT / "artifacts/sprint7" / p) for p in ("ablation_result.json", "comparison_result.json")]
    feature_path = ROOT / "artifacts/features" / summaries[0]["feature_run_id"]
    source = table(feature_path / "battery_features.csv")
    originals = {(r["NamaFile"], r["source_record"]): r for r in source}
    checked = replayed = scalers = sampled = 0
    for summary in summaries:
        output = Path(summary["output_directory"])
        scenarios = table(output / "fault_per_scenario.csv")
        for path in sorted(output.glob("candidates/*/*/model_bundle.joblib")):
            bundle = joblib.load(path)
            folder = path.parent
            assert bundle["engine"] == "battery_v2_sprint7" and bundle["currtot_unit"] == "unverified"
            assert not bundle["unit_conversion_applied"]
            train = [r for flight in bundle["split"]["train"] for r in source if r["NamaFile"] == flight and r["reference_ready"] == "True"]
            predictors = bundle["predictors"]
            provenance = bundle["training_provenance"]
            identities = [{k: int(r[k]) if k != "NamaFile" else r[k] for k in ("NamaFile", "Inst", "TimeUS", "source_record")} for r in train]
            digest = hashlib.sha256(json.dumps({"rows": identities}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            assert digest == provenance["training_identity_sha256"]
            indices = np.arange(len(train))
            if bundle["model_name"] == "ocsvm":
                cap = bundle["config"]["models"]["one_class_svm"]["max_train_samples"]
                if len(train) > cap:
                    indices = np.sort(np.random.default_rng(bundle["random_seed"]).choice(len(train), cap, replace=False))
                x = np.array([[float(r[k]) for k in predictors] for r in train])
                np.testing.assert_allclose(bundle["scaler"].mean_, x.mean(axis=0), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(bundle["scaler"].var_, x.var(axis=0), rtol=1e-10, atol=1e-10)
                assert bundle["scaler"].n_samples_seen_ == len(train)
                assert bundle["model"].shape_fit_[0] == len(indices)
                scalers += 1
                sampled += len(indices)
            else:
                assert bundle["scaler"] is None
            assert provenance["fit_sample_indices"] == indices.tolist()
            assert provenance["fit_sample_identities"] == [identities[i] for i in indices]
            assert provenance["ready_training_samples"] == len(train)
            model = bundle["model"]
            if bundle["model_name"] == "hgb":
                assert model.early_stopping is False and model.n_iter_ == 200
            elif bundle["model_name"] in ("rf", "if"):
                assert len(model.estimators_) == 200 and model.random_state == 42
            if bundle["regression"]:
                assert not set(predictors) & {"Volt", "V_ref", "voltage_sag_actual", "dV_dt", "TimeUS", "Inst", "source_record"}
                assert ("elapsed_time" in predictors) == (bundle["feature_set"] == "battery_e")
            exported = [table(folder / "validation_scores.csv.gz"), table(folder / "healthy_test_scores.csv.gz")]
            scenario = next(r for r in scenarios if r["candidate"] == bundle["candidate"] and r["fold_id"] == bundle["fold_id"] and r["fault_applicable"] == "True")
            exported.append(table(output / scenario["scores_csv"]))
            for scores in exported:
                reconstructed, prior = [], None
                for r in scores:
                    if r["reference_ready"] != "True":
                        prior = r
                        continue
                    base = originals[(r["NamaFile"], r["source_record"])]
                    values = {k: float(base[k]) for k in predictors}
                    if not bundle["regression"]:
                        values["voltage_sag_actual"] = float(r["voltage_sag_actual"]) if r["reference_ready"] == "True" else 0.
                        values["dV_dt"] = (float(r["Volt"]) - float(prior["Volt"])) / float(r["dt_seconds"]) if prior is not None and r["segment_id"] == prior["segment_id"] else 0.
                    if r["reference_ready"] == "True":
                        reconstructed.append((values, r))
                    prior = r
                indices_to_check = np.unique(np.linspace(0, len(reconstructed) - 1, min(100, len(reconstructed)), dtype=int))
                selected = [reconstructed[i] for i in indices_to_check]
                x = pd.DataFrame([[v[k] for k in predictors] for v, _ in selected], columns=predictors)
                with threadpool_limits(limits=1, user_api="openmp"):
                    predicted = model.predict(x) if bundle["regression"] else -model.decision_function(bundle["scaler"].transform(x) if bundle["scaler"] else x)
                column = "predicted_voltage_sag" if bundle["regression"] else "anomaly_score"
                np.testing.assert_allclose(predicted, [float(r[column]) for _, r in selected], rtol=1e-9, atol=1e-9)
                replayed += len(selected)
            checked += 1
    result = dict(status="PASS", bundles_verified=checked, ocsvm_scalers_verified=scalers,
                  sampled_training_identities_verified=sampled, replayed_validation_healthy_and_synthetic_scores=replayed,
                  project_code_imported=False, fit_performed=False,
                  checks="whole healthy-ready source identities, seed/cap sorted indices, exact train-only scaler mean/variance/count, fitted estimator parameters, feature exclusions, saved-object score direction/prediction replay on stratified timeline samples")
    (ROOT / "artifacts/sprint7/bundle_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
