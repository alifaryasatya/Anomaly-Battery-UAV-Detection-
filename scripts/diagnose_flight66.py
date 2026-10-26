"""Read-only diagnosis of frozen historical models. No fit, tuning, or policy edit.

Run from any cwd: .venv/Scripts/python.exe scripts/diagnose_flight66.py
All new outputs are confined to artifacts/flight66_diagnosis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/flight66_diagnosis"
CV = ROOT / "artifacts/cross_validation/4bc978e2773687d3"
FEATURES = ROOT / "artifacts/features/1225e8a76a8ce014"
TARGET = "00000066.BIN"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(name, data):
    (OUT / name).write_text(json.dumps(json_clean(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def csv(name, records):
    data = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    data.to_csv(OUT / name, index=False, float_format="%.17g")


def verify_inputs():
    inputs = {}
    for directory in (CV, FEATURES):
        m = load(directory / "manifest.json")
        for name, expected in m["outputs"].items():
            path = directory / name
            require(path.is_file(), f"Historical artifact unavailable: {path}; stop before reproduction.")
            require(sha(path) == expected, f"Historical manifest mismatch: {path}")
            inputs[path.relative_to(ROOT).as_posix()] = expected
        inputs[(directory / "manifest.json").relative_to(ROOT).as_posix()] = sha(directory / "manifest.json")
    frozen = load(ROOT / "artifacts/audit/freeze_manifest.json")
    for name, expected in frozen["files"].items():
        require(sha(ROOT / name) == expected, f"Frozen input mismatch: {name}")
        inputs[name] = expected
    # Preserve every pre-existing Python engine, configuration and prior final evidence.
    for base, pattern in ((ROOT / "src", "*.py"), (ROOT / "configs", "*"), (ROOT / "artifacts/sprint13", "*")):
        for p in base.rglob(pattern):
            if p.is_file():
                inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    inputs["main.py"] = sha(ROOT / "main.py")
    return inputs


def verify_bundle(bundle, fold, feature_rows):
    split, threshold = bundle["split"], bundle["threshold"]
    groups = [set(split[k]) for k in ("train", "validation", "test")]
    require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "Overlapping flight roles")
    require(TARGET not in groups[0] | groups[1] if TARGET in groups[2] else True, "Flight66 leakage")
    require(set(threshold["source_flights"]) == groups[1] and threshold["source_role"] == "validation", "Calibration membership mismatch")
    require(threshold["percentile"] == 99 and threshold["persistence_seconds"] == .5, "Historical policy mismatch")
    require(bundle["training_provenance"]["synthetic_rows"] == 0, "Synthetic training")
    names = bundle["predictors"]
    require(names == bundle["config"]["features"]["regression_predictors"] == list(bundle["model"].feature_names_in_), "Predictor order mismatch")
    train = feature_rows[feature_rows.NamaFile.isin(groups[0]) & feature_rows.reference_ready].copy()
    columns = ["NamaFile", "Inst", "TimeUS", "source_record", "voltage_sag_actual", *names]
    fingerprint = train[columns].to_dict("records")
    digest = hashlib.sha256(json.dumps({"rows": fingerprint}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(digest == bundle["training_provenance"]["training_rows_sha256"], "Training identity/target fingerprint differs")
    validation = read(CV / "folds" / fold / "validation_scores.csv.gz")
    validation = validation[validation.reference_ready]
    rows = validation[["NamaFile", "Inst", "TimeUS", "source_record", "anomaly_score"]].to_dict("records")
    digest = hashlib.sha256(json.dumps({"rows": rows}, indent=2, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(digest == threshold["source_sha256"], "Calibration fingerprint differs")
    require(np.isclose(np.percentile(validation.anomaly_score, 99), threshold["threshold_v"], atol=1e-14, rtol=0), "Saved P99 verification failed")
    return train


def independent_temporal_audit(df, raw, cfg):
    """Closed-form reference/rolling/derivative/timer oracle, no engine imports."""
    ref = cfg["reference_voltage"]
    warm = (df.TimeUS - df.TimeUS.iloc[0]) / 1e6 <= ref["warmup_seconds"]
    low = df.loc[warm & (df.Curr <= ref["low_current_max_a"]), "Volt"]
    require(len(low) >= ref["min_low_current_samples"], "Insufficient initial reference samples")
    expected_ref = median(low)
    gap_limit = cfg["time"]["max_gap_multiplier"] / cfg["time"]["nominal_frequency_hz"]
    times = df.TimeUS.to_numpy(); curr = df.Curr.to_numpy(); volt = df.Volt.to_numpy()
    dt = np.r_[0., np.diff(times) / 1e6]; boundary = (dt > gap_limit); boundary[0] = True
    seg = np.cumsum(boundary) - 1
    require(np.array_equal(seg, df.segment_id), "Segment boundary mismatch")
    errors = {}
    expected_ready = ~warm
    require(np.array_equal(df.reference_ready, expected_ready), "Warm-up/ready mismatch")
    require(np.allclose(df.loc[df.reference_ready, "V_ref"], expected_ref, atol=1e-12, rtol=0), "Reference differs from initial median")
    expected_dt = np.where(boundary, 0., dt)
    errors["dt_seconds"] = float(np.max(np.abs(df.dt_seconds - expected_dt)))
    for name, values in (("dI_dt", curr), ("dV_dt", volt)):
        expected = np.zeros(len(df)); expected[~boundary] = np.diff(values)[~boundary[1:]] / dt[~boundary]
        errors[name] = float(np.max(np.abs(df[name] - expected)))
    for window, token in ((.5, "0p5"), (1., "1p0")):
        left = 0; means, stds, counts, coverage = [], [], [], []
        segment_start = 0
        for i in range(len(df)):
            if boundary[i]: left = i; segment_start = i
            while (times[i] - times[left]) / 1e6 >= window: left += 1
            values = curr[left:i+1]
            means.append(float(np.mean(values))); stds.append(float(np.std(values, ddof=0))); counts.append(len(values))
            coverage.append(min(1., (times[i] - times[segment_start]) / 1e6 / window))
        for statistic, expected in (("mean", means), ("std", stds), ("count", counts), ("coverage", coverage)):
            name = f"Curr_roll_{statistic}_{token}s"
            errors[name] = float(np.max(np.abs(df[name] - expected)))
        require(np.array_equal(df[f"Curr_roll_ready_{token}s"],np.asarray(coverage)>=1.),"Rolling readiness mismatch")
    require(np.array_equal(df.derivative_ready,~boundary),"Derivative readiness mismatch")
    require(max(errors.values()) < 1e-8, f"Temporal feature errors: {errors}")
    timer_start = None; previous_confirmed = False; expected_durations = []; starts = []; confirmed = []; raws = []; states = []
    for i, row in enumerate(df.itertuples()):
        if boundary[i]: timer_start = None; previous_confirmed = False
        raw_alarm = bool(row.reference_ready and row.anomaly_score > row.threshold_v)
        if raw_alarm:
            if timer_start is None: timer_start = row.TimeUS
            duration = (row.TimeUS - timer_start) / 1e6
        else:
            timer_start = None; duration = 0.
        is_confirmed = raw_alarm and duration >= .5
        raws.append(raw_alarm)
        states.append("NOT_READY" if not row.reference_ready else "WARNING" if is_confirmed else "SUSPECT" if raw_alarm else "NORMAL")
        starts.append(is_confirmed and not previous_confirmed); confirmed.append(is_confirmed); expected_durations.append(duration)
        previous_confirmed = is_confirmed
    require(np.array_equal(df.anomaly_confirmed, confirmed) and np.array_equal(df.confirmed_episode_start, starts), "Independent timer decisions differ")
    require(np.array_equal(df.anomaly_raw,raws) and np.array_equal(df.detector_state,states),"Raw/state decision mismatch")
    require(np.allclose(df.anomaly_duration_seconds, expected_durations, atol=1e-12, rtol=0), "Timer elapsed mismatch")
    original = raw[raw.NamaFile == df.NamaFile.iloc[0]].reset_index(drop=True)
    require(len(original) == len(df), "Raw/retained row count differs")
    raw_mismatches = {n: int(np.sum(original[n].to_numpy() != df[n].to_numpy())) for n in ("TimeUS", "Inst", "Volt", "Curr", "CurrTot")}
    require(not any(raw_mismatches.values()), "Retained telemetry differs from raw")
    ready = df.reference_ready.to_numpy()
    both_ready = ready & np.r_[False, ready[:-1]] & ~boundary
    duration = float(expected_dt[both_ready].sum())
    df["ready_interval_seconds"] = np.where(both_ready, expected_dt, 0.)
    df["near_gap_after_1s"] = df.segment_elapsed_seconds <= 1.
    segment_rows = []
    for sid, block in df.groupby("segment_id", sort=True):
        segment_rows.append(dict(NamaFile=df.NamaFile.iloc[0], segment_id=int(sid), start_time_us=int(block.TimeUS.iloc[0]),
            end_time_us=int(block.TimeUS.iloc[-1]), start_s=float(block.elapsed_time.iloc[0]), end_s=float(block.elapsed_time.iloc[-1]),
            gap_before_s=float(block.gap_before_seconds.iloc[0]), samples=len(block), ready_samples=int(block.reference_ready.sum()),
            confirmed_samples=int(block.anomaly_confirmed.sum()), episodes=int(block.confirmed_episode_start.sum()),
            residual_mean_v=float(block.residual_signed_v.mean()), Curr_mean=float(block.Curr.mean()), CurrTot_mean=float(block.CurrTot.mean()),
            Volt_mean=float(block.Volt.mean()), reference_v=float(expected_ref), first_row_timer_seconds=float(block.anomaly_duration_seconds.iloc[0]),
            first_row_rolling_count_1s=int(block.Curr_roll_count_1p0s.iloc[0])))
    return dict(status="PASS", samples=len(df), raw_telemetry_mismatches=raw_mismatches, segments=int(df.segment_id.nunique()),
        gaps=int(boundary.sum()-1), excluded_gap_seconds=float(dt[boundary].sum()), observed_ready_seconds=duration,
        warmup_samples=int(warm.sum()), reference_low_current_samples=len(low), reference_v=expected_ref,
        reference_lock_elapsed_s=float(df.loc[df.reference_ready, "elapsed_time"].iloc[0]),
        reference_constant_after_lock=True, reference_policy=ref, max_feature_errors=errors,
        duplicate_timestamps=int(original.TimeUS.duplicated().sum()), backward_timestamps=int((original.TimeUS.diff()<0).sum()),
        currtot_decreases=int((original.CurrTot.diff()<0).sum()), instances=original.Inst.unique().tolist(),
        nonfinite_numeric=int((~np.isfinite(original[["TimeUS", "Inst", "Volt", "Curr", "CurrTot"]].to_numpy())).sum()),
        negative_voltage_current_total=int((original[["Volt", "Curr", "CurrTot"]]<0).sum().sum()),
        persistence_decisions_exact=True), segment_rows


def reproduce(flight, fold, features, raw, role="test"):
    path = CV / "folds" / fold / "model_bundle.joblib"
    require(path.exists(), f"Historical model unavailable: {path}. Stop before reproduction.")
    bundle = joblib.load(path)
    train = verify_bundle(bundle, fold, features)
    require(bundle["split"][role] == [flight], "Requested flight role differs from historical split")
    df = features[features.NamaFile == flight].copy().reset_index(drop=True)
    scores = read(CV / "folds" / fold / ("healthy_test_scores.csv.gz" if role=="test" else "validation_scores.csv.gz"))
    require(np.array_equal(df.source_record, scores.source_record) and np.array_equal(df.TimeUS, scores.TimeUS), "Score identities differ")
    for c in scores.columns:
        if c not in df.columns: df[c] = scores[c].to_numpy()
    ready = df.reference_ready
    names = bundle["predictors"]
    with threadpool_limits(limits=1, user_api="openmp"):
        prediction = bundle["model"].predict(df.loc[ready, names])
    error = float(np.max(np.abs(prediction - df.loc[ready, "predicted_voltage_sag"])))
    require(error < 1e-12, "Saved predictions not reproducible")
    df["reproduced_prediction_v"] = np.nan; df.loc[ready, "reproduced_prediction_v"] = prediction
    signed = df.voltage_sag_actual - df.reproduced_prediction_v
    require(np.allclose(signed[ready], df.loc[ready,"residual_signed_v"], atol=1e-12, rtol=0), "Residual mismatch")
    require(np.allclose(signed[ready].clip(lower=0), df.loc[ready,"anomaly_score"], atol=1e-12, rtol=0), "Score mismatch")
    df["threshold_v"] = bundle["threshold"]["threshold_v"]
    df["fold_id"] = fold
    df["VoltageSag_actual"] = df.voltage_sag_actual
    df["VoltageSag_predicted"] = df.reproduced_prediction_v
    df["predicted_voltage_v"] = df.V_ref - df.reproduced_prediction_v
    df["abs_dI_dt"] = df.dI_dt.abs()
    audit, segments = independent_temporal_audit(df, raw, bundle["config"])
    audit.update(fold_id=fold, evaluated_role=role, model_bundle_path=str(path.relative_to(ROOT)), model_bundle_sha256=sha(path), split=bundle["split"],
        threshold=bundle["threshold"], predictors=names, model_parameters=bundle["model"].get_params(), max_prediction_error_v=error)
    return df, train, bundle, audit, segments


def domain_analysis(df, train, bundle):
    names = bundle["predictors"]
    ready = df.reference_ready
    rows, any_out = [], np.zeros(len(df), dtype=bool)
    for name in [*names, "Volt", "voltage_sag_actual"]:
        lo, hi = float(train[name].min()), float(train[name].max())
        outside = ready & ((df[name] < lo) | (df[name] > hi))
        df["outside_train_" + name] = outside
        if name in names: any_out |= outside.to_numpy()
        q = train[name].quantile([.01,.05,.25,.5,.75,.95,.99])
        rows.append(dict(NamaFile=df.NamaFile.iloc[0], variable=name, operational_predictor=name in names,
            train_min=lo, train_max=hi, train_q01=q.iloc[0], train_q05=q.iloc[1],train_q25=q.iloc[2],train_median=q.iloc[3],
            train_q75=q.iloc[4],train_q95=q.iloc[5],train_q99=q.iloc[6], test_min=df.loc[ready,name].min(),test_max=df.loc[ready,name].max(),
            below_train=int((ready & (df[name]<lo)).sum()), above_train=int((ready & (df[name]>hi)).sum()),
            outside_train=int(outside.sum()), outside_fraction=float(outside.sum()/ready.sum()),
            confirmed_outside=int((df.anomaly_confirmed & outside).sum()),
            confirmed_inside=int((df.anomaly_confirmed & ready & ~outside).sum())))
    df["outside_any_train_predictor_range"] = any_out
    # Scaling is only a distance diagnostic. The frozen model receives original predictors.
    center = train[names].median(); scale = train[names].quantile(.75)-train[names].quantile(.25)
    scale = scale.where(scale>1e-10,train[names].std()).where(lambda x:x>1e-10,1.)
    tree = cKDTree(((train[names]-center)/scale).to_numpy())
    distance, index = tree.query(((df.loc[ready,names]-center)/scale).to_numpy(), k=1)
    df["nearest_train_iqr_distance"] = np.nan; df.loc[ready,"nearest_train_iqr_distance"] = distance
    df["nearest_train_flight"] = ""; df.loc[ready,"nearest_train_flight"] = train.iloc[index].NamaFile.to_numpy()
    df["nearest_train_source_record"] = np.nan; df.loc[ready,"nearest_train_source_record"] = train.iloc[index].source_record.to_numpy()
    df["nearest_train_actual_sag_v"] = np.nan; df.loc[ready,"nearest_train_actual_sag_v"] = train.iloc[index].voltage_sag_actual.to_numpy()
    df["nearest_train_target_difference_v"] = df.voltage_sag_actual-df.nearest_train_actual_sag_v
    # Inspect saved HGB discretization without changing model or predictor values.
    mapper = bundle["model"]._bin_mapper
    transformed = mapper.transform(df.loc[ready,names].to_numpy())
    j = names.index("CurrTot"); cuts = mapper.bin_thresholds_[j]
    df["CurrTot_saved_model_bin"] = np.nan; df.loc[ready,"CurrTot_saved_model_bin"] = transformed[:,j]
    return rows, dict(diagnostic_distance="Euclidean in 7 original predictors, training median/IQR (std fallback), nearest actual training row; no operational scaling or OOD decision threshold",
        center=center.to_dict(),scale=scale.to_dict(),CurrTot_last_bin_edge=float(cuts[-1]),CurrTot_training_max=float(train.CurrTot.max()),
        CurrTot_model_bins=int(mapper.n_bins_non_missing_[j]),test_rows_outside_training_range=int(df.outside_train_CurrTot.sum()),
        test_rows_above_training_max=int((ready & (df.CurrTot>train.CurrTot.max())).sum()),
        test_rows_below_training_min=int((ready & (df.CurrTot<train.CurrTot.min())).sum()),
        unique_bins_above_training_max=df.loc[ready & (df.CurrTot>train.CurrTot.max()),"CurrTot_saved_model_bin"].unique().tolist())


def summarize(block):
    a = block[block.reference_ready]
    r=a.residual_signed_v
    return dict(samples=len(a), confirmed_samples=int(a.anomaly_confirmed.sum()), raw_samples=int(a.anomaly_raw.sum()),
        confirmed_fpr=float(a.anomaly_confirmed.mean()) if len(a) else None,
        episodes=int(a.confirmed_episode_start.sum()), ready_seconds=float(a.ready_interval_seconds.sum()),
        residual_mean_v=float(r.mean()),residual_median_v=float(r.median()),residual_q95_v=float(r.quantile(.95)),
        residual_max_v=float(r.max()), mae_v=float(r.abs().mean()),rmse_v=float(np.sqrt((r**2).mean())),
        actual_sag_mean_v=float(a.voltage_sag_actual.mean()),predicted_sag_mean_v=float(a.reproduced_prediction_v.mean()),
        Curr_mean=float(a.Curr.mean()),CurrTot_min=float(a.CurrTot.min()),CurrTot_max=float(a.CurrTot.max()),
        Volt_mean=float(a.Volt.mean()),abs_dI_dt_mean=float(a.abs_dI_dt.mean()),
        outside_any_predictor_samples=int(a.outside_any_train_predictor_range.sum()) if "outside_any_train_predictor_range" in a else None)


def episode_tables(df):
    records=[]
    df["episode_id"]=""
    for k, start in enumerate(np.flatnonzero(df.confirmed_episode_start),1):
        end=start
        while end+1<len(df) and df.anomaly_confirmed.iloc[end+1] and df.segment_id.iloc[end+1]==df.segment_id.iloc[start]: end+=1
        rawstart=start
        while rawstart>0 and df.anomaly_raw.iloc[rawstart-1] and df.segment_id.iloc[rawstart-1]==df.segment_id.iloc[start]:rawstart-=1
        eid=f"{df.NamaFile.iloc[0]}_E{k:02d}";df.loc[start:end,"episode_id"]=eid
        b=df.iloc[start:end+1];rawblock=df.iloc[rawstart:end+1]
        reason="flight_end" if end+1==len(df) else "gap_segment_reset" if df.segment_id.iloc[end+1]!=df.segment_id.iloc[end] else "raw_false"
        rec=dict(NamaFile=df.NamaFile.iloc[0],episode_id=eid,segment_id=int(b.segment_id.iloc[0]),
            raw_start_time_us=int(df.TimeUS.iloc[rawstart]),confirmed_start_time_us=int(b.TimeUS.iloc[0]),last_confirmed_time_us=int(b.TimeUS.iloc[-1]),
            raw_start_s=float(df.elapsed_time.iloc[rawstart]),confirmed_start_s=float(b.elapsed_time.iloc[0]),last_confirmed_s=float(b.elapsed_time.iloc[-1]),
            confirmed_span_seconds=float((b.TimeUS.iloc[-1]-b.TimeUS.iloc[0])/1e6),
            raw_to_last_confirmed_seconds=float((b.TimeUS.iloc[-1]-df.TimeUS.iloc[rawstart])/1e6),
            persistence_elapsed_at_start_seconds=float(b.anomaly_duration_seconds.iloc[0]),confirmed_samples=len(b),end_reason=reason,
            Curr_min=float(b.Curr.min()),Curr_mean=float(b.Curr.mean()),Curr_max=float(b.Curr.max()),
            CurrTot_start=float(b.CurrTot.iloc[0]),CurrTot_end=float(b.CurrTot.iloc[-1]),
            Volt_min=float(b.Volt.min()),Volt_mean=float(b.Volt.mean()),Volt_max=float(b.Volt.max()),V_ref=float(b.V_ref.iloc[0]),
            actual_sag_mean_v=float(b.voltage_sag_actual.mean()),predicted_sag_mean_v=float(b.reproduced_prediction_v.mean()),
            residual_mean_v=float(b.residual_signed_v.mean()),residual_max_v=float(b.residual_signed_v.max()),
            abs_dI_dt_mean=float(b.abs_dI_dt.mean()),abs_dI_dt_max=float(b.abs_dI_dt.max()),
            near_gap_1s_confirmed=int(b.near_gap_after_1s.sum()), outside_CurrTot_samples=int(b.outside_train_CurrTot.sum()),
            transient_context_samples=int(b.transient_context.sum()),low_current_samples=int((b.Curr<=1).sum()))
        records.append(rec)
    require(sum(x["confirmed_samples"] for x in records)==int(df.anomaly_confirmed.sum()),"Episode partition incomplete")
    return records


def grouped_analysis(df,train):
    a=df[df.reference_ready].copy();flight=df.NamaFile.iloc[0]
    threshold_di=float(train.dI_dt.abs().quantile(.95)); threshold_std=float(train.Curr_roll_std_1p0s.quantile(.95))
    df["transient_context"]=(df.abs_dI_dt>threshold_di)|(df.Curr_roll_std_1p0s>threshold_std)
    a["transient_context"]=df.loc[a.index,"transient_context"]
    # Elapsed low-current run is diagnostic only, reset at each telemetry gap.
    low_start=None;low_elapsed=[];previous_segment=None
    for row in df.itertuples():
        if row.segment_id!=previous_segment: low_start=None
        if row.Curr<=1:
            if low_start is None:low_start=row.TimeUS
            low_elapsed.append((row.TimeUS-low_start)/1e6)
        else:low_start=None;low_elapsed.append(0.)
        previous_segment=row.segment_id
    df["diagnostic_low_current_run_seconds"]=low_elapsed
    a["diagnostic_low_current_run_seconds"]=df.loc[a.index,"diagnostic_low_current_run_seconds"]
    summaries=[]; edges={}
    for variable in ("CurrTot","Volt","Curr","abs_dI_dt","elapsed_time"):
        bins=pd.qcut(a[variable],4,duplicates="drop")
        edges[variable]=[float(x) for x in np.unique(a[variable].quantile([0,.25,.5,.75,1]).to_numpy())]
        for label,block in a.groupby(bins,observed=True):summaries.append(dict(NamaFile=flight,variable=variable,group=str(label),**summarize(block)))
    for name,mask in (("CurrTot_above_train_max",a.CurrTot>train.CurrTot.max()),("low_current_le_1A",a.Curr<=1),
                      ("transient_context",a.transient_context),("near_gap_after_1s",a.near_gap_after_1s),
                      ("outside_any_predictor",a.outside_any_train_predictor_range)):
        for flag in (False,True):summaries.append(dict(NamaFile=flight,variable=name,group=str(flag),**summarize(a[mask==flag])))
    for duration in (5,10,30):
        mask=(a.Curr<=1)&(a.diagnostic_low_current_run_seconds>=duration)&~a.transient_context
        summaries.append(dict(NamaFile=flight,variable="stable_low_current_run",group=f">={duration}s",**summarize(a[mask])))
    cross=[]
    for (ood,low),block in a.groupby([a.CurrTot>train.CurrTot.max(),a.Curr<=1]):
        cross.append(dict(NamaFile=flight,CurrTot_above_train_max=bool(ood),low_current_le_1A=bool(low),**summarize(block)))
    windows=[]
    # Fixed descriptive 10-s blocks, not a new detector. No block crosses a telemetry segment.
    for (sid,number),block in a.groupby([a.segment_id,(a.segment_elapsed_seconds//10).astype(int)]):
        span=float((block.TimeUS.iloc[-1]-block.TimeUS.iloc[0])/1e6)
        windows.append(dict(NamaFile=flight,segment_id=int(sid),window_number=int(number),start_s=float(block.elapsed_time.iloc[0]),
            end_s=float(block.elapsed_time.iloc[-1]),observed_span_s=span,threshold_v=float(block.threshold_v.iloc[0]),
            transient_fraction=float(block.transient_context.mean()),raw_fraction=float(block.anomaly_raw.mean()),
            median_residual_above_threshold=bool(block.residual_signed_v.median()>block.threshold_v.iloc[0]),**summarize(block)))
    correlations=[]
    variables=["elapsed_time","CurrTot","Volt","Curr","abs_dI_dt","Curr_roll_std_1p0s","residual_signed_v","anomaly_confirmed"]
    for method in ("pearson","spearman"):
        matrix=a[variables].astype(float).corr(method=method)
        for first in variables:
            for second in variables:
                correlations.append(dict(NamaFile=flight,method=method,first=first,second=second,correlation=matrix.loc[first,second],samples=len(a)))
    # Within-current-stratum correlations expose temporal confounding; no causal importance claim.
    for label,block in a.groupby(pd.qcut(a.Curr,4,duplicates="drop"),observed=True):
        for variable in ("CurrTot","Volt","elapsed_time"):
            correlations.append(dict(NamaFile=flight,method="spearman_within_current_bin",first=variable,second="residual_signed_v",
                correlation=block[variable].corr(block.residual_signed_v,method="spearman"),samples=len(block),current_bin=str(label)))
    return summaries,cross,windows,correlations,dict(abs_dI_dt_training_p95=threshold_di,current_std_1s_training_p95=threshold_std,
        transient_context_definition="abs(dI_dt)>training P95 OR current rolling std(1s)>training P95; diagnostic context only; not detector/tuning",
        quartile_edges=edges,window_definition="Fixed 10s blocks per segment; report duration and continuous bias; not a replacement alarm")


def figures(frames,trains,bundles,episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,"savefig.dpi":170})
    plots=OUT/"plots";plots.mkdir(exist_ok=True)
    colors={TARGET:"#cf4737",frames[1].NamaFile.iloc[0]:"#236caa","00000069.BIN":"#668b44"}
    for df in frames:
        flight=df.NamaFile.iloc[0]; short=flight.split('.')[0];t=df.elapsed_time
        fig,ax=plt.subplots(5,1,figsize=(14,12),sharex=True,constrained_layout=True)
        # NaN breaks at gaps so a connected line never invents telemetry inside a gap.
        def line(axis,column,label,color):
            for j,(_,b) in enumerate(df.groupby("segment_id")):
                axis.plot(b.elapsed_time,b[column],lw=.8,color=color,label=label if j==0 else None)
        line(ax[0],"voltage_sag_actual","Actual sag","#222222");line(ax[0],"reproduced_prediction_v","Frozen HGB predicted sag","#2879b0")
        ax[0].set_ylabel("Voltage sag (V)");ax[0].legend(loc="upper left",ncol=2)
        line(ax[1],"residual_signed_v","Signed residual","#2879b0");ax[1].axhline(df.threshold_v.iloc[0],color="#ce4530",ls="--",label="Frozen threshold")
        ax[1].axhline(0,color="grey",lw=.7);ax[1].set_ylabel("Residual (V)");ax[1].legend(loc="upper left",ncol=2)
        for j,(_,block) in enumerate(df.groupby("segment_id")):
            ax[2].step(block.elapsed_time,block.anomaly_confirmed.astype(int),where="post",color="#ce4530",lw=.9,label="Confirmed warning" if j==0 else None)
        ax[2].set_ylim(-.1,1.2);ax[2].set_yticks([0,1]);ax[2].set_ylabel("Confirmed");ax[2].legend(loc="upper left")
        line(ax[3],"Curr","Current (A)","#236caa");ax[3].set_ylabel("Current (A)")
        twin=ax[3].twinx();line(twin,"CurrTot","CurrTot (source units)","#8e5da8");twin.set_ylabel("CurrTot (source units)",color="#8e5da8")
        line(ax[4],"Volt","Actual voltage","#222222");line(ax[4],"predicted_voltage_v","V_ref - prediction","#2879b0")
        ax[4].set_ylabel("Voltage (V)");ax[4].legend(loc="upper left",ncol=2);ax[4].set_xlabel("Seconds since first flight sample (gaps retained)")
        for axis in ax:
            for _,block in df.groupby("segment_id"):
                if block.index[0]>0:
                    prev=df.iloc[block.index[0]-1].elapsed_time
                    axis.axvspan(prev,block.elapsed_time.iloc[0],color="grey",alpha=.10)
        role="held-out test" if df.split_role.iloc[0]=="test" else "calibration diagnostic (not independent test)"
        fig.suptitle(f"{flight} | {df.fold_id.iloc[0]} {role} | threshold {df.threshold_v.iloc[0]:.6f} V | persistence 0.5 s")
        fig.savefig(plots/f"{short}_timeline.png");plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    a=frames[0][frames[0].reference_ready]
    for axis,var in zip(ax.flat,["CurrTot","Volt","Curr","dI_dt","Curr_roll_std_1p0s","elapsed_time"]):
        normal=~a.anomaly_confirmed
        axis.scatter(a.loc[normal,var],a.loc[normal,"residual_signed_v"],s=3,alpha=.25,color="#2879b0",rasterized=True,label="Unconfirmed")
        axis.scatter(a.loc[~normal,var],a.loc[~normal,"residual_signed_v"],s=3,alpha=.25,color="#cf4737",rasterized=True,label="Confirmed")
        axis.axhline(a.threshold_v.iloc[0],color="black",ls="--",lw=.8)
        axis.set_xlabel(var + (" (source units)" if var=="CurrTot" else ""));axis.set_ylabel("Signed residual (V)")
    ax.flat[0].axvline(trains[0].CurrTot.max(),color="#7b4b9a",ls=":",label="Train max CurrTot");ax.flat[0].legend(fontsize=8)
    fig.suptitle("Flight 66 associations: correlated time / CurrTot / voltage are not causal attribution")
    fig.savefig(plots/"flight66_residual_relationships.png");plt.close(fig)
    names=bundles[0]["predictors"]
    fig,ax=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    for axis,name in zip(ax.flat,[*names,"Volt","voltage_sag_actual"]):
        train=np.sort(trains[0][name].to_numpy());test=np.sort(a[name].to_numpy())
        axis.plot(train,np.arange(1,len(train)+1)/len(train),label="Fold00 healthy train",color="#2879b0")
        axis.plot(test,np.arange(1,len(test)+1)/len(test),label="Flight66 ready",color="#cf4737")
        axis.set_xlabel(name);axis.set_ylabel("Empirical CDF")
    ax.flat[0].legend(fontsize=8);fig.suptitle("Training vs Flight66: original predictors; voltage and target are context only")
    fig.savefig(plots/"flight66_training_feature_ecdf.png");plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for df in frames:
        a=df[df.reference_ready];flight=df.NamaFile.iloc[0];color=colors[flight]
        ax[0,0].hist(a.residual_signed_v,bins=80,histtype="step",density=True,label=flight,color=color)
        ax[0,1].scatter(a.CurrTot,a.residual_signed_v,s=3,alpha=.25,color=color,label=flight)
        ax[1,0].scatter(a.Curr,a.voltage_sag_actual,s=3,alpha=.2,color=color,label=flight)
        ax[1,1].scatter(a.CurrTot,a.Volt,s=3,alpha=.2,color=color,label=flight)
    ax[0,0].set(xlabel="Signed residual (V)",ylabel="Density");ax[0,0].legend()
    ax[0,1].set(xlabel="CurrTot (unverified source units)",ylabel="Signed residual (V)")
    ax[1,0].set(xlabel="Current (A)",ylabel="Actual sag (V)");ax[1,1].set(xlabel="CurrTot (unverified source units)",ylabel="Voltage (V)")
    fig.suptitle("66 vs own-fold70 control and fold00 calibration69 (not an independent test)")
    fig.savefig(plots/"flight66_vs_control.png");plt.close(fig)
    target=frames[0]
    eps=[r for r in episodes if r["NamaFile"]==TARGET]
    picks=[min(eps,key=lambda x:x["confirmed_start_s"]),max(eps,key=lambda x:x["confirmed_span_seconds"])]
    fig,ax=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for j,ep in enumerate(picks):
        b=target[(target.segment_id==ep["segment_id"]) & (target.elapsed_time>=ep["raw_start_s"]-3)&(target.elapsed_time<=ep["last_confirmed_s"]+3)]
        ax[j,0].plot(b.elapsed_time,b.residual_signed_v,label="Signed residual");ax[j,0].axhline(b.threshold_v.iloc[0],color="#cf4737",ls="--",label="Frozen threshold")
        ax[j,0].axvspan(ep["confirmed_start_s"],ep["last_confirmed_s"],color="#cf4737",alpha=.15)
        ax[j,0].set(ylabel="Residual (V)",xlabel="Flight elapsed seconds",title=ep["episode_id"]);ax[j,0].legend(fontsize=8)
        ax[j,1].plot(b.elapsed_time,b.Curr,color="#236caa",label="Current A");ax[j,1].set(ylabel="Current (A)",xlabel="Flight elapsed seconds")
        tw=ax[j,1].twinx();tw.plot(b.elapsed_time,b.Volt,color="#333333",label="Voltage V");tw.set_ylabel("Voltage (V)")
    fig.suptitle("Post-result illustrations: first episode and longest confirmed episode (descriptive only)")
    fig.savefig(plots/"flight66_episode_examples.png");plt.close(fig)


def render_report(frames, results, audit, episodes):
    """Factual narrative and tables derived from this run's verified source data."""
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            return "undefined" if not np.isfinite(v) else f"{v:.6f}"
        return str(v)
    def table(data, columns, labels=None):
        rows=data.to_dict("records") if isinstance(data,pd.DataFrame) else data
        labels=labels or columns
        return "\n".join(["| "+" | ".join(labels)+" |","|"+"---|"*len(columns),
            *["| "+" | ".join(fmt(r.get(k,"")) for k in columns)+" |" for r in rows]])
    target=frames[0]; a=target[target.reference_ready]; summaries=results["flights"]
    ep=pd.DataFrame(episodes);ep=ep[ep.NamaFile==TARGET].copy()
    groups=read(OUT/"condition_strata.csv");groups=groups[groups.NamaFile==TARGET]
    domain=read(OUT/"feature_distribution.csv");domain=domain[domain.NamaFile==TARGET]
    windows=read(OUT/"bias_windows_10s.csv");windows=windows[(windows.NamaFile==TARGET)&(windows.observed_span_s>=5)]
    high=a[a.CurrTot>results["domain"][0]["CurrTot_training_max"]]
    long=ep[ep.confirmed_span_seconds>30]
    maximum=a.loc[a.residual_signed_v.idxmax()]
    ca=a.anomaly_confirmed; confirmed=int(ca.sum())
    low=int((ca & (a.Curr<=1)).sum()); quiet=int((ca & ~a.transient_context).sum())
    stable=(a.Curr<=1)&(a.diagnostic_low_current_run_seconds>=10)&~a.transient_context
    longest=ep.loc[ep.confirmed_span_seconds.idxmax()]
    b0=audit["audits"][0];b70=audit["audits"][1]
    corr=read(OUT/"correlations.csv");corr=corr[(corr.NamaFile==TARGET)&(corr.method=="spearman")&(corr.second=="residual_signed_v")]
    lines=[
        "# ROOT CAUSE ANALYSIS FALSE ALARM FLIGHT 66", "", "Tanggal: 22 Oktober 2026 (Asia/Jakarta). Status: diagnosis historis selesai, tanpa refinement.", "",
        "## Kesimpulan berbasis bukti", "",
        f"False alarm Flight 66 terutama berasal dari **underprediction sag yang bertahan lama pada bagian akhir rekaman**, ketika `CurrTot` melampaui domain training dan level tegangan sudah lebih rendah. Model menganggap tegangan seharusnya lebih tinggi; positive residual kemudian melewati threshold selama >=0,5 s. Ini adalah mekanisme false alarm yang berhasil direproduksi, bukan bukti kerusakan internal baterai.", "",
        f"Reproduksi menghasilkan tepat **{confirmed:,} confirmed samples / 9.506 ready samples (33,326320%) dalam 26 episode**. Sebanyak {int(high.anomaly_confirmed.sum()):,}/{confirmed:,} ({100*high.anomaly_confirmed.sum()/confirmed:.2f}%) confirmed samples berada di atas maksimum `CurrTot` training; lima episode >30 s menyumbang {int(long.confirmed_samples.sum()):,} samples ({100*long.confirmed_samples.sum()/confirmed:.2f}%). Episode terpanjang {longest.confirmed_span_seconds:.6f} s terjadi pada arus rata-rata {longest.Curr_mean:.7f} A.", "",
        "Implementasi reference/features/persistence cocok dengan protokol historis dan data tersimpan. Ada 14 telemetry gaps yang penting bagi interpretasi, tetapi tidak ditemukan salah reset, pencampuran segmen, perubahan raw, atau numeric corruption pada field CSV yang tersedia. Bukti paling kuat menunjuk keterbatasan generalisasi respons terhadap keadaan discharge/riwayat baterai. Waktu, CurrTot, dan tegangan saling terkait; diagnosis ini belum memisahkan kontribusi kausal masing-masing atau penyebab fisik penurunan tegangan.", "",
        "## 1. Artefak, split, dan batas analisis", "",
        "Sumber baseline immutable: `artifacts/cross_validation/4bc978e2773687d3`; feature run `1225e8a76a8ce014`. Artefak tersedia sebelum reproduksi; seluruh source manifest, raw/config freeze dan prior final-audit evidence yang direferensikan diperiksa checksum-nya. Tidak ada pemanggilan fit, tuning, reselection, penghapusan Flight 66, atau perubahan label healthy.", "",
        "| Penggunaan | Model/fold | Train | Calibration | Evaluasi |", "|---|---|---|---|---|",
        "| Target Flight 66 | fold_00 | 70, 71, 73, 74 | 69 | 66 test, unseen model dan calibration |",
        "| Control utama Flight 70 | fold_02 | 66, 69, 73, 74 | 71 | 70 test, unseen model dan calibration |",
        "| Konteks dengan model yang sama | fold_00 yang sama persis | 70, 71, 73, 74 | 69 | 69 validation, calibration diagnostic |", "",
        "Flight 66 **tidak berada dalam training ataupun calibration model fold_00 yang digunakan untuk seluruh prediksi Flight 66**. Control 70 memakai fold historisnya sendiri, yang memang sudah memuat Flight 66 dalam training. Tidak ada fitting baru. Perbandingan 66 vs 70 memakai prosedur sama, tetapi bukan perbandingan dua flight dengan model/threshold/training domain identik. Flight 69 menyediakan konteks model fold_00 yang identik, dengan keterbatasan karena calibration telah melihatnya.", "",
        f"Control utama dipilih sebagai active test lain dengan FPR confirmed minimum, tie-break filename terkecil, sehingga terpilih Flight 70. Ini pemilihan contoh deskriptif setelah hasil historis diketahui. Threshold Flight 66 **{b0['threshold']['threshold_v']:.17g} V**, Flight 70 **{b70['threshold']['threshold_v']:.17g} V**; keduanya P99 healthy-validation historis, persistence **0,5 s**. Threshold tidak diubah.", "",
        "Ordered predictors tetap: `Curr`, `CurrTot`, `dI_dt`, `Curr_roll_mean_0p5s`, `Curr_roll_std_0p5s`, `Curr_roll_mean_1p0s`, `Curr_roll_std_1p0s`. Tidak ada scaler operasional. Kolom tambahan CSV hanya diagnosis. `Volt`, `V_ref`, elapsed time dan actual sag tidak dimasukkan ke prediktor HGB-D. `CurrTot` tetap unit sumber **unverified**, tanpa konversi atau klaim SOC/kapasitas.", "",
        "Fingerprint exact healthy-ready training rows/targets dan validation calibration scores cocok; semua sample Flight 66 dan control diprediksi ulang dari saved object. Maximum error prediksi terhadap CSV historis <1e-12 V; raw/confirmed/episode/state/timer serta exposure historis cocok. Detail: [audit verification](audit_verification.json), [input hashes](input_manifest.json), [diagnostic plan](diagnostic_plan.json).", "",
        "## 2. Actual sag, prediksi, dan residual", "",
        "Definisi yang dipakai tetap:", "",
        "```text", "actual_sag = V_ref - Volt", "signed_residual = actual_sag - HGB_prediction", "score = max(0, signed_residual)", "raw_alarm = reference_ready AND score > historical_threshold", "confirmed = continuous raw run elapsed >= 0.5 s", "```", "",
        table(summaries,["NamaFile","fold_id","evaluated_role","samples","raw_samples","confirmed_samples","confirmed_fpr","episodes","residual_mean_v","mae_v","rmse_v"]), "",
        "FPR di tabel adalah fraction; denominator hanya reference-ready samples pada flight tersebut. Warm-up tetap disimpan sebagai NOT_READY tanpa prediksi. Residual positif berarti prediksi sag terlalu rendah, setara dengan prediksi tegangan terlalu tinggi.", "",
        "![Flight 66: sag, residual, alarm, current, CurrTot dan voltage](plots/00000066_timeline.png)", "",
        "Timestamp `TimeUS` asli dan elapsed seconds tersedia dalam CSV. Sumbu elapsed mempertahankan gaps; garis diputus di gap dan area abu-abu adalah interval tanpa telemetry, bukan sinyal hasil interpolasi. Timeline memperlihatkan beberapa alarm awal sekitar 209-256 s, bias negatif pada sebagian bagian tengah, kemudian bias positif panjang pada bagian akhir. Jadi error bukan offset konstan sepanjang flight.", "",
        f"Error positif terbesar pada elapsed **{maximum.elapsed_time:.6f} s**, TimeUS **{int(maximum.TimeUS)}**: Volt {maximum.Volt:.6f} V, V_ref {maximum.V_ref:.6f} V, actual sag {maximum.voltage_sag_actual:.6f} V, predicted sag {maximum.reproduced_prediction_v:.6f} V, residual **{maximum.residual_signed_v:.6f} V**. Curr hanya {maximum.Curr:.9f} A, |dI/dt| {abs(maximum.dI_dt):.6f} A/s dan rolling std 1 s {maximum.Curr_roll_std_1p0s:.9f} A. Titik ini bukan lonjakan arus besar.", "",
        "## 3. CurrTot tinggi, level tegangan, dan perubahan arus", "",
        table(groups[groups.variable.isin(["CurrTot","Volt"])],["variable","group","samples","confirmed_samples","confirmed_fpr","residual_mean_v"]), "",
        "Quartile dibuat deskriptif dari seluruh ready samples. Ties pada CurrTot menyebabkan jumlah samples antar-bin tidak sama. Hubungan tidak monoton sepanjang flight: quartile CurrTot kedua memiliki FPR lebih rendah daripada quartile pertama, lalu FPR melonjak di bagian akhir. Jadi 'CurrTot tinggi selalu berarti alarm' tidak didukung.", "",
        table(groups[groups.variable.isin(["CurrTot_above_train_max","low_current_le_1A","transient_context","near_gap_after_1s"])],
              ["variable","group","samples","confirmed_samples","confirmed_fpr","residual_mean_v"]), "",
        "Cross-strata current/CurrTot: [CSV](currtot_current_cross_strata.csv). Korelasi Spearman terhadap signed residual:", "",
        table(corr[~corr['first'].isin(["residual_signed_v","anomaly_confirmed"])],["first","correlation","samples"]), "",
        "Korelasi ini adalah asosiasi within-flight, dengan autocorrelation dan temporal confounding; bukan feature importance atau estimasi causal effect. Korelasi antarsemua variabel dan within-current-bin juga disimpan di [correlations.csv](correlations.csv). Tegangan sendiri merupakan bagian dari definisi actual sag/residual, sehingga korelasinya terhadap residual tidak bisa menjadi bukti kausal independen. CurrTot dan elapsed time sama-sama mengikuti perkembangan rekaman. Tidak diberikan p-value yang menganggap ribuan samples sebagai observasi independen.", "",
        "![Hubungan residual dengan kondisi listrik](plots/flight66_residual_relationships.png)", "",
        "## 4. Domain training dan perilaku model pada batas CurrTot", "",
        table(domain,["variable","train_min","train_max","test_min","test_max","outside_train","confirmed_outside"]), "",
        "Volt dan actual sag pada tabel adalah response context, bukan predictors. Outside-range hanya uji marginal min/max, bukan diagnosis kerusakan atau detektor OOD baru. Keluar rentang tidak otomatis menjelaskan setiap error; berada dalam rentang juga tidak membuktikan kesamaan joint distribution atau conditional response.", "",
        f"CurrTot training fold_00 maksimum **{results['domain'][0]['CurrTot_training_max']:.6f}**, sedangkan Flight 66 sampai **{a.CurrTot.max():.6f}**. Ada **{len(high):,}/{len(a):,}** ready rows melampaui maksimum itu; kelompok tersebut menyumbang **{100*(high.residual_signed_v**2).sum()/(a.residual_signed_v**2).sum():.2f}%** total squared prediction error. Seluruh confirmed samples yang keluar rentang predictor adalah kasus CurrTot ini. Outlier marginal Curr/dI_dt/rolling std hanya beberapa rows dan tidak berisi confirmed samples.", "",
        f"Inspeksi bin mapper objek HGB tersimpan menunjukkan last CurrTot cut **{results['domain'][0]['CurrTot_last_bin_edge']:.9f}**, dengan 255 nonmissing bins. Seluruh 5.198 rows di atas maksimum training masuk **bin 254 yang sama**. Artinya penambahan CurrTot di wilayah ini tidak lagi menyediakan perbedaan nilai fitur kepada split trees. Prediksi masih bisa berubah mengikuti enam predictor lain, tetapi model tidak mendapat resolusi tambahan pada sumbu CurrTot di wilayah tersebut.", "",
        "Pada dua segmen low-current akhir, CurrTot tetap (1.408,260132 dan 1.489,035522), dan prediksi sag berada sekitar 1,496-1,514 V. Actual sag tetap lebih tinggi walaupun arus nyaris nol dan tegangan sedikit pulih selama segmen. Training low-current (<=1 A) memiliki actual sag maksimum 1,640516 V; contoh error maksimum Flight 66 mempunyai actual sag 1,951373 V. Ini bukti conditional response mismatch pada keadaan akhir rekaman, bukan semata outlier current.", "",
        "Sebanyak 224 confirmed samples masih berada di dalam rentang CurrTot training, terutama segmen 4. Karena itu ekstrapolasi CurrTot tidak menjelaskan seluruh 26 episode. Distribusi load dan hubungan load-to-sag juga berbeda antarpenerbangan. [Training context](fold00_training_context.csv) menyimpan statistik train per flight/current-state; residual training adalah **in-sample**, bukan unseen FPR.", "",
        "Nearest actual training row pada tujuh predictor juga disimpan per sample (flight/source_record/distance/target difference). Jarak memakai median/IQR training, fallback std untuk IQR kecil, **hanya untuk diagnosis**, tanpa mengubah input model. Tidak dipasang cutoff OOD atau ditafsirkan sebagai confidence probability. Identitas tetangga tidak berarti baterai/timestamp/kondisi fisiknya sepadan.", "",
        "![Distribusi training versus Flight 66](plots/flight66_training_feature_ecdf.png)", "",
        "## 5. Reference voltage dan kualitas data", "",
        f"V_ref Flight 66 = **{b0['reference_v']:.12f} V**, median **{b0['reference_low_current_samples']}** samples dengan Curr <=1 A dalam warm-up 2 s pertama. Ada **20** warm-up rows; lock pada elapsed **{b0['reference_lock_elapsed_s']:.6f} s**. Reference tidak dihitung dari masa depan dan nilainya tetap pada seluruh 15 segmen, sesuai strategi fixed warm-up yang juga dipakai training dan control.", "",
        "V_ref awal Flight 66 berada di dalam rentang V_ref training (15,709306-16,648390 V), sehingga initial reference outlier bukan penjelasan kuat. Fixed reference membuat penurunan level tegangan kumulatif tetap tercermin dalam actual sag setelah gap, termasuk saat current kembali kecil. Ini konsekuensi target historis yang harus dipelajari regressor, bukan kesalahan menghitung median atau salah menerapkan reset.", "",
        "Data CSV Flight 66 memiliki 9.526 rows, satu instance (Inst=0), tanpa nonfinite numeric, duplicate/backward timestamp, negative Volt/Curr/CurrTot, atau CurrTot decrease. Nilai raw yang dipakai features cocok exact dengan source CSV. Ini mendukung tidak adanya numeric/ordering corruption pada ekspor tersedia; audit ini tidak dapat memverifikasi sensor calibration, metadata pack, firmware, atau kebenaran kesehatan baterai dari .BIN asli yang tidak disertakan.", "",
        "Penurunan tegangan yang diamati belum boleh disebut sensor drift, kenaikan internal resistance, battery failure atau SOC ground truth. Healthy tetap **assumed_from_source**, sesuai permintaan dan kontrak dataset.", "",
        "## 6. Gap, segment boundaries, rolling dan persistence", "",
        f"Flight 66: **15 segments / 14 gaps >0,5 s**, total gap **{b0['excluded_gap_seconds']:.6f} s**. Elapsed span 1.824,678372 s, ready observed duration **{b0['observed_ready_seconds']:.6f} s**. Interval dengan gap/warm-up tidak masuk denominator exposure. Tidak dilakukan fill/interpolasi.", "",
        "Oracle independen menghitung ulang derivative actual dt, rolling `(t-window,t]` 0,5/1 s dengan ddof=0, count/coverage/readiness dan segment ID. Maximum feature difference sekitar 1,78e-15. Pada batas segmen derivative menjadi nol, rolling mulai satu sample, persistence run dimulai ulang. Locked V_ref dipertahankan sesuai protokol. Seluruh raw/confirmed/state/duration dan episode starts cocok dengan keluaran historis.", "",
        "Hanya **19/3.168** confirmed samples (0,60%) berada dalam satu detik pertama sesudah segment start, sehingga artefak rolling/persistence sesaat pada batas segmen bukan penyumbang utama. Empat episode berakhir saat gap dan timer benar-benar reset; setelah telemetry muncul lagi, alarm memerlukan >=0,5 s raw run baru. Tidak ada konfirmasi yang menumpuk durasi lintas gap. Gaps tetap membatasi pengetahuan tentang aktivitas/voltage recovery yang tidak terekam.", "",
        "Seluruh detail per-segment, gap_before, ready/confirmed counts, reference dan first-row timer/rolling count: [segments.csv](segments.csv).", "",
        "## 7. Semua confirmed false-alarm episodes Flight 66", "",
        "Durasi adalah last-confirmed minus first-confirmed timestamp. Ini observed span, tidak diekstrapolasi sampai sampel berikutnya atau melintasi gap. Satu confirmed sample memiliki span 0 s tetapi tetap didahului raw run >=0,5 s; raw start dan persistence elapsed disimpan terpisah. Total observed confirmed spans 314,199260 s, bukan 3.168 samples dikali interval nominal. Nilai current/voltage/residual di tabel dirata-ratakan pada confirmed samples.", "",
        table(ep,["episode_id","segment_id","confirmed_start_s","confirmed_span_seconds","confirmed_samples","Curr_mean","CurrTot_start","Volt_mean","residual_mean_v","end_reason"]), "",
        "CSV lengkap memuat absolute TimeUS, raw-start/confirmed-start/end, current/voltage min-mean-max, CurrTot start/end, actual/predicted sag, residual, transient context, low-current counts dan ending reason: [false_alarm_episodes.csv](false_alarm_episodes.csv). Hanya Flight 66 ada pada tabel di atas; CSV juga memuat dua calibration episodes Flight 69 yang dilabel dengan flight ID, sementara control 70 tidak mempunyai episode.", "",
        "![Contoh episode awal dan bias panjang](plots/flight66_episode_examples.png)", "",
        "## 8. Bias lama versus load transient", "",
        f"Deskripsi transient ditetapkan dari fold_00 training: |dI/dt| > **{results['diagnostic_definitions'][0]['abs_dI_dt_training_p95']:.6f} A/s** ATAU rolling std current 1 s > **{results['diagnostic_definitions'][0]['current_std_1s_training_p95']:.6f} A** (masing-masing P95). Ini label konteks untuk analisis, tidak mempengaruhi detector. Sebanyak **{quiet:,}/{confirmed:,} ({100*quiet/confirmed:.2f}%)** confirmed samples berada di luar konteks transient tersebut; **{low:,}/{confirmed:,} ({100*low/confirmed:.2f}%)** pada Curr <=1 A.", "",
        f"Untuk menghindari klaim hanya dari instant derivative, dihitung elapsed consecutive low-current run per segment. Setelah current <=1 A bertahan >=10 s tanpa konteks transient, masih ada **{int((stable&ca).sum()):,} confirmed samples** dari {int(stable.sum()):,} samples kondisi itu. Sensitivitas definisi 5/10/30 s tersedia pada condition_strata.csv; ini pemeriksaan deskriptif, bukan pemilihan persistence baru.", "",
        f"Pada fixed 10-s blocks per segmen yang mempunyai observed span >=5 s, **{int(windows.median_residual_above_threshold.sum())}/{len(windows)}** blocks mempunyai median residual di atas threshold; **{int((windows.median_residual_above_threshold&(windows.transient_fraction==0)).sum())}** di antaranya tanpa sample berkonteks transient. Lima episode >30 s menyumbang {int(long.confirmed_samples.sum()):,} confirmed samples. Episode E25 selama 112,599917 s pada current hampir nol merupakan bukti langsung persistent bias.", "",
        "Load changes tetap berkaitan dengan sebagian error awal dan sesaat. Sebanyak 224 confirmed samples pada segmen 4 muncul saat CurrTot masih dalam rentang training, mayoritas saat arus sekitar 11-13 A. Tidak semua error sementara tersebut melewati batas P95 transient; kedekatan residual dengan threshold juga membuat run terpecah menjadi beberapa episode. Karena itu tidak tepat menyamakan semua short episodes dengan load-transient error atau menyatakan semua alarm disebabkan satu fitur.", "",
        "## 9. Perbandingan dengan active flight ber-FPR rendah", "",
        "Flight 70 memakai prosedur yang sama: raw-to-reference audit, original seven predictors, own historical model, validation-only P99, persistence 0,5 s, independent timer/rolling checks, domain comparison terhadap own training set, dan episode extraction. Hasilnya 0 raw/confirmed alarm dari 1.643 ready samples / 164,100118 ready seconds. V_ref 15,993141 V; dua segmen/satu gap. CurrTot maksimum hanya 223,967438 dan minimum voltage 15,317132 V, jauh dari bagian akhir Flight 66.", "",
        "Mean signed residual Flight 70 **-0,079212 V**: model cenderung memprediksi sag lebih besar, sehingga score positif jarang muncul. Nol FPR tidak berarti regressor bebas bias atau sensitivitas fault sudah ideal. Control mempunyai durasi/domain dan fitted model berbeda; hasil ini mendukung adanya perbedaan generalisasi respons, tetapi tidak mengisolasi satu penyebab.", "",
        "Flight 69 diuji ulang dengan model fold_00 yang sama persis dengan Flight 66, tanpa Flight 66 dalam train/calibration model tersebut. Hasilnya 17 raw samples, tiga confirmed samples, dua episode dari 1.622 ready samples (0,184957% confirmed FPR). Mean residual -0,061415 V; CurrTot maksimum 403,116760. Ini menunjukkan threshold yang sama tidak menghasilkan FPR besar pada domain calibration tersebut. Karena Flight 69 menentukan threshold, hasilnya hanya calibration diagnostic, bukan independent test evidence.", "",
        "![Control Flight 70](plots/00000070_timeline.png)", "",
        "[Calibration Flight 69 timeline](plots/00000069_timeline.png) dan [grafik perbandingan ketiga flight](plots/flight66_vs_control.png).", "",
        "## 10. Temuan terbukti, hipotesis, dan penilaian akhir", "",
        "| Status bukti | Kesimpulan |", "|---|---|",
        "| Terbukti dari reproduksi | FPR tinggi berasal dari residual sag positif yang bertahan melewati threshold/persistence historis, dengan prediksi dan decisions identik sumber. |",
        "| Terbukti dari objek model | Semua CurrTot di atas maksimum training digabung ke uppermost bin yang sama; sumbu CurrTot tidak membedakan perkembangan lebih lanjut di sana. |",
        "| Terbukti dari stratifikasi | 92,93% confirmed samples dan 86,12% total squared prediction error berada pada domain CurrTot di atas maksimum training; bias panjang berarus rendah dominan. |",
        "| Terbukti dari audit | Fixed V_ref, features, segment/gap dan timer sesuai protokol; tidak ditemukan numeric/ordering corruption CSV atau carry-over persistence lintas gap. |",
        "| Didukung kuat, belum causal decomposition | Keterbatasan representasi/generalisasi pada respons akhir discharge dan reference-relative voltage level merupakan penjelasan ML utama. |",
        "| Belum terbukti | CurrTot sebagai penyebab tunggal; pengaruh terpisah dari waktu/voltage/SOC, perbedaan pack/capacity/temperature/aging, calibration sensor, session continuity dan penyebab fisik perubahan voltage. |", "",
        "**Penilaian kualitas data versus generalisasi:** kualitas numeric/ordering ekspor yang tersedia tidak menjelaskan alarm. Gaps cukup besar dan metadata kesehatan/pack belum tersedia, sehingga kualitas observasi fisik belum sepenuhnya terjamin. Dengan batas tersebut, hasil diagnosis lebih kuat menunjukkan masalah generalisasi/conditional bias model pada bagian akhir Flight 66 dibanding bug preprocessing atau reset. Healthy tidak diubah dan kerusakan baterai tidak disimpulkan.", "",
        "**Faktor terbesar:** kondisi CurrTot tinggi bersama level tegangan rendah dan underprediction yang berlangsung lama. Batas bin CurrTot memberi mekanisme representasi yang konkret. Load transient adalah faktor sekunder dan tidak menjelaskan episode terpanjang/error maksimum. 'Voltage drift' di sini hanya perubahan level tegangan teramati terhadap initial reference; tidak identik dengan sensor drift. Korelasi tidak cukup untuk memberi persentase kontribusi kausal CurrTot versus voltage history.", "",
        "## 11. Rekomendasi refinement berikutnya, menurut kekuatan bukti", "",
        "1. **Perluas data development healthy pada akhir discharge dan low-current setelah load**, lintas pack/flight/session dengan metadata kondisi yang terukur. Bukti: support CurrTot training berhenti sebelum >setengah ready Flight 66, dan long low-current bias dominan. Tentukan development/validation baru dan holdout flight/pack yang belum diperiksa sebelum mencoba konfigurasi baru.",
        "2. **Tinjau representasi keadaan baterai dan target fixed-reference sebagai studi terpisah**, termasuk state/history yang bisa tersedia secara causal dan perbedaan initial condition. Bukti: sag low-current tinggi tetap muncul saat short-window current hampir sama; fixed V_ref mengekspresikan perubahan level kumulatif yang tidak sepenuhnya dijelaskan predictor sekarang. Ini rekomendasi penelitian, bukan mengganti reference/fit pada diagnosis ini.",
        "3. **Verifikasi unit/reset/continuity CurrTot dan metadata pack/session**, lalu rancang penanganan domain di luar support sebagai eksperimen yang diprotocol-kan. Bukti: upper-bin saturation langsung teramati. Jangan langsung menafsirkan CurrTot sebagai SOC atau menganggap clipping/normalization pasti menghilangkan bias.",
        "4. **Evaluasi calibration transfer dan uncertainty/OOD monitoring lintas kondisi** setelah representasi/data memadai. Bukti: residual fold00 validation berpusat negatif sementara late Flight 66 positif panjang. Menaikkan threshold atau memperpanjang persistence dengan Flight 66 dapat menyembunyikan gejala dan merusak sensitivity; manfaatnya harus diuji pada development baru dan fault/healthy denominators tetap.",
        "5. **Teliti mismatch respons saat load dan recovery sebagai prioritas berikutnya**, terutama 224 alarms di rentang CurrTot training. Bukti untuk kebutuhan ini ada, tetapi transient correction saja tidak menjangkau sebagian besar confirmed samples yang stabil dan panjang.", "",
        "Semua rekomendasi belum diimplementasikan. Seluruh Flight 66 kini jelas merupakan data diagnosis/development yang telah diperiksa; memakai hasilnya untuk memilih konfigurasi berikutnya lalu menyebut Flight 66 independent holdout tidak sah. Tidak ada ground truth kerusakan internal pada data ini.", "",
        "## Artefak dan reproduksi", "",
        "- [Prediksi/residual semua 9.526 samples Flight 66](00000066_predictions_residuals.csv), termasuk 20 warm-up rows dengan prediction null.",
        "- [Control Flight 70 per sample](00000070_predictions_residuals.csv); [Flight 69 calibration per sample](00000069_predictions_residuals.csv).",
        "- [Seluruh episodes](false_alarm_episodes.csv), [segment/gap audit](segments.csv), [feature distributions](feature_distribution.csv), [training context](fold00_training_context.csv).",
        "- [Condition strata](condition_strata.csv), [current/CurrTot cross-strata](currtot_current_cross_strata.csv), [10-s bias blocks](bias_windows_10s.csv), [correlations](correlations.csv).",
        "- [Summary CSV](flight_summary.csv), [machine-readable results](results.json), [verification](audit_verification.json), [input hashes](input_manifest.json), [evidence hashes](evidence_manifest.json).", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts/diagnose_flight66.py", "```", "",
        f"Script memuat saved models dan melakukan predict tanpa fit. Seluruh **{audit['historical_files_verified_and_preserved']} prior files** pada preservation scope dicocokkan sebelum/sesudah. Pengulangan hanya meregenerasi artefak diagnosis pada folder ini. Grafik diagnostik diperiksa visual untuk label, gap breaks, domain dan episode context.", "",
        "Exposure group pada condition_strata hanya statistik deskriptif dengan sample FPR; episode starts dalam strata berarti lokasi onset episode, bukan episode-rate subgroup yang didefinisikan ulang. Main episode rates/denominator historis tetap pada summary sumber. Tidak ada inferensi independensi antar-sample atau klaim causal feature importance.", "",
        "CSV per sample mempertahankan NamaFile/Inst/TimeUS/source_record, segment dan readiness asli, seluruh tujuh predictors, V_ref/Volt/CurrTot, actual/predicted sag, reproduced_prediction_v, predicted_voltage_v, signed residual, score/threshold, anomaly_raw/duration/confirmed/state dan episode ID. `VoltageSag_actual`/`VoltageSag_predicted` adalah alias yang diminta untuk kolom canonical proyek; nilainya identik. Diagnostic nearest-training/bins/range flags/transient/low-current-run bukan input atau keputusan tambahan. Semua timing telemetry dalam mikrodetik atau detik sesuai suffix; tegangan/residual dalam V, Curr dalam A, dI_dt dalam A/s, dan CurrTot source units unverified.", ""]
    report="\n".join(lines)
    for old,new in (("3,168","3.168"),("2,944","2.944"),("2,726","2.726"),("5,198","5.198"),("9,506","9.506"),("3,032","3.032"),("2,289","2.289"),("1,604","1.604"),("2,965","2.965")):
        report=report.replace(old,new)
    (OUT/"ROOT_CAUSE_ANALYSIS.md").write_text(report,encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    inputs=verify_inputs();write_json("input_manifest.json",inputs)
    metrics=read(CV/"healthy_active_per_fold.csv")
    candidates=metrics[metrics.test_flight!=TARGET].sort_values(["confirmed_fpr","test_flight"],kind="stable")
    control=candidates.iloc[0];control_flight=control.test_flight
    write_json("diagnostic_plan.json",dict(date="2026-10-22",scope="post_hoc_diagnosis_only_no_refinement",target=TARGET,
        control=control_flight,control_rule="minimum historical confirmed FPR among other active held-out tests; filename ascending tie-break; descriptive control selected using already observed results",
        supplementary_same_model_flight="00000069.BIN, original fold00 validation; explicitly calibration diagnostic, not independent test",
        no_fit_tuning_or_policy_changes=True,ready_only_rates=True,predictors_unchanged=True,
        episode_duration="last confirmed minus first confirmed timestamp, no extension across gaps; raw-to-confirm span separate",
        transient_rule="training-only P95 |dI/dt| OR P95 1s current std for descriptive context only",
        bias_rule="fixed nonoverlapping 10s blocks per segment; no threshold or model selection",healthy_status="assumed_from_source"))
    raw=read(ROOT/"data/raw/dataset_baterai.csv");features=read(FEATURES/"battery_features.csv")
    frames=[];trains=[];bundles=[];audits=[];segment_rows=[];domain_rows=[];domains=[]
    groups=[];cross=[];windows=[];correlations=[];definitions=[];episodes=[];flight_summaries=[]
    for flight,fold,role in ((TARGET,"fold_00","test"),(control_flight,control.fold_id,"test"),("00000069.BIN","fold_00","validation")):
        df,train,bundle,audit,segments=reproduce(flight,fold,features,raw,role=role)
        dr,di=domain_analysis(df,train,bundle)
        di.update(NamaFile=flight,fold_id=fold,evaluated_role=role)
        gr,cr,wi,co,de=grouped_analysis(df,train)
        ep=episode_tables(df)
        summary=summarize(df)
        if role=="test":
            record=metrics[metrics.test_flight==flight].iloc[0]
            require(summary["confirmed_samples"]==record.confirmed_alarm_samples and summary["raw_samples"]==record.raw_alarm_samples and len(ep)==record.confirmed_episodes,"Historical alarm totals differ")
            require(abs(summary["ready_seconds"]-record.observed_ready_seconds)<1e-9,"Historical exposure differs")
        summary.update(NamaFile=flight,fold_id=fold,evaluated_role=role,threshold_v=bundle["threshold"]["threshold_v"],reference_v=audit["reference_v"],
            training_flights=bundle["split"]["train"],validation_flights=bundle["split"]["validation"],segments=audit["segments"],gaps=audit["gaps"])
        frames.append(df);trains.append(train);bundles.append(bundle);audits.append(audit);segment_rows+=segments;domain_rows+=dr;domains.append(di)
        groups+=gr;cross+=cr;windows+=wi;correlations+=co;definitions.append(de);episodes+=ep;flight_summaries.append(summary)
        csv(f"{flight.split('.')[0]}_predictions_residuals.csv",df)
        print(json.dumps(dict(flight=flight,status="REPRODUCED",confirmed=summary["confirmed_samples"],episodes=len(ep))),flush=True)
    # Additional calibration context under the exact Flight66 model, NOT an independent control.
    val=read(CV/"folds/fold_00/validation_scores.csv.gz")
    validation_summary=dict(flight="00000069.BIN",role="Flight66 fold00 calibration diagnostic, not independent test",
        ready_samples=int(val.reference_ready.sum()),confirmed_samples=int(val.anomaly_confirmed.sum()),
        raw_samples=int(val.anomaly_raw.sum()),residual_mean_v=float(val.residual_signed_v.mean()),
        residual_q99_v=float(val.loc[val.reference_ready,"residual_signed_v"].quantile(.99)))
    csv("false_alarm_episodes.csv",episodes);csv("segments.csv",segment_rows);csv("feature_distribution.csv",domain_rows)
    csv("condition_strata.csv",groups);csv("currtot_current_cross_strata.csv",cross);csv("bias_windows_10s.csv",windows);csv("correlations.csv",correlations)
    csv("flight_summary.csv",[{k:v for k,v in r.items() if not isinstance(v,list)} for r in flight_summaries])
    # Regressor behavior on original source training rows, explicitly in-sample.
    train=trains[0].copy();b=bundles[0]
    with threadpool_limits(limits=1,user_api="openmp"):train["prediction_v"]=b["model"].predict(train[b["predictors"]])
    train["residual_v"]=train.voltage_sag_actual-train.prediction_v
    tr_rows=[]
    for (flight,low),block in train.groupby([train.NamaFile,train.Curr<=1]):
        tr_rows.append(dict(NamaFile=flight,role="fold00_train_in_sample_not_unseen_FPR",low_current_le_1A=bool(low),samples=len(block),
            CurrTot_min=float(block.CurrTot.min()),CurrTot_max=float(block.CurrTot.max()),Volt_min=float(block.Volt.min()),Volt_max=float(block.Volt.max()),
            V_ref=float(block.V_ref.iloc[0]),actual_sag_min=float(block.voltage_sag_actual.min()),actual_sag_max=float(block.voltage_sag_actual.max()),
            residual_mean_v=float(block.residual_v.mean()),residual_q95_v=float(block.residual_v.quantile(.95))))
    csv("fold00_training_context.csv",tr_rows)
    figures(frames,trains,bundles,episodes)
    for name,digest in inputs.items():require(sha(ROOT/name)==digest,f"Input modified: {name}")
    verification=dict(status="PASS",historical_files_verified_and_preserved=len(inputs),no_fit_performed=True,
        original_model_features_config_threshold_persistence_preserved=True,flight66_not_in_target_fold_training_or_calibration=True,
        historical_counts_exposure_predictions_timers_exact=True,audits=audits)
    write_json("audit_verification.json",verification)
    results=dict(scope="post_hoc_diagnosis_historical_healthy_assumed",flights=flight_summaries,
        domain=domains,diagnostic_definitions=definitions,calibration_context=validation_summary,
        all_historical_active_metrics=metrics.to_dict("records"),training_domain=tr_rows,
        source_cv_run=CV.name,source_feature_run=FEATURES.name,physical_fault_causal_attribution_established=False,
        software_false_alarm_mechanism_reproduced=True)
    write_json("results.json",results)
    render_report(frames,results,verification,episodes)
    write_json("evidence_manifest.json",{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!="evidence_manifest.json"}
        | {"scripts/diagnose_flight66.py":sha(Path(__file__))})
    print(json.dumps(dict(status="DIAGNOSTIC_ANALYSIS_COMPLETE",output=str(OUT),files=len(list(OUT.rglob('*'))))),flush=True)


if __name__=="__main__":
    main()
