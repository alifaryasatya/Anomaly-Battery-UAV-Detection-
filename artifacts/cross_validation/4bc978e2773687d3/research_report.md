# Sprint 6 — HGB Battery-D LOAFO baseline

Scope: **historical development cross-validation**. These historical flights were previously examined in V1; this is not a new independent final holdout.
Run `4bc978e2773687d3`; 6 outer folds; active/reference-ready flight IDs: 00000066.BIN, 00000069.BIN, 00000070.BIN, 00000071.BIN, 00000073.BIN, 00000074.BIN.

## Frozen procedure

Sorted flight i is outer test; cyclic next flight is validation; remaining whole flights train. All models/thresholds were fit, calibrated and saved before ANY outer prediction. HGB uses Battery-D, fixed config/seed, no scaler, no synthetic fitting and no internal random row early-stopping split. Healthy training includes low-current periods; reference-unready rows are excluded. Percentile 99 linear healthy-validation threshold and 0.5 s persistence are unchanged across the reference experiment. No candidate is revised after outer results.
Outer healthy active results below are unseen by their evaluating regressor and unused for that fold's calibration. Validation calibration diagnostics remain in a separate CSV. Low-load evaluation is deferred to Stage 10 and is not inferred from these results.

## Outer results per fold

| Fold | Test | RMSE V | Healthy confirmed FPR | Healthy episodes/min | Sample F1 | Incident detected/applicable | Contaminated | Median detected delay s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fold_00 | 00000066.BIN | 0.144403 | 0.333263 | 1.643663 | 0.314312 | 27/27 | 0 | 0.699954 |
| fold_01 | 00000069.BIN | 0.114124 | 0.004932 | 0.740741 | 0.932606 | 27/27 | 0 | 0.600331 |
| fold_02 | 00000070.BIN | 0.098991 | 0.000000 | 0.000000 | 0.935934 | 27/27 | 0 | 0.699888 |
| fold_03 | 00000071.BIN | 0.031524 | 0.000000 | 0.000000 | 0.950257 | 27/27 | 0 | 0.600039 |
| fold_04 | 00000073.BIN | 0.149698 | 0.000229 | 0.068902 | 0.868625 | 27/27 | 0 | 0.799989 |
| fold_05 | 00000074.BIN | 0.136582 | 0.000000 | 0.000000 | 0.779904 | 21/27 | 0 | 0.800144 |

## Pooled counts and macro statistics

Untouched outer healthy: 26870 ready samples, 3178 confirmed alarm samples, FPR 0.118273; 29 episodes / 2683.400543 s = 0.648431 episodes/min. These healthy counts include each outer flight once, without scenario replication.
Pooled regression: MAE 0.101865 V, RMSE 0.138278 V, R2 0.946481. Macro per-fold RMSE mean 0.112554, population std 0.040257 V.
Synthetic cases: 162 planned, 162 applicable, 0 NOT_APPLICABLE. Incident detected 156; undetected applicable 6 (including 0 PREEXISTING_ALARM). Incident recall 0.962963.
Conventional synthetic samples: TP=66440, FP=85257, TN=557877, FN=15916; precision 0.437978, recall 0.806741, F1 0.567735, sample FPR 0.132565. Macro per-fold F1 mean 0.796940. These expanded sample counts intentionally repeat source flights across distinct synthetic scenarios and are separate from untouched healthy FPR.
Delay on incident-detected cases only: n=156, mean 5.526932 s, median 0.699965 s, population std 10.035913 s. Undetected applicable cases remain failures in all deadline denominators.

| Deadline | Incident successes / applicable | Rate |
|---|---:|---:|
| deadline_le_1s | 101/162 | 0.623457 |
| deadline_le_2s | 102/162 | 0.629630 |
| deadline_le_5s | 115/162 | 0.709877 |
| deadline_le_10s | 129/162 | 0.796296 |
| deadline_le_30s | 149/162 | 0.919753 |

All scalar macro metrics include mean/std/median/min/max and n_defined in summary.csv/results.json. Pooled precision/recall/F1/FPR/deadlines are recomputed from counts, not averages of fold rates. Pooled delays use all incident-detected scenarios; macro delay statistics use defined per-fold delays. Undefined values remain null/blank with their denominators.

## Research targets (fixed before outer evaluation)

- abrupt_severe: rate 1.000000, target 0.800000, applicable=36, status=MET.
- gradual_observable: rate 0.235294, target 0.700000, applicable=51, status=NOT_MET.

Abrupt/severe is increased-resistance or sudden-drop, severe level, <=1 s. Observable gradual is progressive with MEDIUM/HIGH added-sag observability, <=5 s. These development target checks do not authorize deployment or hardware switching.

## Denominators, contamination and failed cases

Only reference-ready rows enter conventional confusion counts. NOT_APPLICABLE has no confusion counts or incident/recall/deadline eligibility and remains visible with its reason. Applicable undetected and PREEXISTING_ALARM stay in every deadline denominator as failures. Support includes aligned onset (possibly zero effect for ramp/progressive); onset and first-effect timestamps are both retained from the frozen scenario definition.
A confirmed warning carried into onset (onset confirmed without a new episode start) contaminates the entire incident. It is never an incident success, even if another episode begins later. Conventional TP samples are retained separately. A genuinely new confirmation exactly at onset can have delay zero. Detection after support ends does not count. Delay already includes persistence; it is not added again.
Healthy/pre-fault exposure uses observed intervals with both endpoints ready in the same stream/segment, excluding gaps and warm-up. Episode starts after gaps are new. Sample FPR and episodes/minute are distinct metrics. failed_cases.csv includes every undetected, contaminated and NOT_APPLICABLE case; no failure is dropped from the full fault table.

## Files and limits

Root CSVs: fold_assignments, regression_per_fold, healthy_active_per_fold, anomaly_per_fold, fault_per_scenario, failed_cases, validation_calibration_per_fold and summary. Each fold saves model bundle, training provenance, threshold/selected policy, calibration scores, untouched test scores and applicable scenario scores (deterministic CSV.gz). all_fold_policies_frozen.json records pre-outer hashes; manifest.json verifies the complete run. results.json contains pooled counts/macro distributions and research target checks. Plot examples include a fixed severe progressive trace and the first unsuccessful applicable case when one exists; they are diagnostic only.
Healthy labels are assumed, CurrTot remains unverified in source units, no real internal battery fault ground truth exists, and flight count/domain are limited. No feature/model selection, tuning, dual-path promotion, low-load false-alarm validation or end-to-end replay is claimed here. Status remains RESEARCH_BASELINE_ONLY; subsequent outer-informed refinement must be labelled exploratory/development selection.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
