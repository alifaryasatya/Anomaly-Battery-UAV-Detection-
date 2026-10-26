# Consolidated research evaluation - Sprint 11

Run `4ad7e153513e7d64`. Scope: `historical_development_research_consolidation`. Status: `RESEARCH_CONSOLIDATION_ONLY`.

## Main findings

Frozen HGB Battery-D/P99/0.5 s baseline: healthy active FPR 11.827317%, 29 episodes over 2683.400543 ready seconds (0.648431/minute). Synthetic incidents detected 156/162, sample F1 0.567735; detected-only median delay 0.699965 s, n=156.

Dual-path result remains **NOT_PROMOTED**. The orchestration does not select a winner from these outer results or change the operational configuration. Completion of this software/reporting stage does not imply that research targets, hardware readiness or the detector promotion gate passed.

## Dataset, audit and causal processing

Raw 32338 rows / 17 flights. Preprocessing retains 32338, drops 0, forms 92 segments with 75 gaps. Current-based labels: ['00000066.BIN', '00000069.BIN', '00000070.BIN', '00000071.BIN', '00000073.BIN', '00000074.BIN'] ACTIVE_LOAD and ['00000061.BIN', '00000062.BIN', '00000063.BIN', '00000064.BIN', '00000065.BIN', '00000067.BIN', '00000068.BIN', '00000072.BIN', '00000075.BIN', '00000076.BIN', '00000077.BIN'] LOW_LOAD_OR_IDLE. Observed duration 3224.801403 s; excluded gaps 2138.817494 s. Features: 31989 ready, 349 warm-up, 0 reference unavailable samples.

Source status healthy is assumed_from_source. CurrTot remains unverified source units without conversion. Raw battery CSV contains only flight/time/instance/Volt/Curr/CurrTot; no throttle or V1 engine/model/threshold is used. Fixed causal warm-up locks V_ref or remains unavailable; actual-dt derivatives and trailing rolling features reset at gaps. Warm-up/readiness/IDs/actual voltage/sag/residual are excluded from regression predictors. Battery-E elapsed_time is ablation only.

## Orchestration and source provenance

10 source stages, 4532 verified outputs and 103 saved-model references. Source run IDs, manifest/summary hashes, pre-outer freeze counts and model bundle hashes are in source_inventory.json/csv and model_bundle_registry.csv. Model weights and training/calibration provenance remain in their immutable source directories; the orchestrator does not copy or refit a historical model merely because current code changed. Current dependency versions and source configurations/versions remain separately traceable.

Stages built at creation: []; training stages built at creation: []. Existing results are verified before reuse. Missing stages use their own fixed engine budgets; corrupt/incompatible results fail without silent rebuild. A completed source catalog supports reproducible repetition without overwriting historical sprint pointers.

The sequence is audit -> segmentation/reference/features -> pilot/fault bank -> reference LOAFO -> fixed ablation/comparison -> validation tuning/dual-path -> low-load -> consolidated grouping/plots/report. Within each modelling experiment, ALL candidate-fold policies/models and selection inputs are frozen before that experiment's first outer evaluation. These are sequential historical development experiments; this report is a retrospective consolidation, not proof of a globally untouched prospective final holdout. Outer outcomes never revise source candidates/policies in this stage.

## Reference baseline

Regression pooled: MAE 0.101865 V; RMSE 0.138278 V; R2 0.946481. Healthy raw alarms 3607/26870; confirmed 3178/26870. Healthy FPR uses ready samples; episode rate uses actual monitored ready intervals, excluding warm-up/gaps and requiring both endpoints ready in one stream/segment.

Synthetic planned 162; applicable 162; NOT_APPLICABLE 0; undetected 6; preexisting-alarm contaminated 0. Incident success requires a fresh confirmation inside fault support. A carried warning at onset is not an incident success. Deadline denominators retain every applicable miss/contamination; NOT_APPLICABLE has null metrics and is excluded. Conventional sample confusion/F1 is distinct from incident recall; repeated synthetic samples are not unique healthy exposure.

| Deadline s | Successes | Applicable denominator | Rate % |
|---:|---:|---:|---:|
| 1 | 101 | 162 | 62.345679 |
| 2 | 102 | 162 | 62.962963 |
| 5 | 115 | 162 | 70.987654 |
| 10 | 129 | 162 | 79.629630 |
| 30 | 149 | 162 | 91.975309 |

Research target checks are preserved from the original baseline:

- abrupt_severe: MET; measured 100.000000%, target 80.000000%, applicable n=36.
- gradual_observable: NOT_MET; measured 23.529412%, target 70.000000%, applicable n=51.

## Fixed ablation and model comparison

HGB/RF Battery-A/B/C/D/E and voltage-behavior IF/OCSVM use matched flight folds/readiness/fault definitions. HGB/RF score is positive sag residual in V; direct IF/OCSVM use signed -decision_function in model units. Only OCSVM has a healthy-training-fitted scaler and deterministic capped train sample. Exact ordered predictors are recorded per bundle. No automatic winner is selected; Battery-E remains ablation only.

| Experiment | Candidate | Healthy FPR % | Episodes/min | Incident detected/applicable | Sample F1 | Regression RMSE V |
|---|---|---:|---:|---:|---:|---:|
| ablation | hgb_battery_a | 26.702642 | 1.788775 | 51/162 | 0.246763 | 0.790650 |
| ablation | hgb_battery_b | 12.132490 | 0.782589 | 156/162 | 0.556668 | 0.138216 |
| ablation | hgb_battery_c | 11.600298 | 0.715510 | 156/162 | 0.564540 | 0.137071 |
| ablation | hgb_battery_d | 11.827317 | 0.648431 | 156/162 | 0.567735 | 0.138278 |
| ablation | hgb_battery_e | 21.622627 | 0.603712 | 117/162 | 0.439440 | 0.208748 |
| ablation | rf_battery_a | 23.289914 | 2.168890 | 51/162 | 0.246725 | 0.807488 |
| ablation | rf_battery_b | 10.930406 | 0.581352 | 156/162 | 0.574067 | 0.152153 |
| ablation | rf_battery_c | 11.213249 | 0.603712 | 156/162 | 0.569863 | 0.153606 |
| ablation | rf_battery_d | 11.254187 | 0.894388 | 156/162 | 0.577312 | 0.150129 |
| ablation | rf_battery_e | 23.074060 | 0.737870 | 108/162 | 0.397812 | 0.315753 |
| comparison | hgb_battery_d | 11.827317 | 0.648431 | 156/162 | 0.567735 | 0.138278 |
| comparison | rf_battery_d | 11.254187 | 0.894388 | 156/162 | 0.577312 | 0.150129 |
| comparison | if_direct | 0.000000 | 0.000000 | 22/162 | 0.002038 | undefined |
| comparison | ocsvm_direct | 0.059546 | 0.089439 | 80/162 | 0.266648 | undefined |

## Validation-only tuning and paired dual-path

Threshold quantiles use only healthy assigned validation; synthetic validation is secondary refinement and never calibrates quantiles or enters model fitting. Per-fold selected policies, exact source run/freeze records and unchanged reference results are retained. Outer-selected rollbacks, winners or threshold increases are not performed.

| Experiment / policy | Healthy FPR % | Episodes/min | Detected/applicable | F1 | Mean detected delay s | Median s | Detected n |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-path tuning / reference | 11.827317 | 0.648431 | 156/162 | 0.567735 | 5.526932 | 0.699965 | 156 |
| single-path tuning / selected | 11.883141 | 0.737870 | 156/162 | 0.568402 | 5.232046 | 0.600039 | 156 |
| dual-path / reference | 11.827317 | 0.648431 | 156/162 | 0.567735 | 5.526932 | 0.699965 | 156 |
| dual-path / selected | 9.151470 | 0.536633 | 162/162 | 0.613357 | 7.264810 | 0.599806 | 162 |

The dual-path promotion gate uses predeclared paired arithmetic macro means with all fold values defined, not the pooled values in the table. Different detected n makes detected-only delay comparisons conditional on different subsets; new late detections do not erase deadline failures.

| Gate | Reference macro | Selected macro | Passed |
|---|---:|---:|---|
| healthy_fpr | 0.056404 | 0.044224 | True |
| episodes_per_minute | 0.408884 | 0.407395 | True |
| f1 | 0.796940 | 0.802624 | True |
| deadline_1s | 0.623457 | 0.611111 | False |
| deadline_2s | 0.629630 | 0.623457 | False |
| deadline_5s | 0.709877 | 0.722222 | True |
| deadline_10s | 0.796296 | 0.783951 | False |
| deadline_30s | 0.919753 | 0.907407 | False |
| mean_incident_delay | 5.656974 | 7.264810 | False |
| refined_dual_candidate_exists | undefined | undefined | True |

## Healthy low-load domain

11 unique LOW_LOAD_OR_IDLE flights are explicitly unseen by the evaluating regressor and detector calibration. Low current does not prove ground operation. Primary fold fold_00 was fixed before prediction; other models are sensitivity on the same cohort, with no cross-model exposure pooling or winner selection.

| Fold | Ready samples | Ready seconds | Raw alarms | Confirmed alarms | Confirmed FPR % | Episodes | Episodes/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold_00 | 5119 | 506.599454 | 0 | 0 | 0.000000 | 0 | 0.000000 |
| fold_01 | 5119 | 506.599454 | 330 | 150 | 2.930260 | 3 | 0.355310 |
| fold_02 | 5119 | 506.599454 | 15 | 4 | 0.078140 | 1 | 0.118437 |
| fold_03 | 5119 | 506.599454 | 0 | 0 | 0.000000 | 0 | 0.000000 |
| fold_04 | 5119 | 506.599454 | 0 | 0 | 0.000000 | 0 | 0.000000 |
| fold_05 | 5119 | 506.599454 | 0 | 0 | 0.000000 | 0 | 0.000000 |

The low-load source report preserves initial voltage, marginal input ranges, response/residual distributions, per-flight membership and actual false-alarm Curr/dI_dt/residual context. A primary no-alarm context is explicitly distinguished from a sensitivity false-alarm illustration added after results. Range overlap/exceedance is descriptive, not an extra alarm or a physical-cause diagnosis. Thresholds remain frozen; low-load has no synthetic recall/F1/delay claims.

## Consolidated groups, failures and checks

synthetic_group_metrics.csv has 361 rows across 19 separate experimental series, grouped by overall, flight, family, severity, onset position and observability. Each series total is recomputed from source scenario CSV and reconciled to its JSON summary. Do not pool alternative candidates/policies as independent incidents. failed_or_not_applicable_cases.csv retains 615 missed/contaminated/NA case records, including explicit reasons and null delays. Missing results are not fabricated zeros.

| Check | Status | Detail |
|---|---|---|
| audit_and_source_integrity | PASS | Frozen raw/config/protocol and all source output hashes verified |
| feature_and_reference_lineage | PASS | Same preprocessing/features, active flight assignments and reference baseline |
| validation_before_outer | PASS | All five outer experiments have verified immutable pre-outer freeze records |
| recomputed_scenario_totals | PASS | 19 experimental series reconcile to their source summaries |
| low_load_unseen_and_exposure | PASS | Explicit source membership; unique cohort exposure retained separately per model |
| dual_promotion | FAIL | NOT_PROMOTED |
| independent_final_holdout | WARNING | Historical development data; no untouched final holdout |
| currtot_units_and_healthy_labels | WARNING | CurrTot unverified; healthy assumed_from_source |
| replay_and_hardware | NOT_EVALUATED | Sprint 12 replay and hardware readiness not established |
| baseline_target_abrupt_severe | PASS | rate=1.0; target=0.8; applicable=36 |
| baseline_target_gradual_observable | FAIL | rate=0.23529411764705882; target=0.7; applicable=51 |

## Source runs

| Stage | Run | Verified outputs | Model bundles | Frozen pre-outer files |
|---|---|---:|---:|---:|
| preprocess | e22890aeb6ffddf6 | 14 | 0 | 0 |
| features | 1225e8a76a8ce014 | 7 | 0 | 0 |
| baseline | fcb19424861476bd | 14 | 1 | 0 |
| inject-faults | 437104037cb0fea5 | 270 | 0 | 0 |
| cross-validate | 4bc978e2773687d3 | 217 | 6 | 18 |
| ablation | 9ae785a0f10298bc | 1997 | 60 | 180 |
| compare-models | 03ebe58933705e3e | 809 | 24 | 72 |
| tune | d4351f20af42a44c | 554 | 6 | 206 |
| tune-dual-path | ad82c3e229bf6510 | 566 | 6 | 215 |
| evaluate-low-load | 56b20c4b7936b122 | 84 | 0 | 0 |

## Scientific limits and remaining work

All seventeen flights are historical development data and have been examined in prior research. Healthy labels are assumed, CurrTot units unverified, and original firmware/pack/BIN metadata and independently labelled real faults are absent. Synthetic applicability and observability are controlled experimental properties, not measured internal battery failures. No independent final holdout, deployment validation, emergency-readiness claim or physical internal-failure diagnosis follows from this package.

Sprint 12 still needs streaming replay/equivalence and latency checks. Sprint 13 final audit is not completed by this report. Baseline software may be replayed without claiming emergency targets or promoting the failed dual-path candidate. No further candidate refinement is justified as independent final validation on these same outer flights.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
