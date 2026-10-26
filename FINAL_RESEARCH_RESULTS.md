# FINAL RESEARCH RESULTS

Audit akhir Sprint 13 - 21 Oktober 2026 (Asia/Jakarta).

**Pipeline penelitian dan audit software selesai. Detector belum divalidasi untuk deployment/emergency switching. Dual-path tetap NOT_PROMOTED.**

Sumber konsolidasi terkini: `4ad7e153513e7d64`; baseline frozen `4bc978e2773687d3`; replay `ee132dc3901fb80a`. Semua angka berasal dari verified CSV/JSON, tanpa memilih ulang kandidat dari outer.

## Data, fitur dan scope

32338 baris / 17 historical flights, 92 continuous segments dan 75 gaps. Enam ACTIVE_LOAD, sebelas LOW_LOAD_OR_IDLE berdasarkan arus; low-load tidak membuktikan UAV berada di ground. Ready 31989 samples, warm-up 349, reference unavailable 0. Observed telemetry 3224.801403 s, gaps 2138.817494 s dikeluarkan dari exposure.

Model utama HGB Battery-D memakai ordered predictors: Curr, CurrTot, dI_dt, Curr rolling mean/std 0,5 s dan 1,0 s. Target V_ref - Volt, score max(0, actual sag - prediction), healthy-validation P99 dan timestamp persistence 0,5 s. Actual Volt/sag/V_ref/dV_dt/residual/readiness/IDs bukan regression predictors. Battery-E elapsed_time hanya ablation. IF/OCSVM memakai ordered direct predictors voltage_sag_actual, Curr, CurrTot, dV_dt, dI_dt dan signed -decision_function; hanya OCSVM memakai scaler healthy-train-only dan deterministic capped train sampling.

Split whole-flight LOAFO: tiap active flight sekali test, berikutnya validation, sisanya train. Semua candidate-fold selection/freeze selesai sebelum outer masing-masing eksperimen. Seluruh 17 flights pernah diperiksa dalam penelitian historis: ini historical development, **bukan independent final holdout baru**. Healthy assumed_from_source; CurrTot unverified tanpa konversi; firmware/BIN/pack metadata dan measured internal-fault ground truth tidak tersedia.

## Baseline frozen

| Metrik pooled | Hasil |
|---|---:|
| MAE / RMSE V | 0.101865 / 0.138278 |
| R2 | 0.946481 |
| Healthy confirmed samples / ready | 3178/26870 |
| Healthy confirmed FPR | 11.827317% |
| Episodes / ready seconds / per minute | 29 / 2683.400543 / 0.648431 |
| Synthetic detected / applicable / missed / contaminated | 156/162 / 6 / 0 |
| Synthetic sample precision / recall / F1 | 0.437978 / 0.806741 / 0.567735 |
| Detected-only delay mean / median / std, n | 5.526932 / 0.699965 / 10.035913 s, n=156 |

| Deadline s | Success / applicable | Rate |
|---:|---:|---:|
| 1 | 101/162 | 62.345679% |
| 2 | 102/162 | 62.962963% |
| 5 | 115/162 | 70.987654% |
| 10 | 129/162 | 79.629630% |
| 30 | 149/162 | 91.975309% |

Macro statistics dipisahkan dari pooled:

| Metrik | Mean | Population std | Median | Min | Max | Defined folds |
|---|---:|---:|---:|---:|---:|---:|
| regression/rmse_v | 0.112554 | 0.040257 | 0.125353 | 0.031524 | 0.149698 | 6 |
| healthy_active/confirmed_fpr | 0.056404 | 0.123828 | 0.000115 | 0.000000 | 0.333263 | 6 |
| healthy_active/confirmed_episodes_per_minute | 0.408884 | 0.612638 | 0.034451 | 0.000000 | 1.643663 | 6 |
| synthetic/f1 | 0.796940 | 0.223454 | 0.900616 | 0.314312 | 0.950257 | 6 |
| synthetic/delay_mean_seconds | 5.656974 | 3.272046 | 5.707445 | 1.988861 | 9.038064 | 6 |

Enam baseline misses tetap terlihat pada flight 74: mild increased-resistance dan mild progressive, masing-masing tiga onset positions. Delay missing tetap null. Abrupt severe <=1s memenuhi target; progressive observable <=5s belum memenuhi target. Flight 66 menyumbang 3.168 dari 3.178 confirmed healthy samples; FPR flight tersebut 33,326320%. FPR active pooled 11,827317% tetap dilaporkan tanpa menaikkan threshold untuk menyembunyikan alarm.

## Ablation dan pembanding tetap

| Experiment / candidate | Healthy FPR % | Episodes/min | Detected/applicable | F1 | RMSE V |
|---|---:|---:|---:|---:|---:|
| ablation/hgb_battery_a | 26.702642 | 1.788775 | 51/162 | 0.246763 | 0.790650 |
| ablation/hgb_battery_b | 12.132490 | 0.782589 | 156/162 | 0.556668 | 0.138216 |
| ablation/hgb_battery_c | 11.600298 | 0.715510 | 156/162 | 0.564540 | 0.137071 |
| ablation/hgb_battery_d | 11.827317 | 0.648431 | 156/162 | 0.567735 | 0.138278 |
| ablation/hgb_battery_e | 21.622627 | 0.603712 | 117/162 | 0.439440 | 0.208748 |
| ablation/rf_battery_a | 23.289914 | 2.168890 | 51/162 | 0.246725 | 0.807488 |
| ablation/rf_battery_b | 10.930406 | 0.581352 | 156/162 | 0.574067 | 0.152153 |
| ablation/rf_battery_c | 11.213249 | 0.603712 | 156/162 | 0.569863 | 0.153606 |
| ablation/rf_battery_d | 11.254187 | 0.894388 | 156/162 | 0.577312 | 0.150129 |
| ablation/rf_battery_e | 23.074060 | 0.737870 | 108/162 | 0.397812 | 0.315753 |
| model_comparison/hgb_battery_d | 11.827317 | 0.648431 | 156/162 | 0.567735 | 0.138278 |
| model_comparison/rf_battery_d | 11.254187 | 0.894388 | 156/162 | 0.577312 | 0.150129 |
| model_comparison/if_direct | 0.000000 | 0.000000 | 22/162 | 0.002038 | undefined |
| model_comparison/ocsvm_direct | 0.059546 | 0.089439 | 80/162 | 0.266648 | undefined |

Tidak ada automatic winner dari tabel outer ini. Battery-E tetap ablation; memilih feature/model family dari hasil tersebut memerlukan label development selection dan independent validation baru.

## Validation tuning dan dual-path

| Experiment / policy | Healthy FPR % | Episodes/min | Detected/applicable | F1 | Mean delay s (detected n) |
|---|---:|---:|---:|---:|---:|
| tuning/reference | 11.827317 | 0.648431 | 156/162 | 0.567735 | 5.526932 (n=156) |
| tuning/selected | 11.883141 | 0.737870 | 156/162 | 0.568402 | 5.232046 (n=156) |
| dual_path/reference | 11.827317 | 0.648431 | 156/162 | 0.567735 | 5.526932 (n=156) |
| dual_path/selected | 9.151470 | 0.536633 | 162/162 | 0.613357 | 7.264810 (n=162) |

Gate promosi memakai predeclared paired arithmetic macro means, bukan mengganti gate dengan pooled improvement. Detected n berbeda; tambahan late detections tidak meniadakan penurunan deadline success.

| Dual gate | Reference macro | Selected macro | Passed |
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

**Keputusan NOT_PROMOTED**. Operational dual tetap disabled. Tidak ada outer reselection/rollback untuk memperbaiki angka setelah hasil terlihat.

## Healthy low-load dan domain

| Fold | Ready | Ready s | Confirmed | FPR % | Episodes/min |
|---|---:|---:|---:|---:|---:|
| fold_00 | 5119 | 506.599454 | 0 | 0.000000 | 0.000000 |
| fold_01 | 5119 | 506.599454 | 150 | 2.930260 | 0.355310 |
| fold_02 | 5119 | 506.599454 | 4 | 0.078140 | 0.118437 |
| fold_03 | 5119 | 506.599454 | 0 | 0.000000 | 0.000000 |
| fold_04 | 5119 | 506.599454 | 0 | 0.000000 | 0.000000 |
| fold_05 | 5119 | 506.599454 | 0 | 0.000000 | 0.000000 |

Primary first-sorted fold00 ditetapkan sebelum low-load prediction. Keenam model memakai cohort sebelas flight yang sama; exposure tidak dipool sebagai 66 independent flights. Semuanya unseen oleh regressor dan calibration detector. Primary nol alarm tidak menjamin semua model/domain bebas alarm. Source domain report menyimpan initial voltage, ranges/response/residual distributions, 462 marginal comparisons dan numeric context; pada primary 642/5119 ready rows berada di luar minimal satu train marginal range. Ini deskriptif, bukan alarm tambahan atau diagnosis physical cause. Contoh actual sensitivity false alarm dipilih setelah run awal tanpa primary alarm dan diberi provenance eksplisit.

## Synthetic masks, observability dan denominator

Bank synthetic: 459 planned, 261 applicable, 198 NOT_APPLICABLE. Konsolidasi 19 separate series / 361 group rows dan 615 failure/NA records; alternatif candidate/policy tidak dipool sebagai independent incidents.

Support mask berbeda dari signal-effect mask. Current-dependent faults hanya valid active segment; sudden-drop pada continuous-ready segment. Semua gaps/warm-up/unavailable dikeluarkan dari ready exposure. NA tidak masuk confusion/recall/deadline denominator; setiap applicable miss/contamination tetap di denominator. Carried confirmed alarm bukan zero-delay success. Undefined rates/delays tetap null, bukan fabricated zero.

## Replay dan latency host

Frozen software replay `ee132dc3901fb80a`: 761095 rows / 179 jobs, 46 columns PASS, max numeric error 0. 31989 healthy-ready single-sample uncached calls; 725490 synthetic exact-input/model cache uses. Feature/reference/slope/residual/timer states selalu dihitung ulang causally; cached speed tidak menggantikan benchmark uncached.

Healthy uncached in-process latency: p50 1.634000 ms, p95 2.184580 ms, p99 2.857084 ms, max 21.833700 ms. Parse/features/prediction/diagnostics+detector saja; model loading/I/O/export dikeluarkan. First calls dipertahankan. Pengukuran Sprint 12 berlangsung bersamaan pytest; bukan isolated idle-host atau hardware benchmark. Final audit memverifikasi recorded timings tanpa mengukur ulang.

Telemetry onset->fresh confirmation sudah mencakup persistence, tidak ditambah 0,5 s lagi. Monotonic processing latency dipisahkan dari signal warning delay; accelerated replay tidak memakai sleep/pacing. Kedua clock tidak dicampur menjadi unsupported hardware delay.

Sumber rinci: [baseline](artifacts/cross_validation/4bc978e2773687d3/research_report.md), [ablation](artifacts/ablation/9ae785a0f10298bc/research_report.md), [comparison](artifacts/model_comparison/03ebe58933705e3e/research_report.md), [consolidated research](artifacts/research_evaluation/4ad7e153513e7d64/research_report.md), [replay](artifacts/replay/ee132dc3901fb80a/replay_report.md).

## Gate akhir

| Check | Scope | Status | Detail |
|---|---|---|---|
| raw_config_artifact_integrity | software_audit | PASS | Verified final-audit evidence |
| full_unit_integration_suite | software_audit | PASS | Verified final-audit evidence |
| CLI_research_and_replay | software_audit | PASS | Verified final-audit evidence |
| predictor_split_scaler_threshold_provenance | software_audit | PASS | Verified final-audit evidence |
| synthetic_masks_and_observability | software_audit | PASS | Verified final-audit evidence |
| validation_only_selection_and_pre_outer_freeze | software_audit | PASS | Verified final-audit evidence |
| readiness_and_metric_denominators | software_audit | PASS | Verified final-audit evidence |
| independent_batch_stream_equivalence | software_audit | PASS | Verified final-audit evidence |
| reproducibility_and_no_remeasurement | software_audit | PASS | Verified final-audit evidence |
| research_target_abrupt_severe | research_target | PASS | measured=1.0; target=0.8; applicable=36 |
| research_target_gradual_observable | research_target | FAIL | measured=0.23529411764705882; target=0.7; applicable=51 |
| dual_path_promotion | research_gate | FAIL | NOT_PROMOTED |
| independent_final_holdout | limitation | WARNING | All flights are historical development; no new untouched final holdout |
| healthy_labels | limitation | WARNING | assumed_from_source; no measured real fault ground truth |
| currtot_units | limitation | WARNING | unverified source units; no conversion |
| low_load_generalization | limitation | WARNING | Primary zero alarms does not erase nonzero sensitivity-model alarms |
| host_latency_context | limitation | WARNING | Accelerated software replay measured under concurrent pytest load, not idle hardware |
| hardware_emergency_readiness | limitation | WARNING | NOT_ESTABLISHED; no deployment or switching validation |

## Keterbatasan dan keputusan akhir

Software/reproducibility audit yang lulus tidak mengubah research gate yang gagal. Dataset kecil dan healthy-assumed, real fault labels dan firmware/pack metadata tidak tersedia; CurrTot belum terverifikasi; seluruh flights historis; low-load operating domain bukan ground truth; synthetic faults tidak mengidentifikasi mekanisme kerusakan internal. Latency host tidak memvalidasi hardware switching. Koleksi data baru yang representatif dan independently labelled serta protokol development/validation baru diperlukan sebelum klaim generalisasi atau emergency readiness. Tidak ada perubahan denominator/threshold/gate setelah outer untuk menutupi kegagalan.

[Final verification](FINAL_VERIFICATION.md) dan [machine-readable results](artifacts/sprint13/final_results.json) memuat sumber/hash/evidence. Scope frozen baseline, tuning/dual candidates dan per-flight/group failures tetap terpisah.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
