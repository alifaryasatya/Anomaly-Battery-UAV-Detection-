# Sprint 3 — causal reference and features

Run `1225e8a76a8ce014`; preprocessing `e22890aeb6ffddf6`.
Rows: 32338; ready: 31989; warm-up: 349; reference unavailable: 0.

Reference collects accepted low-current samples at elapsed <= fixed warm-up duration. On the first sample strictly after that window, lock the median if enough samples exist; otherwise reference stays unavailable. No flight-length percentage, future sample selection or whole-flight fallback. V_ref remains locked across gaps and post-lock voltage changes.
Derivative uses actual dt in seconds; first sample of every segment has dI_dt=dV_dt=0 and derivative_ready=false. Rolling uses (t-window,t], ddof=0, available past plus current, std=0 for one sample. Gap clears rolling/derivative history. Coverage is min(segment elapsed/window,1), not an estimate of sampling completeness. Readiness/count/coverage flags are diagnostics, not automatic predictors.
Batch and incremental processing share CausalFeatureBuilder. elapsed_time is stream elapsed time, used only by Battery-E ablation. No distribution-dependent transformation is fitted. Regression predictors come exclusively from feature_registry.json; actual sag is a target, dV_dt only a direct-baseline input.

## Reference coverage

| Flight | Status | Low-current warm-up samples | V_ref V | Ready samples | Ready observed s |
|---|---|---:|---:|---:|---:|
| 00000061.BIN | READY | 21 | 16.554817 | 400 | 39.800083 |
| 00000062.BIN | READY | 21 | 16.547434 | 465 | 45.698940 |
| 00000063.BIN | READY | 21 | 16.493322 | 332 | 32.699879 |
| 00000064.BIN | READY | 21 | 16.531696 | 1067 | 105.799401 |
| 00000065.BIN | READY | 20 | 16.418150 | 126 | 12.499943 |
| 00000066.BIN | READY | 20 | 16.417031 | 9506 | 949.099397 |
| 00000067.BIN | READY | 21 | 16.542046 | 123 | 12.199987 |
| 00000068.BIN | READY | 20 | 16.533665 | 568 | 56.400415 |
| 00000069.BIN | READY | 20 | 16.507204 | 1622 | 161.999919 |
| 00000070.BIN | READY | 21 | 15.993141 | 1643 | 164.100118 |
| 00000071.BIN | READY | 20 | 15.709306 | 986 | 98.299889 |
| 00000072.BIN | READY | 21 | 16.496592 | 348 | 34.400329 |
| 00000073.BIN | READY | 20 | 16.562006 | 8718 | 870.801230 |
| 00000074.BIN | READY | 18 | 16.648390 | 4395 | 439.099990 |
| 00000075.BIN | READY | 21 | 15.725013 | 307 | 30.599996 |
| 00000076.BIN | READY | 21 | 15.731488 | 718 | 70.500645 |
| 00000077.BIN | READY | 20 | 15.714390 | 665 | 65.999836 |

Coverage samples include only reference-ready rows; observed time includes intervals with both endpoints ready in the same segment. Gaps and warm-up/unavailable intervals are excluded from ready exposure. Truncated warm-up and fully dropped flights remain visible. Activity/reference eligibility does not assert model performance or hardware readiness.

CurrTot remains unverified, without unit conversion. Healthy status is assumed from source. No training, synthetic fault experiments, detector thresholds or fitted model were produced.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
