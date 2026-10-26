# Sprint 2 — battery-only preprocessing and activity audit

Run `e22890aeb6ffddf6`. Input 32338 rows; retained 32338; dropped 0.
17 flights; 92 continuous segments; 75 gaps.
Observed duration: 3224.801403 s; excluded gaps: 2138.817494 s.

## Method

Missing/nonfinite/invalid timestamp or instance rows are dropped with per-field reasons; no fill or interpolation. Sorting occurs after drop; dt/gaps are then recalculated. Duplicate timestamps and multiple instances reject the dataset. Negative telemetry is retained and reported. CurrTot decreases require review before Battery-B–E fitting.
Default activity: Curr >= 1.0 A, run duration >= 1.0 s, flight active-time ratio >= 5.0%.
Observed duration sums intervals within continuous segments. Active exposure counts full durations of qualifying current-only runs, with both interval endpoints active. Run qualification and flight labels are retrospective audit outputs, never regression predictors. Row active_duration_seconds/active_confirmed are causal: no backfilling of the first second. Zero exposure is not eligible and its ratio is undefined (blank CSV/null JSON).
Historical labels are provenance only, loaded after V2 classification. LOW_LOAD_OR_IDLE does not prove the UAV is on the ground. Minimum active-flight gate covers activity only; reference readiness is deferred to Stage 3.

## Activity per flight

| Flight | Observed s | Valid active s | Active % | Valid runs | V2 label |
|---|---:|---:|---:|---:|---|
| 00000061.BIN | 41.9002 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000062.BIN | 47.7986 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000063.BIN | 34.7999 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000064.BIN | 107.8994 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000065.BIN | 14.4999 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000066.BIN | 951.0996 | 439.4999 | 46.2097 | 9 | ACTIVE_LOAD |
| 00000067.BIN | 14.3000 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000068.BIN | 58.4007 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000069.BIN | 164.0000 | 113.3999 | 69.1463 | 2 | ACTIVE_LOAD |
| 00000070.BIN | 166.2001 | 62.5001 | 37.6053 | 3 | ACTIVE_LOAD |
| 00000071.BIN | 100.3002 | 45.1000 | 44.9650 | 1 | ACTIVE_LOAD |
| 00000072.BIN | 36.5003 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000073.BIN | 872.8015 | 432.5999 | 49.5645 | 9 | ACTIVE_LOAD |
| 00000074.BIN | 441.1002 | 265.5001 | 60.1904 | 3 | ACTIVE_LOAD |
| 00000075.BIN | 32.7004 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000076.BIN | 72.5006 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |
| 00000077.BIN | 67.9998 | 0.0000 | 0.0000 | 0 | LOW_LOAD_OR_IDLE |

## Frozen sensitivity audit

No detection threshold or model parameter is selected here.

| Current A | Min duration s | Active flights |
|---:|---:|---:|
| 0.5 | 0.5 | 6 |
| 0.5 | 1.0 | 6 |
| 0.5 | 2.0 | 6 |
| 1.0 | 0.5 | 6 |
| 1.0 | 1.0 | 6 |
| 1.0 | 2.0 | 6 |
| 2.0 | 0.5 | 6 |
| 2.0 | 1.0 | 6 |
| 2.0 | 2.0 | 6 |

## Unit and scope limits

CurrTot remains unverified, without Ah/mAh conversion. TimeUS/Volt/Curr units follow the supplied CSV contract, without independent firmware/log verification. Healthy status is assumed from source; no measured battery fault ground truth exists. No reference voltage, causal features, model training or performance evaluation was performed.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
