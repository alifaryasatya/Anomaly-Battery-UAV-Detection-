# Implementation Plan V2

Status per 21 Oktober 2026: Tahap 0-13 selesai, termasuk konsolidasi research-evaluation, baseline streaming replay/equivalence/host latency dan final audit. Software audit PASS; progressive target belum terpenuhi dan dual-path NOT_PROMOTED. Operational config tetap; deployment/hardware belum divalidasi. Checklist menandai pekerjaan terverifikasi; catatan tiap sprint adalah snapshot historis.

## Tahap 0 — Bekukan scope dan sumber

- [x] Siapkan dataset gabungan dan enam salinan historical active tanpa ThO.
- [x] Simpan manifest checksum dan inventory provenance.
- [x] Audit unit CurrTot, instance, dan asumsi healthy. Bukti log asli tidak disertakan; unit tetap `unverified`, tanpa konversi. Seluruh flight menggunakan Inst=0; tidak ada penurunan CurrTot.
- [x] Bekukan experiment protocol sebelum mengukur performa model V2: `configs/experiment_protocol.md`, salinan `artifacts/audit/experiment_protocol.md` dan `config_used.yaml`, dengan hash di `freeze_manifest.json`.

Output: `artifacts/audit/data_audit.md`, `config_used.yaml`, dependency versions/lockfile.

Hasil aktual: 32.338 baris / 17 flight; delapan checksum ekspor cocok; enam salinan historical active cocok dengan subset combined dataset. Tidak ada missing/nonfinite, invalid timestamp/instance, duplicate/backward timestamp, nilai negatif Volt/Curr/CurrTot, atau penurunan CurrTot. Terdapat 75 gap >0,5 s. Detail tersedia pada `artifacts/audit/data_audit.json` dan `flight_diagnostics.csv`. Environment `.venv` menggunakan Python 3.13.3; versi paket aktual dikunci di `requirements.lock.txt`. Label active-load V2 belum dihitung dan belum ada model dilatih.

## Tahap 1 — Struktur dan CLI

Bangun `main.py` dan modul `src/config.py`, `data_loader.py`, `preprocessing.py`, `activity.py`, `feature_sets.py`, `features.py`, `models.py`, `detectors.py`, `persistence.py`, `fault_injection.py`, `evaluation.py`, `cross_validation.py`, `tuning.py`, `reporting.py`, `replay.py`. Tambahkan `tests/` sesuai invariant tiap tahap, bukan salinan implementasi.

CLI yang direncanakan: `inspect`, `preprocess`, `baseline`, `cross-validate`, `ablation`, `tune`, `compare-models`, `evaluate-low-load`, `tune-dual-path`, `compare-dual-path`, `research-evaluation`, `replay`.

Acceptance: path relatif berakar di V2, validasi config jelas, seed deterministik, tidak mengimpor engine V1.

- [x] `main.py` dan seluruh 15 modul yang direncanakan tersedia, ditambah error types dan package initializer. Modul tahap lanjut merupakan kerangka, bukan engine ML yang sudah berjalan.
- [x] Seluruh 12 perintah terdaftar; `inspect` berjalan terhadap combined CSV tanpa menggabungkan salinan historical active. Perintah tahap lanjut menampilkan `NOT_IMPLEMENTED` dengan exit code 3.
- [x] Validasi YAML lengkap: required/unknown/duplicate keys, tipe/rentang, predictor allowlist, path dalam root V2, dan gate promotion.
- [x] Path config/input/artifact relatif ke root V2, independen dari CWD. Seed Python/NumPy dan RNG eksplisit deterministik; model berikutnya wajib menerima random_state.
- [x] Unit dan integration checks mencakup error paths, CLI dari direktori lain, seluruh perintah, reproducibility, loader tanpa perubahan field/order, isolasi V1, dan integritas 21 berkas freeze Sprint 0.

Verifikasi dan contoh keluaran tersedia di `artifacts/sprint1/`. Tidak ada perubahan protokol/config/data Sprint 0, training, evaluasi performa, atau label active V2 pada sprint ini.

## Tahap 2 — Audit battery-only dan segment

Implement schema/finite checks, duplicate/instance policy, sorting dan no-future-fill cleaning. Assign segment_id menggunakan dt > nominal_dt * multiplier. Hitung aktivitas berbasis current+duration+observed-time ratio. Jangan membaca label historical untuk menentukan label V2.

Output: `flight_inventory.csv`, `battery_activity.csv`, `time_gap_summary.csv`, unit/instance diagnostics. Sensitivity report terhadap min_current/min_duration; belum memilih threshold deteksi dari test.

Acceptance: tidak ada dependency throttle, exposure tidak memasukkan gap, current-based label dijelaskan walau jumlah active berbeda dari enam.

- [x] Numeric/schema/finite/identity checks; drop invalid dengan alasan per field, tanpa fill/interpolasi. Reject duplikasi timestamp dan multi-instance dengan artefak diagnostik; nilai negatif dan penurunan CurrTot tetap dilaporkan.
- [x] Sorting per stream dan gap dihitung ulang setelah drop. Segment baru pada dt >0,5 s; dt awal segment 0, tanpa median masa depan.
- [x] Label aktivitas dari current/duration/observed-time ratio, terpisah dari label V1. Per-row duration/confirmation causal, full exposure run yang lolos minimum duration dilaporkan retrospektif untuk audit.
- [x] Distribusi arus, inventory, segment/run tables, gap summary, unit/instance diagnostics, dan sembilan kombinasi sensitivitas tersedia pada direktori run.
- [x] CLI `preprocess` dijalankan pada combined dataset; raw/config/protokol tidak diubah. Output di `artifacts/preprocessing/<run_id>/` dengan snapshot config, kode/data/provenance hashes dan manifest output.

Hasil aktual: seluruh 32.338 baris dipertahankan; 17 flight; 92 segmen; 75 gap. Observed duration 3.224,801403 s, excluded gap duration 2.138,817494 s. Enam flight ACTIVE_LOAD (`66,69,70,71,73,74`) dan sebelas LOW_LOAD_OR_IDLE. Seluruh sembilan kombinasi sensitivitas menghasilkan daftar active yang sama, tanpa memaksa jumlah enam atau memilih threshold deteksi. Gate minimum tiga flight active lolos; kesiapan reference/features belum dihitung. Bukti pengujian dan lokasi run tersedia pada `artifacts/sprint2/`.

## Tahap 3 — Causal reference dan feature registry

Bangun warm-up V_ref fixed duration dan readiness, Battery-A sampai E, actual dt derivative, rolling trailing berbasis detik, gap reset. V_ref tetap ketika injection dilakukan. Prediktor forbidden divalidasi terpusat.

Output: processed features dan `feature_registry.json`.

Acceptance: memodifikasi sampel masa depan tidak mengubah keluaran sebelumnya, batch/incremental warm-up sama, missing reference terlihat dalam coverage.

- [x] `CausalFeatureBuilder` dan batch `build_features` menggunakan state per flight/instance; warm-up tetap 2 s, low-current <=1 A, minimal lima sampel. Lock pada sampel pertama strictly setelah warm-up; unavailable dan truncated warm-up tetap terlihat tanpa whole-flight fallback.
- [x] Derivative actual dt, trailing rolling `(t-window,t]` dengan ddof=0, std awal 0, serta coverage/count/readiness diagnostics. Gap mereset derivative/rolling; reference terkunci tetap. Elapsed time hanya predictor Battery-E.
- [x] Registry Battery-A–E dan direct baseline, dengan validasi predictor terpusat yang digunakan config dan feature runner; actual voltage/sag/dV_dt/residual/readiness/identifier tidak dapat masuk regresi.
- [x] Perintah pendukung `python main.py features` ditambahkan agar Tahap 3 dapat direproduksi tanpa menjalankan training. Hasil tersimpan di `artifacts/features/<run_id>/` dengan processed CSV, registry, coverage, config/versions, audit dan manifest checksum.
- [x] Pengujian batas warm-up/current/sample count, irreversible unavailable, truncated flight, gap/reference resets, irregular rolling/derivative dengan oracle independen, interleaved streams, future perturbation/prefix invariance, post-lock voltage changes, registry guards dan integrasi CLI.

Hasil aktual: 32.338 baris fitur; 17 flight dengan reference tersedia, 31.989 reference-ready samples, 349 warm-up samples, nol reference-unavailable samples. Ready observed duration 3.189,999997 s; observed duration total 3.224,801403 s. Enam flight active lolos gate gabungan aktivitas/reference. Nilai CurrTot tidak dikonversi; belum ada training atau evaluasi model. Bukti verifikasi tersedia di `artifacts/sprint3/`.

## Tahap 4 — Single-path pilot

Train HGB + Battery-D pada whole-flight train, gunakan healthy validation threshold 99 dan persistence 0,5 s sebagai reference pilot. Simpan model bundle, schema/features/unit policy, scores dan threshold payload. Jangan menyalin model/angka threshold V1.

Acceptance: hanya healthy train untuk fit; score positif satu arah; persistence independent per segment dan timestamp.

- [x] Deterministic pilot fold pertama: test `00000066.BIN` dicadangkan, validation `00000069.BIN`, train `00000070/71/73/74.BIN`; group overlap ditolak. Assignment/parameter disimpan sebelum fit; test tidak diprediksi/dievaluasi.
- [x] HGB Battery-D, seed 42, seluruh ready rows pada whole healthy-assumed training flights, tanpa synthetic/scaler. Early stopping internal dinonaktifkan agar tidak membuat random row validation split. CurrTot decrease menjadi blocking review gate; unit tetap `unverified`.
- [x] Signed residual dan score `max(0,r)`; threshold linear percentile 99 healthy validation saja, persistence fixed 0,5 s. Timer per stream mereset saat false/unready/segment/gap; keputusan dan episode starts tersimpan per row.
- [x] Bundle estimator/config/registry/schema/unit/reference/split/seed/threshold provenance, exact model params, hashes, training provenance, scores/metrics/plot/report dan output manifest tersedia di `artifacts/baseline/<run_id>/`. Run lengkap diverifikasi sebelum reuse; corrupt output ditolak.
- [x] Pengujian group/healthy/synthetic/predictor leakage guards, validation-only calibration, fixed policy, deterministic fit/bundle reload, persistence actual time pada irregular dt/equality/gaps, episode denominator dan integrasi CLI lulus.

Hasil pilot: train 15.742 ready rows; validation 1.622. Validation MAE 0,077292 V, RMSE 0,099383 V, R² 0,811720. P99 threshold 0,111810858843 V; persistence 0,5 s menghasilkan 17 raw alarm samples, 3 confirmed samples (FPR 0,184957%), 2 episode / 161,999919 observed-ready s = 0,740741 episode/menit. Ini calibration diagnostic historical development, bukan unbiased final result. Parameter tidak diubah setelah hasil terlihat. Full detector gate, synthetic detection metrics dan reserved-test evaluation tetap belum diukur. Bukti: `artifacts/sprint4/`.

## Tahap 5 — Synthetic faults dan observability

Implement tiga fault family, 3 severity, 3 start position, current-only applicability dan deterministic segment choice. Progressive memakai waktu aktual, sudden ramp memakai detik. Tulis metadata dan label support per row.

Output: `fault_applicability.csv`, `fault_observability.csv`, `fault_manifest.csv`. Data sintetik ditempatkan di `data/synthetic/`, bukan overwrite raw.

Acceptance: current-dependent fault tidak pernah dimulai/berjalan pada sampel idle atau melintasi gap; NOT_APPLICABLE tidak masuk confusion counts.

- [x] Core `inject_fault` membentuk salinan tiga family × tiga severity × tiga posisi, tanpa mutasi raw/reference/predictor. Progressive dan sudden ramp memakai actual timestamp, bukan indeks baris; dV_dt dihitung ulang dari voltage sintetis.
- [x] Resistance/progressive memilih valid active run terpanjang, tie timestamp awal. Sudden memilih continuous ready segment terpanjang dan boleh low-load. Aligned onset mesti ready; post-onset support minimal 1 s. Tidak ada fallback lokasi diam-diam.
- [x] Metadata selection/bounds/dose/onset/first-effect/end/current/support/max-mean sag/observability/ending lengkap. Support labels, effect flags dan PRE_FAULT/FAULT_SUPPORT/POST_FAULT_ORIGINAL eksplisit; reference tetap terkunci, no synthetic recovery ramp.
- [x] NOT_APPLICABLE beralasan tetap muncul di seluruh tabel; statistik sinyal null, tanpa CSV synthetic dan tanpa eligibility confusion/recall. Applicable berstatus NOT_EVALUATED sampai evaluasi Sprint 6.
- [x] CLI pendukung `inject-faults`, deterministic gzip CSV di `data/synthetic/<run_id>/`, config/protocol/output hashes, cache/integrity guard, laporan applicability/observability dan contoh trace tersedia. Model/pilot threshold tidak dipanggil atau diubah.
- [x] Pengujian formula/time alignment, ramp/progress, support equality/short/idle/gap/tie, frozen reference/all predictors, unavailable/truncated/empty inputs, observability boundaries, synthetic-training guards, gzip/cache/checksum/CLI dan preservasi pilot/raw lulus.

Hasil aktual: 17 flight × 27 = 459 planned scenarios; 261 applicable, 198 NOT_APPLICABLE (NO_VALID_ACTIVE_SEGMENT). Sebanyak 261 CSV.gz, 776.880 baris salinan; applicable observability 225 HIGH / 33 MEDIUM / 3 LOW. Ini audit sinyal sintetis, bukan detection metrics. Reserved-test scenarios hanya disiapkan; tidak ada prediksi/evaluasi detector atau perubahan threshold. Bukti: `artifacts/sprint5/`.

## Tahap 6 — LOAFO baseline V2

Gunakan daftar active-load hasil tahap 2, sorted cyclic validation. Untuk tiap fold: train -> reference/features causal -> calibrate healthy validation -> evaluate untouched test -> synthetic copy test -> metrics.

Output `artifacts/cross_validation/`: `fold_assignments.csv`, `regression_per_fold.csv`, `anomaly_per_fold.csv`, `healthy_active_per_fold.csv`, `fault_per_scenario.csv`, `summary.csv`.

Acceptance: nol overlap; tampilkan macro dan pooled, semua failed cases, deadline denominator, pre-existing alarm dan coverage. Status evaluasi historical development ditulis eksplisit.

- [x] Enam fold sorted cyclic dari active/reference-ready V2; empat whole healthy training flights, satu validation dan satu outer test, tanpa overlap. Semua model, threshold dan selected policy dibekukan sebelum prediksi outer pertama.
- [x] HGB Battery-D dengan seed/parameter tetap, no scaler, internal random-row early stopping off; fit healthy ready saja termasuk periode low-current. P99 healthy validation per fold dan persistence 0,5 s tidak dituning setelah outer terlihat.
- [x] Evaluasi healthy outer asli sekali per flight dan 27 synthetic test copies per fold; reference/predictor tetap, state persistence independen. Seluruh NOT_APPLICABLE tetap tercatat dengan null metrics, tanpa inference/confusion denominator.
- [x] Conventional sample counts terpisah dari incident recall/delay. Carried warning pada onset menjadi PREEXISTING_ALARM dan gagal incident/deadline, termasuk bila ada episode baru kemudian. Undetected tetap denominator semua deadline; delay null. Konfirmasi baru tepat onset boleh delay 0; post-support tidak dihitung.
- [x] Per-fold, pooled counts dan macro mean/population std/median/min/max/n_defined, healthy exposure/episode, pre-fault FPR, coverage, observability, failed cases dan plot tersedia. Model bundles, scores, config/protocol/hashes dan selected-policy freeze disimpan dalam run immutable.
- [x] CLI asli, 256 pengujian proyek dan audit stdlib independen atas 782.739 score rows lulus; 217 output hashes dan 18 pre-outer selection hashes cocok. Raw/freeze dan artefak lama dipertahankan.

Hasil run `4bc978e2773687d3`: outer RMSE pooled 0,138278 V; healthy 26.870 ready samples, 3.178 confirmed (FPR 11,827317%), 29 episode / 2.683,400543 s (0,648431/menit). Semua 162 skenario applicable: 156 detected, 6 undetected, 0 pre-existing. Incident recall 96,296296%; sample F1 pooled 0,567735; median delay detected-only 0,699965 s (n=156). Abrupt severe <=1 s 36/36 memenuhi target; progressive MEDIUM/HIGH <=5 s 12/51 (23,529412%) tidak memenuhi target 70%. Kegagalan tersimpan tanpa revisi threshold. Status tetap historical development / RESEARCH_BASELINE_ONLY, bukan deployment. Detail `artifacts/sprint6/verification.md` dan report run.

## Tahap 7 — Ablation dan pembanding

Bandingkan Battery-A/B/C/D/E untuk HGB dan RF dengan protokol fold identik; model IF/OCSVM memakai voltage behavior. Scaler OCSVM hanya train. Set tambahan tidak otomatis menang. Parameter yang berubah hanya dipilih di development; jika memilih feature family berdasarkan outer results, laporkan sebagai development selection, bukan unbiased final performance estimate.

Output: `artifacts/ablation/`, `artifacts/model_comparison/`, exact model feature list.

- [x] Rencana 10 kandidat HGB/RF Battery-A–E dan empat pembanding HGB-D/RF-D/IF/OCSVM ditetapkan sebelum outer Sprint 7; snapshot plan/code/config tersimpan di `artifacts/sprint7/`. Tidak ada adaptasi dari eksperimen pertama ke kedua.
- [x] Fold, readiness, fault definitions dan applicable/support/observability identik; pairing diverifikasi. Semua candidate-fold models/policies dibekukan sebelum outer pertama pada masing-masing eksperimen.
- [x] HGB/RF hanya exact regression allowlist, target healthy sag, positive residual V. RF memakai config/seed tetap; HGB internal early stopping off. Battery-E elapsed time hanya ablation. Prediksi synthetic digunakan ulang hanya setelah exact predictor/identity/readiness equality.
- [x] IF/OCSVM memakai direct registry dengan score `-decision_function`, signed dan higher=abnormal. Scaler OCSVM fit healthy-ready train saja; SVM sampling uniform tanpa replacement/seed/cap/sorted indices serta source identities tersimpan. Direct models rescore injected voltage behavior; tidak diberi regression metrics palsu.
- [x] P99 healthy validation dan persistence 0,5 s tetap; seluruh denominator/contamination/miss/NA/coverage mengikuti Sprint 6. Per-fold/pooled/macro/paired deltas, failed cases, scores, bundles/provenance/schema/parameter/hash/report/plot tersedia.
- [x] Seluruh 270 pengujian proyek lulus. Audit score stdlib dan audit saved bundles/scaler/sampling/replay independen tersedia pada `artifacts/sprint7/`; raw/config/protocol/artefak Sprint 0–6 dipertahankan.

Hasil dan bukti aktual: `artifacts/sprint7/verification.md`. Kedua perintah tetap historical development comparison; tidak memilih winner otomatis, mengubah operational config atau mempromosikan detector. Tuning threshold/persistence tetap Tahap 8, tanpa memasukkan outer ke selection.

## Tahap 8 — Tuning threshold/persistence

Grid healthy-validation percentiles dan durasi sesuai config, dengan synthetic validation sebagai secondary tuning. Outer test tidak masuk selection. Simpan candidate budget, rationale, selected config per fold sebelum evaluasi outer dijalankan.

Output: `artifacts/tuning/threshold_persistence_grid.csv`, `selected_per_fold.json`. Laporan trade-off sample FPR dan episode/minute harus terpisah.

- [x] Budget fixed 4 percentile × 5 persistence = 20/fold, 120 total pada HGB Battery-D; rencana/code/config hashes disimpan sebelum outer Sprint 8. Tidak ada pemilihan model/feature dari hasil Sprint 7.
- [x] Quantile hanya healthy assigned validation; 27 synthetic validation copies per fold mengevaluasi seluruh kandidat (3.240 evaluations). Synthetic tidak pernah fit atau calibrate quantile, flag used_for_threshold false.
- [x] Healthy confirmed sample FPR dan episodes/minute harus keduanya <= reference validation P99/.5 s; undefined gate ineligible. Ranking/tie/undefined/fallback sesuai frozen protocol, tanpa outer-derived criteria.
- [x] Semua selected/reference policy, model dan selection inputs enam fold di-hash sebelum outer pertama. Full grid, kandidat rejected/reason/rank/denominator dan selected_per_fold.json tersedia. Outer perturbation test membuktikan selection tidak berubah.
- [x] Paired outer original/synthetic, selected versus reference pada model/readiness/fault support yang sama; tidak ada reselection setelah outer terlihat. Per-fold/pooled/macro/paired/failed/coverage tables, score exports, report/plots dan manifest tersedia.
- [x] 289 pengujian proyek lulus; audit stdlib independen mereplay 16.626.456 row decisions dan menghitung ulang enam selection; saved-object audit enam model/3.000 prediction samples tanpa fit. Raw/freeze/artefak Sprint 0–7 tetap cocok; cache CLI tidak menulis ulang keluaran.

Run `d4351f20af42a44c`: tiga fold mempertahankan reference, tiga memilih policy berbeda, seluruhnya dari validation. Outer incident recall tetap 156/162; sample F1 0,567735→0,568402; median detected delay 0,699965→0,600039 s (n=156). Healthy confirmed FPR 11,827317%→11,883141% dan episodes/minute 0,648431→0,737870 meningkat, terutama flight 69 (8→23 confirmed samples, 2→6 episodes). Tidak ada rollback/pemilihan ulang dari outer atau promotion; status RESEARCH_TUNED_CANDIDATE_ONLY. Bukti `artifacts/sprint8/verification.md`.

## Tahap 9 — Diagnosis dan dual-path

Hitung causal residual delta/slope, attribution alarm per path, dan baseline transient context dari development. Pilih Path A lalu shortlist Path B secara deterministik, persistence B awal 0,3/0,5 s. Kandidat tanpa synthetic-validation refinement tidak eligible untuk dipilih.

Output: validation grid, selected policy, `paired_per_fold.csv`, `promotion_decision.md`. Perbandingan pasangan menggunakan flight, fault, severity, posisi, readiness, dan denominator identik.

Acceptance: gate PRD terpenuhi atau NOT_PROMOTED. Setelah hasil outer terlihat, refinement selanjutnya dilabel exploratory; tidak mengklaim independent final result pada test yang sama.

- [x] Incremental causal signed-residual delta/actual-time rate dan OLS trailing `.3/.5/1` s, half-open window, minimum dua timestamp unik, gap/segment/unready reset; diagnostics tidak masuk regression features.
- [x] Independent A/B persistence dengan OR confirmed union, per-path raw/duration/confirmation/episode attribution dan first incident path. Carried warning/NA/undetected semantics tetap; alternating raw tidak menumpuk shared timer.
- [x] Pilih A healthy-validation-only, catat 144-pair healthy grid/fold, screen union FPR+episode terhadap fixed P99/.5 reference, shortlist <=3 pada A terpilih lalu mandatory synthetic validation refinement. Semua six policies/model/input hashes frozen sebelum outer; fallback single reference tersedia.
- [x] Full validation manifest termasuk NA/no-shortlist, refined cases/rank/rationale, quantile fingerprints, baseline validation episode numeric context, paired outer scores/tables/failures/coverage, factual report/plots dan promotion_decision.md/json tersimpan.
- [x] Gate predeclared paired macro means dengan semua fold values defined: healthy FPR/episode no increase, F1/all deadlines no decrease, mean detected delay strictly improves. Hasil NOT_PROMOTED, tanpa outer reselection atau config enable.
- [x] 309 pengujian lulus, termasuk affine/irregular/prefix OLS, timer handoff/independence, validation guards/shortlist/refinement, fallback/NA, outer perturbation selection invariance, undefined/strict/macro gates, bundle/cache/corruption/CLI; audit independen dan preservasi Sprint0–8 tersedia di artifacts/sprint9/.

Run `ad82c3e229bf6510`: 864 healthy pairs, 18 refined candidates/486 validation case evaluations, enam dual selected folds, 324 paired outer case evaluations. Pooled healthy FPR 11,827317%→9,151470%, episode29→24, incident detected156→162 dari162, sampleF1 0,567735→0,613357. Namun deadline1/2/10/30 s turun dan mean detected delay (macro5,656974→7,264810 s) meningkat; gate NOT_PROMOTED. Detected n berubah156→162; enam new late detections tetap dilaporkan. Operational dual tetap disabled. Bukti artifacts/sprint9/verification.md.

## Tahap 10 — Low-load false alarm dan domain diagnostics

Healthy LOW_LOAD_OR_IDLE hanya untuk false alarm. Ukur raw/confirmed samples, FPR, episode/flight, episode/minute dengan monitored duration. Jangan menyebut current rendah sebagai bukti ground.

Plot contoh false alarm dan konteks Curr/dI_dt/voltage residual. Laporkan rentang input, initial voltage, dan distribusi response per flight. Jangan menaikkan threshold hanya untuk menyembunyikan domain mismatch.

- [x] `evaluate-low-load` memverifikasi/reuse enam frozen reference bundles/thresholds Sprint 6; tidak refit, recalibrate, memilih model dari low-load atau memakai kandidat dual NOT_PROMOTED sebagai baseline.
- [x] Sebelas whole LOW_LOAD_OR_IDLE flights, 66 model-flight sensitivity pairs, explicit unseen-by-model/detector dan source memberships. Primary first sorted fold ditetapkan sebelum prediction; repeated cohort exposure tidak dipool sebagai flight independen.
- [x] Ready-only raw/confirmed samples/FPR, episode/flight, observed duration tanpa warm-up/gap dan episodes/minute; zero-ready flight tetap tercatat dengan null rates. Training alarms tidak disebut unseen FPR.
- [x] Rentang/distribusi predictor dan actual/predicted response, initial voltage/V_ref per flight, original train fingerprint/ranges, descriptive marginal exceedance, episode numeric context dan plot actual false alarm tersedia. Tidak ada diagnosis internal atau inferensi ground dari arus rendah.
- [x] Audit independen score/prediction/timer/exposure/domain/context, full tests, cache dari cwd berbeda tanpa writes, preservasi raw/config/protocol/artefak Sprint 0–9 tersedia di `artifacts/sprint10/`.

Run `56b20c4b7936b122`: primary fold00 nol raw/confirmed alarms dari 5.119 ready samples / 506,599454 s pada sebelas unique flights. Sensitivitas fold01 150 confirmed samples / tiga episode (FPR 2,930260%), fold02 empat / satu (0,078140%); empat model termasuk primary nol. Source thresholds dan primary allocation tidak berubah. Setelah run diagnosis awal tanpa primary alarm, ilustrasi tambahan mengambil first sorted sensitivity fold/flight dengan confirmed alarm; revisi deskriptif ini dicatat terpisah, bukan selection model/policy. Detail `artifacts/sprint10/verification.md`.

## Tahap 11 — Research-final orchestration

`python main.py research-evaluation` yang akan dibangun menjalankan audit -> segmentation/reference/features -> development evaluation/ablation/tuning -> frozen outer evaluation -> synthetic/observability/group reporting -> healthy false alarms -> plots -> factual report. Hindari mengevaluasi test lebih dahulu sebelum candidate selection selesai.

Simpan artifact hash/config/versions, model bundles dan factual results summary. Angka laporan harus berasal dari CSV/JSON yang dihasilkan, bukan ditulis manual dari memori.

- [x] `research-evaluation` mengintegrasikan sepuluh tahap melalui source catalog/pointers terverifikasi atau menjalankan engine tahap yang belum memiliki hasil. Semua existing sources dipreflight sebelum build; corrupt/incompatible source tidak diperbaiki dengan silent refit.
- [x] Audit frozen raw/config/protocol, manifest seluruh source outputs, feature/preprocessing lineage dan active groups, matching reference results serta pre-outer freeze records. Sourcecode/version historis tetap terpisah dari code/version orchestrator saat ini.
- [x] Parent preprocessing mengikuti deklarasi feature run; snapshot Sprint 2 lama dipertahankan. Low-load builder menerima explicit completed CV summary, tanpa harus membuat/menimpa pointer Sprint 6.
- [x] Orchestration plan/code/config/version dibekukan sebelum engine dijalankan; urutan/candidate budgets tetap, validation-only selection dan semua candidate-fold freeze sebelum outer pada setiap eksperimen. Tidak memilih ulang kandidat/winner dari outer atau mengklaim global untouched holdout pada data historis.
- [x] Source/model registries, dataset/gap/activity/reference coverage, applicability/observability, baseline per-fold, ablation/comparison, selected policies, promotion, low-load, group/failed/NA tables, lima plot, factual JSON/Markdown tersedia dengan completion manifest.
- [x] Real cold pipeline seluruh sepuluh tahap dan immutable catalog/cache reuse, no implicit build saat source corrupt, raw/lineage/reference/promotion/undefined guards serta CLI teruji; audit aggregation stdlib independen dan preservasi sumber tersedia di `artifacts/sprint11/`.

Run `c7e7b903893a937d`: reuse sepuluh stage, 4.532 verified source outputs, 103 saved-model references, 19 separate experimental series, 361 group rows, 615 failed/NA records dari 3.078 case evaluations. Tidak ada training ulang atau perubahan konfigurasi pada actual project run. Target progressive dan dual promotion yang gagal tetap dinyatakan gagal; status `RESEARCH_CONSOLIDATION_ONLY`. Detail `artifacts/sprint11/verification.md`.

## Tahap 12 — Replay

Implement streaming state per flight/instance/segment, causal warm-up/rolling/slope/persistence dan state NOT_READY/NORMAL/SUSPECT/WARNING. Gunakan timestamp asli; paced replay sekitar 10 Hz opsional. Ukur latency p50/p95/p99 dan end-to-end warning delay.

Acceptance: batch causal dan incremental replay setara untuk input sama, gap/reset benar, timer raw->confirmed benar pada interval irregular. Replay emergency candidate dilanjutkan jika gate detector lolos; baseline dapat direplay sebagai validasi software yang tidak diklaim memenuhi target emergency.

- [x] Raw battery streaming parser/identity audit, causal reference/rolling/derivative/slope/persistence dan NOT_READY/NORMAL/SUSPECT/WARNING; invalid numeric drop tanpa fill/state advance, duplicate/backward/instance guards, per-flight state isolation dan prefix invariance.
- [x] Frozen Sprint 11 source set mempin baseline Sprint 6 dan primary low-load Sprint10. Model/threshold/config/protocol/source hashes dibekukan sebelum replay; tidak refit, recalibrate, enable deployment atau menjalankan dual NOT_PROMOTED sebagai emergency candidate.
- [x] 17 original flights dan seluruh 162 applicable baseline synthetic cases direplay dengan fresh state per case. Causal features, single-row predictions/residuals, diagnostics dan exact timer/decision states dibandingkan ke batch; NA tetap skip dan null semantics teruji.
- [x] Healthy rows selalu uncached single-sample prediction; synthetic exact-input/model cache hanya memreuse prediksi, bukan state. Latency groups terpisah dan actual TimeUS menentukan signal warning delay tanpa double-count persistence.
- [x] Monotonic parse/features/prediction/diagnostics+detector/total latency p50/p95/p99/count/max/nominal-overrun, full traces/jobs/cases/healthy/equivalence tables, report/plots dan immutable completion/cache tersedia.
- [x] 360 project tests lulus; independent raw/reference/rolling/derivative/OLS/model/cache/timer/latency audit, previous-artifact preservation dan cache no-remeasure proof tersedia di `artifacts/sprint12/`.

Run `ee132dc3901fb80a`: 761.095 rows /179 jobs, 46 columns PASS, max numeric error0. Healthy uncached latency p50/p95/p99 1,634/2,184580/2,857084 ms (n31.989, full tests concurrently running), max21,8337 ms. 725.490 synthetic exact-input predictions dipisahkan dari benchmark uncached. Batch incident detected156/162, seluruh deadline/FPR/exposure tetap; persistence sudah termasuk signal delay. Paced replay tidak dilakukan (opsional). Status `BASELINE_SOFTWARE_REPLAY_ONLY`, bukan hardware/emergency readiness atau promotion. Detail `artifacts/sprint12/verification.md`.

## Tahap 13 — Final audit

Jalankan unit/integration suite, CLI research-evaluation dan replay-equivalence. Review unit, input forbidden, split/scaler/threshold provenance, masks, denominator, checksum raw, reproducibility dan readiness coverage.

Output: `FINAL_RESEARCH_RESULTS.md`, `FINAL_VERIFICATION.md`, known limitations dan daftar gate PASS/WARNING/FAIL.

- [x] Full suite 371 tests lulus tanpa failure/error/skip; audit Sprint 0 dan dependency check lulus.
- [x] CLI research-evaluation dan replay dari cwd berbeda serta repeat-cache checks lulus, tanpa stage training baru atau replay remeasurement.
- [x] Sebelas hasil audit independen termasuk seluruh 103 bundles: predictor/unit, flight split, training-only scaler, validation threshold, synthetic masks, metric denominator dan readiness coverage.
- [x] 5.939 prior files / 23 completed manifests cocok; raw/config/protocol/bobot/threshold dan evidence historis dipertahankan.
- [x] Dua laporan final, known limitations dan gate PASS/WARNING/FAIL tersedia dengan machine-readable results dan evidence hashes di artifacts/sprint13/.

Current consolidation `4ad7e153513e7d64` mempertahankan source inventory eksperimen Sprint 11; replay `ee132dc3901fb80a` tetap mempin recorded measurements. Software PASS tidak mengubah progressive NOT_MET atau dual NOT_PROMOTED. Scope penelitian dan baseline software validation selesai; independent final validation dan hardware readiness belum dibuktikan.


## Pengujian inti

- Schema hanya battery telemetry, multi-instance tidak bercampur.
- Current-only segments: threshold equality, irregular dt, gap, durasi terlalu singkat.
- No future leakage pada cleaning/reference/rolling/slope.
- Flight-disjoint deterministic split dan training scaler only.
- Persistence waktu: continuous, interrupted, gap, batas deadline, irregular dt.
- Fault injection menjaga raw, active-load applicability dan reference frozen.
- NOT_APPLICABLE dan undetected denominator; existing alarm di onset bukan deteksi baru.
- False-alarm episode dan observed duration tidak melintasi gap.
- Parameter selected hanya dari validation; replay batch equivalence.

## Kriteria berhenti

Tidak adanya peningkatan bukan alasan mengganti denominator atau menurunkan gate setelah test terlihat. Bila single/dual-path gagal, simpan hasil negatif, pertahankan model yang lolos di scope datanya, dan rancang data development yang representatif sebelum final validation baru.
