# Data Dictionary dan Provenance

## Schema input

| Kolom | Arti | Satuan/status | Peran |
|---|---|---|---|
| NamaFile | Identitas log/flight | String | Group split dan temporal state; bukan predictor |
| TimeUS | Timestamp log | Mikrodetik | Segmentation, derivative, rolling, persistence, delay |
| Inst | Instance sumber battery telemetry | Indeks | Audit instance; bukan predictor |
| Volt | Tegangan pack baterai | V | Pembentuk target dan direct anomaly baseline |
| Curr | Arus baterai | A | Predictor dan penentu active-load |
| CurrTot | Akumulasi pemakaian arus/kapasitas dari sumber | Unit perlu diverifikasi dari format/firmware log asli | Predictor kandidat dan audit monotonicity/reset |

Jangan menyatakan `CurrTot` sebagai Ah atau mAh tanpa memeriksa definisi log asli. CSV saja tidak cukup untuk memastikan unit. Jangan melakukan scaling unit secara diam-diam.

## Isi dan asal data

`dataset_baterai.csv` berasal dari `dataset_baterai_throttle.csv` dengan proyeksi enam kolom di atas. Urutan baris, timestamp, duplikasi, nilai kosong, dan representasi angka dipertahankan saat ekspor. Cleaning dilakukan kemudian pada salinan processed.

Total historis yang diharapkan: 32.338 sampel dari 17 flight. Manifest mencatat hasil aktual dari sumber pada saat ekspor.

Enam salinan active historis adalah subset dataset gabungan. Jangan menggabungkan direktori salinan tersebut dengan dataset gabungan karena akan menggandakan sampel. Metadata label V1 hanya untuk perbandingan/provenance, bukan input model maupun klasifikasi operasional V2.

`data/metadata/source_manifest.json` mencatat SHA-256 sumber sebelum/sesudah ekspor, checksum file keluaran, jumlah baris, dan kolom. File sumber tidak ditulis ulang.

## Active-load pada V2

Definisi awal: `Curr >= 1.0 A` secara kontinu minimal `1.0 s`. Kontinuitas diputus saat timestamp gap melebihi `0.5 s`. Active-load flight ditetapkan menggunakan proporsi waktu active pada interval telemetry yang teramati, bukan durasi wall-clock yang mencakup gap.

Perhitungan exposure memakai interval valid antar-sampel. Interval dihitung active hanya bila kedua endpoint memenuhi syarat arus. Tidak ada interval yang dibentuk melintasi gap. Ambang proporsi awal 5% bersifat configurable dan wajib diaudit efeknya; ambang ini belum terbukti optimal.

## Data log Pixhawk asli

File `.BIN`/`.ulg` dan converter aslinya belum disertakan. Pipeline V2 dimulai dari CSV yang tersedia. Jika converter dibangun kemudian, tentukan firmware/format log terlebih dahulu, baca pesan battery yang sesuai, dokumentasikan unit/instance/timestamp, dan ekspor schema yang sama. Jangan mengarang mapping pesan hanya dari nama kolom CSV.

## Referensi tegangan

V2 memakai warm-up awal berdurasi tetap dalam detik. V_ref dikunci setelah warm-up; pemilihan low-current hanya memakai data yang sudah datang. Jika sampel warm-up tidak cukup, tandai `NOT_READY`. Jika tidak ada low-current yang cukup, tandai `REFERENCE_UNAVAILABLE` dan laporkan coverage. Jangan memakai persentase panjang seluruh flight karena panjang akhir belum diketahui saat replay.

## Batas interpretasi

Label healthy mengikuti asumsi dataset sumber, bukan hasil diagnosis fisik. Jangan menganggap perbedaan pack, jumlah sel, kapasitas, sensor, atau firmware sebagai fault. Metadata tersebut perlu dicatat jika tersedia, dan perbedaan domain perlu dilaporkan terpisah.

## Kolom turunan Sprint 3

`battery_features.csv` menyimpan kolom raw yang telah dibersihkan, identitas source record/segment, flag aktivitas causal dari Sprint 2, dan kolom berikut. Daftar predictor resmi hanya berasal dari `feature_registry.json`; seluruh CSV tidak boleh langsung digunakan sebagai predictor matrix.

| Kolom | Makna |
|---|---|
| `elapsed_time` | Detik sejak sampel pertama stream; tidak reset di gap; predictor Battery-E saja |
| `segment_elapsed_seconds` | Detik sejak awal segment; metadata coverage |
| `dI_dt`, `dV_dt` | Perubahan arus A/s dan voltage V/s memakai actual dt; 0 pada awal segment |
| `derivative_ready` | False pada awal segment; diagnostic, bukan predictor |
| `reference_status` | WARMUP, READY, atau REFERENCE_UNAVAILABLE |
| `readiness`, `reference_ready` | READY/NOT_READY dan flag Boolean; tanpa reference output target kosong |
| `reference_low_current_samples` | Jumlah sampel low-current yang telah diterima di fixed warm-up |
| `reference_lock_time_us` | Timestamp sampel pertama setelah warm-up saat reference dikunci; kosong jika belum/tidak tersedia |
| `V_ref`, `voltage_sag_actual` | Reference terkunci dan target `V_ref-Volt`, dalam V; bukan predictor regresi |
| `Curr_roll_mean_*s`, `Curr_roll_std_*s` | Trailing mean/std arus dalam A, half-open `(t-window,t]`, ddof=0 |
| `Curr_roll_count_*s` | Jumlah sampel yang tersedia dalam trailing window; diagnostic |
| `Curr_roll_coverage_*s` | `min(segment elapsed/window,1)`; tidak menyatakan sampling completeness |
| `Curr_roll_ready_*s` | Apakah segment sudah mencakup durasi window; diagnostic |

Warm-up default menerima low-current pada elapsed <=2 s; lock/penetapan unavailable terjadi pada sampel pertama setelah 2 s. Sampel pada tepat 2 s masih NOT_READY dan dapat masuk median. Tidak ada finalize memakai panjang flight. Reference terkunci bertahan melintasi gap; seluruh rolling/derivative history direset.

`reference_coverage.csv` menyertakan flight tanpa sampel retained, warm-up yang belum selesai, reference unavailable, jumlah ready samples dan denominator waktu. Ready exposure hanya menjumlahkan interval dalam segment yang sama dengan kedua endpoint reference-ready; persentase dengan denominator nol dibiarkan kosong/null.

## Kolom score dan keputusan Sprint 4

`scores.csv` hanya memuat assigned train dan validation flights; reserved test belum diberi prediksi. Field `split_role`, `used_for_fit`, `used_for_threshold`, dan `unseen_by_model` membedakan fit diagnostic dari healthy-validation calibration. Validation unseen oleh regressor tetapi dipakai untuk kalibrasi detector; bukan independent final holdout.

| Kolom | Makna |
|---|---|
| `predicted_voltage_sag` | Prediksi HGB dari Battery-D, dalam V; kosong saat NOT_READY |
| `residual_signed_v` | Actual voltage sag dikurangi predicted sag, V; nilai negatif tetap dipertahankan |
| `anomaly_score` | `max(0, residual_signed_v)`, V; bukan absolute error |
| `anomaly_raw` | Score strictly lebih besar dari healthy-validation threshold |
| `anomaly_duration_seconds` | Elapsed sejak raw pertama dalam run kontinu; 0 pada raw pertama/false/unready/gap |
| `anomaly_confirmed` | Raw kontinu mencapai persistence 0,5 s |
| `confirmed_episode_start` | Konfirmasi baru false->true; timer/episode restart setelah gap |
| `detector_state` | NOT_READY/NORMAL/SUSPECT/WARNING |

FPR sample memakai ready healthy-assumed samples sebagai denominator. Episode/menit memakai observed-ready intervals tanpa gap/warm-up; alarm pada training flights hanya in-sample diagnostics. `threshold.json` menyimpan source validation flight, jumlah sampel, score fingerprint dan metode percentile; model bundle menyimpan unit CurrTot `unverified` tanpa konversi.

## Synthetic copies dan metadata Sprint 5

`fault_manifest.csv` memuat setiap planned scenario. Logical scenario ID di-hash dari flight/family/severity/position; path run induk membedakan data/config/code versions. `scenario_csv` menunjuk full-flight CSV.gz hanya untuk applicable case; `scenario_sha256` memverifikasi byte gzip deterministik (mtime=0). Source feature snapshot tetap terpisah dan tidak ditimpa.

| Kolom metadata | Makna |
|---|---|
| `fault_applicable`, `evaluation_status`, `reason` | True/NOT_EVALUATED atau false/NOT_APPLICABLE dengan alasan; bukan hasil deteksi |
| `configured_dose`, `dose_unit` | Delta resistance dalam ohm atau sudden drop dalam V sesuai severity config |
| `selected_segment_id`, `selected_active_run_id` | Identitas continuity segment/run; active run kosong pada sudden drop |
| `segment_start_time_us`, `segment_end_time_us` | Batas run aktif terpilih atau continuous ready segment, inklusif |
| `requested_onset_time_us` | Fractional timestamp target, disimpan sebagai string Decimal untuk presisi |
| `candidate_onset_time_us`, `post_onset_available_seconds` | Sampel kandidat pertama pada/setelah target dan sisa durasinya, termasuk kasus ditolak |
| `onset_time_us`, `first_effect_time_us` | Aligned support onset dan sampel pertama dengan added sag >0; dapat berbeda untuk ramp/progressive |
| `fault_end_time_us`, `fault_duration_seconds`, `support_samples` | Batas support aktual inklusif, elapsed end-onset dan jumlah sampel support |
| `current_at_onset_a`, `mean_current_after_fault_a` | Arus pada onset dan rata-rata arus seluruh support |
| `max_added_sag_v`, `mean_added_sag_v`, `observability` | Statistik added sag pada support; null/NOT_APPLICABLE untuk kasus tidak berlaku |
| `frozen_v_ref`, `reference_policy` | Reference dari healthy features yang dipertahankan, tanpa refit |
| `post_fault_original_start_time_us`, `ending_policy` | Sampel pertama setelah support kembali memakai original telemetry; tidak ada ramp recovery sintetis |

Kolom tambahan setiap CSV synthetic: `is_synthetic`, `scenario_id`, `fault_family`, `fault_severity`, `fault_start_position`, `fault_applicable`, `fault_support`, `fault_label`, `fault_effect_present`, `fault_phase`, `Volt_healthy`, `voltage_sag_healthy`, `dV_dt_healthy`, `added_sag_v`. Kolom `Volt`, `voltage_sag_actual` dan `dV_dt` berisi hasil injeksi. Timestamp/current/CurrTot/reference/readiness dan seluruh predictor regresi tetap sama dengan healthy features.

`fault_label=1` persis selama fault support, termasuk onset ramp/progressive dengan effect=0. `fault_effect_present` membedakan sinyal nonzero. Phase adalah PRE_FAULT, FAULT_SUPPORT, POST_FAULT_ORIGINAL; core NOT_APPLICABLE copies menggunakan phase NOT_APPLICABLE dan mask false. Exporter tidak menyimpan CSV untuk NOT_APPLICABLE tetapi tetap menyimpan metadata pada ketiga tabel. dV_dt pada sampel post-support dapat memuat transient return-to-original; label support tetap false dan ending dinyatakan eksplisit.

Observability default berdasarkan max added sag: VERY_LOW <0,05 V, LOW [0,05;0,20), MEDIUM [0,20;0,50), HIGH >=0,50 V. Ini kategori efek sinyal sintetis, bukan diagnosis kesehatan fisik atau performa model. CSV synthetic tidak boleh digunakan untuk fitting model/scaler atau healthy-threshold calibration, termasuk baris pre-fault.

## Evaluasi LOAFO Sprint 6

`fold_assignments.csv` memuat semua 17 flight untuk setiap fold: train/validation/test/excluded, jumlah ready/retained samples, label aktivitas, reference coverage dan provenance flags. Hanya enam active/reference-ready flights masuk LOAFO; sebelas low-load belum dievaluasi false alarm pada sprint ini. `healthy_active_per_fold.csv` menghitung outer healthy sekali per flight, terpisah dari penggandaan baris pada synthetic scenarios. Score exports memakai CSV.gz; model/threshold/policy ada per fold.

| Kolom/metrik | Makna |
|---|---|
| `evaluation_status` | DETECTED, UNDETECTED, PREEXISTING_ALARM atau NOT_APPLICABLE pada hasil Sprint 6 |
| `tp`, `fp`, `tn`, `fn` | Conventional confusion pada reference-ready samples; positive label adalah fault support, positive prediction adalah confirmed warning |
| `preexisting_alarm` | Warning sudah confirmed pada onset tanpa episode start baru; seluruh incident contaminated |
| `incident_detected` | Ada episode confirmed baru dalam support dan tidak contaminated; berbeda dari sample TP |
| `candidate_new_confirmation_time_us` | Episode baru pertama dalam support, diagnostic; tidak menjadikan contaminated case sukses |
| `first_confirmation_time_us`, `delay_seconds` | Waktu konfirmasi incident valid dan `(confirmation-onset)/1e6`; sudah termasuk persistence, null jika undetected/contaminated |
| `detection_reason` | Alasan deteksi atau kegagalan, termasuk no raw exceedance versus raw tanpa konfirmasi baru |
| `pre_fault_samples`, `pre_fault_raw_fpr`, `pre_fault_fpr` | Ready samples sebelum aligned onset dan proporsi raw/confirmed alarm pada bagian tersebut |
| `deadline_le_*s` | Boolean keberhasilan incident dalam deadline, denominator seluruh applicable; undetected/contaminated false, NOT_APPLICABLE null |
| `incident_recall` | Detected incidents / applicable scenarios; NOT_APPLICABLE dikeluarkan |
| `synthetic_sample_fpr` | FP/(FP+TN) pada expanded synthetic copies; terpisah dari FPR outer healthy asli |
| `n_defined`, `std` | Jumlah nilai non-null dan population standard deviation pada macro distributions; undefined tetap null |

Support onset mungkin effect=0; first-effect tetap terpisah. Konfirmasi baru tepat onset boleh delay 0; warning yang terbawa ke onset tidak mendapat delay 0. Deteksi setelah akhir support tidak dihitung. Conventional TP tetap tercatat pada contaminated case, tetapi incident/deadline gagal. NOT_APPLICABLE tidak diprediksi dan seluruh confusion/delay/eligibility metrics null.

`summary.csv`/`results.json` membedakan pooled metrics yang dihitung dari gabungan counts/exposure dan macro statistik per fold. Delay pooled hanya incident-detected dengan n tersimpan; mean/median delay tidak mewakili kasus gagal. `failed_cases.csv` mempertahankan undetected, contaminated dan NOT_APPLICABLE. Semua policy/model hashes disimpan dalam `all_fold_policies_frozen.json` sebelum outer inference; `manifest.json` memverifikasi seluruh keluaran run. Evaluasi tetap historical development, tanpa independent final holdout baru.

## Ablation dan model comparison Sprint 7

`candidate` mengidentifikasi estimator dan exact feature set, misalnya `hgb_battery_a`, `rf_battery_e`, `if_direct`, `ocsvm_direct`. HGB/RF Battery-A–E memakai allowlist regresi yang sama untuk tiap set; Battery-E elapsed time hanya ablation. IF/OCSVM memakai registry direct `[voltage_sag_actual, Curr, CurrTot, dV_dt, dI_dt]` yang terpisah. Memakai voltage behavior pada direct anomaly baseline bukan izin memasukkannya ke regresi.

| Field / artefak | Makna Sprint 7 |
|---|---|
| `anomaly_score` | HGB/RF: positive residual V. IF/OCSVM: `-decision_function`, signed dan higher=more abnormal; bukan V, tanpa clipping ke nol |
| `score_unit`, `score_definition` | Metadata pada bundle/policy/threshold dan tabel skenario; membedakan score antar-family |
| `threshold`, `threshold_v` | P99 linear healthy validation per candidate/fold; `threshold_v` hanya regresi, null pada direct models; threshold signed direct sah |
| `predicted_voltage_sag`, `residual_signed_v` | Kosong pada IF/OCSVM, termasuk ready rows; direct models tidak memprediksi voltage |
| `regression_per_fold.csv` | MAE/RMSE/R² hanya HGB/RF; direct models tidak memiliki regression metrics |
| `fit_sample_indices` | Indeks nol relatif ke seluruh assigned healthy-ready training rows dalam urutan split/flight/time; seluruh indeks untuk HGB/RF/IF |
| `fit_sample_identities` | NamaFile/Inst/TimeUS/source_record yang dipakai estimator; OCSVM maksimal configured cap, tanpa validation/test |
| `ready_training_samples`, `model_fit_samples` | Jumlah full healthy-ready training dan jumlah sampel estimator; IF internal max_samples tetap parameter terpisah |
| `scaler_fit_samples` | OCSVM memakai seluruh healthy-ready train untuk StandardScaler, lalu subset deterministik untuk SVM; nol untuk model lain |
| `training_identity_sha256` | Hash urutan identitas seluruh assigned ready training rows; dapat dicocokkan dengan snapshot fitur |
| `paired_per_fold.csv` | Delta metrik candidate terhadap HGB Battery-D pada fold yang sama; selisih FPR/recall adalah proporsi, bukan percentage points yang sudah dikalikan 100 |
| `all_policies_frozen.json` | Hash model/threshold/selected policy semua candidate-fold sebelum outer inference pertama dalam eksperimen tersebut |

Score/threshold IF dan OCSVM tidak sebanding skalanya satu sama lain maupun dengan residual V. Threshold masing-masing dikalibrasi hanya healthy validation; durasi persistence tetap 0,5 s. Regression prediction pada synthetic copy boleh digunakan ulang hanya setelah exact identity/readiness/registered-predictor equality; IF/OCSVM menghitung ulang score karena voltage/sag/dV_dt berubah. Timer dimulai ulang setiap full-flight copy.

Denominator NA/undetected/pre-existing, sample confusion, observed exposure, pre-fault FPR, macro dan pooled mengikuti Sprint 6. Hasil antar-candidate bukan sampel independen: semua menggunakan flight dan skenario yang sama. Tidak ada winner/promosi otomatis atau perubahan operational config. Pemilihan model/feature dari outer results hanya development selection; status `RESEARCH_COMPARISON_ONLY`.

## Validation-only tuning Sprint 8

HGB Battery-D tetap. `candidate_id` seperti `p97p5_d0p2` mengidentifikasi percentile 97,5 dan persistence 0,2 s, bukan feature/model baru. Setiap fold mengkalibrasi empat threshold dari healthy validation scores, lalu mengukur 20 percentile/persistence combinations pada healthy validation dan 27 synthetic validation copies.

| Field / artefak | Makna |
|---|---|
| `source_role`, `validation_flight` | Sumber metrik selection wajib assigned validation; outer berlabel test dan tidak dipakai selection |
| `used_for_threshold` | True hanya ready healthy-validation rows untuk healthy quantile; false pada seluruh synthetic/outer rows |
| `used_for_policy_selection` | Per-row true pada ready validation, termasuk synthetic sebagai secondary metrics; false pada outer. Pada record NA validation, true berarti metadata dipertahankan dalam review budget, bukan inference/confusion eligibility |
| `threshold_candidates.json` | Empat linear percentile payloads dengan source healthy flight/sample count/fingerprint; persistence tidak mengubah quantile |
| `threshold_persistence_grid.csv` | Seluruh candidate metrics, healthy gates, alasan eligible/rejected, rank dan selected flag; 20 baris/fold |
| `healthy_confirmed_fpr`, `healthy_episodes_per_minute` | Dua safety screens validation; keduanya tidak boleh melebihi reference P99/0,5 s |
| `healthy_eligible`, `eligibility_reason`, `eligible_rank` | Lolos kedua healthy gate; undefined gate ditolak. Rank null untuk rejected |
| `abrupt_severe_le_1s_rate` | Severe increased-resistance/sudden-drop successes <=1 s / seluruh applicable subset |
| `gradual_observable_le_5s_rate` | Progressive MEDIUM/HIGH successes <=5 s / seluruh applicable subset |
| `selection_reason` | LEXICOGRAPHIC_VALIDATION_RANK atau NO_ELIGIBLE_CANDIDATE_REFERENCE_RETAINED; reference dapat menang ranking tanpa fallback |
| `selected_per_fold.json` | Assigned split, seluruh selected validation metrics/rationale, selected threshold provenance dan reference policy |
| `validation_fault_per_candidate.csv` | Seluruh candidate × validation scenario metrics/reasons/masks/observability, termasuk gagal dan NA; bukan outer result |
| `policy` | reference atau selected pada outer paired tables; selected tetap frozen meskipun outer memburuk |
| `paired_per_fold.csv` | Selected-minus-reference outer diagnostic deltas pada model/flight/coverage/scenario yang sama; tidak dipakai reselection |

Ranking lexicographic descending empat rate/metric (abrupt severe, observable gradual, incident recall, sample F1), lalu ascending detected median delay, mean delay, percentile, persistence. Undefined rate/F1 di bawah defined zero; undefined delay +infinity. Semua eligible dibandingkan, tanpa weighted aggregate score. Bila tidak ada eligible, reference dipertahankan.

`healthy_validation_scores.csv.gz` dan `validation_scenario_scores/*.csv.gz` berisi base prediction/score serta **reference P99/0,5 s decisions**. Kolom prediksi/score tidak bergantung pada candidate percentile/persistence. Keputusan 20 kandidat direproduksi dari timestamp/score dan parameter grid; metriknya tersedia lengkap di tabel candidate. Ini bukan 20 hasil model yang dilatih ulang. Synthetic predictor/readiness/reference tetap; residual/score dihitung ulang dari injected actual sag.

Model bundle memakai compatible experimental schema dengan `experiment_stage=8`, selected threshold dan training provenance; satu HGB-D per fold dipakai kedua outer policies. `all_policies_frozen.json` mencatat seluruh pre-outer inputs/outputs/model hashes. `manifest.json` meliputi full run. Threshold provenance tetap healthy-validation-only meskipun pemilihan percentile/duration memanfaatkan secondary synthetic validation.

Raw/confirmed sample FPR dan episode rate dibedakan. Validation healthy gates tidak menjamin outer false alarm turun. Tidak ada rollback/reselection berdasarkan outer atau perubahan operational config; status `RESEARCH_TUNED_CANDIDATE_ONLY`. Denominator NA/undetected/pre-existing/detected-only delay, gap/reference coverage dan current-based eligibility tetap seperti Sprint 6.

## Residual diagnostics dan dual-path Sprint 9

Seluruh residual diagnostic berikut dihitung setelah prediksi dari signed `actual sag - predicted sag`, bukan dari clipped anomaly score dan bukan predictor regresi. `dI_dt` tetap fitur arus asli pada registry Battery-D; tidak ada predictor baru yang ditambahkan.

| Field / artefak | Makna |
|---|---|
| `residual_delta_v` | Perubahan signed residual terhadap sampel ready sebelumnya dalam continuity segment; null pada awal/gap/unready |
| `residual_delta_v_per_s`, `residual_delta_ready` | Delta / actual dt dan validitasnya; tidak mengimputasi delta awal dengan nol |
| `residual_slope_*s` | OLS signed residual V/s dalam `(t-window,t]`, local time origin, minimal dua timestamp unik; null jika belum ready |
| `residual_slope_count_*s`, `residual_slope_ready_*s` | Jumlah sampel ready dalam window dan flag >=2; gap/segment/unready mereset history |
| `threshold_high_v`, `threshold_low_v` | Linear percentile positive residual dari assigned healthy validation saja; A/B memakai score > threshold |
| `threshold_slope_v_per_s` | Linear percentile signed slope healthy validation pada ready slope rows; dapat negatif, tanpa clipping; source count/hash/flight tersimpan |
| `path_a_raw`, `path_b_raw` | A: score > high; B: slope ready AND score > low AND slope > slope threshold |
| `path_a_duration_seconds`, `path_b_duration_seconds` | Raw elapsed individual dari actual timestamp; timer independen, masing-masing reset false/unready/gap |
| `path_a_confirmed`, `path_b_confirmed` | Durasi A >=0,5 s dan B >=configured 0,3/0,5 s |
| `path_a_episode_start`, `path_b_episode_start` | False→true masing-masing path; tidak sama dengan union episode count |
| `anomaly_confirmed`, `confirmed_episode_start` | OR confirmed paths dan false→true union/gap; path handoff selama union tetap warning tidak memulai union episode baru |
| `anomaly_raw`, `anomaly_duration_seconds` | OR raw paths dan max individual elapsed sebagai diagnostic; tidak ada persistence timer pada combined raw |
| `alarm_attribution` | A_ONLY/B_ONLY/BOTH/NONE per sample; overlap hanya dihitung sekali pada union confusion |
| `first_confirmation_path` | Attribution pada union episode start saja, null pada sampel lain |
| `incident_first_path` | Path pada first uncontaminated incident confirmation di support; null untuk carried alarm/undetected/NA |
| `confirmed_a_only_samples`, `confirmed_b_only_samples`, `confirmed_both_samples` | Decomposition union confirmed samples; null untuk NA, nol sah bila applicable tanpa warning |
| `mode` | REFERENCE, DUAL atau SINGLE_FALLBACK; fallback memakai P99/0,5 s reference, bukan standalone high percentile terpilih |

`path_a_selection.json` menyimpan dua standalone-A healthy diagnostics dan pilihan FPR/episodes/minute/high percentile ascending. Full `validation_healthy_grid.csv` memuat 144 pairs/fold; A yang tidak dipilih tetap tercatat dengan PATH_A_NOT_SELECTED. Kandidat dengan slope calibration unavailable/healthy gate gagal tidak eligible. Maksimal tiga selected-A pairs di-shortlist memakai ascending union FPR, episodes/minute dan full parameter tuple. `validation_refined_candidates.csv` dan `validation_fault_per_candidate.csv` hanya kandidat shortlist yang mengikuti secondary synthetic validation. Unrefined tidak boleh dipilih; minimal satu applicable validation scenario diperlukan untuk completed refinement.

`validation_fault_manifest.csv` mempertahankan seluruh 27 planned validation cases/fold termasuk NA atau bila tidak ada shortlist. Diagnostic base-score exports memakai REFERENCE decisions dengan seluruh residual windows dan path-B false; kandidat lain direplay dari diagnostic score/time dan policy. Synthetic validation tidak mengkalibrasi quantile. `selected_per_fold.json` memuat split, selected/fallback policy, selected Path A, shortlist IDs, rationale dan freeze-before-outer flag.

`baseline_validation_episode_context.csv` memuat retrospective numeric ranges current/residual dan max abs dI_dt per healthy-assumed baseline warning episode. Ini metadata context, bukan input detector, categorical transient label atau penjelasan kausal kerusakan fisik. Row-level residual deltas/slopes tetap causal.

Promotion memakai predeclared arithmetic **macro mean pada pasangan fold identik**, setiap metric wajib defined di seluruh folds: healthy FPR dan episode rate tidak naik; sample F1 dan deadline <=1/2/5/10/30 s tidak turun; mean per-fold detected mean delay turun strictly. Minimal satu fold harus memiliki refined dual candidate. Pooled tetap dilaporkan terpisah dan tidak dipakai mengganti gate setelah outer terlihat. Keputusan `NOT_PROMOTED` atau `PROMOTED_RESEARCH_ONLY` adalah research gate; operational config dan dual_path.enabled tetap tidak berubah. Low-load dan hardware/replay readiness belum dinilai.

## Sprint 10 — low-load/domain exports

`artifacts/low_load/<run_id>/` memakai frozen Sprint 6 single-path HGB-D/P99/0,5 s. Seluruh flight healthy-assumed LOW_LOAD_OR_IDLE dinilai hanya untuk false alarm; historical development, bukan independent final holdout. Arus rendah tidak membuktikan operasi di ground.

| Field/berkas | Makna |
|---|---|
| `split_role=low_load` | Cohort evaluasi; tidak mengubah peran flight pada training/calibration sumber |
| `source_training_flight`, `source_validation_flight` | Membership flight pada split bundle asli, bukan penggunaan saat Sprint 10 |
| `unseen_by_model` | Flight tidak termasuk source training; tidak berarti historis belum pernah diperiksa peneliti |
| `unseen_by_detector` | Flight tidak termasuk source training maupun source validation/calibration |
| `used_for_fit`, `used_for_threshold`, `used_for_policy_selection` | False pada seluruh low-load exports; stage ini tidak melakukan ketiganya |
| `primary_fold`, `primary` | First sorted frozen fold dipilih sebelum low-load prediction; fold lain sensitivity, tanpa winner selection |
| `ready_samples`, `raw_alarm_samples`, `confirmed_alarm_samples` | Denominator/count samples ready saja; raw memakai score > frozen threshold, confirmed memakai frozen persistence |
| `observed_ready_seconds` | Actual intervals kedua endpoint ready, stream/segment sama dan dt <= gap threshold; warm-up/gap tidak dihitung |
| `confirmed_episodes_per_minute` | 60 × fresh confirmed episode count / observed ready seconds; null bila exposure nol |
| `outside_any_training_range` | Ready predictor mana pun di luar marginal min/max original training; null unready, metadata deskriptif, bukan input/alarm baru |
| `flight_domain_profiles.csv` | Seluruh flight: initial accepted Volt, V_ref dan n/mean/population std/median/min/max/P05/P95; Volt memakai all accepted rows, predictors/actual sag memakai ready rows |
| `response_distributions.csv` | Predicted sag, signed residual dan positive score per model/low-load flight; satuan V |
| `outside_training_ranges.csv` | Per model/flight/predictor: original training min/max dan ready outside count/fraction; equality termasuk di dalam range |
| `episode_context.csv` | Semua confirmed episodes: first/last confirmed timestamp, sample count, Curr/residual range dan maximum abs dI_dt; tanpa physical-cause label |
| `pre_evaluation_freeze.json` | Hash protocol, exact config/registry dan train-only ranges sebelum prediction; protocol menyimpan source model/threshold hashes |

Tidak ada penjumlahan exposure enam model menjadi flight independen. NO_READY_EXPOSURE tetap tercatat dengan null rates dan tanpa prediksi. CurrTot tetap `unverified`, tanpa konversi. Plot contoh sensitivity yang ditambahkan setelah melihat hasil diberi provenance eksplisit; primary allocation, model, threshold dan persistence tidak berubah. Hardware/replay readiness tetap belum dinilai.

## Sprint 11 — consolidated research package

`artifacts/research_evaluation/<run_id>/` adalah konsolidasi hasil historical development. Istilah research-final tidak menyatakan final holdout independen, hardware readiness, keberhasilan semua research targets atau penyelesaian Sprint 13.

| Field/berkas | Makna |
|---|---|
| `orchestration_plan.json` | Audit/data/config/protocol/version/code identity, fixed stage order dan source/selection rules sebelum engine dijalankan |
| `source_inventory.json/csv` | Stage/run/path, manifest/summary hash, jumlah verified outputs, pre-outer frozen files dan model bundles |
| `source_results_frozen.json` | Exact completed source run/manifest IDs yang dipakai report; tidak berubah berdasarkan hasil grouping |
| `execution_trace.csv` | Urutan stage, VERIFIED_REUSE atau BUILT_MISSING_STAGE, dengan training-stage-built flag saat package dibuat |
| `source_stages_built_at_creation`, `training_stages_built_at_creation` | Pekerjaan saat package pertama dibuat; cache mengembalikan summary package asli, bukan mengklaim fitting baru setiap pemanggilan |
| `model_bundle_registry.csv` | Referensi seluruh original model bundles beserta SHA, source manifest dan ordered predictors; tidak menyalin atau refit weights |
| `synthetic_series.csv` | Pooled scenario metrics per experiment/candidate/policy secara terpisah, direkonsiliasi ke source summary |
| `synthetic_group_metrics.csv` | Separate series menurut overall/flight/family/severity/onset position/observability; bukan gabungan kandidat sebagai independent incidents |
| `failed_or_not_applicable_cases.csv` | Missed, contaminated dan NA records; reason/detection reason dan null delay dipertahankan |
| `dataset/`, `baseline/`, `synthetic/`, `selection/`, `low_load/` | Salinan kecil tabel/selected policy/promotion dari completed sources, byte-identical dan di-hash pada package manifest |
| `gate_checks.csv` | Status PASS/FAIL/WARNING/NOT_EVALUATED; software completion tidak mengubah detector/research gate yang gagal |
| `results.json`, `summary.json`, `research_report.md` | Factual consolidated metrics/source scopes; tidak memilih winner atau mengubah threshold/config |
| `sources/<config-data-freeze-key>.json` | Catalog completed sources untuk pengulangan; tidak menimpa sprint pointers historis |

Config/data/source corruption menghentikan proses tanpa silent rebuild. Source feature manifest dan declared preprocessing parent wajib cocok. Current dependency/code versions tidak menghapus provenance versi model historis. Angka sample confusion, incident recall, delay detected-n, deadline denominator, healthy ready exposure dan low-load unique flight count mempertahankan makna stage asal; NULL/undefined tidak diganti nol. Tidak ada klaim replay equivalence sampai Sprint 12.

## Sprint 12 — streaming replay exports

`artifacts/replay/<run_id>/` memverifikasi software baseline single-path HGB-D/P99/0,5 s; tidak mengaktifkan `replay.enabled` atau hardware. Frozen dataset/config/protocol/model/threshold tidak berubah. Dua clock berbeda dipertahankan: telemetry TimeUS untuk causal state/signal warning delay, monotonic host clock untuk processing latency.

| Field/berkas | Makna |
|---|---|
| `replay_protocol.json`, `pre_replay_freeze.json` | Exact source hashes/splits/policies, cache/timing/equivalence rules, dependency versions dan freeze sebelum inference |
| `traces/<cohort>/<job>.csv.gz` | Raw input, causal features/reference/readiness, residual diagnostics, raw/confirmed/episode/state, support mask untuk evaluasi sesudah keputusan dan timing per sample |
| `accepted` | Raw sample lolos parser; pada API invalid numerics menghasilkan dropped output tanpa feature/persistence state advance |
| `prediction_mode=UNCACHED` | Single-sample model call baru, seluruh healthy ready controls memakai mode ini; first call tidak dihapus |
| `prediction_mode=EXACT_INPUT_CACHE` | Prediksi dari identical model identity dan exact ordered predictor tuple yang sudah diamati; hanya synthetic dapat memakai cache pada actual experiment |
| `prediction_mode=NOT_READY` | Tidak ada inference/score; warm-up/reference unavailable tetap terlihat |
| `parse_ns`, `features_ns`, `prediction_ns`, `diagnostics_detector_ns` | Monotonic duration per phase dalam nanodetik; preprocessing dibagi parsing dan causal features, detector mencakup residual/slope diagnostics |
| `end_to_end_ns` | Jumlah empat phase di atas; in-process sample latency, tidak termasuk source/model loading, disk atau CSV export |
| `latency_summary.csv` | Population terpisah menurut cohort/prediction mode/phase; count, mean/p50/p95/p99/max ms dan samples > nominal processing interval |
| `jobs.csv`, `equivalence.json` | Job/cohort/fold/flight, row counts, trace path/hash, batch-stream equivalence dan maximum numeric errors |
| `fault_per_scenario.csv` | Incident/deadline/confusion/preexisting/NA semantics tetap dari frozen source, termasuk detected-delay n melalui pooled summary |
| `confirmation_processing_latency_ms` | Host sample latency pada first new uncontaminated incident confirmation saja; null untuk missed/NA/preexisting |
| `raw_run_start_time_us` | Awal actual raw run yang menghasilkan confirmation, dapat mendahului onset jika saat onset masih SUSPECT; bukan bukti carried confirmed warning |
| `persistence_elapsed_at_confirmation_seconds` | Durasi raw run yang sudah dipakai dalam confirmation; tidak ditambahkan lagi ke `delay_seconds` |
| `delay_seconds` | Actual telemetry onset→first fresh confirmed warning dalam fault support; persistence sudah tercakup, bukan elapsed accelerated wall execution |
| `healthy_per_flight.csv` | Unique-flight ready samples/FPR/exposure/episode rate untuk active own-fold dan low-load primary-fold; exposure tidak digandakan antar model |

Numeric equivalence abs/rel tolerance 1e-10; Boolean/string/null states harus exact. Cache tidak menyimpan residual, reference, slope, timing atau alarm state. Cached synthetic latency tidak dipresentasikan sebagai uncached model performance. Pengulangan completed CLI memverifikasi/reuse recorded measurements, tanpa diam-diam mengukur ulang atau menimpa latency. Paced replay tidak dilakukan. Status `BASELINE_SOFTWARE_REPLAY_ONLY`, dual-path NOT_PROMOTED, tanpa hardware/emergency-readiness claim.


## Sprint 13 - final audit evidence

| Berkas/field | Makna |
|---|---|
| `FINAL_RESEARCH_RESULTS.md` | Factual results seluruh seri, denominators, negative results, limitations dan gate PASS/WARNING/FAIL |
| `FINAL_VERIFICATION.md` | Command/test/audit coverage dan batas software validation |
| `artifacts/sprint13/before_snapshot.json` | SHA256 prior completed run outputs, frozen Sprint 0 inputs dan prior sprint evidence |
| `pre_execution_manifest.json`, `source_manifest.json` | Program hashes sebelum dan sesudah final audit |
| `commands_verification.json`, `tests.xml` | Actual argv/cwd/return codes, cache checks dan JUnit test counts tanpa failure/error/skip |
| `audits/*.json` | Hasil verifier independen yang ditangkap tanpa overwrite evidence historis |
| `research_result.json`, `replay_result.json` | Pointer current research consolidation dan recorded baseline replay yang diverifikasi |
| `final_results.json`, `verification.json` | Verified source results dan coverage audit, terpisah dari keputusan promotion |
| `gate_checks.csv` | `check`, `status`, `scope`, `detail`; FAIL riset tidak diganti PASS karena software berhasil |
| `preservation_verification.json` | Hasil pencocokan prior input/model/threshold/artifact/evidence hashes |
| `document_manifest.json`, `evidence_manifest.json` | Hash final documents dan evidence audit, tanpa self-hash sirkular |
| `attempt1/` | Catatan transparan percobaan pytest dari parent cwd yang dihentikan; digantikan full suite dari project root |

`software_audit_status=PASS` hanya menyatakan pemeriksaan teknis selesai. `research_gate_status=NOT_PROMOTED` dan progressive target NOT_MET tetap hasil negatif. `hardware_ready=false` bukan metrik latency dan tidak berubah oleh sukses replay. Undefined rate/delay tetap null; repeated model/cohort/policy bukan independent flights atau incidents tambahan.
