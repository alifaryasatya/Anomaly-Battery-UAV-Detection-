# PRD — UAV Battery Electrical Anomaly Detection V2

## 1. Tujuan dan ruang lingkup

Membangun ulang program penelitian deteksi respons listrik baterai abnormal dengan telemetry baterai saja. HGB memodelkan voltage sag healthy, residual mendeteksi penyimpangan, dan persistence mengonfirmasi warning. Pengguna utama adalah peneliti S1 yang membutuhkan eksperimen dapat direproduksi dan hasil per flight yang dapat ditelusuri.

Target penelitian: deteksi mayoritas abrupt/severe anomaly dalam <=1 s dan gradual anomaly yang sufficiently observable dalam <=5 s, dengan false alarm rendah dan konsisten antar-flight. Target ini belum merupakan hasil V2. Nilai metrik V1 tidak boleh diklaim sebagai performa V2.

Keluaran adalah warning anomali listrik, bukan aktuasi emergency switching. Hardware switching berada di luar ruang lingkup implementasi awal.

## 2. Prinsip ilmiah

- HGB + residual adalah arsitektur operasional tetap. RF residual, Isolation Forest, dan OCSVM adalah pembanding.
- Tidak ada actual voltage, voltage sag aktual, dV_dt, residual, anomaly score, atau Volt × Curr pada predictor regresi.
- Direct anomaly baselines boleh melihat voltage_sag_actual dan dV_dt.
- Train hanya healthy training flights; synthetic anomalies tidak pernah masuk training model/scaler.
- Seluruh split dilakukan per flight. Tidak ada random row split dan tidak ada overlap group.
- Tuning hanya pada development/validation. Outer test mengukur hasil, tidak merevisi kandidat.
- Data historis sudah diperiksa pada V1. Tulis hasil V2 sebagai historical development cross-validation.
- Health labels merupakan asumsi sumber. Tidak ada real battery fault ground truth.

## 3. Data

Input wajib: `NamaFile, TimeUS, Inst, Volt, Curr, CurrTot`. Throttle tidak menjadi input, fitur, syarat fault, maupun sumber reference voltage.

Paket awal menyertakan 17 historical flights dan enam salinan active historis. Inventory dan manifest adalah sumber jumlah aktual. Jangan menduplikasi sampel dengan membaca combined dataset dan salinan per flight secara bersamaan.

Schema/unit harus diaudit, terutama CurrTot dan kemungkinan instance baterai lebih dari satu. Jika satu flight memiliki beberapa instance, pilih secara eksplisit atau perlakukan sebagai stream berbeda; jangan mencampur stream pada timestamp yang sama. Tanda arus negatif dan reset CurrTot harus dilaporkan.

Data eksternal dan konversi `.BIN`/`.ulg` tidak menjadi dependency paket awal. Tambahan dataset di masa depan harus memiliki provenance dan identitas group yang jelas.

## 4. Cleaning dan continuity

Urutkan per stream dan TimeUS. Catat missing/nonfinite, duplicate timestamps, nilai negatif, dan timestamp tidak valid. Jangan mengisi data memakai sampel masa depan. Pilih reject/drop dengan laporan, atau causal forward-fill dengan batas waktu configurable. Gap terhitung ulang setelah drop.

`nominal_dt = 1 / nominal_frequency_hz`; `gap_threshold = nominal_dt * max_gap_multiplier`, awalnya 0,5 s. Sampel setelah gap memulai segment baru. Semua derivative, rolling, lag, residual trend, persistence, dan episode counter bekerja per flight/instance/segment.

Boundary derivative bernilai 0 dengan flag `derivative_ready=false`. Rolling hanya memakai histori yang tersedia; std awal 0 dengan flag coverage. Flag readiness bukan predictor otomatis. Interval dt aktual dipakai; baris pertama tidak dihitung memakai median masa depan.

## 5. Definisi active-load tanpa throttle

Sampel active bila Curr >= min_current_a. Segmen valid memerlukan kontinuitas minimal min_duration_seconds berdasarkan timestamp aktual. Laporan mencakup distribusi arus, proporsi waktu active, durasi segmen valid, dan sensitivitas terhadap ambang.

Label V2: `ACTIVE_LOAD` atau `LOW_LOAD_OR_IDLE`. Istilah low load tidak membuktikan UAV berada di ground. Label historis V1 ditampilkan terpisah sebagai provenance. Jumlah active V2 harus dihitung ulang, tidak dipaksa menjadi enam.

## 6. V_ref dan target

Warm-up awal berdurasi tetap 2 s sebagai kandidat awal. Pilih sampel low-current yang sudah diterima; setelah jendela berakhir, kunci median Volt sebagai V_ref jika jumlah sampel cukup. Sebelum reference tersedia, output `NOT_READY`. Jika low-current tidak cukup, reference unavailable dan flight dilaporkan tidak evaluable; jangan diam-diam memilih reference dari seluruh flight.

`voltage_sag_actual = V_ref - Volt`. Sintetik mengubah Volt pada salinan setelah reference terkunci; jangan menghitung ulang V_ref dari fault-contaminated telemetry. Delay dan denominators harus menjelaskan exclusion warm-up dan coverage.

## 7. Feature registry dan ablation

Battery-D adalah adaptasi Set D, bukan exact Set D V1.

| Set | Predictor |
|---|---|
| Battery-A | Curr |
| Battery-B | Curr, CurrTot |
| Battery-C | Battery-B + dI_dt |
| Battery-D | Battery-C + trailing mean/std Curr pada 0,5 s dan 1,0 s |
| Battery-E | Battery-D + elapsed_time, hanya studi ablation |

Operasional awal Battery-D. `dt`, TimeUS, NamaFile, Inst, elapsed_time, readiness, Volt dan V_ref tidak otomatis menjadi predictor. Feature registry mengontrol seluruh eksperimen. Lag opsional harus causal dan reset gap. CurrTot hanya dimasukkan setelah audit unit/reset; ablation A adalah baseline tanpa CurrTot.

Rolling window memakai detik dan timestamp aktual, `ddof=0`; bukan asumsi lima baris sama dengan 0,5 s. dI_dt dan dV_dt dalam unit per detik.

## 8. Model dan score

HGB dilatih pada target voltage sag healthy. RF memakai predictor yang identik untuk perbandingan. Residual signed `r = actual - predicted`; anomaly score `max(0, r)`. Absolute error tidak dipakai sebagai score default.

Isolation Forest/OCSVM memakai `[voltage_sag_actual, Curr, CurrTot, dV_dt, dI_dt]`. OCSVM StandardScaler fit hanya train. Sample cap OCSVM harus deterministik dan hanya dari train. Threshold score baseline memakai healthy validation; arah score harus konsisten (lebih tinggi = lebih abnormal).

## 9. Threshold, persistence, dual-path

Single-path baseline memakai healthy-validation percentile 99 dan persistence referensi 0,5 s. Setelah baseline dibekukan, studi threshold 95/97,5/99/99,5 serta persistence 0,2/0,3/0,5/0,7/1,0 s dilakukan di validation saja.

`raw = score > threshold`. Run mulai pada timestamp raw pertama; confirmed setelah elapsed >= duration. False atau gap mereset timer. Laporkan `anomaly_raw`, `anomaly_duration_seconds`, `anomaly_confirmed`.

Dual-path adalah eksperimen opsional: A `r > T_high`, B `r > T_low AND causal_slope > T_slope`; timer terpisah, keputusan OR. Slope OLS trailing timestamp window. Parameter dipilih validation-only dengan search budget yang dibekukan sebelum outer evaluation.

Tolak kandidat yang menaikkan healthy validation FPR atau episodes/minute terhadap baseline. Di kandidat eligible, prioritaskan abrupt/severe <=1 s, lalu observable gradual <=5 s, F1/recall, kemudian median/mean delay. Tie-break deterministik. Jika tidak ada kandidat aman, pertahankan single-path. Keputusan historical LOAFO tidak menjadi klaim final independent holdout.

## 10. Evaluasi per flight

Primary: Leave-One-Active-Load-Flight-Out. Outer test satu flight, validation flight berikutnya secara cyclic dari daftar terurut, sisa train. Minimum tiga active-load flights. Fit preprocessing yang belajar distribusi hanya train; threshold/persistence hanya validation. Simpan assignment, konfigurasi, seed, feature list, hash data dan fitted objects tiap fold.

Sehat low-load dievaluasi terpisah. Tandai per-flight apakah benar-benar unseen oleh model yang mengevaluasinya. Jangan melaporkan alarm pada training flights sebagai unseen FPR.

## 11. Synthetic experimental design

Tiga fault family x tiga severity x tiga start position (40/50/60%) pada setiap eligible test flight. Semua nilai configurable. Pilih segmen deterministik, misalnya segmen valid terpanjang dengan tie-break timestamp paling awal.

- Increased resistance: added_sag(t) = Curr(t) * delta_R.
- Progressive: added_sag(t) = Curr(t) * delta_R_max * progress(t); progress memakai waktu dari start sampai akhir segmen aktif dan dipotong ke [0,1].
- Sudden drop: configured drop dalam V, opsional ramp dalam detik (awal 0,4 s).

Current-dependent injection hanya pada sisa segmen active-load yang dipilih dan tidak melintasi gap. Bila tidak ada segmen valid atau waktu pasca-start tidak cukup, `fault_applicable=false`, `evaluation_status=NOT_APPLICABLE`, tulis reason. Sudden drop dapat berlaku pada low-load yang siap dievaluasi.

Simpan onset, mask fault support, duration, current start, segment bounds, severity, posisi, max/mean added sag, dan mean current after fault. Fault ending/recovery dan label per row harus eksplisit.

Observability VERY_LOW/LOW/MEDIUM/HIGH berdasarkan max added sag configurable (0,05/0,20/0,50 V awal). Ini kategori efek sinyal sintetis, bukan klasifikasi kesehatan fisik baterai.

## 12. Metrik dan denominator

Regression: MAE/RMSE V, R², per flight/fold serta mean/std/median/min/max.

Detection: precision, recall, F1, synthetic sample FPR, healthy untouched active FPR, pre-fault FPR, episodes/flight, episodes/minute, number applicable/detected/undetected, mean/median/std delay dan deadline rates <=1/2/5/10/30 s. Laporkan macro per-fold dan pooled counts terpisah.

NOT_APPLICABLE tidak masuk TP/FP/TN/FN atau recall denominator. Undetected applicable masuk denominator semua deadline dan dinyatakan gagal. Delay dihitung dari actual onset timestamp sampai first confirmed sample di dalam fault support. Pre-existing alarm saat onset tidak boleh disebut deteksi baru dengan delay nol: tandai `PREEXISTING_ALARM`, tampilkan conventional sample metrics terpisah, dan keluarkan dari numerator incident detection. Laporkan jumlah contaminated scenarios; jangan sembunyikan sebagai success.

Delay rata-rata hanya pada detected disertai jumlah undetected dan deadline rates. False alarms/minute memakai observed telemetry duration, tidak menghitung gap sebagai waktu monitored. Episode baru setelah gap selalu dihitung baru.

## 13. Keluaran

Wajib: dataset/time-gap/activity audit, feature registry, assignments, regression/anomaly/fault per fold, applicable/observability tables, model comparison, ablation tables, threshold/persistence grid, selected parameters, healthy low-load false alarms, paired dual-path comparison, promotion decision, plots dan factual research report.

Plots matplotlib: count/sample/current per flight, regression actual/prediction/residual, per-fold RMSE/F1/recall/FPR/delay, fault traces + threshold + onset + confirmation, severity/deadline matrices, ablation RMSE/F1. Unit terbaca dan failed cases tetap terlihat.

## 14. Replay dan final gate

Replay setelah detector dibekukan dan gate sehat terlewati. State: NOT_READY/NORMAL/SUSPECT/WARNING. Batch causal dan incremental replay harus menghasilkan nilai/keputusan yang setara. Ukur preprocessing+prediction+detector latency p50/p95/p99, total warning delay, persistence dan gap resets; jangan menjumlahkan persistence lagi bila sudah termasuk delay.

Latency diukur dengan monotonic clock. Timestamp telemetry menentukan delay sinyal. Deadline warning belum membuktikan kesiapan emergency switching hardware.

## 15. Acceptance criteria

1. Semua input V2 tidak memerlukan ThO; source data/checksum tetap utuh.
2. Schema/instance/unit/activity audit tersedia sebelum training.
3. Semua temporal state causal dan reset di gap; replay warm-up tidak bergantung panjang akhir flight.
4. Leakage tests menolak predictor voltage aktual dan group overlap.
5. Setiap threshold/persistence memiliki sumber validation tercatat.
6. Failed/NOT_APPLICABLE/pre-existing-alarm scenarios dilaporkan dengan denominator benar.
7. Promosi dual-path memerlukan FPR dan episodes/minute tidak meningkat, F1/deadline tidak turun, dan mean delay membaik dibanding baseline pada pasangan eksperimen identik.
8. Jika gate gagal, status NOT_PROMOTED; jangan mengaktifkan otomatis.
9. Laporan menyatakan keterbatasan data healthy-only, jumlah flight, unit belum terverifikasi, dan domain operasi.

Pernyataan wajib laporan:

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
