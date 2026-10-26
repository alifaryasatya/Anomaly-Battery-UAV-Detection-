# Frozen experiment protocol — sprint0-v1

Tanggal pembekuan: 20 Oktober 2026 (Asia/Jakarta), sebelum training atau pengukuran model V2. Scope adalah battery-only electrical anomaly warning, bukan diagnosis kerusakan fisik atau emergency switching. Parameter numerik model, fitur, aktivitas, reference, fault dan gate berasal dari salinan `config_used.yaml`. Protocol baru diperlukan jika keputusan eksperimen diubah; hasil setelah inspeksi outer test harus disebut exploratory/development selection.

## Data dan preprocessing

Satu input: `data/raw/dataset_baterai.csv`, 17 historical flights. Salinan enam active flights dan label V1 hanya provenance; tidak menentukan eligibility V2. Unit CurrTot `unverified`, nilai sumber tanpa konversi. Battery-A adalah kontrol tanpa CurrTot; penurunan CurrTot memerlukan review sebelum Battery-B–E dipakai. Healthy merupakan asumsi sumber. Tidak ada real battery fault ground truth atau final holdout baru.

State dipisah per flight/instance/segment. Reject multi-instance tanpa pilihan eksplisit dan reject duplicate timestamp. Drop missing/nonfinite atau timestamp/instance tidak valid dengan laporan, tanpa fill/interpolasi. Curr negatif dipertahankan dan dilaporkan. Urutkan stream; hitung gap ulang setelah drop. Timestamp valid adalah integer nonnegatif, instance integer nonnegatif. Gap `dt > 0.5 s` reset seluruh temporal state. Interval pertama tidak mendapat dt dari median masa depan.

Active adalah `Curr >= 1 A` kontinu minimal 1 s. Exposure hanya interval valid dengan kedua endpoint active, pada run yang memenuhi minimum duration. Observed exposure tidak melintasi gap. Flight ACTIVE_LOAD bila active exposure / observed exposure >= 5%; zero-exposure tidak eligible. Sensitivitas audit: current [0.5, 1.0, 2.0] A x duration [0.5, 1.0, 2.0] s, ratio tetap 5%; default tidak dipilih ulang dari performa outer. Eligibility, coverage dan minimum tiga active flights menjadi gate sebelum LOAFO.

Reference per stream menggunakan warm-up pertama 2 s: terima sampel bertimestamp relatif <= 2 s dengan Curr <= 1 A; lock median hanya setelah jendela selesai dengan minimal lima sampel. Jika tak cukup, REFERENCE_UNAVAILABLE untuk stream. Gap mereset rolling/derivative/persistence, reference yang telah terkunci tetap; gap selama warm-up tidak memperpanjang jendela. Output sebelum lock NOT_READY. Synthetic dibuat setelah lock dan tidak menghitung ulang V_ref. Laporan menyertakan coverage dan pengecualian warm-up.

Battery-A/B/C/D/E sesuai PRD. Rolling trailing memakai timestamp dengan interval `(t-window, t]`, histori tersedia saja, ddof=0. Derivative pada awal segment 0 dengan readiness false. Elapsed time hanya Battery-E. Regression predictor tidak boleh memakai actual voltage/sag, dV_dt, residual, score, V_ref, voltage x current, identifier, dt atau readiness. Feature registry menjadi satu sumber fitur. Semua transform belajar distribusi fit hanya train.

## Split, model, dan baseline

Seed 42. Urutkan eligible flight IDs; outer test flight i, validation flight (i+1) modulo N, sisanya healthy train. Simpan actual assignment setelah audit aktivitas; tidak menetapkan enam flight V1 sebagai active V2. Nol overlap flight. Evaluasi outer hanya setelah kandidat fold terpilih dan dibekukan. Seluruh hasil disebut historical development cross-validation.

HGB Battery-D adalah model utama, RF predictor identik sebagai pembanding. Score `max(0, actual sag - predicted sag)`. Single-path pilot: percentile 99 healthy validation, `score > threshold`, persistence 0.5 s sejak timestamp raw pertama. False/gap reset timer; konfirmasi ketika elapsed >= duration. Parameter model dari config, tanpa tuning hyperparameter tersembunyi.

IF/OCSVM memakai registry direct baseline pada config. Higher score harus lebih abnormal; scaler OCSVM train-only. Sample cap 2500: pilihan uniform tanpa replacement dari train memakai seed 42, indeks hasil diurutkan dan disimpan. Threshold healthy validation. Battery-A–E dan semua pembanding memakai fold/fault/coverage identik dan exact feature lists tercatat. Pemilihan family dari hasil outer adalah development selection.

## Synthetic dan selection budget

Tiga family x tiga severity x tiga posisi = 27 skenario per eligible flight. Nilai resistance, progressive delta R, drop, ramp, dan observability mengikuti config. Pilih active segment valid terpanjang; tie timestamp awal. Posisi adalah fraksi durasi aktual segment; onset sampel pertama pada/setelah waktu posisi. Current-dependent fault hanya di sisa segment active yang sama. Progressive progress memakai waktu dari onset aktual ke akhir segment. Sudden drop memakai longest continuous ready segment (tie paling awal), boleh low-load. Minimum post-onset support 1 s; selebihnya NOT_APPLICABLE dengan alasan. Support berakhir di akhir segment tanpa recovery sintetis; mask eksplisit. Observability dari max added sag: <0.05 VERY_LOW, [0.05,0.20) LOW, [0.20,0.50) MEDIUM, >=0.50 V HIGH.

Single-path grid 4 percentile x 5 persistence = 20 kandidat/fold pada validation saja. Healthy validation mengkalibrasi percentile, synthetic validation memberi metrik sekunder. Tolak kandidat dengan healthy FPR atau confirmed episodes/minute melebihi reference baseline. Ranking eligible, lexicographic descending: severe sudden/resistance deadline <=1 s, progressive MEDIUM/HIGH deadline <=5 s, incident recall, sample F1; lalu ascending median delay, mean delay. Undefined deadline/recall/F1 bukan sukses, undefined delay +infinity. Tie terakhir ascending tuple percentile, persistence. Jika tidak ada eligible, baseline dipertahankan.

Dual-path tetap disabled secara operasional. Eksperimen: Path A memakai high percentile [99,99.5] dan persistence baseline 0.5 s; Path B grid low percentile 3 x slope percentile 4 x slope window 3 x persistence 2. Total maksimal 144 pasangan/fold, bukan budget adaptif setelah test terlihat. Residual slope OLS causal dalam `(t-window,t]`, minimal dua timestamp unik; belum ready berarti Path B false. Slope percentile dari healthy validation yang ready. Timer A/B terpisah, keputusan OR. Healthy-screen kandidat, shortlist maksimal tiga memakai FPR, episodes/minute lalu ascending parameter tuple; refinement synthetic validation wajib memakai ranking di atas. Config terpilih disimpan sebelum outer. Jika tidak aman, pertahankan single-path.

## Metrics, gate, dan replay

Regression MAE/RMSE V, R2 per fold serta mean/std/median/min/max. Detection melaporkan sample confusion counts, precision/recall/F1, healthy active FPR, pre-fault FPR, episodes/flight, episodes/minute, applicable/detected/undetected, observability, coverage dan deadline <=1/2/5/10/30 s. Macro dan pooled counts terpisah. Standard deviation populasi (ddof=0); undefined rates/R2 ditulis null beserta denominator, tidak dipaksa nol. Episode berdasar confirmed false->true; gap selalu memulai episode baru. Monitored time adalah jumlah interval telemetry valid pada coverage evaluable, tanpa gap/warm-up.

NOT_APPLICABLE dikeluarkan dari confusion/recall denominator. Undetected applicable gagal semua deadline. Delay hanya first confirmation baru di dalam support sejak onset aktual. Alarm sudah confirmed saat onset adalah PREEXISTING_ALARM, tidak masuk numerator incident detection; tetap dalam denominator applicable/deadline dan conventional sample metrics terpisah. Delay mean/median/std hanya pada incident detected, selalu disertai undetected dan contaminated counts.

Low-load healthy hanya false alarm terpisah; label unseen per flight harus eksplisit. Alarm training flights bukan unseen FPR. Promotion dual-path pada pasangan outer identik memerlukan mean healthy FPR dan episodes/minute tidak naik, F1 dan seluruh deadline tidak turun, mean incident delay membaik; undefined gate berarti tidak lolos. Jika gagal status NOT_PROMOTED. Outer results tidak merevisi kandidat. Target 80% abrupt/severe <=1 s dan 70% observable gradual <=5 s masih aspirational, bukan hasil.

Replay dilakukan setelah freeze detector: NOT_READY/NORMAL/SUSPECT/WARNING, batch causal setara incremental. Telemetry timestamp untuk signal delay, monotonic clock untuk latency p50/p95/p99. Persistence sudah termasuk end-to-end delay. Baseline replay boleh untuk validasi software; emergency candidate mensyaratkan gate sehat dan tidak membuktikan hardware readiness.

Simpan data hashes, config, dependency versions, features, fitted objects, seeds, assignments, selected candidate provenance, failed cases dan factual CSV/JSON. Jangan mengubah denominator/gate setelah hasil diketahui.

> Synthetic faults represent controlled battery-related electrical anomalies for evaluation and must not be interpreted as measured internal battery failures.

> The model detects abnormal battery electrical response and does not diagnose the underlying internal physical failure mechanism.
