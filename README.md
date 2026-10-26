# Deteksi anomali kelistrikan baterai UAV

Repository penelitian untuk mendeteksi penyimpangan respons tegangan baterai dengan model regresi HGB Battery-D dan residual tegangan. Baseline memakai validasi silang berbasis flight (LOAFO), threshold persentil ke-99 dari flight validasi, serta konfirmasi persistence 0,5 detik berdasarkan timestamp.

## Mulai

Gunakan Python 3.13 dan pasang versi dependensi yang dikunci:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Dataset baterai ada di `data/raw/dataset_baterai.csv`. Untuk menjalankan pipeline dari awal:

```powershell
.\.venv\Scripts\python.exe main.py inspect
.\.venv\Scripts\python.exe main.py preprocess
.\.venv\Scripts\python.exe main.py features
.\.venv\Scripts\python.exe main.py cross-validate
```

Keluaran eksperimen baru dibuat di `artifacts/`. Hasil historis yang dipaketkan di bawah adalah snapshot analisis terverifikasi; perintah pipeline tidak menimpa hasil dengan identitas berbeda.

## Demo mandiri untuk dosen

Demo ini berjalan offline dari CSV yang disertakan. Jalankan perintah dari folder root repository. Untuk menyiapkan salinan baru:

```powershell
git clone https://github.com/alifaryasatya/Anomaly-Battery-UAV-Detection-.git
cd Anomaly-Battery-UAV-Detection-
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Lihat command yang tersedia, periksa dataset, lalu jalankan evaluasi LOAFO:

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py inspect
.\.venv\Scripts\python.exe main.py cross-validate
```

`inspect` menunjukkan schema dan cakupan dataset. `cross-validate` memakai pemisahan seluruh flight; bila identitas run cocok, CLI memeriksa lalu memakai ulang artefak baseline yang sudah disertakan. Setelah itu buka laporan lokal:

```powershell
Start-Process .\artifacts\reports\research_v2_20261026\research_report.html
```

Untuk demonstrasi singkat, tunjukkan grafik tegangan sag aktual dan prediksi, residual terhadap threshold P99, serta hasil alarm. Buka indeks skenario di browser dari `artifacts/reports/research_v2_20261026/scenario_index.html`, lalu cari `00000074 progressive mild` untuk melihat contoh gangguan progressive yang tidak terdeteksi beserta skor per sampelnya.

Model HGB Battery-D memperkirakan voltage sag berdasarkan fitur arus. Skor alarm adalah residual positif `max(0, sag aktual − sag prediksi)`; alarm dikonfirmasi setelah skor melewati P99 validation secara kontinu selama 0,5 detik. Fault sintetis diterapkan ke salinan telemetry untuk menguji respons perangkat lunak, bukan untuk menyatakan baterai mengalami kerusakan nyata.

Hasil baseline yang perlu dijelaskan: 156/162 event sintetis terdeteksi, healthy-assumed confirmed FPR 11,83%, dan target progressive dalam ≤5 detik tercapai 12/51. Hasil ini adalah evaluasi historis pada enam flight, bukan validasi keselamatan atau demonstrasi switching fisik. Siapkan environment dan jalankan perintah sebelum presentasi agar waktu demo dapat dipakai untuk membahas hasil.

## Laporan dan hasil yang disertakan

- [Laporan penelitian mendalam (HTML)](artifacts/reports/research_v2_20261026/research_report.html) dan [Markdown](artifacts/reports/research_v2_20261026/research_report.md).
- [Indeks skenario interaktif](artifacts/reports/research_v2_20261026/scenario_index.html), dengan tautan ke 162 file skor per sampel.
- [Hasil konsolidasi historis](FINAL_RESEARCH_RESULTS.md) dan [verifikasi final](FINAL_VERIFICATION.md).
- [Baseline LOAFO](artifacts/cross_validation/4bc978e2773687d3/research_report.md), termasuk konfigurasi fold, model tersimpan, skor, metrik, dan manifest.
- [Diagnosis false alarm Flight 66](artifacts/flight66_diagnosis/ROOT_CAUSE_ANALYSIS.md), dengan CSV per sampel, episode, segmentasi, serta grafik.
- [Ringkasan hasil konsolidasi yang menjadi konteks laporan](artifacts/research_evaluation/4ad7e153513e7d64/research_report.md).

Laporan utama mencatat 156/162 event sintetis terdeteksi dan healthy-assumed confirmed FPR 11,83%. Target progressive ≤5 detik tercapai 12/51; target tersebut belum terpenuhi. Flight 66 menyumbang 3.168 dari 3.178 confirmed healthy-assumed false-positive samples pada baseline. Angka alarm tidak digunakan untuk mengubah label kesehatan baterai.

## Batas interpretasi

Enam flight active sudah dikenal selama penelitian dan seluruhnya pernah digunakan dalam evaluasi historis. Hasilnya adalah bukti historical development, bukan independent final holdout. Label healthy adalah asumsi sumber, unit `CurrTot` belum diverifikasi, dan fault sintetis bukan ground truth kerusakan elektrokimia. Hasil perangkat lunak tidak menyatakan kesiapan deployment, switching fisik, atau keselamatan penerbangan.

Pipeline dimulai dari CSV telemetry baterai; decoder log `.BIN` tidak disertakan. Dataset menyimpan identitas flight, timestamp, tegangan, arus dan counter arus tanpa kanal GPS/throttle.
