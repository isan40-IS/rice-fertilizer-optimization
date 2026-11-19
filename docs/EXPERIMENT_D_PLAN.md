# Experiment D — Surrogate Model Suitability and Fertilizer Response Analysis

Laporan hasil: [diagnosis dan interpretasi Experiment D](EXPERIMENT_D_RESULTS.md).

Tanggal rancangan: 2026-11-17. Status: diimplementasikan dan diverifikasi setelah persetujuan peneliti.

Implementasi: `evaluation/surrogate_diagnostics.py` dan notebook target di bawah.
Keluaran verifikasi lengkap: `artifacts/experiment_D/exp_D_20261117T123759Z_33092529/`.
Notebook dijalankan tanpa fit (fit model/preprocessor dinonaktifkan saat verifikasi);
85 tes suite lulus, termasuk 15 tes D yang juga diulang pada helper akhir.
Model/data/split/config C dipertahankan. Holdout diagnostik: MAE 1042.957,
RMSE 1466.330 kg/ha, R2 0.727121 (n=192; bukan pristine test).
Terdapat 33 kombinasi NPK unik training; pairing terkontrol belum dapat dikonfirmasi.
Lihat FIND-016, notebook dan `suitability_summary.csv` untuk interpretasi.

Keputusan implementasi: jarak adalah diagnostik kontinu tanpa cutoff baru; marker
pelanggaran dipakai untuk range/kategori yang terverifikasi. Kontrol `No fertilizer`
tidak di-sweep menjadi dosis positif tanpa perubahan kategori eksplisit. Pairing
bersifat deskriptif sampai metadata/desain dikonfirmasi, tanpa agregasi replikasi
otomatis. Kategori pupuk tidak di-sweep sebagai state tanah. Threshold/bobot final
serta bukti literatur tetap kosong; tidak ada eksekusi DE.

Cara pakai: buka notebook dengan environment proyek, jalankan berurutan. Ubah
`GROUP_DIR` secara eksplisit bila kandidat berubah. Playground C2 bisa dijalankan
ulang dengan `BASE_MODE`, ID/index dan `OVERRIDES`; worksheet E menerima pilihan
manual tanpa menulis configs. Menjalankan setup analisis kembali membuat direktori
D baru; tidak menimpa keluaran sebelumnya. Rancangan detail berikut dipertahankan.
Target notebook: `notebooks/2_model_experiment/experiment_D.ipynb`.
Dokumen ini merupakan instruksi untuk agent pengembang. Gunakan AGENTS.md dan konteks agent/ yang relevan. Bahasa penjelasan notebook: Indonesia. Pertahankan pendekatan bare minimum: gunakan model dan utilitas proyek yang sudah tersedia, tanpa arsitektur prediktif baru.

## 1. Tujuan, kebutuhan, dan batas lingkup

Experiment C memilih kandidat berdasarkan performa prediksi. Experiment D menguji apakah kandidat tersebut dapat menjalankan peran surrogate untuk optimasi dosis N, P, K. Model harus (1) menghasilkan prediksi yield yang dapat dievaluasi; (2) responsif terhadap perubahan decision variables pada state tetap; (3) mampu membedakan perlakuan dosis yang sebanding; (4) mengevaluasi kombinasi input yang didukung data; (5) konsisten dalam preprocessing dan feature engineering. Respons terhadap state juga perlu diperiksa untuk memahami sifat site-specific model.

Pertanyaan utama: bagaimana prediksi yield berubah saat NPK berubah pada berbagai kondisi tanah dan lingkungan? Bagaimana model bereaksi ketika satu state variable berubah? Apakah perubahan yang diprediksi konsisten dengan perbedaan yield perlakuan yang benar-benar teramati, jika pasangan yang sebanding tersedia?

Tidak melakukan training/tuning ulang, seleksi fitur baru, penghapusan baris, imputasi baru, penggantian split, Differential Evolution, atau penetapan dosis optimal. Tidak menetapkan parameter agronomis tanpa literatur/persetujuan. Pengujian D adalah diagnostik, bukan validasi lapangan atau bukti kausal. Jangan mengubah configs aktif atau artefak C. Stabilitas solusi optimizer baru diuji pada tahap optimasi berikutnya.

## 2. A — Model terpilih dan provenance

Sumber utama: `docs/EXPERIMENT_C_USAGE.md`, `docs/EXPERIMENT_C_TUNING_PLAN.md` (rancangan historis), dan artefak C aktual. Menurut usage section 7, grup yang selesai adalah `artifacts/experiment_C/exp_C_20261117T055252Z_b321014c/`. Kandidat XGBoost search seed 11, mean CV RMSE 1805.25, MAE 1267.71 kg/ha, R2 0.6475, SD RMSE 519.30. Profil `minimalist_features` (11 input); cohort 1.000; native numerical missing handling; model/split seed 42. Best params tercatat pada usage dan artefak; jangan merekonstruksi model dari angka ini.

### A1. Load artefak

- Sediakan `GROUP_DIR` eksplisit pada sel konfigurasi; default grup di atas. Jangan memilih direktori terbaru otomatis. Baca `experiment_manifest.json`, `frozen_candidate.json`, `study_comparison.csv`, snapshot dan training manifest subrun. Gunakan schema aktual, jangan mengarang key JSON.
- Resolusi bundle mengikuti frozen candidate. Lokasi yang diharapkan untuk grup tersebut: `optuna_seed_11/models/xgboost.joblib`; pastikan manifest menunjuk ke model yang sama. Gunakan serializer resmi proyek untuk memuat fitted pipeline. Hanya load joblib tepercaya dari repository sendiri.
- Tampilkan path, identitas model/metode/seed, status run, parameter, feature profile, target, hashes dan versi library, CV metrics dan fold SD. Tampilkan daftar kolom input mentah, hasil transformer dan fitur yang dikeluarkan karena all-missing pada refit. Gunakan bundle/pipeline aktual, bukan asumsi jumlah kolom.
- Cocokkan dataset hash dan split assignment dengan manifest. Jika file hilang, hash mismatch, schema tidak cocok atau pipeline tidak bisa diload, hentikan bagian yang bergantung padanya dan laporkan blocker. Jangan fallback ke model lain atau fit ulang diam-diam.
- Jika `selected_holdout/evaluation.json` dan `predictions.csv` tersedia, baca dan validasi kesesuaiannya. Jika belum, buat prediksi read-only menggunakan bundle pada holdout asli dalam D, tanpa fitting; simpan hasil hanya pada D. Jangan mengaktifkan runner tuning. Holdout telah pernah diperiksa pada eksperimen proyek; label sebagai diagnostik, bukan pristine test. CV C bukan nested independent CV.

## 3. B — Data, schema dan range

Dataset C menurut usage: `data/processed/experiment_B_candidates.csv`, 1.000 baris read-only. Gunakan dataset yang dicatat manifest bila berbeda dan laporkan perbedaan. Load split assignments/CV assignments asli; validasi row index/record_id, uniqueness yang memang disyaratkan, train/test membership, group overlap dan jumlah baris. Jangan membuat split baru. Tampilkan training, holdout, full cohort secara terpisah. Statistik calon batas dan skala optimasi menggunakan training saja; full/holdout hanya deskriptif.

### B1. Data dictionary yang benar-benar digunakan

Buat tabel `feature_schema` seluruh kolom terkait model (bukan hanya state): `column`, `role`, `dtype`, `unit`, `description`, `required_input`, `used_after_preprocessing`, `missing_n_train`, `missing_pct_train`, `source_or_formula`.

Klasifikasi: target (`yield.wb.kg_ha` bila manifest mengonfirmasi), decision (`N.rate.kg_ha`, `P.rate.kg_ha`, `K.rate.kg_ha`), derived decision (total, rasio, proporsi yang benar-benar ada), state numeric, state categorical, identifier/group/metadata (bukan prediktor kecuali bundle menyatakan), unused. Cocokkan metadata aktual, configs snapshot dan fitted transformer. Unit dan definisi yang tidak diketahui tandai unknown. Tampilkan fitur training-requested vs efektif sesudah all-missing filtering dan encoding; jangan menganggap semua 11 fitur efektif.

### B2. Statistik dan visualisasi

- Semua kolom numerik terkait: n observed, missing count/%, min, max, mean, SD, Q01, Q05, Q25, median, Q75, Q95, Q99, IQR, unique count. Untuk dosis termasuk jumlah nol dan nilai positif; tampilkan juga statistik positive-only dengan denominator jelas. Jangan menghapus nol perlakuan.
- Kategorikal: level, frekuensi, missing count; pisahkan missing asli dari kategori string `nan` bila dapat diidentifikasi; tampilkan level training yang dikenal transformer. Target juga ditampilkan distribusi dan satuan.
- Distribusi N/P/K, total dan proporsi: histogram/boxplot dan scatter pasangan N-P, N-K, P-K; plot distribusi state terpilih. Tampilkan pola missingness state, jumlah observasi dan jumlah grup per pola. Tidak perlu plot seluruh metadata.
- Tabel kombinasi dosis yang benar-benar teramati: frekuensi, jumlah grup, contoh treatment; periksa keterkaitan dosis dan fitur turunan. Bedakan joint support dari sekadar min-max marginal.
- Gunakan ringkasan tabel yang mudah dibaca, bukan dump 1.000 baris. Simpan tabel lengkap sebagai CSV.

## 4. C — Mencoba fitted predictive model

### C1. Prediksi kasus nyata

Gunakan pipeline.predict pada kolom mentah sesuai urutan/schema bundle. Tampilkan default 8 kasus holdout yang dipilih deterministik (seed 42, usahakan variasi dosis dan pola missing; jika tidak cukup tampilkan yang tersedia). Pemilihan contoh tidak berdasarkan error terbaik. Sertakan record_id, group, seluruh state input, N/P/K, fitur turunan (tabel detail bila lebar), actual yield, predicted yield, residual = actual - predicted, absolute error, squared error, absolute percentage error hanya bila actual != 0. Tampilkan MAE/RMSE/R2 pada seluruh holdout dan jumlah kasus; metrik subset contoh diberi label terpisah. Jangan menghitung R2 per baris. Plot actual vs predicted dengan garis y=x dan residual vs predicted; baca horizontal bands tanpa menyimpulkan penyebab hanya dari grafik. Jika tersedia, cocokkan prediksi artefak C dengan prediksi ulang menggunakan toleransi numerik yang wajar.

### C2. Manual prediction playground (sel mudah diedit)

Prioritaskan sel Python biasa, tidak perlu ipywidgets. Sediakan dua cara memilih baseline: `BASE_MODE = 'record'` dengan `BASE_RECORD_ID` (atau row_index eksplisit jika ID ambigu), dan `BASE_MODE = 'default'` memakai satu baris training nyata yang representatif, misalnya paling dekat dengan median training pada fitur utama yang teramati dengan aturan deterministik dan dilaporkan. Jangan menyusun median setiap kolom menjadi kombinasi lahan yang tidak pernah ada tanpa label synthetic. Default dipilih tanpa menggunakan yield atau error. Jika tidak dapat menentukan, gunakan baris training pertama yang valid dan jelaskan.

Contoh sel (nama kolom harus diverifikasi):

```python
BASE_MODE = 'record'  # record | default
BASE_RECORD_ID = None # isi ID; None memilih default training case secara eksplisit
OVERRIDES = {
    # 'avprec.mm': 850,
    # 'N.rate.kg_ha': 120,
    # 'P.rate.kg_ha': 15,
    # 'K.rate.kg_ha': 40,
    # 'initial.ph.soil': np.nan,
}
```

Helper `predict_scenario(base_row, overrides)` atau nama sesuai proyek:

1. Salin baseline (jangan mutate dataframe). Tampilkan ID/split, observed/missing fields dan nilai awal. Tolak ID ambigu, kolom tidak dikenal, target/identifier overrides dan direct override derived decision features. Kolom yang tidak efektif harus diperingatkan, bukan dianggap memengaruhi prediksi. Validasi tipe, unit dan kategori; jangan silently clip atau imputasi. NPK harus finite dan nonnegative; state numerik boleh NaN bila pipeline menerima.
2. Terapkan hanya override yang diberikan, pertahankan kolom lain. Hitung ulang seluruh fitur turunan NPK sesuai fungsi/formula proyek dan snapshot (pseudocount, perlakuan zero denominator, proporsi). Jika formula belum dapat diverifikasi, blokir perubahan NPK dan laporkan. Jangan memakai MICE/fit preprocessing baru.
3. Prediksi baseline dan modified dengan fitted pipeline yang sama. Tampilkan tabel before/after/delta untuk fitur dan flag perubahan missingness. Tampilkan yhat_base, yhat_modified, delta_y, persentase bila baseline prediksi positif; tampilkan peringatan domain dan kategori tidak dikenal.
4. Actual yield hanya milik kasus observasi awal. Jika ada perubahan efektif input, actual baru dan error modified adalah N/A (kecuali perlakuan aktual yang benar-benar sebanding ditemukan dan ditampilkan terpisah). Jangan membandingkan prediksi skenario hipotetis dengan yield asli sebagai error. Jika tidak ada override, output harus identik dengan baseline.

Pastikan helper yang sama digunakan C2, D dan ekspor. Target tidak pernah menjadi input. Sel dapat dijalankan ulang tanpa training. Sediakan contoh perubahan avprec, N saja, NPK bersamaan dan state missing jika sesuai schema; nilai contoh hanya ilustrasi, bukan rekomendasi.

## 5. D — Surrogate suitability checks otomatis

Gunakan model beku yang sama. Semua perubahan input melalui helper konsistensi fitur. Gunakan skenario dasar dari training (untuk diagnosis fungsi, bukan generalisasi). Default 6–10 skenario, deterministik, beragam state dan missingness serta grup bila tersedia; jangan memilih berdasarkan prediksi tertinggi. Tampilkan tabel baseline dan missingness. Untuk pengujian empiris D4 gunakan holdout asli. Semua jumlah skenario/grid adalah anggaran pengembangan yang boleh diubah pada sel konfigurasi, bukan threshold agronomis.

### D1. Dose-response / individual conditional response

Untuk setiap skenario dan N/P/K: variasikan satu dosis, tahan dua dosis lain dan seluruh state tetap. Gunakan sekitar 21 titik pada rentang training (misalnya Q05–Q95, tambahkan baseline dan titik nol jika teramati; rentang data eksploratif, bukan rentang rekomendasi). Simpan juga titik dosis aktual yang relevan bila tersedia. Jangan sampling rentang tak diketahui. Selalu hitung ulang total/rasio/proporsi. Plot garis respons per skenario dan unsur, penanda baseline, titik observasi sebanding bila ada, highlight unsupported points.

Output `dose_response.csv` dan tabel per skenario/unsur: baseline yield prediksi, min/max predicted yield, response amplitude (max-min), jumlah perubahan prediksi/nonflat steps dengan toleransi numerik, persentase flat steps, finite differences per interval (kg yield/kg nutrient), proporsi arah naik/turun/tetap, apakah maksimum pada endpoint grid. Catat jumlah titik didukung data dan ringkas hasil valid-support terpisah bila cukup. Jangan menilai kurva bertangga otomatis buruk atau mengharuskan monotonicity/diminishing return di semua skenario. Toleransi numerik hanya mengatasi floating point, bukan batas relevansi agronomis.

### D2. Interaksi dosis / response surface

Untuk 2–3 skenario representatif, grid dua dosis (N–P, N–K, P–K), dosis ketiga dan state tetap. Default 15 x 15 per pasangan; rentang eksploratif seperti D1. Buat heatmap predicted yield dan, bila layak, delta dari baseline; gunakan mask/penanda unsupported points. Tampilkan baseline dan titik kombinasi yang teramati bila sebanding. Jangan menjalankan pencarian optimum atau mengeluarkan rekomendasi dari maksimum heatmap. Simpan tabel grid. Periksa apakah variasi respons satu unsur berubah saat unsur lain berbeda. Ini interaksi dalam fungsi model, bukan bukti mekanisme fisiologis.

### D3. State responsiveness dan site-specific behavior

Untuk fitur state numerik efektif, pada beberapa baseline yang nilai state-nya teramati, lakukan sweep satu state pada rentang training Q05–Q95 atau nilai teramati sambil mempertahankan NPK dan state lain; tampilkan kurva dan amplitude. Untuk kategorikal gunakan level training yang wajar bila tersedia; jangan membuat kategori baru. Analisis missing state secara terpisah: observed vs mask-to-NaN untuk kasus yang dapat diuji; jangan menyamakan missing dengan besaran numerik nol atau menafsirkan efek mask sebagai efek agronomis.

Untuk pertanyaan khusus optimasi, ulangi N/P/K sweep pada 2–3 nilai state teramati (misalnya Q25, median, Q75) dengan baseline NPK sama. Plot overlay response curves. Laporkan apakah perubahan state memengaruhi level prediksi saja atau juga bentuk/amplitude respons dosis. Hindari kombinasi state tidak realistis; flag dukungan data. Tidak menjalankan optimasi ulang atau menghitung indeks sensitivitas dosis optimal proposal (memerlukan DE).

### D4. Empirical dose ranking dan delta-yield error

Cari pasangan perlakuan berbeda dosis di holdout dengan group eksperimen/blok sama, kondisi non-dose yang benar-benar sebanding berdasarkan metadata yang tersedia. Prioritas pasangan one-nutrient-change; laporkan multi-nutrient-change terpisah. Periksa varietas, musim, lokasi, sumber pupuk, manajemen dan kondisi lain sejauh tersedia. Kesamaan missingness bukan bukti kondisi sama; tandai keterbatasan. Jangan membuat pasangan antar lokasi/eksperimen hanya karena nilai state mirip. Jangan mengarang pairing jika metadata tidak memadai.

Untuk pasangan A,B, hitung delta_actual = y_B-y_A, delta_pred = f(x_B;s_B)-f(x_A;s_A) memakai input aktual kedua observasi; jika state berbeda secara relevan, eksklusi dari klaim controlled comparison. Jika pengulangan perlakuan tersedia, definisikan agregasi berdasarkan desain dan laporkan, jangan pseudo-replikasi. Laporkan jumlah treatment, pasangan dan grup, aturan matching, jumlah yang tidak memenuhi. Hitung sign/ranking agreement pada pasangan non-tie, jumlah actual ties dan predicted ties secara eksplisit, delta MAE/RMSE serta group-level summaries (pasangan dalam satu grup tidak independen). Boleh gunakan Spearman per grup bila >=3 treatment berbeda dan nonconstant. Plot delta actual vs delta predicted. Jika tidak ada pasangan layak, beri status insufficient evidence, bukan nol skor atau pass otomatis. Tidak gunakan hasil ini untuk retune kandidat diam-diam.

### D5. Domain support dan input consistency

- Range check numerik terhadap training min–max dan quantile; kategori unknown; missingness pattern frequency. Bedakan missing dari out-of-range. Periksa nonfinite prediksi.
- Joint NPK support: nearest-neighbor distance pada NPK training yang diskalakan (IQR jika >0; fallback range jika >0; fitur konstan ditandai). Tampilkan nearest training combinations dan jarak. Skala/reference fit training saja. Jika state support diperiksa, bandingkan pada observed state yang sama, laporkan jumlah dimensi dan reference rows yang tersedia; jarak antar pola tidak otomatis comparable. Jangan menyembunyikan hasil karena sparse data.
- Untuk referensi jarak, hitung distribusi leave-one-out nearest-neighbor training bila memadai; perhatikan duplikasi treatment dapat membuat jarak nol. Laporkan Q50/Q90/Q95 dan opsi distinct-combination reference. Ini diagnostik, bukan hard constraint. Ambang final memerlukan persetujuan. Jangan klaim in-range berarti reliable.
- Uji helper: baseline unchanged prediction; recomputed derived features sesuai formula; non-NPK fixed pada NPK sweep; non-state fixed pada state sweep; zero dose handling; repeated same input deterministik; invalid/unknown override tidak diam-diam diterima; input/model/data tidak dimutasi. Catat mismatch sebagai error. Bila model tidak responsif pada sebagian skenario, laporkan flatness, bukan langsung ubah model.

### D6. Ringkasan kelayakan

Buat tabel `check`, `evidence`, `result`, `limitation`, `status`, `next_action`. Status: pass untuk kontrak teknis yang terbukti; warning untuk indikasi empiris; insufficient_evidence jika data kurang; fail untuk kontrak yang rusak. Untuk respons/ranking/domain jangan tetapkan ambang lolos ilmiah tanpa peneliti; laporkan angka. Kesimpulan terpisah untuk kesiapan teknis, bukti responsiveness, bukti ranking dan keterbatasan. Rekomendasikan proceed / proceed with cautions / investigate berdasarkan bukti, sebagai usulan bukan approval agronomis. Jangan mengklaim kestabilan solusi DE sebelum DE dijalankan.

## 6. E — Persiapan parameter optimasi (tanpa menjalankan DE)

Tujuan: menyediakan bukti data untuk peneliti saat memilih parameter fungsi objektif, constraint, penalty, perturbation, dan domain warning. Pisahkan statistik hasil data, usulan aturan, nilai dari literatur dan keputusan final. Jangan menulis nilai otomatis ke `configs/decisions.yaml`. Baca konfigurasi optimasi/evaluasi dan proposal aktual; jika terdapat nilai lama, tampilkan sebagai konfigurasi sebelumnya yang memerlukan peninjauan, bukan nilai final.

### E1. Formulasi yang menjadi referensi

Decision vector x = [N, P, K]; state s tetap dalam setiap skenario. Predicted improvement = f(x;s) - f(x_base;s). Proposal menggunakan hard bounds, total-dose constraint, total-dose penalty dan extreme-composition penalty. Jika implementasi memakai objective ternormalisasi, skala yield, definisi penalty dan satuan bobot harus diverifikasi; jangan mencampur normalized score dengan kg/ha. Baseline yang digunakan untuk perbandingan optimasi kelak harus mengikuti kebijakan yang disetujui peneliti. Experiment D menggunakan observed case sebagai baseline diagnosis, bukan menetapkan `baseline_policy` final.

### E2. Data yang disiapkan untuk setiap parameter

| Parameter / keputusan | Bukti yang dihitung dari training | Literatur / keputusan peneliti |
|---|---|---|
| N, P, K lower/upper bound | min/max, Q01/Q05/Q95/Q99, positive-only distribution, zero frequency, dukungan kombinasi dan jumlah grup | rentang agronomis dan konteks satuan; lower/upper final merupakan irisan domain yang disepakati |
| `total_soft`, `total_max` | distribusi total NPK; min/max, Q50/Q75/Q90/Q95/Q99, jumlah/fraction observasi melampaui setiap calon batas | ambang agronomis dan toleransi; tidak otomatis mengambil quantile tertentu |
| `proportion_bounds` N/P/K | distribusi N/T, P/T, K/T untuk T>0; quantile, frekuensi nol, scatter komposisi, jumlah grup | rentang komposisi yang dapat diterima; jangan menyamakan proporsi dengan ratio |
| `yield_scale` jika normalisasi aktif | IQR, SD, median, min/max yield training dan satuan | pilihan skala dan justifikasi; jika IQR nol laporkan |
| `lambda_total`, `lambda_composition` | kisaran delta predicted yield dari D1/D2 sebagai konteks skala; besaran penalty pada kombinasi teramati bila formula dan threshold telah ditentukan | pilihan trade-off; bukan parameter yang langsung dapat disimpulkan dari distribusi data |
| Perturbation state (`delta_m`) | min/max, IQR, SD, unique values, resolusi nilai bila dapat dipastikan | besaran perturbation sesuai unit dan kewajaran agronomis/pengukuran |
| Marginal nutrient step (`h_j`) | range dosis, distribusi selisih antardosis teramati (utamakan perlakuan sebanding), response flatness dan batas domain | langkah finite difference yang masuk akal; tidak otomatis memilih nilai terkecil |
| `max_training_distance` | distribusi jarak reference dari D5 dan coverage calon batas | ambang warning yang disetujui, bukan jaminan validitas |
| `boundary_warning_fraction` | distribusi posisi relatif dosis terhadap calon bounds jika telah diisi | toleransi warning |
| `max_input_stability_index` | belum tersedia; memerlukan hasil optimasi dan perturbasi | isi setelah evaluasi optimasi, bukan dari prediksi D saja |
| DE population size, mutation factor, crossover, max iterations, tolerance, seeds | jumlah decision variables=3, biaya inference batch (opsional timing singkat) | literatur/default algoritma dan budget eksperimen; statistik data tidak menentukan nilai ini secara langsung |

Gunakan satuan nutrient rate yang terverifikasi; jangan menyamakan P dengan P2O5 atau K dengan K2O tanpa konversi yang sah. Untuk total=0, proporsi secara matematis tidak terdefinisi; laporkan jumlahnya terpisah dan jangan memasukkan ke statistik komposisi sebagai proporsi valid. Untuk quantile, tampilkan jumlah observasi nonmissing dan jumlah grup. Bila treatment berulang mendominasi distribusi, tampilkan pula ringkasan berbasis kombinasi unik atau grup sebagai diagnostik tambahan, tidak mengganti statistik utama diam-diam.

### E3. Worksheet parameter yang dapat diisi manual

Buat tabel `optimization_parameter_worksheet.csv` dengan kolom: `parameter`, `unit`, `data_evidence`, `candidate_data_value_or_range`, `literature_value_or_range`, `literature_reference`, `previous_config_value`, `proposed_final_value`, `rationale`, `status`. Nilai literatur dan final dibiarkan kosong jika belum diberikan. Gunakan `pending_literature`, `pending_decision`, `not_applicable` atau `requires_optimization_results` sesuai konteks. Nilai data yang berupa beberapa opsi jangan dipaksakan menjadi satu rekomendasi.

Tambahkan sel editable dengan contoh struktur berikut (sesuaikan schema aktual; tidak menulis config):

```python
LITERATURE_BOUNDS = {
    'N': {'min': None, 'max': None, 'reference': None},
    'P': {'min': None, 'max': None, 'reference': None},
    'K': {'min': None, 'max': None, 'reference': None},
}
MANUAL_PARAMETERS = {
    'total_soft': None,
    'total_max': None,
    'proportion_bounds': None,
    'lambda_total': None,
    'lambda_composition': None,
    'yield_scale': None,
}
```

Jika peneliti mengisi rentang literatur dan memilih rentang data secara eksplisit, tampilkan intersection per unsur: L=max(L_data,L_literature), U=min(U_data,U_literature). Validasi L<=U, unit konsisten, total constraints kompatibel dengan bounds, dan jika proporsi diperlakukan sebagai hard bounds kelak, periksa kelayakan bersama. Untuk soft proportion penalty jangan mengubahnya menjadi hard constraint. Laporkan intersection kosong dan kombinasi yang tidak/kurang didukung. Tampilkan fraksi observasi training di dalam calon domain jika semua parameter diperlukan tersedia. Ini analisis persiapan, bukan eksekusi optimizer atau persetujuan parameter final.

## 7. Struktur notebook dan output

Notebook harus dapat dibaca berurutan dan sel analisis dapat dijalankan ulang tanpa fitting. Susunan:

1. Tujuan, scope dan configuration cell (GROUP_DIR, seed, sample sizes, grid size, output dir, pilihan diagnostik).
2. A — Load frozen model, manifest, pipeline dan dataset/split; integrity checks.
3. B — Feature schema, missingness, distribusi dan joint dose support.
4. C1 — Observed-case predictions dan holdout evaluation.
5. C2 — Manual playground, baseline ID/default + overrides.
6. D1 — N/P/K response curves.
7. D2 — Pairwise response surfaces.
8. D3 — State sensitivity/conditional dose response.
9. D4 — Empirical ranking dan delta-yield evaluation.
10. D5/D6 — Domain support, consistency tests dan summary.
11. E — Data-derived optimization parameter evidence dan editable worksheet.
12. Kesimpulan: apa yang sudah teruji, belum teruji, dan keputusan berikutnya.

Simpan hanya artefak D, contoh struktur (nama file boleh mengikuti konvensi proyek; dokumentasikan path aktual):

```text
artifacts/experiment_D/<run_id>/
  manifest.json
  feature_schema.csv
  numeric_statistics.csv
  categorical_statistics.csv
  missingness_patterns.csv
  observed_dose_combinations.csv
  holdout_metrics.json
  holdout_predictions.csv
  example_cases.csv
  dose_response.csv
  dose_response_summary.csv
  response_surface.csv
  state_response.csv
  conditional_dose_response.csv
  dose_ranking_pairs.csv
  dose_ranking_summary.csv
  domain_reference_statistics.csv
  optimization_parameter_worksheet.csv
  suitability_summary.csv
```

Manifest mencatat path/hash model, dataset/split, config snapshot, seed, library versions, skenario/ID, grid dan definisi metrik. Jangan menimpa C atau output D lama tanpa instruksi. Simpan grafik yang berguna bila sesuai kebiasaan proyek; tampilkan juga dalam notebook. Tidak perlu dependency visualisasi baru, UI tambahan, dashboard atau runner kompleks. Jika utilitas reusable sudah tersedia, panggil; bila helper kecil diperlukan, buat di lokasi yang sesuai proyek dan gunakan satu implementasi, bukan salinan fungsi berbeda untuk setiap section. Hindari refactor pipeline C kecuali ada bug yang benar-benar memblokir; laporkan dan minta keputusan sebelum perubahan metodologis.

## 8. Acceptance criteria dan handoff agent

- Notebook tersedia di path target; source data dan model C tetap tidak berubah.
- Model loaded dari frozen artifact, bukan training ulang; feature schema dan provenance dapat diperiksa.
- Statistik data lengkap dengan pemisahan train/holdout/full dan unit yang jelas.
- Prediksi kasus aktual dan metrik holdout dihitung dengan pipeline yang sama; playground dapat memilih ID/default dan mengubah hanya kolom yang diinginkan.
- Semua perubahan dosis merekalkulasi fitur turunan, tidak mengubah state; perubahan hipotetis tidak dilaporkan sebagai error terhadap yield asli.
- Dose response, response surface, state response, empirical ranking (atau alasan insufficient evidence), domain check dan summary tersedia.
- Worksheet parameter terisi bukti data; literatur/keputusan yang belum tersedia tetap kosong. Tidak ada eksekusi DE.
- Uji minimal pada helper dan notebook: schema/hash mismatch, invalid ID, invalid override, baseline no-op, derived consistency, missing numeric input yang didukung, reproducible selection, finite predictions, dan tidak ada fit. Laporkan tes yang benar-benar dilakukan, yang belum dijalankan, blocker dan perubahan file. Jika artefak tidak tersedia, siapkan notebook dengan validasi/fail-fast; jangan membuat hasil sintetis seolah hasil penelitian.
- Setelah implementasi, jalankan smoke test dan bagian diagnostik sejauh artefak tersedia dan budget memungkinkan; catat bagian yang belum dijalankan. Perbarui `agent/05_current_state.md` serta findings/backlog yang relevan berdasarkan bukti aktual; keputusan penelitian baru hanya dicatat sebagai approved setelah persetujuan peneliti.

Prioritas: A, B, C, D1, D4/D5, E adalah inti. D2/D3 menggunakan grid/skenario terbatas; jangan memperluas otomatis ke seluruh populasi atau menambah model baru. Jika D mengungkap kelemahan surrogate, laporkan diagnosis spesifik sebelum mengusulkan perubahan. Tujuan akhir adalah melanjutkan optimasi secara bertanggung jawab, bukan mengejar model sempurna.
