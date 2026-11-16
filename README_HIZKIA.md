# Panduan Operasional untuk Hizkia

Dokumen ini menjelaskan cara kerja proyek, hubungan antarfile, keputusan yang masih perlu diisi, dan urutan penggunaan kode.

## 1. Gambaran besar sistem

```text
CSV mentah
  ↓
Audit dan pra-pemrosesan data
  ↓
Pelatihan serta evaluasi model prediktif
  ↓
Model terbaik menjadi surrogate model
  ↓
Differential Evolution mencari dosis NPK
  ↓
Evaluasi produktivitas, efisiensi, sensitivitas, dan stabilitas
```

### Lapisan data

`src/rice_fertilizer/data/` membaca CSV, memfilter padi, memeriksa kualitas data, menangani missing value NPK yang bersifat struktural, dan membuat fitur turunan.

### Lapisan model prediktif

`src/rice_fertilizer/modeling/` membangun Linear Regression, Random Forest, XGBoost, dan CatBoost. Imputasi dan encoding ditempatkan dalam pipeline agar tidak mempelajari data validasi atau data uji.

### Lapisan surrogate dan optimasi

`src/rice_fertilizer/surrogate/` mengubah kandidat `[N, P, K]` menjadi satu baris fitur lengkap yang konsisten dengan model. `src/rice_fertilizer/optimization/` menghitung objective, constraint, penalty, dan menjalankan Differential Evolution.

### Lapisan evaluasi

`src/rice_fertilizer/evaluation/` membandingkan baseline dan optimum, menghitung efisiensi, sensitivitas, stabilitas input, serta stabilitas algoritma.

## 1.5 Buat Virtual Environtment dan Install Requirement
a. Aktifkan Virtual Environment
Menggunakan Parameter Default
(Nama: rfo, Python: 3.14, Req: requirements.txt, Mode: biasa)
Script `setup_env.py` digunakan untuk membuat dan menyinkronkanvirtual environment secara otomatis. Script ini bersifat *idempotent*: jika environment belum ada maka akan dibuat baru, dan jika sudah ada maka langkah pembuatan akan dilewati (hanya memperbarui dependensi).
```bash
python scripts/setup_env.py
```
b. Menjalankan untuk Mode Dev / Test
```bash
python scripts/setup_env.py --mode dev
```
c. Custom Semua Parameter
```bash
python scripts/setup_env.py -n rice-fertilizer -p 3.11 -r requirements.txt -m dev
```
d. Melihat Help / Bantuan Options
```bash
python scripts/setup_env.py --help
```

## 1.6 Aktifkan Environment
```bash
.\.rfo\Scripts\activate
```

## 2. Folder data sengaja dikosongkan

Pada hasil generate ini, folder `data/` hanya mempunyai `.gitkeep`. Saat pipeline pertama kali dijalankan, sistem membuat:

```text
data/input/
data/interim/
data/processed/
```

Pertama, jalankan:

```bash
python scripts/prepare_data_directory.py
```

Setelah folder input muncul, masukkan CSV ke:

```text
data/input/data.csv
```

Nama file dapat diubah melalui `configs/paths.yaml`. Folder `interim`, `processed`, dan hasil `artifacts` baru dibuat ketika pipeline dijalankan.

## 3. Hubungan antarfile ketika pipeline berjalan

### A. `scripts/preprocess.py`

```text
loader.py
  → validator.py
  → crop_filter.py
  → missing_values.py
  → cleaner.py
  → feature_engineering.py
  → preprocessing_pipeline.py
```

Output utama:

```text
data/processed/model_dataset.csv
artifacts/<run_id>/data_audit/
```

### B. `scripts/train_models.py`

```text
splitters.py
  → pipeline_factory.py
  → model_registry.py
  → tuner.py
  → evaluator.py
  → interpretation.py
  → model_selector.py
  → serializer.py
```

Output utama:

```text
artifacts/<run_id>/models/*.joblib
artifacts/<run_id>/metrics/model_comparison.csv
artifacts/<run_id>/selected_model/model_bundle.joblib
artifacts/<run_id>/splits/split_assignments.csv
```

### C. `scripts/optimize_scenarios.py`

```text
scenario.py
  → candidate_builder.py
  → predictor.py
  → objective.py
  → penalties.py
  → differential_evolution.py
  → optimizer.py
```

Output utama:

```text
artifacts/<run_id>/optimization/optimization_results.csv
artifacts/<run_id>/optimization/convergence_history.csv
```

### D. `scripts/evaluate_optimization.py`

```text
productivity.py
fertilizer_efficiency.py
sensitivity.py
input_stability.py
algorithm_stability.py
agronomic_validation.py
```

## 4. Keputusan yang wajib kamu isi

Semua keputusan utama dikumpulkan pada `configs/decisions.yaml`. Program berhenti dengan pesan jelas ketika nilai wajib masih kosong.

### 4.0 Pseudocount fitur rasio

```yaml
feature_engineering:
  ratio_pseudocount: null
```

Nilai ini digunakan pada `N/(P+c)`, `N/(K+c)`, dan `P/(K+c)`. Nilai yang terlalu kecil dapat menghasilkan rasio sangat ekstrem ketika denominator nol, sedangkan nilai terlalu besar dapat mengubah makna rasio. Uji beberapa nilai, periksa distribusi rasio, dan dokumentasikan alasan pemilihannya.

### 4.1 Profil fitur

Pilih profil dari `configs/features.yaml`:

```yaml
feature_profile: core_candidate
```

Kandidat yang tersedia:

- `core_candidate`: NPK, curah hujan, jenis tanah, dan tipe pupuk.
- `extended_candidate`: menambah pH, SOC, SOM, dan temperatur.
- `source_enriched_candidate`: menambah sumber N, P, dan K.

Pilih setelah membandingkan cakupan data, performa, leakage, dan interpretabilitas.

### 4.2 Strategi validasi

Contoh:

```yaml
validation_strategy: group_kfold
group_column: group_article_experiment
```

`exp.id` belum tentu unik lintas artikel. Karena itu kode juga membentuk `group_article_experiment` dan `group_article_experiment_year`.

### 4.3 Baseline

Pilih:

```yaml
baseline_policy: observed
```

Kode menyediakan `observed`, `zero`, `group_mean`, dan `historical_best`. `observed` merupakan kandidat utama karena membandingkan dosis aktual dan dosis optimum pada state yang sama.

### 4.4 Bounds NPK

```yaml
optimization:
  bounds:
    N: {min: null, max: null}
    P: {min: null, max: null}
    K: {min: null, max: null}
```

Isi berdasarkan domain historis, literatur, dan/atau validasi pakar. Angka pada file contoh hanyalah dummy.

### 4.5 Penalti total dosis

```yaml
optimization:
  total_soft: null
  total_max: null
  lambda_total: null
```

- `total_soft`: batas nyaman sebelum penalti.
- `total_max`: hard constraint total NPK.
- `lambda_total`: kekuatan penalti.

### 4.6 Penalti komposisi ekstrem

```yaml
optimization:
  proportion_bounds:
    N: {min: null, max: null}
    P: {min: null, max: null}
    K: {min: null, max: null}
  lambda_composition: null
```

Proporsi dihitung terhadap total NPK. Rentang ini tidak boleh diperlakukan sebagai rasio universal semua lahan tanpa dasar agronomis.

### 4.7 Skala objective

Yield dapat bernilai ribuan, sedangkan penalty 0–1. Isi `yield_scale`, misalnya IQR target pada data training, agar bobot penalty dapat ditafsirkan secara konsisten.

### 4.8 Perturbasi sensitivitas

```yaml
perturbations:
  initial.ph.soil:
    method: absolute
    value: null
  avprec.mm:
    method: iqr_fraction
    value: null
```

Perturbasi terlalu kecil dapat tidak terlihat pada model berbasis pohon; terlalu besar tidak lagi bersifat lokal.

### 4.9 Aturan validasi agronomis

```yaml
agronomic_validation:
  boundary_warning_fraction: null
  max_training_distance: null
  max_input_stability_index: null
```

Kode memberi status dan peringatan. Keputusan kelayakan tetap memerlukan interpretasi agronomis.

## 5. Bagian yang dinilai melalui interpretabilitas

Gunakan keluaran untuk menilai:

1. Apakah N, P, dan K berkontribusi terhadap prediksi?
2. Apakah model terlalu bergantung pada jenis tanah, sumber pupuk, artikel, atau lokasi?
3. Apakah prediksi selalu naik ketika dosis dinaikkan?
4. Apakah solusi hampir selalu berada di batas atas?
5. Apakah performa group-based atau temporal jatuh jauh dibanding random split?

File utama:

```text
modeling/interpretation.py
validation/leakage_checks.py
validation/domain_checks.py
surrogate/domain_guard.py
```

## 6. Bagian yang memerlukan keputusan penelitian

- pseudocount rasio NPK;
- profil fitur;
- group key;
- strategi validasi;
- baseline;
- bounds;
- penalty weights;
- perturbation size;
- kriteria pemilihan surrogate.

Semua keputusan tersebut perlu dijelaskan di Bab 3 dan dianalisis di Bab 4.

## 7. Bagian yang memerlukan validasi agronomis

- batas NPK;
- batas total dosis;
- rentang proporsi NPK;
- interpretasi solusi dekat batas;
- arah perubahan dosis terhadap state variable;
- penilaian apakah hasil hanya simulasi metodologis atau layak dijadikan bahan rekomendasi.

## 8. Checklist sebelum menjalankan

```text
[ ] `python scripts/prepare_data_directory.py` sudah dijalankan
[ ] CSV tersedia di data/input/data.csv
[ ] ratio_pseudocount sudah diisi dan diperiksa distribusinya
[ ] Nama kolom sesuai configs/data.yaml
[ ] feature_profile sudah dipilih
[ ] validation_strategy sudah dipilih
[ ] group_column tersedia jika group validation
[ ] baseline_policy sudah dipilih
[ ] bounds NPK sudah diisi
[ ] total_soft dan total_max sudah diisi jika penalty aktif
[ ] proportion bounds sudah diisi jika penalty aktif
[ ] lambda penalty sudah diisi
[ ] perturbation sudah diisi sebelum analisis sensitivitas
```

## 9. Urutan pengerjaan yang disarankan

1. Jalankan preprocessing dan baca audit.
2. Bandingkan beberapa profil fitur.
3. Jalankan model prediktif tanpa optimasi.
4. Periksa leakage dan interpretabilitas.
5. Pilih surrogate.
6. Jalankan optimasi tanpa penalty sebagai diagnosis.
7. Tambahkan constraint dan penalty.
8. Lakukan analisis bobot penalty.
9. Jalankan sensitivitas dan stabilitas.
10. Validasi bersama pakar agronomi.

Optimizer dapat menemukan kelemahan model. Karena itu objective tinggi tidak boleh dibaca sendirian tanpa domain check, stabilitas, dan interpretasi.
## Pengaturan fitur dan imputasi aktif

Training melalui `scripts/train_models.py` dan bagian training
`notebooks/2_model_experiment/experiment_A.ipynb` memakai trainer yang sama.
Pengendali imputasinya adalah `configs/decisions.yaml`:

```yaml
feature_profile: minimalist_features
state_imputation:
  mode: none
  default_method: none
  columns: {}
```

`mode: none` menonaktifkan seluruh imputer, termasuk jika konfigurasi lama
masih memiliki metode per kolom. NaN numerik diteruskan ke estimator;
kategorikal di-one-hot tanpa imputasi (NaN menjadi kategori tersendiri).
Model aktif di `configs/models.yaml` adalah XGBoost dan CatBoost.

Untuk imputasi, gunakan `mode: plan`, lalu tentukan `default_method`,
`columns` dan, jika diperlukan, `mice_shared`. Contoh lengkap MICE dengan
profil `advanced_imputation` ada di `configs/examples/decision.example.yaml`.
Contoh tidak dimuat otomatis: salin bagian yang diperlukan ke konfigurasi
aktif secara eksplisit, dan pastikan dataset memiliki semua kolom profil.
Dalam mode plan, default kategorikal adalah `most_frequent`; metode `none`
per kolom mempertahankan missing tanpa imputer.

Pipeline, mode dan konfigurasi imputasi disimpan dalam bundle baru agar
inferensi/optimasi mengikuti preprocessing saat training. Bundle dan metrik
historis tidak berubah; label `native` pada eksperimen lama bukan bukti
bahwa bundle tersebut dilatih tanpa imputasi. Diperlukan training baru
untuk memperoleh model riset yang memakai alur ini.

## Konfigurasi aktif experiment B

Konfigurasi saat ini memakai `feature_importance_candidates` (97 input), dua
model XGBoost/CatBoost, dan tanpa imputasi. `configs/paths.yaml` membaca raw CSV
serta baseline 1.000 baris sebagai referensi cohort; output baru adalah
`data/processed/experiment_B_candidates.csv`. Artefak run tersimpan di
`artifacts/experiment_B/<run_id>/`.

Dari root proyek, dengan environment Python proyek aktif:

```powershell
python scripts/preprocess.py
python scripts/train_models.py
```

Atau `python scripts/run_full_experiment.py` untuk preprocessing dan training
dalam satu run. Preprocessing mempertahankan ID, urutan, target, dosis dan kolom
baseline yang sama; hash/ID/nilai yang tidak cocok menyebabkan proses berhenti.
Parsing seluruh numerik profil tidak mengimputasi missing. Baseline lama tidak
ditulis ulang. Rincian konfigurasi dan keterbatasan importance ada di
`docs/EXPERIMENT_B_PLAN.md`. Dokumen pengaturan minimalist sebelumnya adalah
contoh baseline terdahulu; profil aktif kini ditentukan oleh decisions.yaml.
