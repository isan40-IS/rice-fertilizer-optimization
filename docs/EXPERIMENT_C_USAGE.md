# Experiment C: penggunaan implementasi

Status: implementasi tersedia; studi riset penuh **belum dijalankan oleh agen**.
Rancangan asal: [EXPERIMENT_C_TUNING_PLAN.md](EXPERIMENT_C_TUNING_PLAN.md).

## 1. Environment

Dari root repository dengan environment proyek aktif:

```powershell
python -m pip install -e ".[tuning]"
```

Alternatif: `python -m pip install -r requirements-tuning.txt`.
Optuna opsional, diimpor hanya pada backend Optuna. Random/grid tidak memerlukannya.
Implementasi diverifikasi menggunakan Optuna 5.1.0; versi library ikut disimpan
dalam manifest. Jangan menyamakan hasil versi/platform berbeda tanpa pemeriksaan.

## 2. Konfigurasi dan data

Konfigurasi C ada di `configs/experiment_C/`, terpisah dari konfigurasi B/default.
Profil C dibekukan dari konfigurasi aktif peneliti pada saat implementasi:
**`minimalist_features` (11 input)**. Ini bukan profil broad 97 input. Tidak ada
profil baru yang dipilih berdasarkan hasil holdout. Untuk menguji profil lain,
ubah `configs/experiment_C/decisions.yaml` secara eksplisit dan gunakan profil yang
sama untuk semua backend/seed dalam satu rangkaian; laporkan sebagai eksperimen baru.

- Dataset: `data/processed/experiment_B_candidates.csv`, 1.000 observasi yang
  sama, digunakan read-only; target, dosis dan state baseline tidak diganti.
- Tidak perlu menjalankan preprocessing lagi. `processed_dataset_read_only: true`
  menolak preprocessing C agar tidak menimpa dataset/artefak B.
- `drop_all_missing: true`: ColumnTransformer memilih kolom yang teramati pada
  **setiap fit training fold**, termasuk refit akhir pada seluruh training.
  Observasi tidak dibuang dan fitur missing sebagian tetap dipakai. Nilai holdout
  tidak dapat menghidupkan kolom yang kosong pada fit training. Ini tidak secara
  fisik menghapus kolom CSV; bundle mencatat kolom yang benar-benar ditransformasi.
- Tanpa imputasi, XGBoost/CatBoost saja; model seed 42 dan group split seed 42 tetap.
- Pencarian CV memakai fold yang sama; CV ulang sesudah tuning adalah analisis
  development, bukan nested evaluation independen.

Pilihan backend untuk pemanggilan trainer biasa ada pada `models.yaml`:

```yaml
tuning:
  method: random       # random | grid | optuna
  search_seed: 42
  cv_n_jobs: 1
  random:
    n_iter: 30
  grid:
    max_candidates: 1000
  optuna:
    n_trials: 30
    timeout_seconds: null
    sampler: tpe
    pruner: none
    n_jobs: 1
    storage: null
    resume: false
```

Runner **perbandingan C** memilih daftar metode/seed dari
`configs/experiment_C/experiment.yaml` dan mengisi `tuning.method/search_seed`
untuk setiap subrun. Nilai efektif, bukan hanya file dasar, disimpan pada snapshot.

```yaml
methods: [random, optuna]
search_seeds: [11, 29, 47]
```

Untuk satu backend saja, misalnya gunakan `methods: [optuna]`. Untuk satu seed,
gunakan satu elemen pada `search_seeds`. Grid dijalankan sekali karena tidak
memiliki variasi sampling; grid penuh XGBoost ditolak batas default 1.000 kandidat.
Sediakan grid kecil secara eksplisit jika ingin memakai grid; jangan memotong
ruang parameter diam-diam lalu mengklaim domain/budget sama.

Versi awal mendukung **daftar nilai diskret** bersama pada semua backend. Rentang
kontinu/typed distributions belum didukung dan menghasilkan error. Pruning,
persistent storage/resume dan trial paralel juga ditolak secara eksplisit pada
adapter awal. Menggunakan Optuna bukan berarti semua kemampuan Optuna diaktifkan.

## 3. Jalankan studi

```powershell
python scripts/run_experiment_c.py
```

Opsional: `--config-dir configs/experiment_C`. Notebook
`notebooks/2_model_experiment/experiment_C.ipynb` memanggil fungsi runner yang sama;
ubah `RUN_TUNING = True` hanya jika siap. Menjalankan ulang membuat grup baru,
bukan melanjutkan trial lama. Grafik/analisis tidak otomatis melatih ulang model.

Default: dua model x dua metode x tiga seed x 30 kandidat x lima folds = hingga
1.800 fold fits; ada refit dan CV pelaporan tambahan. Seluruhnya belum dieksekusi
pada data penelitian oleh agen. Hentikan/ubah budget **sebelum** mulai jika terlalu
besar; perbandingan adil memerlukan budget/ruang yang setara, dan timeout dapat
menghasilkan jumlah trial aktual berbeda yang wajib dilaporkan.

Output:

```text
artifacts/experiment_C/exp_C_<timestamp>_<id>/
  experiment_manifest.json
  study_comparison.csv
  frozen_candidate.json
  random_seed_11/
    config_snapshot/
    training_manifest.json
    splits/cv_assignments.json
    splits/split_assignments.csv
    tuning/<model>_search_results.csv
    tuning/<model>_search_summary.json
    models/<model>.joblib
    metrics/model_comparison.csv
    interpretation/<model>_built_in_importance.csv
  optuna_seed_11/
  ...
```

Search history mencatat parameter, nomor/state, skor tiap fold, mean/std skor,
waktu fit/score dan identitas metode/seed/model. Optuna mencatat durasi trial;
random/grid memiliki mean fit/score times dari sklearn, bukan wall time trial.
Ringkasan memuat kandidat diminta/selesai/unik, waktu tuning+refit, search-space
hash dan fold hash. Runtime pipeline keseluruhan lebih besar daripada runtime tuner.

Seed pencarian saja yang berubah, sehingga variasi antarseed mengukur pencarian,
bukan perubahan split/model seed. Ulangan seed sama hanya cek reproducibility,
bukan replikasi independen. Trial bisa mengulang kombinasi; laporkan `n_unique`.

Jika fitting gagal, proses fail-fast dan grup bertanda failed. Error pencarian
dicatat pada `<model>_search_failure.json`; pada Optuna termasuk trial gagal dan
hasil parsial yang sempat selesai. Tidak ada parameter fallback tersembunyi.
Konfigurasi invalid dapat ditolak sebelum study/berkas failure dibuat. Untuk
group failed, baca manifest/error; jangan memperlakukannya sebagai rangkaian lengkap.

## 4. Eksplorasi pada notebook C

Isi `GROUP_DIR` dengan grup hasil di atas, biarkan `RUN_TUNING = False`, lalu
jalankan sel analisis:

1. Periksa satu fingerprint dataset/fitur/fold untuk semua study.
2. Bandingkan seluruh mean CV RMSE per seed/model/metode. Lebih kecil lebih baik;
   tuner menyimpan `best_score` sebagai negative RMSE (lebih besar lebih baik).
3. Bandingkan median/rentang/SD antarseed, variasi CV fold, runtime dan kandidat unik.
   Tiga seed bukan tiga replikasi dataset atau bukti signifikansi kuat.
4. Baca best-so-far versus trial/waktu. Penurunan minimum tidak membuktikan
   generalisasi membaik terus atau optimum global ditemukan.
5. Baca best parameters dan frekuensi nilainya. Ini analisis hyperparameter,
   bukan importance kolom agronomi; jangan menggabungkan parameter antar-study.

Seluruh hasil pada notebook yang disediakan masih kosong; tidak ada skor C buatan.
Manifest/snapshot/version dan dataset hash harus menyertai pelaporan hasil.

## 5. Kandidat CV dan pelaporan holdout terpisah

Runner memaksa `report_holdout: false`: tidak menghitung skor, prediksi atau
permutation importance holdout selama perbandingan search seed/metode. Built-in
importance berasal dari model full-training sehingga tetap diekspor.

Kandidat pada `frozen_candidate.json` adalah model/study eligible dengan mean CV
RMSE terkecil; ties diurutkan deterministik menurut model, metode, search seed.
Semua seed tetap dilaporkan. Threshold mean CV R2 >= 0.60 dipertahankan. Jika
tidak ada kandidat eligible, study selesai tanpa kandidat; jangan menurunkan
threshold secara otomatis. Ranking model berbobot per subrun tetap tersimpan,
tetapi aturan pembekuan antarstudy C dinyatakan terpisah dan memakai CV RMSE.

Setelah kandidat dibekukan, jalankan secara eksplisit:

```powershell
python scripts/run_experiment_c.py --evaluate-selected "artifacts/experiment_C/exp_C_<timestamp>_<id>"
```

Atau aktifkan `EVALUATE_FROZEN_HOLDOUT = True` di notebook untuk grup tersebut.
Ini memuat model terlatih, bukan mengulang tuning. Hasil tersimpan di
`selected_holdout/evaluation.json`, `predictions.csv` dan `permutation_importance.csv`.
Perubahan hash dataset membuat evaluasi ditolak. Jangan memilih study berdasarkan
holdout. Holdout proyek sudah pernah diperiksa, sehingga bukan pristine test.

## 6. Verifikasi dan batas implementasi

Tes mencakup tiga backend, API legacy, refit/prediksi, tanda skor, seed tetap,
group leakage, grid cap, invalid params, missing optional dependency, trial gagal,
filtering train-only, serangkaian study sintetis dan holdout eksplisit.

Resume, domain kontinu, nested CV, pruning/early stopping, feature selection dari
importance dan diagnosis respons NPK sebelum DE belum diimplementasikan oleh tugas
ini. Dataset/cohort, hasil B, dan koefisien optimasi tidak diubah. Tuning C tidak
otomatis menetapkan surrogate untuk penggunaan agronomis atau DE.
