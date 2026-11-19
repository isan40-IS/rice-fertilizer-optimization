# Rancangan tuning multi-backend dan analisis experiment C

Update implementasi: backend random/grid/Optuna diskret, runner berulang, konfigurasi
C terpisah dan notebook sudah tersedia. Lihat [petunjuk implementasi aktual](EXPERIMENT_C_USAGE.md).
Peneliti telah menyelesaikan Optuna 30 trial x 3 seed x 2 model x 5 fold;
hasil terverifikasi dan batas interpretasi tercatat pada bagian 7 petunjuk penggunaan. Dokumen di bawah mempertahankan rancangan awal;
resume/storage dan domain kontinu adalah pengembangan tahap berikutnya, bukan
kemampuan adapter saat ini. Profil C mengikuti konfigurasi peneliti saat implementasi
(`minimalist_features`).

Tanggal rancangan: 2026-11-17. Bagian di bawah adalah **rancangan historis**;
status aktual mengikuti pembaruan di atas. Percobaan 60 trial masih usulan.
Dokumen ini tidak mengaktifkan Optuna, mengganti konfigurasi, menghapus fitur,
menginstal dependency, atau menghasilkan skor baru.

## 1. Tujuan dan kondisi awal

Tujuan engineering: satu pipeline dengan pilihan `random`, `grid`, atau `optuna`
melalui konfigurasi. Tujuan experiment C: membandingkan kualitas pencarian,
biaya komputasi, dan kestabilannya dengan kondisi evaluasi yang sama.

Implementasi yang sudah diperiksa:

- `modeling/trainer.py` membangun pipeline preprocessing + estimator dan memanggil
  `modeling/tuner.py::tune_pipeline` dengan training data, CV, dan groups.
- Tuner hanya menggunakan `RandomizedSearchCV`, 30 kandidat secara default,
  `refit=True`, dan `error_score="raise"`. Search space kosong langsung melakukan fit.
- Output adalah fitted pipeline dan dictionary `best_params` / `best_score`.
- Trainer menghitung kembali CV, holdout, importance dan ranking antarmodel,
  lalu menyimpan bundle. CV ulang setelah tuning memakai fold yang sama;
  ini bukan evaluasi nested yang independen dari proses pemilihan parameter.
- `configs/models.yaml` berisi daftar nilai `search_space` dengan prefix `model__`,
  negative RMSE sebagai scoring, dan hanya XGBoost/CatBoost aktif.
- Optuna belum terdaftar pada `requirements.txt` maupun `pyproject.toml`.

Pertahankan 1.000 observasi, target, native numerical missing handling, group-aware
holdout dan lima CV folds. Jangan menjadikan peningkatan skor alasan mengubah split.
Experiment C menggunakan artefak terpisah dari B dan tidak menimpa study/model lama.

## 2. Benarkah Optuna harus dijalankan 2-3 kali?

**Tidak ada kewajiban teknis menjalankan Optuna beberapa kali.** Satu study dapat
berjalan dan memberikan parameter terbaik. Pengulangan berguna untuk mengukur
sensitivitas hasil terhadap randomness pencarian, bukan agar algoritmanya valid.

| Aktivitas | Makna | Pengaturan |
| --- | --- | --- |
| Satu study dengan 30 trial | 30 percobaan parameter dalam satu pencarian | Bukan 30 eksperimen independen |
| Study baru dengan seed sama | Pemeriksaan reproducibility | Data/fold/seed model/versi sama; trial sequential |
| Study baru dengan seed berbeda | Mengukur variasi strategi pencarian | Ubah sampler seed saja; fold dan seed model tetap |
| Melanjutkan study yang sama | Menambah anggaran pencarian lama | Storage dan study name sama; bukan replikasi independen |
| Melatih ulang best parameters | Mengukur variasi estimator jika seed model berubah | Bukan variasi sampler; eksperimen terpisah |
| Mengubah seed split | Mengubah pembagian kelompok/data | Bukan isolasi variasi tuning; tidak dilakukan pada studi awal |

Seed sampler dan objective yang deterministik membantu reproducibility. Eksekusi
paralel/distributed dan objective yang nondeterministik masih dapat memberi hasil
berbeda. Gunakan sequential trial untuk pemeriksaan awal dan catat lingkungan
komputasi; seed saja tidak menjamin kesamaan bit-per-bit pada semua platform.
[FAQ resmi Optuna tentang reproducibility](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results).

Untuk penelitian ini, **usulan** adalah tiga seed pencarian, misalnya `[11, 29, 47]`,
untuk Optuna dan random search. Ini proposal anggaran, bukan keputusan eksperimen
yang sudah dijalankan. Dua atau tiga pengulangan hanya memberi gambaran awal
kestabilan; tidak cukup untuk klaim statistik kuat tentang keunggulan algoritma.

## 3. Rancangan konfigurasi

Contoh berikut adalah kontrak usulan, **belum dapat dibaca implementasi sekarang**:

```yaml
# configs/models.yaml -- rancangan
scoring: neg_root_mean_squared_error
random_state: 42

tuning:
  method: optuna                    # random | grid | optuna
  search_seed: 11                   # berbeda dari seed model dan split
  cv_n_jobs: 1
  random:
    n_iter: 30
  grid:
    max_candidates: 1000            # usulan batas komputasi; bukan batas ilmiah
  optuna:
    n_trials: 30
    timeout_seconds: null
    sampler: tpe
    pruner: none
    n_jobs: 1
    storage: null                  # in-memory pada versi awal
    study_name: null                # diisi otomatis dari run/model/seed
    resume: false

models:
  xgboost:
    enabled: true
    class: xgboost
    params:
      random_state: 42
    search_space:
      model__max_depth: [2, 3, 4, 6, 8]
      model__learning_rate: [0.01, 0.03, 0.05, 0.1]
```

Contoh hanya menampilkan sebagian parameter; implementasi memakai seluruh search
space yang dibekukan. `tuning.method` berlaku untuk model aktif. Notebook tidak
mengganti dictionary model atau menyusun objective yang berbeda dari trainer.

Kompatibilitas:

- Tanpa bagian `tuning`, pertahankan random search dengan `search_iterations`
  lama. Saat konfigurasi baru tersedia, gunakan pengaturan baru secara konsisten
  dan catat nilai efektifnya; jangan mencampur dua anggaran tanpa penjelasan.
- Metode tidak dikenal, budget invalid, atau nama parameter yang tidak terdapat
  pada `pipeline.get_params()` harus gagal dengan pesan jelas sebelum training.
- Search space kosong tetap melakukan direct fit; tandai backend efektif sebagai
  `direct`, tanpa mengarang skor tuning. Validasi nama backend tetap dilakukan.
- Seed split di `validation.yaml` dan seed estimator di params model tetap terpisah
  dari seed pencarian. Jangan mengubah semuanya dengan satu sakelar seed.

## 4. Search space bersama dan batas grid

Tahap pertama mempertahankan daftar nilai yang sudah ada:

- Random: kandidat sampel dengan `RandomizedSearchCV`.
- Grid: seluruh kombinasi `GridSearchCV`.
- Optuna: daftar yang sama melalui `trial.suggest_categorical`.

Dengan demikian, perbandingan awal menguji cara memilih kandidat pada domain
yang sama. Nilai learning rate pada daftar tidak diperlakukan sebagai rentang
kontinu secara diam-diam.

Search space saat ini berisi 46.080 kombinasi XGBoost dan 768 CatBoost. Dengan
lima folds, grid penuh membutuhkan 230.400 dan 3.840 fold fits, sebelum refit dan
evaluasi lain. Grid harus menghitung jumlah kombinasi lebih dahulu dan menolak
budget melampaui batas konfigurasi; jangan memotong grid lalu menamainya grid penuh.

Pengembangan tahap kedua dapat menerima tipe eksplisit `categorical`, `int`, dan
`float` dengan `low/high/step/log`. Random memerlukan distribusi yang ekuivalen;
grid wajib mempunyai `grid_values` eksplisit untuk domain rentang. Validator harus
menguji endpoint, tipe, dan kombinasi step/log. Jika domain diperluas, perbandingan
baru adalah perubahan strategi **dan** search space, sehingga harus dilabeli terpisah.

## 5. Dispatcher dan objective yang konsisten

```text
trainer: pipeline + X_train + y_train + CV/groups + tuning config
  -> tune_pipeline
      -> validasi / normalisasi konfigurasi dan search space
      -> materialisasi indeks fold training yang sama
      -> random adapter | grid adapter | optuna adapter
      -> pilih parameter berdasarkan mean CV score
      -> refit clone pipeline pada seluruh training
      -> fitted_pipeline + search_info
  -> evaluasi / interpretasi / serialization seperti sebelumnya
```

Untuk setiap Optuna trial: pilih parameter, clone seluruh pipeline, fit preprocessing
dan estimator hanya pada training fold, hitung validation score, lalu rata-ratakan
lima skor. Jangan fit preprocessing di seluruh training sebelum CV. Group splitting
tetap berasal dari splitter proyek, bukan random split baru di objective.

Gunakan konvensi scorer scikit-learn: semakin besar semakin baik. Karena negative
RMSE aktif, study menggunakan **`direction="maximize"`**. Nilai -1400 lebih baik
daripada -1800; laporan RMSE positif adalah `-best_score`. Jangan membalik tanda
lagi pada model selector. Objective tuning tetap mean CV RMSE; ranking berbobot
antarmodel adalah tahap lain yang sudah ada.
[Arah objective dan storage Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html).

Holdout tidak boleh dipakai sebagai objective, pruning signal atau eval_set untuk
early stopping. Implementasi awal memakai `NopPruner` secara eksplisit dan tidak
menambah early stopping estimator. Fitur tersebut dapat direncanakan terpisah
setelah fairness dan penanganan fold sudah diuji. Sekadar memakai Optuna tidak
otomatis mempercepat training jika pruning tidak diimplementasikan.

Gunakan satu tingkat paralelisme yang terkendali. Awal: trial dan CV sequential,
sementara thread estimator dikonfigurasi eksplisit. Catat seed, thread, CPU/GPU,
versi Python/NumPy/sklearn/XGBoost/CatBoost/Optuna. Kesalahan konfigurasi harus
fail-fast; trial gagal/nonfinite tidak boleh menjadi best trial. Jika tidak ada
trial valid, hentikan dengan ringkasan error, bukan fallback tersembunyi.

## 6. Kontrak output dan artefak

Pertahankan `best_params` dan `best_score` agar trainer/consumer lama tetap bekerja.
Tambahkan backend, scoring/direction, seed efektif, waktu, jumlah trial/kandidat,
status, search space, skor per fold, dan lokasi artefak.

```text
artifacts/experiment_C/<run_id>/
  config_snapshot/
  splits/                  # termasuk fold assignments/manifest
  tuning/
    xgboost_search_results.csv
    xgboost_search_summary.json
    catboost_search_results.csv
    catboost_search_summary.json
  models/
  metrics/
  interpretation/
  selected_model/
```

Tabel kandidat/trial: model, method, study/run ID, search seed, nomor, params,
status, skor tiap fold, mean/std CV, durasi dan pesan kegagalan. Pisahkan jumlah
trial yang diminta, dicoba, selesai, gagal/pruned dan jumlah konfigurasi unik.
Optuna dapat mengunjungi parameter yang sama; jangan mengklaim 30 trial selalu
berarti 30 konfigurasi unik. Catat juga versi dependency dan hash dataset/profile/fold.

Bundle menyimpan metode dan best parameters, bukan mewajibkan objek Optuna untuk
prediksi. Impor Optuna hanya saat dipilih; dependency usulan adalah optional extra
`tuning` pada `pyproject.toml`, dengan versi diuji dan dicatat saat implementasi.
Random/grid tetap bekerja tanpa Optuna terpasang.

## 7. Langkah penggunaan Optuna setelah implementasi

1. **Siapkan environment:** implementasikan adapter dan optional dependency dahulu;
   install versi Optuna yang diuji pada environment proyek. Catat versi dan jalankan
   tes backend sebelum memakai data penelitian. Dokumen ini tidak menginstal paket.
2. **Bekukan data/fitur:** gunakan cohort 1.000 baris, pilih profil C yang eksplisit,
   dan catat penghapusan kolom all-missing sesuai arahan peneliti. Jangan hapus baris.
   Gunakan training untuk menentukan mask fitur; untuk mask yang dipelajari dalam
   CV, tempatkan pemilihnya di pipeline fold. Selaraskan dengan pemeriksaan
   preservation preprocessing agar pengurangan schema tidak diam-diam mengubah nilai.
   Kedua metode tuning wajib memakai profil/mask yang sama.
3. **Bekukan split:** simpan holdout IDs dan lima indeks fold pada training. Tetapkan
   seed estimator 42; ubah hanya `search_seed` untuk studi variasi pencarian.
4. **Pilih backend:** setelah implementasi, set `tuning.method: optuna`, TPE,
   `pruner: none`, jumlah trial dan seed. Buat run/study baru untuk setiap model/seed.
5. **Smoke test sintetis:** dua trial pada dua fold untuk memeriksa skor, refit,
   artefak dan reload. Ini uji kode, bukan hasil penelitian dan tidak digabungkan
   ke study penelitian.
6. **Jalankan studi riset:** instruksi yang direncanakan adalah memakai entry point
   bersama `python scripts/train_models.py`, setelah path/config C disiapkan.
   `experiment_C.ipynb` memakai fungsi yang sama atau membaca hasilnya; jangan
   menyalin ulang objective khusus notebook. Perintah ini **saat ini masih menjalankan
   random search**, jadi jangan menganggap Optuna aktif sebelum adapter dibuat.
7. **Ulangi untuk seed pencarian berikutnya:** study baru dengan budget sama;
   pertahankan dataset, fold, estimator seed dan ruang parameter.
8. **Bandingkan dengan random:** jalankan seed/budget yang setara. Grid hanya pada
   domain kecil yang dideklarasikan; jangan mengubah domain diam-diam agar muat budget.
9. **Analisis C dan freeze kandidat:** pilih dengan aturan berbasis CV yang ditetapkan
   sebelum melihat holdout. Refit best parameters hanya pada seluruh training.
10. **Evaluasi/report:** holdout menjadi pelaporan terbatas setelah pemilihan dibekukan.
    Holdout proyek sudah pernah diperiksa; jangan mengklaim pengujian konfirmatori
    yang benar-benar belum disentuh. Lanjutkan diagnosis respons NPK sebelum DE.

### Contoh inti objective (ilustrasi API, bukan kode integrasi yang sudah tersedia)

Kode berikut membutuhkan `pipeline`, `X_train`, `y_train`, dan `fold_indices`
yang sudah disiapkan dengan group-aware splitter proyek. Ini belum dijalankan.

```python
import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        name: trial.suggest_categorical(name, choices)
        for name, choices in search_space.items()
    }
    candidate = clone(pipeline).set_params(**params)
    scores = cross_val_score(
        candidate, X_train, y_train, cv=fold_indices,
        scoring="neg_root_mean_squared_error", n_jobs=1,
        error_score="raise",
    )
    trial.set_user_attr("fold_scores", scores.tolist())
    return float(scores.mean())

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=search_seed),
    pruner=optuna.pruners.NopPruner(),
    direction="maximize",
)
study.optimize(objective, n_trials=n_trials, n_jobs=1)
best_pipeline = clone(pipeline).set_params(**study.best_params)
best_pipeline.fit(X_train, y_train)
```

### Resume tidak sama dengan pengulangan

Storage SQLite/RDB dan `load_if_exists=True` dapat melanjutkan study bernama sama.
Memanggil `optimize(n_trials=30)` lagi meminta 30 trial tambahan, bukan menetapkan
total keseluruhan ke 30. Kontrak config yang direncanakan harus menjelaskan apakah
budget adalah tambahan atau total. Usulan runner: budget adalah total target, hitung
trial yang sudah ada terlebih dahulu dan laporkan seluruh status.

Penyimpanan riwayat trial tidak otomatis menjamin kelanjutan RNG sampler persis
seperti proses yang tidak terputus; state sampler dan lingkungan perlu dikelola
jika kesetaraan urutan diperlukan. Periksa hash data, fold, fitur, scoring dan
search space sebelum resume; tolak ketidakcocokan. Sebagai tahap pertama,
utamakan studi sequential utuh daripada resume lintas sesi.
[Panduan resmi penyimpanan/resume Optuna](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).

## 8. Rencana isi experiment_C.ipynb

Notebook harus memisahkan sel eksekusi tuning yang eksplisit dari sel baca/analisis
artefak agar rerun grafik tidak otomatis menjalankan studi baru.

1. **Manifest:** daftar run/model/method/seed, data hash, profil, mask fitur, CV folds,
   estimator seed, budget, scoring dan versi. Tolak perbandingan yang tidak sepadan.
2. **Sanity checks:** group overlap, target/schema, no-imputation, sign score,
   trial count/status dan hasil refit/reload. Jangan memakai hasil smoke test.
3. **Riwayat pencarian:** plot best-so-far CV RMSE versus nomor trial dan waktu
   kumulatif. Turunnya best-so-far adalah sifat pemilihan minimum, bukan bukti
   bahwa generalisasi selalu membaik. Kurva mendatar bukan bukti optimum global.
4. **Ringkasan tiap study:** best mean CV RMSE, fold std, durasi, trial valid/unik,
   best parameters dan eligibility. Contoh kolom:

   | model | method | search_seed | trials_complete | best_cv_rmse | fold_std | runtime_s |
   | --- | --- | --- | --- | --- | --- | --- |
   | belum dijalankan | | | | | | |

5. **Variasi antarseed:** tampilkan semua tiga hasil per model/metode, median,
   rentang dan SD. Bedakan `fold_std` dalam study dan variasi hasil terbaik antarstudy.
   Parameter berbeda dengan skor serupa dapat berarti solusi alternatif pada
   ruang pencarian, bukan otomatis implementasi salah.
6. **Perbandingan backend:** laporkan perbedaan CV RMSE dan biaya per seed, jumlah
   kandidat unik, serta batas domain/budget. Jangan melaporkan hanya seed paling
   menguntungkan. Tiga seed yang dipasangkan bukan tiga replikasi dataset independen.
7. **Parameter diagnostics:** tampilkan parameter terbaik, frekuensi nilai batas,
   sebaran trial dan hubungan parameter-skor. Optuna hyperparameter importance
   menjelaskan variasi antartrial pada ruang yang dipilih, **bukan feature importance
   kolom agronomi**. Jangan menafsirkannya sebagai efek kausal parameter.
8. **Reproducibility check:** satu fresh study kecil dengan seed sama dapat menjadi
   cek implementasi; hasil exact hanya diharapkan bila objective/lingkungan memang
   deterministik. Jangan menghitung study ulang identik sebagai seed independen.
9. **Kandidat final:** tetapkan aturan CV sebelumnya; misalnya mean CV terbaik dengan
   pelaporan semua seed. Membandingkan best-of-three dengan satu random run tidak
   adil karena total anggaran berbeda. Tidak ada penggabungan best parameters antarstudy.
10. **Holdout dan respons:** setelah kandidat dibekukan, baca metrik/importance dan
    residual kelompok. Jangan memilih study berdasarkan holdout, lalu mengklaimnya
    sebagai final test independen. Tetap periksa respons NPK, dukungan domain dan
    kestabilan rekomendasi untuk kebutuhan surrogate DE.

## 9. Anggaran dan aturan berhenti yang diusulkan

Contoh: dua model x dua metode (random/Optuna) x tiga search seeds x 30 trial
x lima folds = **1.800 fold fits**, belum termasuk refit/evaluasi tambahan.
Nilai ini usulan, bukan persetujuan untuk menjalankan seluruhnya.

Mulai dengan smoke test, lalu estimasi biaya dari pilot yang dilabeli jelas.
Tetapkan budget sebelum studi utama. Jika search space atau budget diubah sesudah
melihat hasil, tandai sebagai eksperimen lanjutan dan bandingkan secara setara.
Tidak perlu menetapkan ambang peningkatan RMSE arbitrer sekarang; laporkan ukuran
perbaikan, variasi dan biaya, lalu pertimbangkan kepentingan praktisnya untuk DE.
Jangan menjadikan tuning berulang penghalang tanpa batas menuju optimasi.

## 10. Berkas pengembangan dan acceptance criteria

| Berkas | Perubahan yang direncanakan |
| --- | --- |
| `modeling/tuner.py` | Dispatcher, adapter, validasi, refit dan search_info seragam |
| `modeling/trainer.py` | Teruskan config/seed, ekspor search history dan bundle metadata |
| `configs/models.yaml` | Pilihan metode dan budget; kompatibilitas konfigurasi lama |
| `requirements.txt` / `pyproject.toml` | Kebijakan dependency Optuna yang diuji, lazy import |
| `tests/` | Backend/scoring/group/refit/error/budget/reproducibility tests sintetis |
| `notebooks/2_model_experiment/experiment_C.ipynb` | Orkestrasi eksplisit dan analisis manifest/history/seed/biaya |

Selesai implementasi bila ketiga backend memberi fitted pipeline dengan kontrak
lama, CV training tetap terjaga, holdout tidak masuk objective, hasil dapat ditelusuri,
grid besar ditolak, missing dependency hanya memblokir Optuna, dan tes terkait lulus.
Selesai experiment C hanya setelah hasil benar-benar dijalankan/diperiksa dan
keterbatasannya dilaporkan. Saat ini tidak ada hasil C yang diklaim.
