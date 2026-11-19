# NO_ENGINEERED ? Hasil, Interpretasi, dan Implikasi untuk Optimasi

Diperbarui: **19 November 2026**. Sumber hasil adalah **kode cell dan output terbaru**
`notebooks/2_model_experiment/experiment_B.ipynb`, lalu diverifikasi terhadap CSV,
bundle, dan snapshot run. Judul/narasi Markdown lama notebook tidak digunakan
sebagai bukti konfigurasi atau hasil saat ini. Dokumen ini menggantikan
`NO_ENGINEERED_USAGE.md`; petunjuk running tetap tersedia di bagian akhir.

## 1. Identitas eksperimen dan batas bukti

| Komponen | Terverifikasi pada run ini |
| --- | --- |
| Run | `artifacts/no_engineered/ta_20261119T040155Z` |
| Notebook | Cell 1 menunjuk run tersebut; cell 3?9, 14?23 memuat hasil terkait |
| Dataset | `data/processed/no_engineered.csv`: 1.000 baris, 44 kolom tersimpan |
| Input model | **31 fitur asli**: 17 numerik + 14 kategorikal; 88 kolom setelah transformasi |
| Pengecualian input | Semua tujuh turunan NPK dan semua empat koordinat tidak masuk model |
| Cohort | Seluruh nilai/ID/urutan reference cohort dipertahankan; 0 baris dibuang karena target/dosis missing pada run ini |
| Split | 808 training / 192 holdout; 212 / 54 group artikel-eksperimen; tanpa overlap record/group |
| Overlap artikel | 7 artikel muncul pada kedua partisi; evaluasi bukan article-disjoint |
| Tuning | RandomizedSearchCV, 30 kandidat unik selesai per model, 0 gagal; scoring negative RMSE; 5-fold GroupKFold |
| Seed | **Search seed 11**; estimator dan split seed 42; permutation seed 42, 10 pengacakan |
| Missing | Tidak ada imputer; numerik native NaN, kategori melalui one-hot encoding |
| Model terpilih oleh pipeline | **CatBoost**, sesuai `selected_model/selection.json` |

Snapshot dan search summary run adalah sumber konfigurasi efektif. Angka search
seed 42 pada penjelasan persiapan sebelumnya bukan seed hasil run ini. Dataset
SHA-256: `2162ca28d54b8620397b300d44dc47b3fb6bbb83821c5da0cfd60d407892b743`.
Reference SHA-256:
`94ab8b338849d0c46cabbff786e232d233eceaf9f69c74b47297f50ca56e27b6`.
Keduanya sudah dicocokkan dengan file aktual. Split ID juga identik dengan run
B luas `ta_20261116T203731Z`.

Nomor cell pada dokumen mengikuti indeks cell berkas notebook (mulai dari 0).

**Catatan reproduksibilitas notebook:** cell 12 yang tersimpan menghasilkan
`AssertionError` karena masih berisi `assert len(features) == len(set(features)) == 97`.
Bundle aktual berisi 31 fitur. Output tabel setelah cell itu konsisten dengan
artefak 31 fitur dan telah diverifikasi terpisah, tetapi notebook belum dapat
menyelesaikan Run All dalam keadaan sekarang. Ini kesalahan assertion konteks
lama, bukan bukti perubahan cohort atau gagalnya training. Cell 25 tidak dieksekusi
pada run urutan saat ini; flag joint permutation tetap False. Jika kelak diaktifkan,
blok koordinatnya juga perlu disesuaikan karena koordinat tidak tersedia sebagai
input. Tidak ada hasil joint permutation/kurva respons/DE yang disimpulkan di sini.
Notebook tidak diubah dalam tugas dokumentasi ini.

Sumber utama: [notebook](../notebooks/2_model_experiment/experiment_B.ipynb),
[model comparison](../artifacts/no_engineered/ta_20261119T040155Z/metrics/model_comparison.csv),
[snapshot](../artifacts/no_engineered/ta_20261119T040155Z/config_snapshot/),
[training manifest](../artifacts/no_engineered/ta_20261119T040155Z/training_manifest.json),
[selection](../artifacts/no_engineered/ta_20261119T040155Z/selected_model/selection.json).

## 2. Hasil prediksi: berguna, dengan variasi antarkelompok yang besar

MAE/RMSE dalam kg/ha. Kolom ? adalah SD antarlima fold, bukan confidence interval.

| Metrik | XGBoost | CatBoost |
| --- | ---: | ---: |
| CV MAE | 1.341,887 ? 452,098 | 1.358,640 ? 388,533 |
| CV RMSE | 1.902,103 ? 652,454 | **1.898,157 ? 523,927** |
| CV R? | 0,603020 ? 0,221431 | **0,612106 ? 0,171078** |
| Holdout MAE | 1.099,415 | **1.059,520** |
| Holdout RMSE | 1.520,929 | **1.432,413** |
| Holdout R? | 0,706421 | **0,739598** |
| Eligibility CV R? ? 0,60 | Lolos | Lolos |
| Selection score; lebih kecil lebih baik | 0,5 | **?0,5; dipilih** |

Baseline konstan menggunakan mean target training mempunyai holdout RMSE
**2.812,128 kg/ha**. Kedua model mengalahkannya, sehingga ada kegunaan prediktif
pada holdout ini. CatBoost menjadi kandidat paling masuk akal dari run ini:
CV RMSE sedikit lebih rendah, variasi fold lebih kecil, serta holdout lebih baik.
XGBoost tetap mempunyai CV MAE lebih rendah. Seleksi otomatis memakai gabungan
metrik CV yang dikonfigurasi, bukan memilih berdasarkan holdout.

Selisih CV RMSE kedua model hanya **3,946 kg/ha**, jauh lebih kecil daripada
variasi antarfold. Jadi selection flag bukan bukti superioritas statistik atau
kemenangan universal CatBoost. Keduanya juga hanya sedikit melampaui ambang
eligibility; ambang itu syarat administratif pemilihan kandidat, bukan validasi
respons agronomis. CV untuk konfigurasi hasil tuning ini memakai fold yang juga
dipakai saat pencarian, sehingga merupakan bukti pengembangan, bukan evaluasi
nested yang independen.

Cell 21 menunjukkan heterogenitas yang perlu dibawa ke optimasi:

- R? validation XGBoost berkisar **0,292?0,814**; CatBoost **0,341?0,777**.
- Fold 2 memiliki gap RMSE validation minus training sekitar **1.400,428**
  untuk XGBoost dan **1.169,228** untuk CatBoost. Ada persoalan generalisasi
  kelompok/fold tertentu; bukan sekadar semua fold mengalami pola yang sama.
- Fold 3?5 justru memiliki RMSE validation lebih rendah daripada training.
  Ini dapat konsisten dengan perbedaan kesulitan/distribusi kelompok; tidak
  boleh langsung dinyatakan sebagai leakage atau bukti tidak ada overfitting.
- Group `41::5` dan `41::6`, masing-masing n=5, mengalami overprediction:
  bias XGBoost ?3.961,077 / ?3.653,189; CatBoost ?3.056,903 / ?3.026,860 kg/ha.
  Bias adalah actual minus predicted. Periksa konteks kelompok tersebut sebelum
  menyatakan model andal untuk semua skenario; tidak ada penghapusan baris.
- Ada **67 kombinasi input unik** dari 192 observasi holdout. XGBoost memberi
  55 prediksi unik, CatBoost 67. Input berulang menjelaskan sebagian prediksi
  berulang; angka itu sendiri tidak menguji plateau sepanjang perubahan dosis.

## 3. Importance: N/P lebih menonjol, missingness tetap kuat

### 3.1 Built-in importance pada fitur asli

Agregasi menggunakan mapping fitted encoder ke fitur asal, kemudian menjumlahkan
kategori. Persentase berikut dinormalisasi **di dalam setiap model**. XGBoost
menggunakan importance gain pada tree; CatBoost menggunakan ukuran built-in
berbeda. Persentasenya membantu membaca masing-masing model, bukan skala efek
agronomis yang dapat dibandingkan langsung antarmodel.

| Fitur | XGBoost (%) | CatBoost (%) |
| --- | ---: | ---: |
| `soil_type` | **39,429** | 11,771 |
| `N.rate.kg_ha` | 1,448 | **27,772** |
| `P.rate.kg_ha` | **21,226** | **13,098** |
| `K.rate.kg_ha` | 1,013 | 1,470 |
| `avprec.mm` | 0,705 | **11,779** |
| `K_source` | 9,558 | 6,511 |
| `N_source` | 8,543 | 2,767 |
| `P_source` | 6,056 | 3,678 |

Pada XGBoost, dummy **`soil_type_nan` sendiri menyumbang 34,376%** dari total
built-in importance; pada CatBoost **9,381%**. Maka dominasi soil type tidak
boleh ditulis sebagai bukti bahwa sifat fisik tanah paling menentukan hasil.
Model memakai indikator tidak tercatatnya soil type, yang dapat merekam pola
pelaporan/penelitian. Penyebab pola missing itu belum ditetapkan.

CatBoost menyebarkan reliance lebih jelas pada N, P, dan curah hujan, sedangkan
XGBoost lebih terkonsentrasi pada missing soil type, P dan kategori sumber pupuk.
Ini menguatkan alasan memeriksa CatBoost sebagai kandidat surrogate, tetapi
importance tidak menjelaskan arah, saturasi, interaksi, maupun konsistensi respons.

Cell 4 terbaru memiliki **19 kolom encoding** XGBoost ?0,5% (kumulatif 96,09%)
dan **23** pada CatBoost (92,36%). Ini hasil run baru, bukan 46 kolom threshold
B luas yang dahulu dipakai membentuk profil. Tidak ada seleksi fitur baru yang
dilakukan dari tabel terbaru tersebut.

### 3.2 Permutation importance pada holdout

Nilai adalah kenaikan RMSE setelah satu fitur asli diacak, dalam kg/ha; ? adalah
SD dari 10 pengacakan, bukan CI atau variasi group/fold.

| Fitur | XGBoost | CatBoost |
| --- | ---: | ---: |
| `N.rate.kg_ha` | **596,853 ? 80,793** | **437,247 ? 72,555** |
| `P.rate.kg_ha` | **439,309 ? 50,043** | **429,060 ? 41,264** |
| `K.rate.kg_ha` | **?6,431 ? 1,850** | **48,662 ? 7,557** |
| `soil_type` | 129,988 ? 38,394 | 185,041 ? 30,640 |
| `avprec.mm` | 61,889 ? 25,455 | 140,350 ? 24,036 |
| `K_source` | 7,389 ? 9,127 | 116,931 ? 20,068 |
| `P_source` | 63,221 ? 15,428 | 49,346 ? 8,205 |
| `ph.method` | 56,108 ? 16,713 | 22,378 ? 6,224 |

Pada kedua model, N/P menduduki urutan teratas permutation. Ini menunjukkan
prediksi bergantung pada informasi dosis langsung, walaupun built-in N pada
XGBoost relatif kecil. Dua ukuran tersebut menanyakan hal berbeda: kontribusi
split tree versus kerusakan skor ketika informasi suatu kolom diacak.

Sinyal K belum konsisten antarmodel. Nilai negatif XGBoost berarti pengacakan K
memperbaiki skor pada pengujian ini; **bukan** bukti bahwa penambahan K menurunkan
hasil, bahwa K tidak penting secara agronomis, atau bahwa K boleh dibuang dari
variabel keputusan. CatBoost mempunyai reliance K positif tetapi lebih kecil
daripada N/P. K membutuhkan pemeriksaan respons dan dukungan kombinasi khusus.

XGBoost memiliki 19 skor positif / 8 negatif / 4 nol; CatBoost 27 / 2 / 2.
Korelasi rank Spearman permutation keduanya **0,531436**, jadi hanya ada kesepakatan
parsial. Contohnya, `P_fertilizer.conc` XGBoost memiliki permutation ?74,792 ?
20,133 meskipun muncul dalam built-in importance. Ini alasan untuk diagnostik,
bukan aturan menghapus fitur. Tidak ada uji signifikansi atau stabilitas lintas
fold/seed dari hitungan tersebut.

Tanpa total/ratio/prop, pengacakan dosis tidak lagi meninggalkan turunan NPK yang
stale. Namun N/P/K masih berkorelasi dan terkait sumber/konsentrasi/laju produk.
Permutation individual tetap dapat membuat kombinasi sintetis di luar pola data.
Ia mengukur predictive reliance, bukan efek intervensi pupuk.

## 4. Coverage dan missingness membatasi penggunaan surrogate

Semua 31 input mempunyai setidaknya satu nilai terukur; **0 all-missing**.
`ph.method` adalah satu-satunya fitur dengan satu nilai observed unik, yaitu
`pH in water`, sementara 65,1% missing. Sinyal kolom itu membedakan tercatat/tidak
tercatatnya metode, bukan membandingkan beberapa metode pH. Ini contoh konkret
mengapa importance besar belum berarti pengetahuan agronomis yang kuat.

| Fitur/contoh | Missing cohort | Dukungan yang perlu diperhatikan |
| --- | ---: | --- |
| N/P/K | 0% | Dosis lengkap, tetapi bukan kombinasi eksperimen yang padat |
| `soil_type` | 29,7% | Dummy missing dominan, terutama XGBoost |
| `avprec.mm` | 22,0% | 198 group training dan 53 holdout memiliki nilai |
| `initial.ph.soil` / `soc.percentage` | 46,6% / 50,6% | State penting masih sering tidak terukur |
| `N_fertilizer.conc` | 92,6% | Hanya 74 nilai; 13 group training / 7 holdout |
| `bd.2` dan `bd.3` | 89,5% | Masing-masing 6 group training, **0 group holdout terukur** |
| `clay.perc2` | 89,4% | 6 group training / 1 holdout terukur |
| `K.fertilizer.rate.kg_ha` | 84,7% | 27 group training / 10 holdout terukur |
| `prev_crop` | 80,6% | 19 group training / 5 holdout terukur |

Kedua fitted preprocessor mempertahankan **1.954 NaN numerik holdout** tanpa
imputasi. Ini memverifikasi mekanisme missing handling, bukan menghilangkan bias
missingness. Nol permutation `bd.2`/`bd.3` tidak menguji pengaruh nilai terukurnya
pada holdout karena seluruh nilainya missing di sana. Mengisi nilai baru saat
optimasi/sensitivity merupakan kondisi yang belum dievaluasi oleh holdout itu.

## 5. Apakah fitur turunan sebelumnya menutupi fitur asli?

**Observasi:** pada run ini N/P menjadi utama menurut permutation, dan CatBoost
juga memberikan built-in importance terbesar pada N/P. Dependensi model berubah
setelah profil diganti. Namun XGBoost tetap didominasi missing soil type; menghapus
turunan tidak otomatis membuat model lebih berpusat pada variabel agronomis.

Sebagai konteks deskriptif, CSV artefak B luas lama memberikan hasil berikut:

| Model | CV RMSE B luas ? no_engineered | Holdout RMSE B luas ? no_engineered |
| --- | ---: | ---: |
| XGBoost | 1.853,726 ? 1.902,103 | 1.440,205 ? 1.520,929 |
| CatBoost | 1.899,908 ? 1.898,157 | 1.440,017 ? 1.432,413 |

CatBoost relatif mempertahankan akurasi; XGBoost memburuk secara deskriptif.
Tidak ada peningkatan akurasi yang jelas dan konsisten pada kedua model.
Perbandingan ini berasal dari file hasil, bukan ringkasan Markdown lama, dan
**bukan ablasi terkontrol**: profil sekaligus menghapus koordinat serta fitur
lain, dan search seed berubah dari konfigurasi lama 42 menjadi 11 sehingga
parameter terpilih dapat berbeda. Cohort/split yang sama tidak menghilangkan
confounding tersebut. Klaim bahwa turunan merusak/menutupi informasi asli tetap
**hipotesis**, bukan kesimpulan kausal dari hasil ini.

Ablasi dengan input asli, split, seed dan budget sama lalu hanya menyalakan/
mematikan tujuh turunan dapat menguji hipotesis lebih langsung. Itu usulan
opsional, belum dijalankan dan tidak perlu otomatis menunda pekerjaan optimasi.

## 6. Implikasi untuk surrogate dan Differential Evolution

**Rekomendasi:** gunakan fitted CatBoost run ini sebagai **kandidat untuk
pemeriksaan respons/domain**, bukan langsung menggantikan surrogate final C/D.
Status selected_model mengidentifikasi pemenang pipeline pada run ini saja.

| Temuan | Implikasi bagi optimasi | Pemeriksaan minimum berikutnya |
| --- | --- | --- |
| Reliance N/P tinggi | Ada informasi dosis yang berguna, tetapi belum diketahui bentuk respons saat state tetap | Sweep satu dosis dan grid pasangan dosis pada fitted model; periksa arah, saturasi, lonjakan dan plateau |
| Reliance K berbeda antarmodel | Rekomendasi K dapat lebih bergantung pada pilihan model, support atau penalti | Periksa respons K pada beberapa state dan N/P; bandingkan dua model tanpa mengklaim salah satunya benar secara agronomis |
| 808 training hanya **33 kombinasi NPK unik** | DE menjelajah ruang kontinu yang jauh lebih luas daripada kombinasi teramati | Laporkan kombinasi terdekat, jarak dan dukungan konteks; rentang marginal saja tidak membuktikan dukungan joint |
| Korelasi training P?K 0,797172; N?P 0,530414; N?K 0,137538 | Mengubah satu dosis dapat keluar dari pola perlakuan, walau masih di dalam min/max masing-masing | Tinjau grid N?P/N?K/P?K serta komposisi pada kelompok relevan |
| Missing/category/source memberi sinyal | Optimizer bisa mengeksploitasi fungsi prediksi yang memuat pola penelitian, bukan respons pupuk teridentifikasi | Pertahankan state/mask missing per skenario dan laporkan ketidaklengkapannya; jangan optimalkan metode/depth atau missingness |
| Product rate, concentration dan nutrient source masih menjadi input | Menahan semua fitur itu tetap sambil mengubah NPK berisiko membuat perlakuan tidak konsisten | Konfirmasi unit, formulasi dan kebijakan fitur: mana tetap, mana terikat dosis, mana perlu dibatasi/diperbarui; jangan menciptakan konversi tanpa bukti |
| CV bervariasi dan beberapa group overpredicted | Satu skor global tidak menjamin rekomendasi andal untuk setiap konteks | Sertakan konteks/group error serta domain scenario pada pelaporan hasil |

Rentang marginal N/P/K training yang diperiksa adalah N 0?300, P 0?78, K
0?100 dalam kolom dosis yang ada. Ini **rentang observasi**, bukan batas agronomis
yang disetujui dan bukan izin otomatis memakai seluruh kotak pencarian tersebut.

Ada risiko implementasi konkret: `pipelines/run_optimization.py` menganggap
seluruh input bundle selain dosis dan engineered sebagai `state_features`.
Jadi `K.fertilizer.rate.kg_ha`, konsentrasi dan kategori sumber akan tetap saat
N/P/K berubah. Misalnya, bila semantik data mengharuskan dosis K berasal dari
laju produk ? fraksi K, mengubah K saja berpotensi melanggar hubungan itu.
Kesetaraan/unit tersebut **belum** ditetapkan oleh analisis ini. Demikian pula,
kategori `No fertilizer` tidak konsisten dengan dosis positif jika artinya
kontrol tanpa pupuk. Jangan menganggap generic optimization runner otomatis
menegakkan seluruh semantik yang diperiksa helper diagnostik D.

Keuntungan implementasi profil ini adalah tidak ada total/ratio/prop sebagai
input, sehingga risiko turunan stale di model berkurang. Candidate builder masih
menghitung turunan secara umum lalu memilih `bundle.feature_columns`; fitur
lain tidak otomatis masuk model. Keuntungan ini tidak menyelesaikan hubungan
produk/sumber, keterbatasan support, atau persoalan validitas intervensi.

Prediksi tree juga dapat memiliki plateau; DE tidak memerlukan gradien, tetapi
banyak dosis dapat memberi prediksi sama atau hampir sama. Saat itu hasil akhir
bisa ditentukan penalti, batas, tie, dan seed. Banyaknya prediksi unik CatBoost
pada holdout tidak membuktikan fungsi respons dosisnya bebas plateau.

## 7. Langkah selanjutnya yang disarankan

1. **Rapikan reproduksibilitas analisis:** ganti assertion 97 dengan pemeriksaan
   jumlah/schema dari snapshot dan bundle, lalu sesuaikan optional cell koordinat.
   Ini pekerjaan notebook yang belum dilaksanakan dalam pembaruan dokumentasi ini.
2. **Bekukan identitas kandidat secara terpisah:** pertahankan bundle CatBoost
   `selected_model/model_bundle.joblib`, snapshot, hash dataset dan split run ini.
   Jangan menimpa model/data/hasil C/D atau menggunakan hasil respons D sebagai
   bukti bagi CatBoost baru. Loader D saat ini membaca manifest frozen C; mengganti
   path bundle saja belum menjadikannya diagnostik kompatibel untuk run biasa ini.
3. **Lakukan diagnostik respons tanpa retraining:** pada beberapa skenario nyata
   dengan state tetap, periksa N/P/K satu per satu serta interaksinya, baseline
   yang sama, domain dan konsistensi produk/sumber. Gunakan titik dosis yang
   didukung data terlebih dahulu; beri label jelas pada eksplorasi di luar pola.
   Belum ada kurva respons untuk model no_engineered dalam bukti yang dibaca.
4. **Tetapkan keputusan optimasi bersama peneliti:** kebijakan laju produk/sumber,
   nutrient basis/unit, baseline, bounds dan penalty trade-off. `yield_scale`,
   `lambda_total`, `lambda_composition` masih null pada snapshot dan diperlukan
   oleh objective yang saat ini diaktifkan. Jangan mengisi angka arbitrer atau
   mematikan penalti semata-mata agar program berjalan.
5. **Setelah skenario/keputusan siap, jalankan DE dan evaluasinya:** bandingkan
   baseline dan kandidat dengan model/state yang sama, dukungan domain, total
   pupuk, sensitivitas input, serta stabilitas lintas seed. Laporkan sebagai
   **perubahan hasil yang diprediksi model**, bukan efek lapangan terverifikasi.

Prioritasnya pemeriksaan respons dan skenario, bukan otomatis menambah trial,
mengganti ke Optuna, menghapus fitur lagi, mengubah validasi atau membuang baris.
Importance fold-training/ablasi terkontrol tetap usulan tambahan bila diperlukan.
Holdout sudah dilihat dan memengaruhi iterasi analisis, sehingga tidak dapat
dilabeli sebagai test final yang belum pernah disentuh.

## 8. Konfigurasi dan cara menjalankan normal

Profil ada pada `configs/features.yaml` ? `profiles.no_engineered` dan dipilih
melalui `configs/decisions.yaml`. Dataset disiapkan dengan cohort reference;
44 kolom CSV termasuk kolom arsip/passthrough tidak berarti semuanya masuk X.
Hanya 31 fitur profil plus decision features yang dideduplikasi menjadi input.

Dari root proyek, pilih **satu** jalur:

```powershell
.\.rfo\Scripts\Activate.ps1
python scripts/preprocess.py
python scripts/train_models.py
```

Atau setelah environment aktif:

```powershell
python scripts/run_full_experiment.py
```

Training saja cukup bila dataset sesuai sudah tersedia dan tidak berubah.
Perintah tersebut menjalankan eksperimen baru; tidak diperlukan untuk membaca
hasil run yang dianalisis di sini. Skrip normal memuat `configs/*.yaml` otomatis,
tanpa membaca rekursif subfolder dan tanpa opsi CLI `--config-dir`. Tidak perlu
memasukkan setiap YAML atau menjalankan cohort sebagai program terpisah.
`cohort.mode: reference` menjaga ID/urutan/nilai/hash; bukan menjalankan MILP lagi.
`configs/examples/` tidak aktif otomatis. Runner C memakai konfigurasi tersendiri
dan masih memakai profil/dataset C, bukan otomatis profil no_engineered ini.
Optimasi tetap terpisah dari full experiment preprocessing/training.

### Daftar fitur konfigurasi

Numeric:

- `exp_year`
- `N.rate.kg_ha`
- `P.rate.kg_ha`
- `K.rate.kg_ha`
- `K.fertilizer.rate.kg_ha`
- `N_fertilizer.conc`
- `P_fertilizer.conc`
- `K_fertilizer.conc`
- `bd.2`
- `bd.3`
- `som.percentage`
- `soc.percentage`
- `n.total.percentage`
- `clay.perc1`
- `clay.perc2`
- `initial.ph.soil`
- `avprec.mm`

Categorical:

- `crop_rotation`
- `prev_crop`
- `fertilizer_type`
- `N_source`
- `P_source`
- `K_source`
- `soil_type`
- `texture1`
- `som.or.soc.depth`
- `soc.method`
- `n.total.method`
- `clay.depth1`
- `ph.soil_water.ratio`
- `ph.method`


## 9. Riwayat dan verifikasi pembaruan ini

- Pembentukan profil sebelumnya: 46 kolom encoding B luas ?0,5% dipetakan ke
  38 fitur asli, dikurangi tujuh turunan dan dua koordinat yang lolos menjadi
  29, lalu ditambah initial pH dan fertilizer type sesuai keputusan penelitian.
  Ketidaksesuaian agregasi 11-input yang dicatat saat persiapan adalah riwayat
  notebook saat itu, bukan deskripsi output agregasi run 31-input sekarang.
- Persiapan sebelumnya lulus 86 tes software dan preprocessing sementara.
  Catatan belum adanya training/export berlaku **pada tahap persiapan tersebut**.
  Sekarang training/export peneliti sudah terbukti ada pada run 19 November ini.
- Pembaruan ini memeriksa kode/output cell terbaru, CSV hasil, snapshot dan
  selected bundle; mengulang pemeriksaan hash, preservasi reference, split,
  schema, NaN dan prediksi saved model tanpa fit/tuning. Prediksi bundle cocok
  dengan CSV holdout. Angka model comparison dan importance sesuai artefak.
- Tidak ada perubahan kode penelitian, konfigurasi, notebook, dataset atau model.
  Tidak ada training, permutation baru, kurva respons baru atau DE dijalankan;
  tes software tidak diulang karena tugas ini hanya dokumentasi/read-only audit.
- Pemilihan surrogate final, kebijakan fitur terkait dosis, validitas agronomis,
  cutoff domain dan coefficient optimasi masih memerlukan keputusan peneliti.
