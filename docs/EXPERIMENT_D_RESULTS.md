# Experiment D — Diagnosis, interpretasi, dan kesimpulan

Tanggal laporan: 2026-11-18. Status: analisis artefak yang telah selesai; tidak ada training, tuning, perubahan konfigurasi, atau eksekusi Differential Evolution (DE) dalam penyusunan laporan ini.

## 1. Dasar bukti dan ruang kesimpulan

Sumber utama adalah run lengkap yang tercantum pada keluaran notebook:

`artifacts/experiment_D/exp_D_20261117T124700Z_d1e89993/`

Run ini berstatus `complete`. Kandidat berasal dari Experiment C `exp_C_20261117T055252Z_b321014c`, yaitu XGBoost dari pencarian Optuna seed 11. Model/split seed tetap 42; seed pencarian bukan seed pembagian data. Profilnya `minimalist_features`, tanpa imputasi, dengan 11 input mentah. Data berjumlah 1.000 observasi, terbagi menjadi 808 training dan 192 holdout.

Seluruh path sumber yang dicatat pada manifest D diperiksa ulang; hash masih cocok saat laporan dibuat. Run verifikasi sebelumnya `exp_D_20261117T123759Z_33092529` menghasilkan metrik dan ID skenario yang sama. Pengulangan pada input/model yang sama ini bukan replikasi penelitian independen.

Run lebih baru `exp_D_20261118T061047Z_4f1c32ee` masih berstatus `running` saat pemeriksaan. Hasil parsialnya tidak digabungkan dengan laporan ini. Status tersebut sendiri belum membedakan proses masih berjalan dari eksekusi yang terhenti sebelum manifest diperbarui.

**Bahasa sederhana:** laporan ini membaca hasil yang benar-benar sudah selesai. Angka dari percobaan yang belum selesai tidak dicampurkan.

Pembedaan bukti dalam dokumen ini:

- **Terverifikasi:** angka atau perilaku yang tercatat pada artefak/kode.
- **Interpretasi:** implikasi yang masuk akal dari bukti tersebut.
- **Hipotesis:** penjelasan yang belum diuji secara khusus.
- **Rekomendasi:** usulan langkah berikutnya, bukan keputusan parameter final.

## 2. Kesimpulan utama

**Model siap secara teknis untuk menjadi objek eksperimen optimasi, dan terbukti responsif terhadap perubahan dosis pada skenario yang diuji. Kelayakan agronomis serta ketepatan respons perlakuannya belum terkonfirmasi.**

Tidak ditemukan dasar untuk menyatakan model gagal hanya karena kurvanya bertangga. Sebaliknya, akurasi holdout dan respons yang tidak datar juga belum cukup untuk menyatakan dosis hasil pencarian kelak valid di lapangan. Risiko utama berada pada cakupan kombinasi NPK yang jarang, perilaku rasio dekat dosis nol, pola missingness, dan keterbatasan metadata pembanding perlakuan.

**Bahasa sederhana:** model bisa dipakai untuk menyelidiki calon dosis, tetapi kita harus menjaga agar optimizer tidak memanfaatkan bagian model yang datanya lemah. Hasilnya tetap prediksi model, bukan resep pupuk yang sudah terbukti.

## 3. Integritas dan kesiapan teknis

Notebook memuat fitted pipeline dari artefak C; model tidak direkonstruksi dengan menyalin parameter. Pemeriksaan mencakup identitas kandidat, schema, hash dataset, pembagian train/holdout, keanggotaan CV, dan pemisahan grup eksperimen. Ringkasan D mencatat 17 pemeriksaan konsistensi yang lulus, antara lain:

- Baseline tanpa perubahan menghasilkan prediksi identik.
- Perubahan NPK menghitung ulang fitur turunan dengan formula training.
- State tetap ketika dosis diubah; NPK tetap ketika state diubah.
- Missing numerik yang didukung pipeline tetap menghasilkan prediksi finite.
- Target, identifier, fitur turunan langsung, dan dosis negatif ditolak sebagai override yang tidak sah.
- Prediksi berulang deterministik dan sumber data tidak dimutasi.

Pseudocount rasio yang digunakan bundle aktual adalah **1e-5**, bukan sekadar default fungsi 1e-8. Proporsi menggunakan epsilon fungsi 1e-8. Proporsi tambahan dihitung untuk analisis, tetapi bukan seluruhnya input efektif profil minimalist.

**Interpretasi:** kontrak inference dan rekalkulasi fitur sudah memiliki bukti teknis. Ini tidak memverifikasi mekanisme respons tanaman atau kausalitas data observasional.

## 4. Kinerja prediksi holdout

Sumber: `holdout_metrics.json`, `holdout_predictions.csv`.

| Metrik | Hasil |
| --- | ---: |
| Jumlah observasi | 192 |
| MAE | 1.042,96 kg/ha |
| RMSE | 1.466,33 kg/ha |
| R² | 0,7271 |
| Rata-rata residual, actual − predicted | −234,14 kg/ha |
| Jumlah nilai prediksi unik | 44 |

**Interpretasi teknis:** MAE menunjukkan rata-rata besar kesalahan absolut. RMSE yang lebih tinggi daripada MAE konsisten dengan adanya kesalahan yang lebih besar pada sebagian observasi; kasus penyumbangnya perlu dilihat dari residual per baris. Residual rata-rata negatif berarti model secara agregat cenderung memprediksi lebih tinggi pada holdout ini. R² membandingkan jumlah kuadrat error dengan variasi target terhadap rata-rata target holdout; bukan persentase prediksi yang benar.

**Bahasa sederhana:** kesalahan prediksi rata-rata sekitar 1,04 ton/ha, dan sebagian kasus meleset lebih jauh. Angka R² 0,727 tidak berarti 72,7% kasus diprediksi dengan benar.

Prediksi unik yang lebih sedikit daripada jumlah baris dapat berasal dari input berulang dan keluaran ensemble pohon yang sama. Angka 44 saja tidak mengidentifikasi penyebab atau membuktikan kegagalan; perlu pemeriksaan input dan leaf untuk atribusi lebih lanjut.

Holdout telah diperiksa dalam rangkaian penelitian sebelumnya. Karena itu, hasil ini **diagnostik, bukan pristine confirmatory test**. CV C juga digunakan selama pemilihan hyperparameter, bukan nested independent evaluation. Laporan ini tidak menyatakan generalisasi ke artikel/lokasi baru atau peningkatan terhadap eksperimen lama melalui ablation terkontrol.

## 5. Distribusi dan cakupan training

Sumber: `numeric_statistics.csv`, `observed_dose_combinations.csv`, `domain_reference_statistics.csv`.

| Input | Missing training | Nilai unik teramati | Min–max |
| --- | ---: | ---: | --- |
| pH | 42,70% | 12 | 4,3–7,2 |
| Curah hujan | 25,87% | 19 | 0–814,8 mm |
| SOC | 45,92% | 24 | 0,575–4,6% |
| N | 0% | 23 | 0–300 kg/ha |
| P | 0% | 18 | 0–78 kg/ha |
| K | 0% | 8 | 0–100 kg/ha |

Terdapat **33 kombinasi NPK unik di antara 808 baris training**. Jadi banyak baris tidak sama dengan banyak kombinasi perlakuan. Statistik min–max marginal juga tidak menjamin seluruh ruang tiga dimensi di dalamnya memiliki contoh.

Basis nutrient rate mengikuti dataset; P tidak otomatis diperlakukan sebagai P₂O₅, begitu pula K terhadap K₂O. Nilai minimum curah hujan nol adalah nilai teramati dalam pipeline, bukan bukti otomatis bahwa pengukurannya benar atau missing terselubung. Tidak ada pengubahan nilai tersebut dalam D.

**Bahasa sederhana:** model melihat banyak catatan, tetapi variasi resep NPK-nya terbatas. Mencampurkan tiga dosis yang masing-masing pernah muncul belum menjamin campuran tersebut pernah diuji.

## 6. D1 — Respons satu dosis

Sumber: `baseline_scenarios.csv`, `skipped_scenarios.csv`, `dose_response.csv`, `dose_response_summary.csv`.

Delapan baseline dipilih dari training tanpa menggunakan yield sebagai kriteria. Kontrol `No fertilizer`, record 1427, tidak diubah menjadi dosis positif sambil mempertahankan kategori kontrol. Tersisa tujuh baseline kompatibel dan **21 kurva**, dengan **451 titik pemeriksaan**. Ini pembatasan skenario diagnostik, bukan penghapusan observasi dataset.

Grid memakai Q05–Q95 training, ditambah baseline dan nol jika teramati. Setiap sweep mengubah satu nutrient rate, mempertahankan state dan dua dosis lain, lalu merekalkulasi fitur turunan. Baseline yang berbeda dapat mempunyai dosis pendamping berbeda; selisih antargaris tidak dapat diatribusikan hanya kepada state.

| Unsur yang divariasikan | Median amplitude prediksi | Rentang amplitude antarskenario |
| --- | ---: | ---: |
| N | 4.042,80 kg/ha | 3.946,99–6.779,11 |
| P | 3.041,14 kg/ha | 1.473,55–4.750,14 |
| K | 998,84 kg/ha | 801,38–1.838,37 |

Amplitude adalah maksimum dikurangi minimum sepanjang grid; bukan kenaikan dari baseline. Angka ini berasal dari seluruh titik grid, termasuk titik yang dapat melanggar domain. Besarnya amplitude juga bergantung pada rentang dosis yang diuji, sehingga bukan perbandingan importance atau efektivitas agronomis yang setara antarunsur.

**Interpretasi:** model responsif terhadap ketiga dosis pada baseline yang diuji. Kekhawatiran bahwa model sama sekali tidak berubah ketika dosis diubah tidak didukung hasil ini. Namun, perubahan tersebut melibatkan total/rasio bersama dosis langsung, sehingga tidak mengisolasi kontribusi satu kolom.

Kurva memuat interval naik, turun, dan datar. Maksimum berada pada endpoint pada satu dari tujuh kurva K; tidak pada kurva N/P dalam grid ini. Ketiadaan maksimum di endpoint tidak membuktikan optimum agronomis sudah ditemukan. Grid masih diskret dan mencakup kombinasi dengan dukungan terbatas.

**Bahasa sederhana:** model bereaksi ketika dosis diubah. Reaksi besar tidak berarti hasil panen sungguhan akan berubah sebesar itu, dan puncak grafik belum menjadi dosis yang sebaiknya dipakai.

### Diagnosis khusus: pelanggaran rasio

Sebanyak **109/451 titik (24,17%)** memiliki pelanggaran rentang training pada fitur numerik. Pada keluaran ini, nama fitur yang ditandai seluruhnya merupakan rasio NPK: `ratio_n_p`, `ratio_n_k`, atau `ratio_p_k`.

Ini menunjukkan bahwa menjaga dosis dasar dalam rentang marginal belum cukup untuk menjaga seluruh fitur turunan dalam domain training. Dengan formula N/(P+1e-5), perubahan P dekat nol dapat menghasilkan rasio sangat besar. Perilaku ekstrem ini merupakan konsekuensi formula yang dipertahankan dari training, bukan imputasi baru atau bukti otomatis bug inference.

Contoh record 1145 pada sweep P: amplitude seluruh grid **1.473,55 kg/ha**, tetapi hanya **319,59 kg/ha** ketika dihitung pada tujuh titik tanpa pelanggaran terverifikasi. Pada record 1623 untuk N, amplitude berkurang dari **4.042,80** menjadi **1.568,18 kg/ha** pada subset tersebut.

**Interpretasi:** sebagian amplitude bergantung pada wilayah di luar rentang fitur yang pernah dilihat model. Bahkan subset tanpa pelanggaran belum dapat disebut reliable, karena joint support dan state support tetap perlu diperiksa.

**Hipotesis yang belum diuji:** optimizer dapat menemukan skor tinggi di wilayah yang dipengaruhi rasio ekstrem atau kombinasi jarang. D belum menjalankan optimizer, sehingga eksploitasi tersebut belum menjadi hasil pengamatan.

## 7. D2 — Interaksi dua dosis

Sumber: `response_surface.csv` dan grafik `surface_*.png`.

Terdapat **1.350 titik** dari dua baseline dan tiga pasangan unsur pada grid 15 × 15. Sebanyak **220 titik (16,30%)** memiliki pelanggaran terverifikasi. Heatmap prediksi harus dibaca bersama heatmap jarak, baseline, dan marker pelanggaran.

**Interpretasi:** D menyediakan pemeriksaan permukaan fungsi model secara berpasangan. Laporan ini tidak menetapkan skor interaksi formal, mengklaim mekanisme fisiologis, atau memilih koordinat maksimum sebagai rekomendasi.

**Bahasa sederhana:** warna tinggi berarti model memprediksi yield tinggi untuk kombinasi itu. Jika kombinasi jauh dari contoh training, angka tersebut tetap berisiko walaupun terlihat menarik.

## 8. D3 — Respons state dan missingness

Sumber: `state_response_summary.csv`, `conditional_response_summary.csv`, `state_missing_response.csv`.

Pada tiga baseline per state yang diuji, amplitude sweep state berada pada rentang:

| State | Rentang amplitude prediksi |
| --- | ---: |
| pH | 257,62–838,50 kg/ha |
| Curah hujan | 1.327,60–2.207,56 kg/ha |
| SOC | 687,11–700,40 kg/ha |

Nilai ini bukan importance global; baseline dan rentang state berbeda. Overlay juga menunjukkan amplitude respons dosis dapat berubah ketika state berubah. Contoh record 1174: amplitude respons N sebesar 4.392,36 kg/ha pada pH 5,6 dan 4.157,55 pada pH 6,5. Jadi respons state tidak hanya dapat menggeser level; ringkasan amplitude dosis juga berubah pada contoh tersebut. Tidak ada uji kausal atau signifikansi interaksi.

Masking informasi menghasilkan perubahan prediksi yang perlu diperhatikan. Untuk curah hujan, tiga perubahan prediksi adalah **−1.273,03**, **+152,34**, dan **+714,77 kg/ha**. Pada sembilan uji masking state, dua skenario menghasilkan pola missing yang tidak tercatat di training.

**Diagnosis:** native missing handling menjaga inference tetap berjalan, tetapi tidak menjamin prediksi tidak sensitif terhadap missingness, tidak bias, atau didukung pola yang sama di training.

**Bahasa sederhana:** model memberikan jawaban meskipun informasi hilang. Jawabannya bisa berubah cukup besar ketika informasi yang semula ada disembunyikan; hilangnya informasi bukan perubahan kondisi lahan sebenarnya.

## 9. D4 — Ranking dosis dan perbedaan yield empiris

Sumber: `dose_ranking_pairs.csv`, `dose_ranking_summary.csv`, `dose_ranking_by_group.csv`.

| Kandidat pasangan | Jumlah | Grup | Agreement arah perubahan | Delta MAE | Delta RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Satu nutrient berubah | 63 | 32 | 92,06% | 647,13 | 931,29 |
| Beberapa nutrient berubah | 143 | 6 | 90,91% | 962,18 | 1.218,21 |

Satuan delta error adalah kg/ha. **Seluruh angka tabel bersifat deskriptif, bukan validasi respons terkontrol.** Pasangan dalam grup yang sama saling bergantung; denominator bukan jumlah eksperimen independen.

Sebanyak 102 pasangan ditandai karena konteks missing, 78 karena konteks berbeda, dan 26 karena metadata/desain belum dikonfirmasi. Dengan kebijakan konservatif default, jumlah pasangan eligible adalah **nol**. Ini tidak membuktikan seluruh pasangan mustahil sebanding; metadata/desain belum cukup untuk menyatakannya.

**Interpretasi:** agreement arah terlihat menjanjikan sebagai observasi awal, tetapi dapat dipengaruhi perbedaan state/manajemen dan pengulangan perlakuan. Tidak sah menyatakan model telah memvalidasi efek respons pupuk sebesar 92%.

**Bahasa sederhana:** model sering mengurutkan dua hasil panen dengan arah yang sama seperti data. Namun, kita belum memastikan apakah kedua perlakuan hanya berbeda pupuk atau juga berbeda kondisi lain. Karena itu, bukti respons terkontrol masih belum cukup.

## 10. D5 — Domain dan risiko extrapolation

Pada reference 808 baris training, **800 baris mempunyai jarak leave-one-out NPK nol** karena kombinasi berulang. Akibatnya Q50/Q90/Q95 reference per baris semuanya nol. Menggunakan Q95 itu langsung sebagai cutoff akan memberi aturan yang menyesatkan.

Reference 33 kombinasi unik memberikan Q50 **0,3995**, Q90 **0,5993**, dan Q95 **0,7024** pada jarak Euclidean setelah penskalaan training. Angka tersebut tetap diagnostik; belum menjadi threshold yang disetujui atau probabilitas keandalan.

Pada 192 holdout, tidak ditemukan pelanggaran range/kategori terverifikasi maupun pola missing baru; jarak NPK maksimum **0,3703**. Ini hanya menggambarkan kedekatan terhadap reference training dalam representasi yang diperiksa. Tidak membuktikan generalisasi ke lahan baru atau meniadakan ketergantungan antarpublikasi/kondisi.

**Bahasa sederhana:** data holdout relatif dekat dengan data yang dikenal model. Saat optimizer membuat kombinasi baru, kedekatan tersebut belum tentu tetap berlaku. Semua nilai individual yang terlihat wajar belum menjamin kombinasinya wajar.

## 11. Worksheet optimasi dan keputusan yang belum selesai

Sumber: `optimization_parameter_worksheet.csv`, `total_threshold_evidence.csv`, `manual_parameter_evidence.csv`, `chosen_bounds_intersection.csv`.

Worksheet menyediakan bukti distribusi, bukan parameter final. Pada run ini, pilihan bounds data/literatur belum lengkap sehingga intersection masih pending. Batas/bobot lama ditampilkan untuk ditinjau, bukan otomatis disetujui.

Keputusan tersisa mencakup basis satuan nutrient; domain N/P/K dan kebijakan zero-dose; batas total dan komposisi; baseline optimasi; yield scale; bobot trade-off; serta kebijakan peringatan domain dan missingness. Bila objective dinormalisasi, improvement/yield_scale harus dibedakan dari improvement mentah kg/ha. Bobot penalty tidak boleh dipilih seolah semua komponen bersatuan kg/ha.

Ada 34 training rows dengan total NPK nol. Proporsinya tidak terdefinisi secara matematis dan dikeluarkan dari statistik komposisi, walaupun representasi engineering menyimpan nol. Ini perbedaan antara konvensi input model dan definisi analisis, bukan alasan menghapus kontrol.

## 12. Rekomendasi dan prioritas

1. **Pertahankan model beku sebagai baseline D.** Respons dosis sudah teramati; tidak ada kebutuhan otomatis untuk retuning atau menghapus data berdasarkan hasil ini.
2. **Tinjau risiko domain sebelum DE.** Gunakan jarak kontinu, kombinasi unik, pelanggaran rasio, dan state/missingness bersama. Jangan menjadikan cutoff reference berduplikasi sebagai aturan otomatis. Kebijakan final tetap keputusan peneliti.
3. **Lengkapi worksheet dengan literatur dan keputusan eksplisit.** Bounds, baseline, yield scale dan trade-off belum dapat disimpulkan hanya dari quantile atau skor prediksi.
4. **Tinjau metadata perlakuan bila membutuhkan klaim ranking terkontrol.** Penambahan konteks read-only dan keputusan agregasi replikasi harus menjaga lineage/split; tidak membuat pasangan atau membuang baris diam-diam. Jika data tetap tidak memadai, nyatakan keterbatasan, bukan memaksakan validasi.
5. **Setelah keputusan parameter selesai, lanjutkan eksperimen DE terbatas dan dapat diaudit.** Kelak laporkan predicted improvement, dosis, penalty, boundary/domain/missingness warnings, serta stabilitas antarseed. Ini usulan tahap berikutnya; D sendiri belum menghasilkan solusi DE.

Kesimpulan teknisnya adalah **proceed with cautions untuk persiapan optimasi**, dengan kesiapan inference dan responsiveness yang terbukti pada skenario uji, tetapi validasi respons agronomis, cakupan kombinasi, dan keputusan objective masih terbatas. Kesimpulan sederhananya: **model sudah bisa ditanya “bagaimana jika dosis diubah?”, tetapi jawaban “dosis mana yang sebaiknya dipakai?” memerlukan batas, tujuan, dan pemeriksaan tambahan yang disepakati.**

## 13. Penelusuran bukti

Semua CSV/JSON/grafik yang disebut di atas berada dalam run utama pada bagian 1. Dokumen terkait:

- [Rancangan dan batas lingkup D](EXPERIMENT_D_PLAN.md).
- [Penggunaan dan hasil tuning C](EXPERIMENT_C_USAGE.md).
- Notebook: `notebooks/2_model_experiment/experiment_D.ipynb`.
- Implementasi: `src/rice_fertilizer/evaluation/surrogate_diagnostics.py`.

Laporan disusun dari tabel numerik, manifest, dan pemeriksaan implementasi; tidak mengklaim uji lapangan, uji signifikansi, atau eksperimen optimasi yang belum dilakukan. Run yang dirujuk secara eksplisit diperlukan karena eksekusi notebook berikutnya dapat menghasilkan direktori dan hasil berbeda.
