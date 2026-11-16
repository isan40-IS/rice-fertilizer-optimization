# Validation Status

## Yang telah diperiksa

- Seluruh modul Python berhasil dikompilasi tanpa syntax error.
- `pytest` berhasil menjalankan 19 unit dan integration tests.
- Smoke test end-to-end menggunakan data sintetis sementara berhasil melewati:
  1. pembuatan folder runtime,
  2. preprocessing dan rekonstruksi structural zero,
  3. feature engineering,
  4. GroupKFold training,
  5. pemilihan surrogate model,
  6. Differential Evolution,
  7. evaluasi produktivitas dan efisiensi,
  8. sensitivity analysis,
  9. input dan algorithm stability,
  10. ekspor CSV, grafik, dan Excel.

Data sintetis smoke test tidak disertakan dalam proyek. Folder `data/` tetap kosong selain `.gitkeep`.

## Yang belum dapat divalidasi pada saat generate

- Hasil terhadap CSV asli pengguna.
- Daftar fitur final.
- Arti seluruh label perlakuan yang ambigu.
- Batas agronomis NPK dan total dosis.
- Rentang proporsi NPK.
- Bobot penalti.
- Pseudocount fitur rasio.
- Nilai perturbasi.
- Validasi pakar dan validasi lapangan.

Bagian tersebut sengaja dikendalikan melalui `configs/decisions.yaml` dan tidak diisi secara sepihak oleh kode.
