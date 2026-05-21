# Panduan Deploy Streamlit Online

Project ini sudah disiapkan untuk deploy ke Streamlit Community Cloud.

## File yang wajib ada di GitHub

- `app.py`
- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`
- `models/xgboost_4kategori.joblib`
- `models/lightgbm_4kategori.joblib`
- `outputs/model_4kategori/dataset_harian_4kategori.csv`
- `outputs/model_4kategori/evaluasi_model_4kategori.csv`
- File lain di `outputs/model_4kategori` yang masih dipakai aplikasi

Folder seperti `myenv`, `.ml_deps`, `__pycache__`, `.matplotlib_cache`, log, dan backup lokal tidak perlu diupload.

## Langkah Deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Upload isi folder project ini ke repository tersebut.
3. Buka https://share.streamlit.io.
4. Login memakai akun GitHub.
5. Klik `Create app`.
6. Pilih `Yup, I have an app`.
7. Isi:
   - Repository: repository GitHub project ini
   - Branch: `main`
   - Main file path: `app.py`
8. Buka `Advanced settings`.
9. Pilih Python version `3.12`.
10. Klik `Deploy`.

Setelah deploy selesai, aplikasi akan mendapat link online dengan domain `streamlit.app`.

## Jika Deploy Gagal

- Pastikan `requirements.txt` ada di root repository, sejajar dengan `app.py`.
- Pastikan folder `models` dan `outputs/model_4kategori` ikut terupload.
- Jika error `ModuleNotFoundError`, tambahkan package yang hilang ke `requirements.txt`.
- Jika error model tidak ditemukan, cek kembali nama file model dan struktur folder di GitHub.
- Jika error Python version, deploy ulang dan pilih Python `3.12` di `Advanced settings`.

## Jalankan Lokal

```powershell
cd "C:\Skripsi\FINAL\New"
.\myenv\Scripts\streamlit.exe run app.py
```
