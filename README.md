# 🌾 PadiSight — Prediksi Produksi Padi Jawa Tengah

Aplikasi web prediksi produksi padi berbasis Machine Learning (Random Forest)
untuk 30 Kabupaten di Jawa Tengah, periode data 1990–2025.

---

## 📁 Struktur Folder

```
prediksi-padi/
├── app.py                    ← Aplikasi Streamlit utama
├── train_model.py            ← Script training model
├── requirements.txt          ← Dependensi Python
├── dataset_terintegrasi.csv  ← Dataset (LETAKKAN DI SINI)
├── model_rf.pkl              ← Model (dibuat otomatis)
├── scaler.pkl                ← Scaler (dibuat otomatis)
├── le_wilayah.pkl            ← Label encoder (dibuat otomatis)
├── wilayah_stats.json        ← Statistik wilayah (dibuat otomatis)
├── wilayah_defaults.json     ← Default nilai iklim (dibuat otomatis)
└── README.md
```

---

## 🚀 Cara Deploy (Langkah demi Langkah)

### LANGKAH 1 — Persiapan Lokal

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Mac/Linux)
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### LANGKAH 2 — Letakkan Dataset

Salin file `dataset_terintegrasi.csv` ke dalam folder `prediksi-padi/`.

### LANGKAH 3 — Training Model

```bash
python train_model.py
```

Ini akan membuat file: `model_rf.pkl`, `scaler.pkl`, `le_wilayah.pkl`,
`wilayah_stats.json`, `wilayah_defaults.json`.

### LANGKAH 4 — Jalankan Lokal (Uji Coba)

```bash
streamlit run app.py
```

Buka browser di: `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Community Cloud (GRATIS)

### Langkah 1 — Upload ke GitHub

1. Buat repository baru di github.com
2. Upload **semua file** dalam folder `prediksi-padi/` ke repo tersebut
   (termasuk .pkl, .json, .csv, app.py, requirements.txt)

```bash
git init
git add .
git commit -m "Initial commit PadiSight"
git branch -M main
git remote add origin https://github.com/USERNAME/prediksi-padi.git
git push -u origin main
```

### Langkah 2 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **"New app"**
4. Isi:
   - Repository: `USERNAME/prediksi-padi`
   - Branch: `main`
   - Main file path: `app.py`
5. Klik **"Deploy!"**
6. Tunggu 2–3 menit → aplikasi online!

URL akan seperti: `https://prediksi-padi-USERNAME.streamlit.app`

---

## 🐳 Deploy dengan Docker (Opsional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t padisight .
docker run -p 8501:8501 padisight
```

---

## 📊 Informasi Model

| Parameter         | Nilai                      |
|-------------------|---------------------------|
| Algoritma         | Random Forest Regressor   |
| n_estimators      | 200 pohon                 |
| Train/Test split  | 80% / 20%                 |
| MAE               | ±67.563 ton               |
| RMSE              | ±87.941 ton               |
| R² Score          | 0.686                     |
| Fitur terpenting  | Wilayah (73.6%)           |

---

## 📦 Dependensi

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.18.0
```

---

## ⚠️ Catatan

- File `.pkl` dan `.json` HARUS ada sebelum menjalankan `app.py`
- Jika upload ke GitHub, pastikan file `dataset_terintegrasi.csv` ikut diupload
- Untuk Streamlit Cloud: semua file di repo otomatis tersedia
