"""
train_model.py
Jalankan script ini SEKALI sebelum deploy untuk membuat file model:
  - model_rf.pkl
  - scaler.pkl
  - le_wilayah.pkl
  - wilayah_stats.json
  - wilayah_defaults.json

Cara pakai:
  python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

print("=== TRAINING MODEL PREDIKSI PRODUKSI PADI ===\n")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("[1/5] Memuat dataset...")
df = pd.read_csv("dataset_terintegrasi.csv")
print(f"      ✓ Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"      ✓ Missing values: {df.isnull().sum().sum()}")
print(f"      ✓ Wilayah: {df['wilayah'].nunique()} kabupaten")
print(f"      ✓ Periode: {df['tahun'].min()} – {df['tahun'].max()}\n")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("[2/5] Preprocessing...")

le_wilayah = LabelEncoder()
le_stasiun = LabelEncoder()
df['wilayah_enc'] = le_wilayah.fit_transform(df['wilayah'])
df['stasiun_enc'] = le_stasiun.fit_transform(df['stasiun'])

features = [
    'tahun', 'wilayah_enc', 'curah_hujan_mm',
    'suhu_rata_celsius', 'suhu_max_celsius', 'suhu_min_celsius',
    'kelembapan_pct', 'penyinaran_jam', 'kecepatan_angin_ms'
]

X = df[features]
y = df['produksi_ton']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"      ✓ Train: {len(X_train)} baris | Test: {len(X_test)} baris")
print(f"      ✓ Fitur: {len(features)}\n")

# ── 3. Training ───────────────────────────────────────────────────────────────
print("[3/5] Melatih model Random Forest (n_estimators=200)...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)
print("      ✓ Training selesai!\n")

# ── 4. Evaluasi ───────────────────────────────────────────────────────────────
print("[4/5] Evaluasi model...")
y_pred = model.predict(X_test_sc)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"      ✓ MAE  : {mae:,.2f} ton")
print(f"      ✓ RMSE : {rmse:,.2f} ton")
print(f"      ✓ R²   : {r2:.4f}\n")

# Feature importance
fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("      Feature Importance:")
for feat, imp in fi.items():
    bar = "█" * int(imp * 40)
    print(f"      {feat:<22} {imp:.4f} {bar}")
print()

# ── 5. Simpan Artefak ─────────────────────────────────────────────────────────
print("[5/5] Menyimpan model dan artefak...")

joblib.dump(model,      "model_rf.pkl")
joblib.dump(scaler,     "scaler.pkl")
joblib.dump(le_wilayah, "le_wilayah.pkl")

# Stats per wilayah
stats = df.groupby('wilayah').agg(
    produksi_mean=('produksi_ton', 'mean'),
    produksi_min=('produksi_ton', 'min'),
    produksi_max=('produksi_ton', 'max'),
    curah_mean=('curah_hujan_mm', 'mean'),
    suhu_mean=('suhu_rata_celsius', 'mean')
).reset_index()

defaults = df.groupby('wilayah').agg(
    curah_hujan_mm=('curah_hujan_mm', 'mean'),
    suhu_rata_celsius=('suhu_rata_celsius', 'mean'),
    suhu_max_celsius=('suhu_max_celsius', 'mean'),
    suhu_min_celsius=('suhu_min_celsius', 'mean'),
    kelembapan_pct=('kelembapan_pct', 'mean'),
    penyinaran_jam=('penyinaran_jam', 'mean'),
    kecepatan_angin_ms=('kecepatan_angin_ms', 'mean')
).reset_index()

with open("wilayah_stats.json", "w") as f:
    json.dump(stats.set_index('wilayah').to_dict(orient='index'), f, indent=2)

with open("wilayah_defaults.json", "w") as f:
    json.dump(defaults.set_index('wilayah').to_dict(orient='index'), f, indent=2)

print("      ✓ model_rf.pkl")
print("      ✓ scaler.pkl")
print("      ✓ le_wilayah.pkl")
print("      ✓ wilayah_stats.json")
print("      ✓ wilayah_defaults.json")
print("\n=== SELESAI! Jalankan: streamlit run app.py ===")
