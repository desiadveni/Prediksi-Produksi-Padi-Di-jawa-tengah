import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PadiSight — Prediksi Produksi Padi Jawa Tengah",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model_rf.pkl")
    scaler   = joblib.load("scaler.pkl")
    le_wil   = joblib.load("le_wilayah.pkl")
    return model, scaler, le_wil

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_terintegrasi.csv")
    with open("wilayah_stats.json") as f:
        stats = json.load(f)
    with open("wilayah_defaults.json") as f:
        defaults = json.load(f)
    return df, stats, defaults

model, scaler, le_wil = load_model()
df, stats, defaults   = load_data()
WILAYAH_LIST          = le_wil.classes_.tolist()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #F8FAF5; }
  [data-testid="stSidebar"]          { background: #1A3A2A; }
  [data-testid="stSidebar"] * { color: #D4EAD0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #9FD4A0 !important; font-size:13px !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #E8F5E4 !important; }

  /* Hero */
  .hero { background: linear-gradient(135deg,#1A6B3A 0%,#2D9B5A 60%,#4CAF70 100%);
          border-radius: 16px; padding: 32px 36px; margin-bottom: 24px; color: white; }
  .hero h1 { margin:0; font-size:28px; font-weight:700; letter-spacing:-0.5px; }
  .hero p  { margin:6px 0 0; opacity:.85; font-size:15px; }

  /* Metric cards */
  .metric-row { display:flex; gap:14px; margin-bottom:20px; flex-wrap:wrap; }
  .metric-card { background:white; border-radius:12px; padding:18px 22px;
                  flex:1; min-width:140px; box-shadow:0 1px 4px rgba(0,0,0,.07); }
  .metric-label { font-size:12px; color:#7B9E87; font-weight:500; text-transform:uppercase; letter-spacing:.5px; }
  .metric-value { font-size:26px; font-weight:700; color:#1A3A2A; margin:4px 0 0; }
  .metric-unit  { font-size:12px; color:#7B9E87; }

  /* Prediction result */
  .pred-box { background: linear-gradient(135deg,#1A6B3A,#2D9B5A);
               border-radius:14px; padding:28px 32px; text-align:center;
               color:white; margin:20px 0; }
  .pred-label { font-size:14px; opacity:.8; margin-bottom:6px; }
  .pred-value { font-size:48px; font-weight:800; letter-spacing:-1px; }
  .pred-unit  { font-size:16px; opacity:.75; margin-top:4px; }
  .pred-range { font-size:13px; opacity:.7; margin-top:10px; }

  /* Section title */
  .sec-title { font-size:17px; font-weight:700; color:#1A3A2A;
               border-left:4px solid #2D9B5A; padding-left:12px;
               margin:24px 0 14px; }

  /* Info pill */
  .pill { display:inline-block; background:#E8F5E4; color:#1A6B3A;
          border-radius:99px; padding:3px 12px; font-size:12px; font-weight:600; }

  /* Warning */
  .warn-box { background:#FFFBEB; border:1px solid #FCD34D; border-radius:10px;
               padding:12px 16px; font-size:13px; color:#92400E; margin:10px 0; }

  div[data-testid="stButton"] > button {
    background: #2D9B5A; color: white; border: none;
    border-radius: 10px; padding: 12px 32px; font-size:16px; font-weight:600;
    width: 100%; cursor: pointer; transition: background .2s;
  }
  div[data-testid="stButton"] > button:hover { background: #1A6B3A; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 PadiSight")
    st.markdown("Prediksi Produksi Padi\nJawa Tengah")
    st.markdown("---")
    st.markdown("### Pilih Kabupaten & Tahun")

    wilayah_sel = st.selectbox("Kabupaten / Kota", WILAYAH_LIST, index=0)
    tahun_sel   = st.slider("Tahun Prediksi", 2024, 2035, 2025)

    def_vals = defaults.get(wilayah_sel, {})

    st.markdown("---")
    st.markdown("### Data Iklim")
    st.caption("Nilai default = rata-rata historis wilayah")

    curah   = st.number_input("💧 Curah Hujan (mm)", 1000.0, 4000.0,
                               float(round(def_vals.get('curah_hujan_mm', 2300), 1)), step=10.0)
    suhu_r  = st.number_input("🌡 Suhu Rata-rata (°C)", 22.0, 30.0,
                               float(round(def_vals.get('suhu_rata_celsius', 26.0), 1)), step=0.1)
    suhu_x  = st.number_input("🔺 Suhu Maks (°C)", 28.0, 36.0,
                               float(round(def_vals.get('suhu_max_celsius', 32.5), 1)), step=0.1)
    suhu_n  = st.number_input("🔻 Suhu Min (°C)", 16.0, 24.0,
                               float(round(def_vals.get('suhu_min_celsius', 20.0), 1)), step=0.1)
    lembap  = st.number_input("💦 Kelembapan (%)", 60.0, 95.0,
                               float(round(def_vals.get('kelembapan_pct', 79.0), 1)), step=0.5)
    sinar   = st.number_input("☀️ Penyinaran (jam/hari)", 4.0, 9.0,
                               float(round(def_vals.get('penyinaran_jam', 6.6), 2)), step=0.1)
    angin   = st.number_input("💨 Kecepatan Angin (m/s)", 1.0, 5.0,
                               float(round(def_vals.get('kecepatan_angin_ms', 3.0), 2)), step=0.1)

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Sekarang")

    st.markdown("---")
    st.caption("Model: Random Forest (200 pohon)\nR² = 0.686 | Dataset: 1990–2025")

# ── MAIN AREA ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌾 PadiSight — Prediksi Produksi Padi Jawa Tengah</h1>
  <p>Model Machine Learning berbasis Random Forest · 30 Kabupaten · Data 1990–2025</p>
</div>
""", unsafe_allow_html=True)

# ── Overview metrics ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Data", "1.080 baris", "36 tahun historis")
with col2:
    st.metric("Kabupaten", "30 wilayah", "Jawa Tengah")
with col3:
    st.metric("Akurasi Model (R²)", "0.686", "Random Forest")
with col4:
    st.metric("MAE Model", "±67.563 ton", "Mean Absolute Error")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_model, tab_info = st.tabs(
    ["🔮 Prediksi", "📊 Eksplorasi Data", "🤖 Performa Model", "ℹ️ Tentang"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f'<div class="sec-title">Input Prediksi</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table style="width:100%;font-size:14px;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Kabupaten</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{wilayah_sel}</td>
          </tr>
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Tahun Prediksi</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{tahun_sel}</td>
          </tr>
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Curah Hujan</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{curah:,.1f} mm</td>
          </tr>
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Suhu Rata-rata</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{suhu_r:.1f} °C</td>
          </tr>
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Kelembapan</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{lembap:.1f} %</td>
          </tr>
          <tr style="border-bottom:1px solid #E8F5E4;">
            <td style="padding:8px 0;color:#7B9E87;">Penyinaran</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{sinar:.2f} jam</td>
          </tr>
          <tr>
            <td style="padding:8px 0;color:#7B9E87;">Kecepatan Angin</td>
            <td style="padding:8px 0;font-weight:600;color:#1A3A2A;">{angin:.2f} m/s</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-title">Hasil Prediksi</div>', unsafe_allow_html=True)

        # ── Run prediction ───────────────────────────────────────────────────
        wil_enc = le_wil.transform([wilayah_sel])[0]
        X_input = np.array([[tahun_sel, wil_enc, curah, suhu_r,
                              suhu_x, suhu_n, lembap, sinar, angin]])
        X_sc    = scaler.transform(X_input)

        # Predict with all trees for confidence interval
        preds_all = np.array([tree.predict(X_sc) for tree in model.estimators_])
        pred_mean = preds_all.mean()
        pred_std  = preds_all.std()
        pred_low  = max(0, pred_mean - 1.96 * pred_std)
        pred_high = pred_mean + 1.96 * pred_std

        # Historical context
        hist = df[df['wilayah'] == wilayah_sel]['produksi_ton']
        hist_mean = hist.mean()
        delta_pct = ((pred_mean - hist_mean) / hist_mean) * 100

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Estimasi Produksi Padi — {wilayah_sel} ({tahun_sel})</div>
          <div class="pred-value">{pred_mean:,.0f}</div>
          <div class="pred-unit">ton / tahun</div>
          <div class="pred-range">Rentang 95%: {pred_low:,.0f} – {pred_high:,.0f} ton</div>
        </div>
        """, unsafe_allow_html=True)

        # Delta vs historical
        arrow = "▲" if delta_pct > 0 else "▼"
        color = "#16A34A" if delta_pct > 0 else "#DC2626"
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,.07);">
          <div style="font-size:12px;color:#7B9E87;font-weight:600;margin-bottom:8px;">PERBANDINGAN HISTORIS</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#555;font-size:14px;">Rata-rata historis</span>
            <span style="font-weight:700;font-size:14px;color:#1A3A2A;">{hist_mean:,.0f} ton</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#555;font-size:14px;">Prediksi {tahun_sel}</span>
            <span style="font-weight:700;font-size:14px;color:#1A3A2A;">{pred_mean:,.0f} ton</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:#555;font-size:14px;">Selisih</span>
            <span style="font-weight:700;font-size:14px;color:{color};">{arrow} {abs(delta_pct):.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Trend chart per wilayah ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">Tren Produksi Historis</div>', unsafe_allow_html=True)
    df_wil = df[df['wilayah'] == wilayah_sel].sort_values('tahun')
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_wil['tahun'], y=df_wil['produksi_ton'],
        mode='lines+markers', name='Produksi Aktual',
        line=dict(color='#2D9B5A', width=2.5),
        marker=dict(size=5, color='#1A6B3A'),
        hovertemplate='%{x}: <b>%{y:,.0f} ton</b><extra></extra>'
    ))
    fig_trend.add_hline(y=pred_mean, line_dash="dash", line_color="#F59E0B",
                        annotation_text=f"Prediksi {tahun_sel}: {pred_mean:,.0f} ton",
                        annotation_position="bottom right")
    fig_trend.update_layout(
        height=320, margin=dict(t=10,b=10,l=10,r=10),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False, title='Tahun'),
        yaxis=dict(showgrid=True, gridcolor='#F0F0F0', title='Produksi (ton)'),
        font=dict(family='sans-serif', size=13),
        showlegend=False
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-title">Produksi Rata-rata per Kabupaten</div>', unsafe_allow_html=True)
    prod_by_wil = df.groupby('wilayah')['produksi_ton'].mean().sort_values(ascending=True).reset_index()
    fig_bar = go.Figure(go.Bar(
        x=prod_by_wil['produksi_ton'], y=prod_by_wil['wilayah'],
        orientation='h',
        marker=dict(
            color=prod_by_wil['produksi_ton'],
            colorscale=[[0,'#C8E6C9'],[0.5,'#2D9B5A'],[1,'#1A3A2A']]
        ),
        hovertemplate='<b>%{y}</b><br>%{x:,.0f} ton<extra></extra>'
    ))
    fig_bar.update_layout(
        height=750, margin=dict(t=10,b=10,l=10,r=10),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#F5F5F5', title='Rata-rata Produksi (ton)'),
        yaxis=dict(showgrid=False),
        font=dict(size=12)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-title">Distribusi Produksi</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(df, x='produksi_ton', nbins=40, color_discrete_sequence=['#2D9B5A'])
        fig_hist.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                                plot_bgcolor='white', paper_bgcolor='white',
                                xaxis_title='Produksi (ton)', yaxis_title='Frekuensi')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-title">Tren Produksi Jawa Tengah</div>', unsafe_allow_html=True)
        trend_jateng = df.groupby('tahun')['produksi_ton'].mean().reset_index()
        fig_line = px.line(trend_jateng, x='tahun', y='produksi_ton', color_discrete_sequence=['#2D9B5A'])
        fig_line.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                                 plot_bgcolor='white', paper_bgcolor='white',
                                 xaxis_title='Tahun', yaxis_title='Produksi Rata-rata (ton)')
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="sec-title">Korelasi Antar Variabel</div>', unsafe_allow_html=True)
    num_cols = ['produksi_ton','curah_hujan_mm','suhu_rata_celsius','suhu_max_celsius',
                'suhu_min_celsius','kelembapan_pct','penyinaran_jam','kecepatan_angin_ms']
    corr_mat = df[num_cols].corr()
    fig_corr = px.imshow(corr_mat, text_auto='.2f', aspect='auto',
                          color_continuous_scale='RdYlGn', zmin=-1, zmax=1)
    fig_corr.update_layout(height=450, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="sec-title">Perbandingan Performa Model</div>', unsafe_allow_html=True)

    perf_data = {
        'Model': ['Linear Regression', 'Random Forest ★', 'Gradient Boosting'],
        'MAE (ton)': [131285, 67563, 74415],
        'RMSE (ton)': [156049, 87941, 93378],
        'R² Score': [0.012, 0.686, 0.646]
    }
    perf_df = pd.DataFrame(perf_data)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_r2 = go.Figure(go.Bar(
            x=perf_df['Model'], y=perf_df['R² Score'],
            marker_color=['#94A3B8','#2D9B5A','#60A5FA'],
            text=[f"{v:.3f}" for v in perf_df['R² Score']], textposition='outside'
        ))
        fig_r2.update_layout(title='R² Score (lebih tinggi = lebih baik)',
                              height=320, margin=dict(t=40,b=10,l=10,r=10),
                              plot_bgcolor='white', paper_bgcolor='white',
                              yaxis=dict(range=[0,1], showgrid=True, gridcolor='#F0F0F0'),
                              xaxis=dict(showgrid=False))
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_p2:
        fig_mae = go.Figure(go.Bar(
            x=perf_df['Model'], y=perf_df['MAE (ton)'],
            marker_color=['#94A3B8','#2D9B5A','#60A5FA'],
            text=[f"{v:,.0f}" for v in perf_df['MAE (ton)']], textposition='outside'
        ))
        fig_mae.update_layout(title='MAE (lebih rendah = lebih baik)',
                               height=320, margin=dict(t=40,b=10,l=10,r=10),
                               plot_bgcolor='white', paper_bgcolor='white',
                               yaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                               xaxis=dict(showgrid=False))
        st.plotly_chart(fig_mae, use_container_width=True)

    st.dataframe(perf_df.style.highlight_max(subset=['R² Score'], color='#D1FAE5')
                                .highlight_min(subset=['MAE (ton)','RMSE (ton)'], color='#D1FAE5'),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fi_data = {
        'Fitur': ['wilayah','tahun','suhu_max','suhu_rata','suhu_min',
                  'kecepatan_angin','curah_hujan','kelembapan','penyinaran'],
        'Importance': [0.7358, 0.0612, 0.0317, 0.0308, 0.0307, 0.0304, 0.0293, 0.0265, 0.0236]
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance')
    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Fitur'], orientation='h',
        marker=dict(color=fi_df['Importance'],
                    colorscale=[[0,'#C8E6C9'],[1,'#1A3A2A']]),
        text=[f"{v*100:.1f}%" for v in fi_df['Importance']], textposition='outside'
    ))
    fig_fi.update_layout(height=380, margin=dict(t=10,b=10,l=10,r=80),
                          plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='#F0F0F0', title='Importance Score'),
                          yaxis=dict(showgrid=False))
    st.plotly_chart(fig_fi, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INFO
# ════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown('<div class="sec-title">Tentang PadiSight</div>', unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
**🎯 Tujuan Aplikasi**

PadiSight adalah sistem prediksi produksi padi berbasis machine learning untuk mendukung perencanaan ketahanan pangan di Jawa Tengah. Pengguna dapat memasukkan kondisi iklim dan mendapatkan estimasi produksi padi per kabupaten.

**📊 Dataset**
- Sumber: Data terintegrasi produksi & iklim Jawa Tengah
- Periode: 1990 – 2025 (36 tahun)
- Cakupan: 30 Kabupaten/Kota
- Total rekord: 1.080 baris
- Fitur: 9 variabel iklim + lokasi

**🤖 Algoritma**

Random Forest Regressor dengan 200 pohon keputusan dipilih karena:
- Mampu menangkap hubungan nonlinear
- Robust terhadap outlier
- Memberikan feature importance
- Performa terbaik di antara model yang diuji
        """)
    with col_i2:
        st.markdown("""
**📐 Metodologi (CRISP-DM)**

1. Business Understanding — identifikasi kebutuhan prediksi
2. Data Understanding — eksplorasi 1.080 baris data
3. Data Preparation — encoding, scaling, split 80:20
4. Modeling — uji 3 algoritma regresi
5. Evaluation — MAE, RMSE, R² Score
6. Deployment — aplikasi web Streamlit

**⚠️ Catatan Penting**

- Prediksi bersifat estimasi berbasis pola historis
- Faktor ekstrem (bencana, kebijakan mendadak) tidak dimodelkan
- Gunakan sebagai referensi perencanaan, bukan keputusan tunggal
- Model dievaluasi dengan R² = 0.686 (68.6% variansi dijelaskan)

**📬 Kontak**

Proyek Data Mining — Universitas / Institusi Anda
        """)

    st.markdown('<div class="sec-title">Statistik Dataset per Wilayah</div>', unsafe_allow_html=True)
    stats_df = df.groupby('wilayah').agg(
        Produksi_Rata=('produksi_ton','mean'),
        Produksi_Min=('produksi_ton','min'),
        Produksi_Max=('produksi_ton','max'),
        CurahHujan_Rata=('curah_hujan_mm','mean'),
        Suhu_Rata=('suhu_rata_celsius','mean')
    ).round(1).reset_index()
    stats_df.columns = ['Kabupaten','Prod. Rata-rata (ton)','Prod. Min','Prod. Max','Curah Hujan (mm)','Suhu Rata (°C)']
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
