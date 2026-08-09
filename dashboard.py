import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Keriting Banten",
    page_icon="🌶️",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS KUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .metric-value.red   { color: #e84855; }
    .metric-value.blue  { color: #2e86ab; }
    .metric-value.green { color: #16a34a; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #e84855;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-green  { background: #dcfce7; color: #16a34a; }
    .badge-yellow { background: #fef9c3; color: #ca8a04; }
    .badge-red    { background: #fee2e2; color: #dc2626; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }

    hr.divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────
def interpretasi_mape(mape):
    if mape < 10:
        return "Sangat Baik", "badge-green"
    elif mape < 20:
        return "Baik", "badge-green"
    elif mape < 50:
        return "Cukup", "badge-yellow"
    else:
        return "Kurang Baik", "badge-red"


def warna_mape(mape):
    if mape < 10:
        return "green"
    elif mape < 20:
        return "green"
    elif mape < 50:
        return "red"
    else:
        return "red"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌶️ Prediksi Harga Cabai Merah Keriting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Provinsi Banten • Algoritma ARIMA • Data PIHPS Nasional (hargapangan.id)</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UPLOAD FILE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Upload Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload file Excel dataset harga dari PIHPS Nasional (.xlsx)",
    type=["xlsx"],
    help="File Excel berformat Tabel Harga Berdasarkan Komoditas dari hargapangan.id"
)

if uploaded_file is None:
    st.info("Silakan upload file Excel dataset terlebih dahulu untuk memulai analisis.")
    st.stop()


# ─────────────────────────────────────────────
# BACA & PROSES DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    df_raw   = pd.read_excel(file, header=0)
    banten_row = df_raw[df_raw['Komoditas (Rp)'] == 'Banten'].iloc[0]
    date_cols  = [c for c in df_raw.columns if '/' in str(c)]

    df_harian = pd.DataFrame({
        'tanggal': pd.to_datetime(date_cols, format='%d/ %m/ %Y'),
        'harga'  : banten_row[date_cols].values
    })

    df_harian['harga'] = df_harian['harga'].replace('-', np.nan)
    df_harian['harga'] = df_harian['harga'].astype(str).str.replace(',', '').str.strip()
    df_harian['harga'] = pd.to_numeric(df_harian['harga'], errors='coerce')

    df_bulanan = df_harian.set_index('tanggal').resample('MS').mean()
    df_bulanan['harga'] = df_bulanan['harga'].interpolate(method='linear')

    return df_bulanan

with st.spinner("Memproses data..."):
    df_bulanan = load_and_process(uploaded_file)

st.success(f"✅ Data berhasil dimuat: **{len(df_bulanan)} observasi bulanan** "
           f"({df_bulanan.index[0].strftime('%b %Y')} – {df_bulanan.index[-1].strftime('%b %Y')})")


# ─────────────────────────────────────────────
# TABS NAVIGASI
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Historis",
    "🔬 Pemodelan ARIMA",
    "📈 Evaluasi Model",
    "🔮 Prediksi 2026"
])


# ══════════════════════════════════════════════
# TAB 1 — DATA HISTORIS
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)

    stats = df_bulanan['harga'].describe()

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Rata-rata",       f"Rp {stats['mean']:,.0f}",  "blue"),
        (c2, "Std Deviasi",     f"Rp {stats['std']:,.0f}",   ""),
        (c3, "Minimum",         f"Rp {stats['min']:,.0f}",   "green"),
        (c4, "Maksimum",        f"Rp {stats['max']:,.0f}",   "red"),
        (c5, "Median",          f"Rp {stats['50%']:,.0f}",   "blue"),
    ]
    for col, label, value, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Grafik Harga Historis</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_bulanan.index, df_bulanan['harga'],
            color='#2E86AB', linewidth=2, marker='o', markersize=4)
    ax.fill_between(df_bulanan.index, df_bulanan['harga'],
                    alpha=0.12, color='#2E86AB')
    ax.set_title('Harga Eceran Cabai Merah Keriting di Provinsi Banten\nJanuari 2021 – Desember 2025',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Harga (Rp/kg)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Tabel Data Bulanan</div>', unsafe_allow_html=True)
    df_tampil = df_bulanan.copy()
    df_tampil.index = df_tampil.index.strftime('%B %Y')
    df_tampil.columns = ['Harga (Rp/kg)']
    df_tampil['Harga (Rp/kg)'] = df_tampil['Harga (Rp/kg)'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(df_tampil, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — PEMODELAN ARIMA
# ══════════════════════════════════════════════
with tab2:

    split_idx = int(len(df_bulanan) * 0.8)
    train = df_bulanan.iloc[:split_idx]
    test  = df_bulanan.iloc[split_idx:]

    st.markdown('<div class="section-header">Pembagian Data Training & Testing</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Training (80%)</div>
            <div class="metric-value blue">{len(train)} bulan</div>
            <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">
                {train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Testing (20%)</div>
            <div class="metric-value red">{len(test)} bulan</div>
            <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">
                {test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Uji Stasionaritas (Augmented Dickey-Fuller)</div>', unsafe_allow_html=True)

    result_asli = adfuller(train['harga'].dropna())
    p_asli      = result_asli[1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ADF Statistic (Data Asli)</div>
            <div class="metric-value">{result_asli[0]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">p-value (Data Asli)</div>
            <div class="metric-value {'red' if p_asli > 0.05 else 'green'}">{p_asli:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        status = "❌ Tidak Stasioner → perlu differencing" if p_asli > 0.05 else "✅ Stasioner"
        warna  = "red" if p_asli > 0.05 else "green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kesimpulan</div>
            <div class="metric-value {warna}" style="font-size:1rem">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    if p_asli > 0.05:
        result_diff = adfuller(train['harga'].diff().dropna())
        p_diff      = result_diff[1]
        st.markdown("**Hasil Differencing Orde 1:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ADF Statistic (Diff 1)</div>
                <div class="metric-value">{result_diff[0]:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">p-value (Diff 1)</div>
                <div class="metric-value {'red' if p_diff > 0.05 else 'green'}">{p_diff:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            status2 = "✅ Stasioner setelah Differencing" if p_diff <= 0.05 else "❌ Masih tidak stasioner"
            warna2  = "green" if p_diff <= 0.05 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Kesimpulan</div>
                <div class="metric-value {warna2}" style="font-size:1rem">{status2}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Plot ACF & PACF</div>', unsafe_allow_html=True)

    data_plot = train['harga'].diff().dropna() if p_asli > 0.05 else train['harga']
    label_acf = "(Setelah Differencing)" if p_asli > 0.05 else "(Data Asli)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(data_plot,  lags=20, ax=axes[0], alpha=0.05)
    plot_pacf(data_plot, lags=20, ax=axes[1], alpha=0.05)
    axes[0].set_title(f'ACF {label_acf}',  fontweight='bold')
    axes[1].set_title(f'PACF {label_acf}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Pemilihan Model ARIMA Terbaik (Auto ARIMA)</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Mencari model ARIMA terbaik berdasarkan AIC..."):
        model_auto = auto_arima(
            train['harga'],
            seasonal=False,
            information_criterion='aic',
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True,
            max_p=5, max_q=5, max_d=2
        )

    p, d, q = model_auto.order

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Terbaik</div>
            <div class="metric-value blue">ARIMA({p},{d},{q})</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AIC</div>
            <div class="metric-value">{model_auto.aic():.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BIC</div>
            <div class="metric-value">{model_auto.bic():.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Diagnostik Residual</div>', unsafe_allow_html=True)
    model_fit = ARIMA(train['harga'], order=(p, d, q)).fit()
    fig_diag  = model_fit.plot_diagnostics(figsize=(14, 7))
    plt.suptitle(f'Diagnostik Residual ARIMA({p},{d},{q})', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig_diag)
    plt.close()


# ══════════════════════════════════════════════
# TAB 3 — EVALUASI MODEL
# ══════════════════════════════════════════════
with tab3:

    split_idx  = int(len(df_bulanan) * 0.8)
    train      = df_bulanan.iloc[:split_idx]
    test       = df_bulanan.iloc[split_idx:]

    model_auto = auto_arima(
        train['harga'], seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True,
        max_p=5, max_q=5, max_d=2
    )
    p, d, q   = model_auto.order
    model_fit = ARIMA(train['harga'], order=(p, d, q)).fit()

    pred_test       = model_fit.forecast(steps=len(test))
    pred_test.index = test.index

    aktual   = test['harga'].values
    prediksi = pred_test.values

    rmse = np.sqrt(mean_squared_error(aktual, prediksi))
    mae  = mean_absolute_error(aktual, prediksi)
    mape = np.mean(np.abs((aktual - prediksi) / aktual)) * 100

    interp, badge_class = interpretasi_mape(mape)
    warna_m             = warna_mape(mape)

    st.markdown('<div class="section-header">Metrik Evaluasi Akurasi</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">Rp {rmse:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">Rp {mae:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAPE</div>
            <div class="metric-value {warna_m}">{mape:.2f}%</div>
            <span class="badge {badge_class}">{interp}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Grafik Aktual vs Prediksi</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train.index, train['harga'],
            label='Data Training', color='#2E86AB', linewidth=2)
    ax.plot(test.index, test['harga'],
            label='Data Testing (Aktual)', color='#333333', linewidth=2)
    ax.plot(pred_test.index, pred_test,
            label=f'Prediksi ARIMA({p},{d},{q})', color='#E84855',
            linewidth=2, linestyle='--')
    ax.set_title(f'Aktual vs Prediksi ARIMA({p},{d},{q}) | MAPE: {mape:.2f}%',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Harga (Rp/kg)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Tabel Perbandingan Aktual vs Prediksi</div>', unsafe_allow_html=True)
    df_eval = pd.DataFrame({
        'Bulan'         : test.index.strftime('%B %Y'),
        'Aktual (Rp/kg)': [f"Rp {v:,.0f}" for v in aktual],
        'Prediksi (Rp/kg)': [f"Rp {v:,.0f}" for v in prediksi],
        'Selisih (Rp)'  : [f"Rp {abs(a-p):,.0f}" for a, p in zip(aktual, prediksi)]
    })
    st.dataframe(df_eval, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — PREDIKSI 2026
# ══════════════════════════════════════════════
with tab4:

    split_idx  = int(len(df_bulanan) * 0.8)
    train      = df_bulanan.iloc[:split_idx]

    model_auto = auto_arima(
        train['harga'], seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True,
        max_p=5, max_q=5, max_d=2
    )
    p, d, q    = model_auto.order
    model_full = ARIMA(df_bulanan['harga'], order=(p, d, q)).fit()

    forecast  = model_full.get_forecast(steps=12)
    pred_mean = forecast.predicted_mean
    pred_ci   = forecast.conf_int(alpha=0.05)

    bulan_2026 = pd.date_range(start='2026-01-01', periods=12, freq='MS')
    pred_mean.index = bulan_2026
    pred_ci.index   = bulan_2026

    nama_bulan = ['Januari','Februari','Maret','April','Mei','Juni',
                  'Juli','Agustus','September','Oktober','November','Desember']

    st.markdown('<div class="section-header">Ringkasan Prediksi 2026</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Prediksi</div>
            <div class="metric-value blue">Rp {pred_mean.mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        idx_max = pred_mean.values.argmax()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Harga Tertinggi</div>
            <div class="metric-value red">Rp {pred_mean.max():,.0f}</div>
            <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">{nama_bulan[idx_max]}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        idx_min = pred_mean.values.argmin()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Harga Terendah</div>
            <div class="metric-value green">Rp {pred_mean.min():,.0f}</div>
            <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">{nama_bulan[idx_min]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Grafik Prediksi 2026</div>', unsafe_allow_html=True)

    historis_terakhir = df_bulanan.iloc[-12:]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(historis_terakhir.index, historis_terakhir['harga'],
            label='Data Historis (2025)', color='#2E86AB',
            linewidth=2, marker='o', markersize=5)
    ax.plot(pred_mean.index, pred_mean.values,
            label='Prediksi 2026', color='#E84855',
            linewidth=2.5, marker='s', markersize=6, linestyle='--')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                    alpha=0.2, color='#E84855', label='Interval Kepercayaan 95%')
    ax.axvline(x=pd.Timestamp('2026-01-01'), color='gray', linestyle=':', linewidth=1.5)
    ax.set_title(f'Prediksi Harga Cabai Merah Keriting Provinsi Banten 2026\n'
                 f'ARIMA({p},{d},{q}) | Interval Kepercayaan 95%',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Harga (Rp/kg)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Tabel Prediksi Harga Per Bulan 2026</div>', unsafe_allow_html=True)

    df_pred = pd.DataFrame({
        'Bulan'              : nama_bulan,
        'Periode'            : bulan_2026.strftime('%Y-%m'),
        'Prediksi (Rp/kg)'  : [f"Rp {v:,.0f}" for v in pred_mean.values],
        'Batas Bawah (Rp/kg)': [f"Rp {v:,.0f}" for v in pred_ci.iloc[:, 0].values],
        'Batas Atas (Rp/kg)' : [f"Rp {v:,.0f}" for v in pred_ci.iloc[:, 1].values],
    })
    st.dataframe(df_pred, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Unduh Hasil Prediksi</div>', unsafe_allow_html=True)

    df_download = pd.DataFrame({
        'bulan'          : nama_bulan,
        'periode'        : bulan_2026.strftime('%Y-%m'),
        'prediksi_harga' : pred_mean.values.round(0).astype(int),
        'batas_bawah'    : pred_ci.iloc[:, 0].values.round(0).astype(int),
        'batas_atas'     : pred_ci.iloc[:, 1].values.round(0).astype(int),
    })

    st.download_button(
        label="⬇️ Download Prediksi 2026 (.csv)",
        data=df_download.to_csv(index=False).encode('utf-8'),
        file_name='prediksi_cabai_banten_2026.csv',
        mime='text/csv'
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.8rem; padding: 1rem 0;">
    Prediksi Harga Cabai Merah Keriting Provinsi Banten •
    Algoritma ARIMA • Data PIHPS Nasional (hargapangan.id) •
    Diolah menggunakan Python & Streamlit
</div>
""", unsafe_allow_html=True)
