import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Keriting Banten",
    page_icon="🌶️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {font-size:32px; font-weight:700; color:#1a1a2e; margin-bottom:4px;}
.sub-title {font-size:16px; color:#6b7280; margin-bottom:20px;}
.section-title {font-size:21px; font-weight:700; color:#1a1a2e; margin-top:28px; margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

nama_bulan = {
    1:"Januari", 2:"Februari", 3:"Maret", 4:"April",
    5:"Mei", 6:"Juni", 7:"Juli", 8:"Agustus",
    9:"September", 10:"Oktober", 11:"November", 12:"Desember"
}

def format_rupiah(nilai):
    return "Rp {:,.0f}".format(float(nilai)).replace(",", ".")

# Hasil forecast ARIMA(3,0,3) dari model yang dilatih ulang
# menggunakan seluruh data Januari 2021-Desember 2025.
hasil_forecast = [
    50632.0, 47604.0, 40030.0, 42695.0,
    49476.0, 52131.0, 49977.0, 47293.0,
    46936.0, 48205.0, 49129.0, 48964.0
]

prediksi_2026 = pd.Series(
    hasil_forecast,
    index=pd.date_range("2026-01-01", periods=12, freq="MS"),
    name="Prediksi Harga"
)

# Ringkasan dihitung dari seluruh prediksi Januari-Desember 2026.
rata_rata = prediksi_2026.mean()
tertinggi = prediksi_2026.max()
bulan_tertinggi = prediksi_2026.idxmax()
terendah = prediksi_2026.min()
bulan_terendah = prediksi_2026.idxmin()

st.markdown(
    '<div class="main-title">🌶️ Prediksi Harga Cabai Merah Keriting</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Provinsi Banten • Data Historis 2021–2025 • Prediksi 2026</div>',
    unsafe_allow_html=True
)
st.divider()

st.markdown(
    '<div class="section-title">🔮 Prediksi Harga Tahun 2026</div>',
    unsafe_allow_html=True
)
st.info(
    "Dashboard ini menyajikan hasil prediksi harga eceran cabai merah keriting "
    "di Provinsi Banten untuk tahun 2026. Prediksi menggunakan model "
    "ARIMA(3,0,3) yang dilatih ulang dengan data historis Januari 2021 "
    "sampai Desember 2025."
)

# 1. RINGKASAN PERKIRAAN
st.markdown(
    '<div class="section-title">📌 Ringkasan Perkiraan</div>',
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.metric("Rata-rata Perkiraan 2026", f"{format_rupiah(rata_rata)}/kg")
        st.caption("Rata-rata prediksi Januari-Desember 2026")

with col2:
    with st.container(border=True):
        st.metric("Perkiraan Harga Tertinggi", f"{format_rupiah(tertinggi)}/kg")
        st.caption(f"Diperkirakan pada {nama_bulan[bulan_tertinggi.month]} 2026")

with col3:
    with st.container(border=True):
        st.metric("Perkiraan Harga Terendah", f"{format_rupiah(terendah)}/kg")
        st.caption(f"Diperkirakan pada {nama_bulan[bulan_terendah.month]} 2026")

st.write(
    f"Rata-rata harga cabai merah keriting sepanjang tahun 2026 diperkirakan "
    f"sebesar **{format_rupiah(rata_rata)}/kg**. Harga tertinggi diperkirakan "
    f"terjadi pada **{nama_bulan[bulan_tertinggi.month]}** sebesar "
    f"**{format_rupiah(tertinggi)}/kg**, sedangkan harga terendah diperkirakan "
    f"terjadi pada **{nama_bulan[bulan_terendah.month]}** sebesar "
    f"**{format_rupiah(terendah)}/kg**."
)

# 2. PREDIKSI HARGA TAHUN 2026
st.markdown(
    '<div class="section-title">📈 Perkiraan Pergerakan Harga Tahun 2026</div>',
    unsafe_allow_html=True
)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    prediksi_2026.index, prediksi_2026.values,
    marker="o", linewidth=2.5, markersize=6, label="Prediksi 2026"
)
ax.set_title("Prediksi Harga Cabai Merah Keriting Tahun 2026", fontsize=14, fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Harga (Rp/kg)")
ax.set_xticks(prediksi_2026.index)
ax.set_xticklabels(
    [nama_bulan[t.month] for t in prediksi_2026.index],
    rotation=45
)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: f"Rp {x:,.0f}".replace(",", "."))
)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Grafik menampilkan pergerakan hasil prediksi harga untuk Januari-Desember 2026.")

# 3. TABEL PREDIKSI JANUARI-JULI 2026
st.markdown(
    '<div class="section-title">📋 Tabel Prediksi Harga Tahun 2026 (Januari-Juli)</div>',
    unsafe_allow_html=True
)
prediksi_jan_jul = prediksi_2026.iloc[:7]
tabel_prediksi = pd.DataFrame({
    "Bulan": [nama_bulan[t.month] for t in prediksi_jan_jul.index],
    "Prediksi Harga (Rp/kg)": [format_rupiah(v) for v in prediksi_jan_jul.values]
})
st.dataframe(tabel_prediksi, use_container_width=True, hide_index=True)
st.caption(
    "Tabel menampilkan hasil prediksi harga per kilogram untuk Januari "
    "sampai Juli 2026 berdasarkan model ARIMA(3,0,3)."
)

st.warning(
    "Hasil prediksi merupakan perkiraan berdasarkan pola data historis "
    "Januari 2021 sampai Desember 2025. Harga aktual dapat berbeda karena "
    "perubahan kondisi pasar, pasokan, permintaan, musim, cuaca, dan faktor lainnya."
)
st.divider()
st.caption("Sumber data historis: PIHPS Nasional (hargapangan.id)")
st.caption("Periode data historis: Januari 2021-Desember 2025")
st.caption("Metode peramalan: ARIMA(3,0,3)")
st.caption("Periode prediksi: Januari-Desember 2026")
