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

# Prediksi ARIMA(3,0,3), model dilatih ulang menggunakan data Januari 2021-Desember 2025.
hasil_forecast = [
    50632, 47604, 40030, 42695, 49476, 52131,
    49977, 47293, 46936, 48205, 49129, 48964
]

# Data aktual yang tersedia untuk Januari-Juli 2026.
# Bulan Agustus-Desember tidak ditampilkan dalam perbandingan karena data aktual belum tersedia.
aktual_jan_jul = [36745, 42900, 42655, 45384, 56543, 49255, 39196]

tanggal_2026 = pd.date_range("2026-01-01", periods=12, freq="MS")
prediksi_2026 = pd.Series(hasil_forecast, index=tanggal_2026, name="Prediksi")
aktual_2026 = pd.Series(
    aktual_jan_jul,
    index=tanggal_2026[:7],
    name="Aktual"
)

# Ringkasan prediksi dihitung dari seluruh bulan Januari-Desember 2026.
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

st.markdown('<div class="section-title">🔮 Perkiraan Harga Cabai Tahun 2026</div>', unsafe_allow_html=True)
st.info(
    "Dashboard ini menyajikan hasil prediksi harga cabai merah keriting di Provinsi Banten "
    "untuk tahun 2026 menggunakan model ARIMA(3,0,3) yang dilatih ulang dengan data historis "
    "Januari 2021 sampai Desember 2025."
)

# Ringkasan perkiraan dalam tiga kotak.
st.markdown('<div class="section-title">📌 Ringkasan Perkiraan</div>', unsafe_allow_html=True)
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
    f"Rata-rata harga cabai merah keriting sepanjang tahun 2026 diperkirakan sebesar "
    f"**{format_rupiah(rata_rata)}/kg**. Harga tertinggi diperkirakan terjadi pada "
    f"**{nama_bulan[bulan_tertinggi.month]}** sebesar **{format_rupiah(tertinggi)}/kg**, "
    f"sedangkan harga terendah diperkirakan terjadi pada "
    f"**{nama_bulan[bulan_terendah.month]}** sebesar **{format_rupiah(terendah)}/kg**."
)

# Grafik gabungan: data aktual Januari-Juli dan prediksi Januari-Desember 2026.
# Garis aktual berhenti pada Juli karena data aktual setelah Juli belum tersedia.
# Garis prediksi diteruskan sampai Desember.
st.markdown(
    '<div class="section-title">📊 Perbandingan Data Aktual dan Prediksi Harga Tahun 2026</div>',
    unsafe_allow_html=True
)
st.caption(
    "Grafik menampilkan data aktual dari Januari-Juli 2026 dan hasil prediksi "
    "ARIMA(3,0,3) dari Januari-Desember 2026."
)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    aktual_2026.index, aktual_2026.values,
    marker="o", linewidth=2.2, markersize=6,
    label="Data Aktual 2026"
)
ax.plot(
    prediksi_2026.index, prediksi_2026.values,
    marker="s", linestyle="--", linewidth=2.2, markersize=6,
    label="Prediksi ARIMA(3,0,3)"
)
ax.set_title(
    "Perbandingan Data Aktual dan Prediksi Harga Cabai Merah Keriting Banten Tahun 2026",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Bulan")
ax.set_ylabel("Harga (Rp/kg)")
ax.set_xticks(prediksi_2026.index)
ax.set_xticklabels(
    [nama_bulan[t.month] + " 2026" for t in prediksi_2026.index],
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

# Tabel perbandingan aktual dan prediksi untuk Januari-Juli 2026.
st.markdown('<div class="section-title">📋 Tabel Perbandingan Data Aktual dan Prediksi 2026</div>', unsafe_allow_html=True)

selisih = prediksi_2026.iloc[:7].values - aktual_2026.values
persen_selisih = (selisih / aktual_2026.values) * 100

tabel = pd.DataFrame({
    "Bulan": [nama_bulan[t.month] for t in aktual_2026.index],
    "Harga Aktual 2026": [format_rupiah(v) for v in aktual_2026.values],
    "Harga Prediksi 2026": [format_rupiah(v) for v in prediksi_2026.iloc[:7].values],
    "Selisih (Rp/kg)": [format_rupiah(abs(v)) for v in selisih],
    "Perbedaan (%)": [f"{abs(v):.2f}%".replace(".", ",") for v in persen_selisih],
    "Keterangan": ["Prediksi lebih tinggi" if v > 0 else "Prediksi lebih rendah" if v < 0 else "Sama" for v in selisih]
})
st.dataframe(tabel, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">📌 Kesimpulan Perbandingan</div>', unsafe_allow_html=True)
rata_selisih = abs(persen_selisih).mean()
st.success(
    f"Berdasarkan perbandingan Januari-Juli 2026, rata-rata perbedaan absolut antara "
    f"harga aktual dan hasil prediksi adalah **{rata_selisih:.2f}%**. "
    "Perbedaan ini menunjukkan bahwa hasil prediksi model tidak selalu sama dengan "
    "harga yang terjadi di pasar."
)

st.markdown('<div class="section-title">📘 Cara Membaca Dashboard</div>', unsafe_allow_html=True)
st.info(
    "Grafik pertama menampilkan seluruh hasil prediksi harga tahun 2026. Grafik kedua "
    "memperlihatkan perbandingan antara data aktual dan prediksi untuk Januari-Juli 2026. "
    "Tabel digunakan untuk melihat nilai aktual, nilai prediksi, selisih, serta arah "
    "perbedaannya pada setiap bulan."
)

st.markdown('<div class="section-title">⚠️ Catatan</div>', unsafe_allow_html=True)
st.warning(
    "Prediksi tahun 2026 merupakan hasil perkiraan berdasarkan pola data historis "
    "Januari 2021 sampai Desember 2025. Perbandingan dengan data aktual hanya ditampilkan "
    "untuk Januari-Juli 2026 karena data aktual yang tersedia pada penelitian ini baru "
    "mencakup periode tersebut. Harga sebenarnya dapat berbeda karena kondisi pasar, "
    "pasokan, permintaan, musim, cuaca, dan faktor lainnya."
)

st.divider()
st.caption("Sumber data historis: PIHPS Nasional (hargapangan.id)")
st.caption("Periode data historis: Januari 2021-Desember 2025")
st.caption("Metode peramalan: ARIMA(3,0,3)")
st.caption("Periode prediksi: Januari-Desember 2026")
st.caption("Periode perbandingan aktual dan prediksi: Januari-Juli 2026")
