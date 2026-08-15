import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Keriting Banten",
    page_icon="🌶️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAMA BULAN
# ============================================================

nama_bulan = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}


# ============================================================
# FUNGSI FORMAT RUPIAH
# ============================================================

def format_rupiah(nilai):

    return (
        "Rp {:,.0f}".format(float(nilai))
        .replace(",", ".")
    )


def nama_bulan_indonesia(tanggal):

    return nama_bulan[tanggal.month]


# ============================================================
# DATA HISTORIS
# JANUARI 2021 - DESEMBER 2025
# ============================================================

periode = pd.date_range(
    start="2021-01-01",
    periods=60,
    freq="MS"
)


harga = [

    51555.0,
    59057.89,
    54881.82,
    46061.9,
    35382.35,
    27095.24,
    30054.76,
    23490.0,
    23506.82,
    34692.5,
    47568.18,
    45293.48,

    33378.57,
    38447.22,
    48431.82,
    41394.74,
    44426.67,
    82885.71,
    95711.9,
    68111.36,
    69936.36,
    52440.48,
    35993.18,
    38131.82,

    45504.55,
    48780.0,
    41473.91,
    39552.5,
    32031.82,
    33104.55,
    35090.48,
    42217.39,
    38178.57,
    47190.91,
    81620.45,
    80007.89,

    70978.26,
    79464.29,
    60657.14,
    43093.18,
    50360.87,
    51740.0,
    43708.7,
    41538.64,
    30750.0,
    27963.04,
    27857.14,
    47045.45,

    65150.0,
    52517.5,
    54923.81,
    61797.73,
    40809.09,
    42547.62,
    41469.57,
    38852.38,
    57172.73,
    60336.96,
    64962.5,
    57947.62

]


df = pd.DataFrame(
    {
        "harga": harga
    },
    index=periode
)


# ============================================================
# PEMBAGIAN DATA
#
# 80% TRAINING
# 20% TESTING
#
# Training : Januari 2021 - Desember 2024
# Testing  : Januari 2025 - Desember 2025
# ============================================================

split_idx = int(
    len(df) * 0.8
)


train = df.iloc[
    :split_idx
].copy()


test = df.iloc[
    split_idx:
].copy()


# ============================================================
# MODEL UNTUK EVALUASI
#
# ARIMA(3,0,3)
#
# Model dilatih menggunakan data 2021-2024.
# Kemudian digunakan untuk memprediksi 2025.
# ============================================================

with st.spinner(
    "🔍 Mengevaluasi model ARIMA..."
):

    model_evaluasi = ARIMA(
        train["harga"],
        order=(3, 0, 3)
    ).fit()


    prediksi_testing = model_evaluasi.forecast(
        steps=len(test)
    )


    prediksi_testing.index = test.index


# ============================================================
# NILAI AKTUAL DAN PREDIKSI
# ============================================================

aktual_testing = test["harga"].values

hasil_testing = prediksi_testing.values


# ============================================================
# EVALUASI RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        aktual_testing,
        hasil_testing
    )
)


# ============================================================
# EVALUASI MAE
# ============================================================

mae = mean_absolute_error(
    aktual_testing,
    hasil_testing
)


# ============================================================
# EVALUASI MAPE
# ============================================================

mape = np.mean(
    np.abs(
        (
            aktual_testing - hasil_testing
        )
        /
        aktual_testing
    )
) * 100


# ============================================================
# KATEGORI MAPE
# ============================================================

if mape < 10:

    kategori_mape = "Sangat Baik"

elif mape < 20:

    kategori_mape = "Baik"

elif mape < 50:

    kategori_mape = "Cukup"

else:

    kategori_mape = "Kurang Baik"


# ============================================================
# MODEL FINAL UNTUK PREDIKSI 2026
#
# Setelah evaluasi selesai, model dilatih kembali
# menggunakan seluruh data 2021-2025.
# ============================================================

with st.spinner(
    "🔮 Menghitung perkiraan harga tahun 2026..."
):

    model_final = ARIMA(
        df["harga"],
        order=(3, 0, 3)
    ).fit()


    prediksi_2026 = model_final.forecast(
        steps=12
    )


# ============================================================
# PERIODE PREDIKSI 2026
# ============================================================

periode_2026 = pd.date_range(
    start="2026-01-01",
    periods=12,
    freq="MS"
)


prediksi_2026.index = periode_2026


# ============================================================
# DATA AKTUAL TAHUN 2025
# ============================================================

aktual_2025 = df[
    df.index.year == 2025
]["harga"].copy()


# ============================================================
# RINGKASAN PREDIKSI
# ============================================================

rata_rata_prediksi = prediksi_2026.mean()


harga_tertinggi = prediksi_2026.max()


bulan_tertinggi = prediksi_2026.idxmax()


harga_terendah = prediksi_2026.min()


bulan_terendah = prediksi_2026.idxmin()


rata_rata_aktual_2025 = aktual_2025.mean()


perubahan_rata_rata = (
    (
        rata_rata_prediksi
        -
        rata_rata_aktual_2025
    )
    /
    rata_rata_aktual_2025
) * 100


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🌶️ Prediksi Harga Cabai Merah Keriting'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="sub-title">'
    'Provinsi Banten • Data Historis 2021–2025 • Prediksi 2026'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# INFORMASI UTAMA
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔮 Perkiraan Harga Cabai Tahun 2026'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Dashboard ini menampilkan perkiraan harga cabai merah "
    "keriting untuk setiap bulan tahun 2026 berdasarkan pola "
    "harga yang tercatat dari Januari 2021 sampai Desember 2025."
)


# ============================================================
# RINGKASAN PERKIRAAN
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📌 Ringkasan Perkiraan'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.metric(
            label="Rata-rata Perkiraan 2026",
            value=(
                format_rupiah(
                    rata_rata_prediksi
                )
                + "/kg"
            )
        )

        st.caption(
            "Rata-rata seluruh perkiraan harga 2026"
        )


with col2:

    with st.container(border=True):

        st.metric(
            label="Perkiraan Harga Tertinggi",
            value=(
                format_rupiah(
                    harga_tertinggi
                )
                + "/kg"
            )
        )

        st.caption(
            "Terjadi pada "
            + nama_bulan_indonesia(
                bulan_tertinggi
            )
            + " 2026"
        )


with col3:

    with st.container(border=True):

        st.metric(
            label="Perkiraan Harga Terendah",
            value=(
                format_rupiah(
                    harga_terendah
                )
                + "/kg"
            )
        )

        st.caption(
            "Terjadi pada "
            + nama_bulan_indonesia(
                bulan_terendah
            )
            + " 2026"
        )


# ============================================================
# PREDIKSI SETIAP BULAN
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📅 Prediksi Harga Setiap Bulan Tahun 2026'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Setiap kotak menunjukkan perkiraan harga cabai merah "
    "keriting per kilogram."
)


def kartu_prediksi(index):

    tanggal = prediksi_2026.index[index]

    nilai = prediksi_2026.iloc[index]

    with st.container(border=True):

        st.markdown(
            "### "
            + nama_bulan_indonesia(tanggal)
            + " 2026"
        )

        st.metric(
            label="Perkiraan harga",
            value=(
                format_rupiah(nilai)
                + "/kg"
            )
        )


# Januari - Maret

col1, col2, col3 = st.columns(3)

with col1:
    kartu_prediksi(0)

with col2:
    kartu_prediksi(1)

with col3:
    kartu_prediksi(2)


# April - Juni

col1, col2, col3 = st.columns(3)

with col1:
    kartu_prediksi(3)

with col2:
    kartu_prediksi(4)

with col3:
    kartu_prediksi(5)


# Juli - September

col1, col2, col3 = st.columns(3)

with col1:
    kartu_prediksi(6)

with col2:
    kartu_prediksi(7)

with col3:
    kartu_prediksi(8)


# Oktober - Desember

col1, col2, col3 = st.columns(3)

with col1:
    kartu_prediksi(9)

with col2:
    kartu_prediksi(10)

with col3:
    kartu_prediksi(11)


# ============================================================
# GRAFIK PREDIKSI 2026
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Perkiraan Pergerakan Harga Tahun 2026'
    '</div>',
    unsafe_allow_html=True
)


fig, ax = plt.subplots(
    figsize=(13, 5)
)


ax.plot(
    prediksi_2026.index,
    prediksi_2026.values,
    marker="o",
    linewidth=2.5,
    markersize=6,
    label="Prediksi 2026"
)


ax.set_title(
    "Perkiraan Harga Cabai Merah Keriting Tahun 2026",
    fontsize=14,
    fontweight="bold"
)


ax.set_xlabel("Bulan")


ax.set_ylabel("Harga (Rp/kg)")


ax.set_xticks(
    prediksi_2026.index
)


ax.set_xticklabels(
    [
        nama_bulan_indonesia(tanggal)
        for tanggal in prediksi_2026.index
    ],
    rotation=45
)


ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, pos:
        f"Rp {x:,.0f}"
    )
)


ax.grid(
    True,
    alpha=0.25
)


ax.legend(
    frameon=False
)


plt.tight_layout()


st.pyplot(fig)


plt.close(fig)


# ============================================================
# HASIL EVALUASI MODEL
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🎯 Hasil Evaluasi Model'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Evaluasi dilakukan menggunakan data tahun 2025 sebagai "
    "data pengujian. Model terlebih dahulu mempelajari data "
    "tahun 2021 sampai 2024, kemudian hasil prediksinya "
    "dibandingkan dengan harga aktual tahun 2025."
)


# ============================================================
# KARTU EVALUASI
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.metric(
            label="RMSE",
            value=format_rupiah(rmse)
        )

        st.caption(
            "Semakin kecil, semakin baik."
        )


with col2:

    with st.container(border=True):

        st.metric(
            label="MAE",
            value=format_rupiah(mae)
        )

        st.caption(
            "Rata-rata besar kesalahan prediksi."
        )


with col3:

    with st.container(border=True):

        st.metric(
            label="MAPE",
            value=f"{mape:.2f}%"
        )

        st.caption(
            "Kategori: "
            + kategori_mape
        )


# ============================================================
# PENJELASAN EVALUASI
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📖 Apa Arti Hasil Evaluasi?'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    f"Model menghasilkan nilai **RMSE sebesar "
    f"{format_rupiah(rmse)}**. Nilai ini menunjukkan "
    f"besarnya kesalahan prediksi dengan mempertimbangkan "
    f"selisih antara harga aktual dan hasil prediksi. "
    f"Semakin kecil nilai RMSE, semakin dekat hasil prediksi "
    f"dengan harga aktual."
)


st.write(
    f"Nilai **MAE sebesar {format_rupiah(mae)}** menunjukkan "
    f"bahwa rata-rata terdapat selisih sekitar "
    f"**{format_rupiah(mae)}** antara harga yang diprediksi "
    f"model dengan harga aktual pada data pengujian."
)


st.write(
    f"Nilai **MAPE sebesar {mape:.2f}%** menunjukkan rata-rata "
    f"persentase kesalahan prediksi model terhadap harga aktual. "
    f"Nilai tersebut termasuk kategori **{kategori_mape}**."
)


# ============================================================
# INTERPRETASI MAPE
# ============================================================

if mape < 10:

    st.success(
        f"✅ MAPE sebesar {mape:.2f}% termasuk kategori "
        "**sangat baik**. Artinya, secara rata-rata kesalahan "
        "prediksi relatif kecil terhadap harga aktual."
    )

elif mape < 20:

    st.success(
        f"✅ MAPE sebesar {mape:.2f}% termasuk kategori "
        "**baik**. Artinya, model memiliki tingkat kesalahan "
        "yang relatif rendah dalam melakukan prediksi."
    )

elif mape < 50:

    st.warning(
        f"⚠️ MAPE sebesar {mape:.2f}% termasuk kategori "
        "**cukup**. Artinya, model masih dapat digunakan "
        "sebagai gambaran perkiraan, tetapi hasil prediksi "
        "memiliki selisih yang cukup terhadap harga aktual."
    )

else:

    st.error(
        f"⚠️ MAPE sebesar {mape:.2f}% termasuk kategori "
        "**kurang baik**. Artinya, hasil prediksi memiliki "
        "kesalahan yang relatif besar terhadap harga aktual."
    )


# ============================================================
# GRAFIK EVALUASI
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Perbandingan Aktual dan Prediksi pada Data Pengujian'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Grafik ini menunjukkan perbandingan harga aktual tahun "
    "2025 dengan hasil prediksi model pada periode pengujian."
)


fig_eval, ax_eval = plt.subplots(
    figsize=(13, 5.5)
)


ax_eval.plot(
    test.index,
    test["harga"],
    marker="o",
    linewidth=2.5,
    label="Harga Aktual 2025"
)


ax_eval.plot(
    prediksi_testing.index,
    prediksi_testing.values,
    marker="s",
    linestyle="--",
    linewidth=2.5,
    label="Hasil Prediksi Model"
)


ax_eval.set_title(
    "Evaluasi Model ARIMA pada Data Pengujian Tahun 2025",
    fontsize=14,
    fontweight="bold"
)


ax_eval.set_xlabel("Bulan")


ax_eval.set_ylabel("Harga (Rp/kg)")


ax_eval.set_xticks(
    test.index
)


ax_eval.set_xticklabels(
    [
        nama_bulan_indonesia(tanggal)
        for tanggal in test.index
    ],
    rotation=45
)


ax_eval.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, pos:
        f"Rp {x:,.0f}"
    )
)


ax_eval.grid(
    True,
    alpha=0.25
)


ax_eval.legend(
    frameon=False
)


plt.tight_layout()


st.pyplot(fig_eval)


plt.close(fig_eval)


# ============================================================
# PENJELASAN GRAFIK EVALUASI
# ============================================================

st.info(
    "Cara membacanya sederhana. Garis harga aktual menunjukkan "
    "harga yang benar-benar tercatat pada tahun 2025. Garis "
    "hasil prediksi menunjukkan perkiraan model untuk periode "
    "yang sama. Semakin dekat kedua garis tersebut, semakin "
    "baik model mengikuti pola harga aktual."
)


# ============================================================
# PERBANDINGAN 2025 VS 2026
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Perbandingan Harga Aktual 2025 dan Prediksi 2026'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Bagian ini berbeda dengan evaluasi model. Evaluasi model "
    "digunakan untuk mengetahui kinerja model, sedangkan "
    "perbandingan 2025 dan 2026 digunakan untuk memberikan "
    "gambaran perubahan harga dari tahun 2025 ke tahun 2026."
)


# ============================================================
# GRAFIK PERBANDINGAN
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(13, 5.5)
)


ax2.plot(
    aktual_2025.index,
    aktual_2025.values,
    marker="o",
    linewidth=2.5,
    label="Aktual 2025"
)


ax2.plot(
    prediksi_2026.index,
    prediksi_2026.values,
    marker="s",
    linestyle="--",
    linewidth=2.5,
    label="Prediksi 2026"
)


ax2.set_title(
    "Perbandingan Harga Aktual 2025 dan Prediksi 2026",
    fontsize=14,
    fontweight="bold"
)


ax2.set_xlabel("Bulan")


ax2.set_ylabel("Harga (Rp/kg)")


ax2.set_xticks(
    prediksi_2026.index
)


ax2.set_xticklabels(
    [
        nama_bulan_indonesia(tanggal)
        for tanggal in prediksi_2026.index
    ],
    rotation=45
)


ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, pos:
        f"Rp {x:,.0f}"
    )
)


ax2.grid(
    True,
    alpha=0.25
)


ax2.legend(
    frameon=False
)


plt.tight_layout()


st.pyplot(fig2)


plt.close(fig2)


# ============================================================
# TABEL PERBANDINGAN
# MENGGUNAKAN PANDAS STYLER
# TIDAK MENGGUNAKAN HTML TABLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📋 Tabel Perbandingan Setiap Bulan'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Warna hijau menunjukkan harga prediksi 2026 lebih tinggi "
    "daripada harga aktual 2025. Warna merah menunjukkan "
    "harga prediksi 2026 lebih rendah."
)


# ============================================================
# MEMBUAT DATAFRAME TABEL
# ============================================================

data_tabel = []


for i in range(12):

    tanggal = aktual_2025.index[i]

    aktual = aktual_2025.iloc[i]

    prediksi = prediksi_2026.iloc[i]

    perubahan = (
        (
            prediksi - aktual
        )
        /
        aktual
    ) * 100


    if perubahan > 0:

        status = "↑ Naik"

    elif perubahan < 0:

        status = "↓ Turun"

    else:

        status = "Tetap"


    data_tabel.append(
        {
            "Bulan":
                nama_bulan_indonesia(tanggal),

            "Harga Aktual 2025":
                aktual,

            "Harga Prediksi 2026":
                prediksi,

            "Perubahan (%)":
                perubahan,

            "Keterangan":
                status
        }
    )


df_perbandingan = pd.DataFrame(
    data_tabel
)


# ============================================================
# FUNGSI FORMAT RUPIAH PADA STYLER
# ============================================================

def format_rupiah_styler(nilai):

    return (
        "Rp {:,.0f}/kg"
        .format(float(nilai))
        .replace(",", ".")
    )


# ============================================================
# FUNGSI WARNA PREDIKSI
# ============================================================

def warna_prediksi(row):

    styles = pd.Series(
        "",
        index=row.index
    )


    if row["Perubahan (%)"] > 0:

        styles["Harga Prediksi 2026"] = (
            "background-color: #dcfce7;"
            "color: #166534;"
            "font-weight: 600;"
        )

        styles["Perubahan (%)"] = (
            "background-color: #dcfce7;"
            "color: #166534;"
            "font-weight: 600;"
        )

        styles["Keterangan"] = (
            "background-color: #dcfce7;"
            "color: #166534;"
            "font-weight: 600;"
        )


    elif row["Perubahan (%)"] < 0:

        styles["Harga Prediksi 2026"] = (
            "background-color: #fee2e2;"
            "color: #991b1b;"
            "font-weight: 600;"
        )

        styles["Perubahan (%)"] = (
            "background-color: #fee2e2;"
            "color: #991b1b;"
            "font-weight: 600;"
        )

        styles["Keterangan"] = (
            "background-color: #fee2e2;"
            "color: #991b1b;"
            "font-weight: 600;"
        )


    return styles


# ============================================================
# FORMAT TABEL
# ============================================================

styled_table = (

    df_perbandingan
    .style
    .apply(
        warna_prediksi,
        axis=1
    )
    .format(
        {
            "Harga Aktual 2025":
                format_rupiah_styler,

            "Harga Prediksi 2026":
                format_rupiah_styler,

            "Perubahan (%)":
                lambda x:
                f"{x:+.2f}%"
        }
    )
)


# ============================================================
# TAMPILKAN TABEL
# ============================================================

st.dataframe(
    styled_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PENJELASAN TABEL
# ============================================================

st.caption(
    "Harga aktual 2025 merupakan harga yang benar-benar "
    "tercatat, sedangkan harga prediksi 2026 merupakan "
    "hasil perkiraan model ARIMA."
)


# ============================================================
# KESIMPULAN PERBANDINGAN
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📌 Kesimpulan Perbandingan'
    '</div>',
    unsafe_allow_html=True
)


if perubahan_rata_rata > 0:

    st.warning(
        f"Rata-rata harga aktual tahun 2025 adalah "
        f"{format_rupiah(rata_rata_aktual_2025)}/kg. "
        f"Rata-rata harga tahun 2026 diperkirakan "
        f"{format_rupiah(rata_rata_prediksi)}/kg. "
        f"Secara rata-rata, harga tahun 2026 diperkirakan "
        f"naik sebesar {perubahan_rata_rata:.2f}% "
        f"dibandingkan tahun 2025."
    )


elif perubahan_rata_rata < 0:

    st.success(
        f"Rata-rata harga aktual tahun 2025 adalah "
        f"{format_rupiah(rata_rata_aktual_2025)}/kg. "
        f"Rata-rata harga tahun 2026 diperkirakan "
        f"{format_rupiah(rata_rata_prediksi)}/kg. "
        f"Secara rata-rata, harga tahun 2026 diperkirakan "
        f"turun sebesar {abs(perubahan_rata_rata):.2f}% "
        f"dibandingkan tahun 2025."
    )


else:

    st.info(
        "Rata-rata harga tahun 2025 dan perkiraan tahun 2026 "
        "berada pada tingkat yang relatif sama."
    )


# ============================================================
# CARA MEMBACA DASHBOARD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📖 Cara Membaca Dashboard'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Pertama, lihat bagian Perkiraan Harga 2026 untuk mengetahui "
    "harga yang diperkirakan pada setiap bulan. Kedua, lihat "
    "Hasil Evaluasi Model untuk mengetahui seberapa baik model "
    "melakukan prediksi. Ketiga, lihat Tabel Perbandingan untuk "
    "melihat perbedaan harga aktual tahun 2025 dengan perkiraan "
    "harga tahun 2026."
)


# ============================================================
# CATATAN
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⚠️ Catatan'
    '</div>',
    unsafe_allow_html=True
)


st.warning(
    "Harga tahun 2026 merupakan hasil perkiraan berdasarkan "
    "pola data historis Januari 2021 sampai Desember 2025. "
    "Harga sebenarnya dapat berbeda karena kondisi pasar, "
    "pasokan, permintaan, musim, cuaca, dan faktor lainnya."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Sumber data: PIHPS Nasional (hargapangan.id)"
)


st.caption(
    "Data historis: Januari 2021 - Desember 2025"
)


st.caption(
    "Metode prediksi: ARIMA(3,0,3)"
)


st.caption(
    "Evaluasi model: Data testing tahun 2025"
)


st.caption(
    "Periode prediksi: Januari - Desember 2026"
)
