"""
Aplikasi Streamlit - Klasifikasi Hujan Harian Kolkata (XGBoost Classifier)

Cara menjalankan:
    1. Jalankan notebook klasifikasi_1.ipynb sampai selesai agar file berikut terbentuk
       di folder yang sama dengan app.py ini:
           model.pkl, scaler.pkl, le_season.pkl, le_wind.pkl, eval_data.pkl
    2. Jalankan:  streamlit run app.py
"""

import os
import pickle
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Klasifikasi Hujan Kolkata",
    page_icon="🌧️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUMERIC_EDA = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "sunshine_duration",
]


# ------------------------------------------------------------------
# Fungsi bantu (sama dengan yang dipakai di notebook)
# ------------------------------------------------------------------
def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    return "Monsoon"


def get_wind_compass(degree: float) -> str:
    if degree >= 337.5 or degree < 22.5:
        return "N"
    elif degree < 67.5:
        return "NE"
    elif degree < 112.5:
        return "E"
    elif degree < 157.5:
        return "SE"
    elif degree < 202.5:
        return "S"
    elif degree < 247.5:
        return "SW"
    elif degree < 292.5:
        return "W"
    return "NW"


@st.cache_resource
def load_artifacts():
    """Memuat model, scaler, encoder, dan data evaluasi dari file .pkl."""
    files = {
        "model": "model.pkl",
        "scaler": "scaler.pkl",
        "le_season": "le_season.pkl",
        "le_wind": "le_wind.pkl",
        "eval": "eval_data.pkl",
    }
    loaded = {}
    for key, name in files.items():
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            return None, name
        with open(path, "rb") as f:
            loaded[key] = pickle.load(f)
    return loaded, None


art, missing_file = load_artifacts()
if art is None:
    st.error(
        f"File **{missing_file}** tidak ditemukan. Jalankan notebook sampai sel "
        "penyimpanan artifact, lalu letakkan semua file .pkl di folder yang sama "
        "dengan app.py."
    )
    st.stop()

model = art["model"]
scaler = art["scaler"]
le_season = art["le_season"]
le_wind = art["le_wind"]
ev = art["eval"]

data = ev["data"]
features = ev["features"]
metrics_df = ev["metrics_df"]
cm = ev["cm"]
fpr, tpr, auc_score = ev["fpr"], ev["tpr"], ev["auc"]
importances = ev["importances"]


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🌧️ Klasifikasi Hujan Kolkata")
page = st.sidebar.radio(
    "Menu",
    ["Prediksi", "Dataset dan EDA", "Evaluasi Model"],
)


# ------------------------------------------------------------------
# Halaman 1: Prediksi
# ------------------------------------------------------------------
def page_prediksi():
    st.title("Prediksi Hujan Harian")
    st.write(
        "Masukkan kondisi cuaca harian di Kolkata, lalu model akan memprediksi "
        "apakah hari tersebut **hujan** atau **tidak hujan**."
    )

    med = data[features].median(numeric_only=True)

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Suhu dan Penyinaran")
            tanggal = st.date_input("Tanggal", value=date.today())
            t_mean = st.number_input(
                "Suhu rata-rata (°C)", 0.0, 50.0,
                float(med["temperature_2m_mean"]), 0.1,
            )
            t_max = st.number_input(
                "Suhu maksimum (°C)", 0.0, 55.0,
                float(med["temperature_2m_max"]), 0.1,
            )
            t_min = st.number_input(
                "Suhu minimum (°C)", 0.0, 45.0,
                float(med["temperature_2m_min"]), 0.1,
            )
            sunshine = st.number_input(
                "Durasi penyinaran matahari (detik)", 0.0, 50000.0,
                float(med["sunshine_duration"]), 100.0,
            )

        with col2:
            st.subheader("Angin")
            w_speed = st.number_input(
                "Kecepatan angin maksimum", 0.0, 100.0,
                float(med["wind_speed_10m_max"]), 0.1,
            )
            w_gust = st.number_input(
                "Hembusan angin maksimum", 0.0, 150.0,
                float(med["wind_gusts_10m_max"]), 0.1,
            )
            w_dir = st.slider(
                "Arah angin dominan (derajat)", 0, 360,
                int(med["wind_direction_10m_dominant"]),
            )

        submit = st.form_submit_button("Prediksi", type="primary")

    if not submit:
        return

    # Validasi logika input
    errors = []
    if not (t_min <= t_mean <= t_max):
        errors.append("Suhu harus memenuhi: minimum ≤ rata-rata ≤ maksimum.")
    if w_gust < w_speed:
        errors.append("Hembusan angin maksimum tidak boleh lebih kecil dari kecepatan angin maksimum.")
    if errors:
        for e in errors:
            st.error(e)
        return

    # Fitur turunan (sama seperti di notebook)
    season = get_season(tanggal.month)
    compass = get_wind_compass(w_dir)

    row = pd.DataFrame([{
        "temperature_2m_mean": t_mean,
        "temperature_2m_max": t_max,
        "temperature_2m_min": t_min,
        "wind_speed_10m_max": w_speed,
        "wind_gusts_10m_max": w_gust,
        "wind_direction_10m_dominant": w_dir,
        "sunshine_duration": sunshine,
        "season": le_season.transform([season])[0],
        "wind_compass": le_wind.transform([compass])[0],
    }])[features]  # urutan kolom harus sama dengan saat training

    X_scaled = scaler.transform(row)
    prob_rain = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(prob_rain >= 0.5)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hasil Prediksi", "🌧️ Hujan" if pred == 1 else "☀️ Tidak Hujan")
    c2.metric("Probabilitas Hujan", f"{prob_rain * 100:.1f}%")
    c3.metric("Musim / Arah Angin", f"{season} / {compass}")

    st.progress(prob_rain)
    if pred == 1:
        st.success("Model memprediksi hari ini **hujan**.")
    else:
        st.info("Model memprediksi hari ini **tidak hujan**.")

    st.caption(
        "Ambang klasifikasi 0,5. Hasil ini adalah prediksi statistik dari data historis "
        "Kolkata, bukan prakiraan cuaca resmi."
    )


# ------------------------------------------------------------------
# Halaman 2: Dataset dan EDA
# ------------------------------------------------------------------
def page_eda():
    st.title("Dataset dan Eksplorasi Data")

    n_rain = int((data["is_rainy"] == 1).sum())
    n_dry = int((data["is_rainy"] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(data):,}")
    c2.metric("Jumlah Fitur Model", len(features))
    c3.metric("Hari Hujan", f"{n_rain:,} ({n_rain / len(data) * 100:.2f}%)")
    c4.metric("Hari Tidak Hujan", f"{n_dry:,} ({n_dry / len(data) * 100:.2f}%)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data", "Distribusi Target", "Musim", "Distribusi Fitur", "Korelasi"]
    )

    with tab1:
        st.subheader("Cuplikan Data")
        st.dataframe(data.head(20), use_container_width=True)
        st.subheader("Statistik Deskriptif")
        cols = [c for c in NUMERIC_EDA if c in data.columns]
        st.dataframe(data[cols].describe().T, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="is_rainy", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Tidak Hujan", "Hujan"])
        ax.set_xlabel("Kelas Cuaca")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Kelas Target Cuaca")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=data, x="season", order=["Winter", "Summer", "Monsoon"], ax=ax
        )
        ax.set_xlabel("Musim")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Data Berdasarkan Musim")
        st.pyplot(fig)

    with tab4:
        feat = st.selectbox(
            "Pilih fitur", [c for c in NUMERIC_EDA if c in data.columns]
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=data, x=feat, kde=True, ax=ax)
        ax.set_title(f"Distribusi {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    with tab5:
        cols = [c for c in NUMERIC_EDA if c in data.columns] + ["is_rainy"]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            data[cols].corr(), annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5, ax=ax,
        )
        ax.set_title("Correlation Heatmap Fitur Numerik Cuaca Kolkata")
        st.pyplot(fig)
        st.caption(
            "Kolom precipitation_sum, rain_sum, dan precipitation_hours tidak dipakai "
            "sebagai fitur model karena menentukan label secara langsung (target leakage)."
        )


# ------------------------------------------------------------------
# Halaman 3: Evaluasi Model
# ------------------------------------------------------------------
def page_evaluasi():
    st.title("Evaluasi Model XGBoost")

    st.subheader("Metrik Evaluasi (Data Uji)")
    cols = st.columns(len(metrics_df))
    for col, (_, r) in zip(cols, metrics_df.iterrows()):
        col.metric(r["Metric"], f"{r['Score']:.4f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues",
            annot_kws={"size": 16, "weight": "bold"},
            xticklabels=["Prediksi Negatif (0)", "Prediksi Positif (1)"],
            yticklabels=["Aktual Negatif (0)", "Aktual Positif (1)"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix XGBoost")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        st.pyplot(fig)
        st.caption(
            f"Prediksi benar: {tn + tp} dari {cm.sum()} data. "
            f"Kesalahan: {fp} FP (alarm palsu) dan {fn} FN (hujan tidak terdeteksi)."
        )

    with right:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"XGBoost (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve XGBoost")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importances, x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature Importance XGBoost")
    st.pyplot(fig)
    st.dataframe(importances, use_container_width=True, hide_index=True)


if page == "Prediksi":
    page_prediksi()
elif page == "Dataset dan EDA":
    page_eda()
elif page == "Evaluasi Model":
    page_evaluasi()
