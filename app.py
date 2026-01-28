import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.feature_extraction import extract_features

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Earthquake Early Warning System",
    page_icon="🌍",
    layout="centered"
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("lgbm_final.pkl")
features = joblib.load("features.pkl")

# =========================================================
# HEADER
# =========================================================
st.title("🌍 Earthquake Early Warning System (EWS)")
st.caption("AI-based Vibration Signal Analysis for Early Earthquake Detection")

st.markdown("---")

# =========================================================
# ABOUT SYSTEM
# =========================================================
with st.expander("ℹ️ About This System", expanded=True):
    st.markdown("""
**Earthquake Early Warning System (EWS)** ini menggunakan **Artificial Intelligence**
untuk menganalisis **sinyal getaran (acoustic / vibration data)** dari sensor.

Model ini mempelajari **pola statistik & spektral (FFT)** dari data getaran
untuk **memprediksi waktu menuju potensi kegagalan / kejadian besar (Time to Failure)**.

💡 **Tujuan sistem:**
- Deteksi dini potensi gempa
- Memberikan peringatan lebih awal
- Mendukung sistem mitigasi bencana

⚙️ **Model yang digunakan:**
- LightGBM Regression
- Feature Engineering (Statistik + FFT)
- Trained on segmented seismic signal data
""")

# =========================================================
# DATA DESCRIPTION
# =========================================================
with st.expander("📄 Data Input Description", expanded=True):
    st.markdown("""
### 📥 Format Data yang Diperlukan

Silakan upload file **CSV** dengan ketentuan berikut:

- **Harus memiliki kolom:** `acoustic_data`
- Setiap baris merepresentasikan **sinyal getaran**
- Data berasal dari:
  - Sensor getaran
  - Accelerometer
  - Seismic / acoustic sensor

### Contoh Struktur CSV:
acoustic_data
12
-8
15
-20
...

📌 **Catatan penting:**
- Semakin panjang sinyal, semakin stabil prediksi
- Sistem ini **tidak memerlukan label**
- Data diproses secara otomatis oleh AI
""")

st.markdown("---")

# =========================================================
# FILE UPLOADER
# =========================================================
st.subheader("📤 Upload Vibration Data")

uploaded_file = st.file_uploader(
    "Upload file CSV berisi data getaran",
    type=["csv"]
)

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "acoustic_data" not in df.columns:
            st.error("❌ Kolom `acoustic_data` tidak ditemukan di file CSV.")
            st.stop()

        x = df["acoustic_data"].values

        with st.spinner("🔍 Analyzing vibration signal..."):
            feat = extract_features(x)
            X = pd.DataFrame([feat])[features]
            pred_log = model.predict(X)[0]
            prediction = np.expm1(pred_log)

        st.success("✅ Prediction Completed")

        # =================================================
        # OUTPUT
        # =================================================
        st.markdown("### 📊 Prediction Result")

        st.metric(
            label="Estimated Time to Failure",
            value=f"{prediction:.2f} seconds"
        )

        # Simple risk interpretation
        if prediction < 3:
            st.error("🚨 HIGH RISK — Immediate attention required")
        elif prediction < 7:
            st.warning("⚠️ MEDIUM RISK — Monitor closely")
        else:
            st.success("🟢 LOW RISK — Condition appears stable")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "⚠️ This system is a **decision-support tool** and should be used together "
    "with professional monitoring systems and expert judgment."
)
