import streamlit as st
import pandas as pd
from gradio_client import Client

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Earthquake Early Warning System",
    page_icon="🌍",
    layout="centered"
)

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

Model dijalankan secara **remote di Hugging Face** dan mempelajari  
**pola statistik & spektral (FFT)** dari sinyal getaran untuk memprediksi:

> ⏱️ **Estimated Time to Failure**

💡 **Tujuan sistem:**
- Deteksi dini potensi gempa
- Memberikan peringatan lebih awal
- Mendukung mitigasi risiko bencana

⚙️ **Model:**
- LightGBM Regression
- Feature Engineering (Statistical + FFT)
- Deployed on Hugging Face (Gradio)
""")

# =========================================================
# DATA DESCRIPTION
# =========================================================
with st.expander("📄 Data Input Description", expanded=True):
    st.markdown("""
### 📥 Format Data yang Diperlukan

Upload file **CSV** dengan ketentuan:

- Kolom wajib: **`acoustic_data`**
- Setiap baris = satu sinyal getaran
- Data berasal dari:
  - Sensor seismik
  - Accelerometer
  - Acoustic / vibration sensor

### Contoh Struktur CSV
acoustic_data
12
-8
15
-20
...

📌 **Catatan:**
- Tidak memerlukan label
- Semakin panjang sinyal → prediksi lebih stabil
- Seluruh proses feature extraction dilakukan oleh AI
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
        # basic validation (optional, ringan)
        df = pd.read_csv(uploaded_file)
        if "acoustic_data" not in df.columns:
            st.error("❌ Kolom `acoustic_data` tidak ditemukan di file CSV.")
            st.stop()

        with st.spinner("🔍 Sending data to AI model..."):
            client = Client("suyagi/earthquakes-try")

            # kirim file langsung ke Hugging Face
            prediction = client.predict(
                uploaded_file,
                api_name="/predict"
            )

        st.success("✅ Prediction Completed")

        # =================================================
        # OUTPUT
        # =================================================
        st.markdown("### 📊 Prediction Result")

        st.metric(
            label="Estimated Time to Failure",
            value=f"{prediction:.2f} seconds"
        )

        # Risk interpretation
        if prediction < 3:
            st.error("🚨 HIGH RISK — Immediate attention required")
        elif prediction < 7:
            st.warning("⚠️ MEDIUM RISK — Monitor closely")
        else:
            st.success("🟢 LOW RISK — Condition appears stable")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "⚠️ This system is a **decision-support tool**. "
    "Predictions should be combined with official seismic monitoring systems."
)
