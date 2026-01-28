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

st.divider()

# =========================================================
# ABOUT SYSTEM
# =========================================================
with st.expander("ℹ️ About This System", expanded=True):
    st.markdown("""
**Earthquake Early Warning System (EWS)** menggunakan **Artificial Intelligence**
untuk menganalisis **sinyal getaran (acoustic / vibration data)** dari sensor seismik.

Model dijalankan **secara remote di Hugging Face (Gradio API)** dan mempelajari  
pola **statistik & spektral (FFT)** untuk memprediksi:

> ⏱️ **Estimated Time to Failure (TTF)**

**Teknologi:**
- LightGBM Regression
- Feature Engineering (Statistical + FFT)
- Cloud-based inference (Hugging Face)
""")

# =========================================================
# DATA INPUT DESCRIPTION
# =========================================================
with st.expander("📄 Data Input Description", expanded=True):
    st.markdown("""
### 📥 CSV Requirements
- Wajib memiliki kolom: **`acoustic_data`**
- Setiap baris = satu sinyal getaran
- Tidak memerlukan label

Contoh:

acoustic_data
12
-8
15
-20
""")

st.divider()

# =========================================================
# FILE UPLOADER
# =========================================================
st.subheader("📤 Upload Vibration Data (CSV)")

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

        # Validation
        if "acoustic_data" not in df.columns:
            st.error("❌ Kolom `acoustic_data` tidak ditemukan.")
            st.stop()

        if df.empty:
            st.error("❌ File CSV kosong.")
            st.stop()

        st.success(f"✅ Data loaded ({len(df)} samples)")
        st.dataframe(df.head())

        with st.spinner("🔍 Sending data to AI model..."):
            client = Client("suyagi/earthquakes-try")
        
            import tempfile, os
        
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
        
            result = client.predict(
                tmp_path,
                api_name="/predict"
            )
        
            os.remove(tmp_path)

        # =================================================
        # OUTPUT HANDLING
        # =================================================
        if isinstance(result, (list, tuple)):
            prediction = float(result[0])
        elif isinstance(result, dict):
            prediction = float(list(result.values())[0])
        else:
            prediction = float(result)

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
st.divider()
st.caption(
    "⚠️ This system is a **decision-support tool**. "
    "Predictions should be combined with official seismic monitoring systems."
)

