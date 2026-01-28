import streamlit as st
import pandas as pd
import tempfile, os
from gradio_client import Client, file

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Earthquake Early Warning System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Earthquake Early Warning System (EWS)")
st.caption("AI-based Vibration Signal Analysis for Early Earthquake Detection")
st.divider()

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload CSV berisi kolom `acoustic_data`",
    type=["csv"]
)

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            result = client.predict(
                file(tmp_path),        # 🔥 WAJIB pakai file()
                api_name="/predict"
            )

            os.remove(tmp_path)

        # =================================================
        # OUTPUT
        # =================================================
        prediction = float(result)

        st.markdown("### 📊 Prediction Result")
        st.metric(
            label="Estimated Time to Failure",
            value=f"{prediction:.2f} seconds"
        )

        if prediction < 3:
            st.error("🚨 HIGH RISK — Immediate attention required")
        elif prediction < 7:
            st.warning("⚠️ MEDIUM RISK — Monitor closely")
        else:
            st.success("🟢 LOW RISK — Condition appears stable")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
