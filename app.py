import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.feature_extraction import extract_features

# =====================
# Load model & features
# =====================
MODEL_PATH = "lgbm_final.pkl"
FEATURES_PATH = "features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# =====================
# FastAPI app
# =====================
app = FastAPI(
    title="EWS Acoustic AI",
    description="Predict Time to Failure from Acoustic Signal",
    version="1.0"
)

# =====================
# Health check
# =====================
@app.get("/")
def health():
    return {"status": "ok", "message": "EWS AI is running"}

# =====================
# Prediction endpoint
# =====================
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if "acoustic_data" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain 'acoustic_data' column"
        )

    x = df["acoustic_data"].values.astype(np.float32)

    if len(x) < 1000:
        raise HTTPException(
            status_code=400,
            detail="acoustic_data too short"
        )

    # Feature extraction
    feat = extract_features(x)

    # Build dataframe in correct order
    X = pd.DataFrame([feat])[features]

    # Predict (log target → inverse)
    y_log_pred = model.predict(X)[0]
    y_pred = np.expm1(y_log_pred)

    return {
        "prediction_time_to_failure_seconds": float(y_pred),
        "prediction_time_to_failure_minutes": float(y_pred / 60)
    }


# =====================
# Local run (Railway compatible)
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
