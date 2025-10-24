
import os, pandas as pd, numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load

st.set_page_config(page_title="Khulna Rainfall–Runoff Predictor", page_icon="🌧️", layout="wide")
st.title("🌧️ Khulna Rainfall–Runoff Predictor")

MODEL_PATH = "models/LinearRegression_2023_2025.joblib"
FEATURES = ["rain_mm", "temp_c", "pet_mm", "api7_mm", "smi", "doy", "month"]

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

model = load_model()

st.sidebar.header("Data")
mode = st.sidebar.radio("Pick input:", ["Sample 2026 features", "Upload CSV"])

if mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload features CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
    else:
        st.stop()
else:
    df = pd.read_csv("data/features_2026.csv")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

pred = model.predict(df[FEATURES])
out = df.copy()
out["runoff_pred_mm"] = pred

st.subheader("Preview")
st.dataframe(out.head(20), use_container_width=True)

st.subheader("Hydrograph")
plt.figure()
x = out["date"] if "date" in out.columns else out.index
plt.plot(x, out["runoff_pred_mm"], label="Predicted runoff (mm)")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Runoff (mm)")
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())

st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "runoff_predictions.csv", "text/csv")
