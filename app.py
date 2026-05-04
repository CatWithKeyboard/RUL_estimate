from __future__ import annotations

from pathlib import Path
import tempfile
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import IMSDataLoader, IMS_CONFIG, parse_ims_timestamp
from feature_engineering import extract_features


st.set_page_config(page_title="Bearing RUL Estimator", layout="wide")
st.title("Bearing Remaining Useful Life Estimator")
st.caption("Upload IMS/NASA bearing vibration snapshots and estimate remaining life.")

model_path = st.sidebar.text_input("Model path", value="artifacts/bearing_rul_model.joblib")
dataset = st.sidebar.selectbox("Dataset layout", ["set1", "set2", "set3"])
bearing_id = st.sidebar.selectbox("Bearing ID", sorted(IMS_CONFIG[dataset]["bearing_channels"].keys()))

uploaded_files = st.file_uploader(
    "Upload one or more vibration snapshot files",
    accept_multiple_files=True,
    type=None,
)

if uploaded_files:
    if not Path(model_path).exists():
        st.error("Model file not found. Train the model first using train.py.")
    else:
        model = joblib.load(model_path)
        tmp_dir = Path(tempfile.mkdtemp())
        rows = []
        loader = IMSDataLoader(tmp_dir)
        expected_channels = 8 if dataset == "set1" else 4

        for uploaded in uploaded_files:
            dest = tmp_dir / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            df = loader.load_signal_file(dest, expected_channels=expected_channels)
            signal = loader.get_bearing_signal(df, dataset, bearing_id, combine_channels=True)
            feats = extract_features(signal)
            rows.append({
                "timestamp": parse_ims_timestamp(uploaded.name),
                "file_name": uploaded.name,
                **feats,
            })

        pred_df = pd.DataFrame(rows).sort_values("timestamp")
        feature_cols = [c for c in pred_df.columns if c not in {"timestamp", "file_name"}]
        pred_df["predicted_rul_hours"] = model.predict(pred_df[feature_cols])
        pred_df["predicted_rul_days"] = pred_df["predicted_rul_hours"] / 24.0

        latest = pred_df.iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Latest predicted RUL (hours)", f"{latest['predicted_rul_hours']:.2f}")
        c2.metric("Latest predicted RUL (days)", f"{latest['predicted_rul_days']:.2f}")

        fig = px.line(pred_df, x="timestamp", y="predicted_rul_hours", markers=True, title="Predicted RUL Trend")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predictions")
        st.dataframe(pred_df[["timestamp", "file_name", "predicted_rul_hours", "predicted_rul_days"]], use_container_width=True)

        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="bearing_rul_predictions.csv", mime="text/csv")
else:
    st.info("Upload IMS bearing snapshot files to estimate remaining useful life.")
