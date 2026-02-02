from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import predict_single, predict_dataframe, DEFAULT_MODEL_PATH


st.set_page_config(
    page_title="Klasifikasi Status Gizi Balita (Stunting)",
    page_icon="🧒",
    layout="centered",
)

st.title("🧒 Klasifikasi Status Gizi Balita (Stunting)")
st.caption("Input umur, jenis kelamin, dan tinggi badan untuk memprediksi **stunted** vs **tidak stunted**.")

with st.expander("ℹ️ Tentang model", expanded=False):
    st.markdown(
        """
        Model ini dilatih dari dataset `data/data_balita.csv` dengan label digabung menjadi 2 kelas:
        - **stunted**: `stunted` + `severely stunted`
        - **tidak stunted**: `normal` + `tinggi`

        Algoritma: **RandomForestClassifier** (scikit-learn) dalam pipeline preprocessing.
        """
    )
    metrics_path = Path(__file__).resolve().parent / "models" / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.write(
                {
                    "accuracy (test)": round(metrics.get("accuracy", 0.0), 6),
                    "roc_auc (test)": round(metrics.get("roc_auc", 0.0), 6),
                }
            )
        except Exception:
            st.info("metrics.json ditemukan, tapi tidak bisa dibaca.")

st.subheader("Prediksi satu data")
col1, col2, col3 = st.columns(3)
with col1:
    umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=12, step=1)
with col2:
    jk = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
with col3:
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=128.0, value=75.0, step=0.1)

if st.button("Prediksi", type="primary"):
    try:
        res = predict_single(umur, jk, tinggi, model_path=DEFAULT_MODEL_PATH)
        if res.label == "stunted":
            st.error(f"Prediksi: **STUNTED** (P={res.proba_stunted:.3f})")
        else:
            st.success(f"Prediksi: **TIDAK STUNTED** (P(stunted)={res.proba_stunted:.3f})")

        st.write(
            {
                "probability_stunted": round(res.proba_stunted, 6),
                "probability_tidak_stunted": round(res.proba_tidak_stunted, 6),
            }
        )
    except FileNotFoundError as e:
        st.warning(str(e))
        st.info("Solusi: jalankan `python train.py` untuk membuat model, lalu jalankan Streamlit lagi.")
    except Exception as e:
        st.exception(e)

st.divider()

st.subheader("Prediksi batch (upload CSV)")
st.caption("CSV minimal berisi kolom: `Umur (bulan)`, `Jenis Kelamin`, `Tinggi Badan (cm)` (boleh pakai variasi nama kolom umum).")

uploaded = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head(10))

        out_df, counts = predict_dataframe(df, model_path=DEFAULT_MODEL_PATH)
        st.success(f"Selesai. Ringkas: {counts}")

        st.write(out_df.head(20))

        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button(
            "Download hasil (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.exception(e)

st.divider()
st.caption("Made with Streamlit • Repo-ready for GitHub • Jalankan lokal: `streamlit run streamlit_app.py`")
