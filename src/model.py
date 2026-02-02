from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "model.joblib"


@dataclass(frozen=True)
class PredictionResult:
    label: str
    proba_stunted: float
    proba_tidak_stunted: float


def _normalize_gender(gender: str) -> str:
    g = (gender or "").strip().lower()
    # allow common variants
    if g in {"laki-laki", "laki laki", "male", "m", "l"}:
        return "laki-laki"
    if g in {"perempuan", "female", "f", "p"}:
        return "perempuan"
    return gender  # let the model's OneHotEncoder handle unknowns


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to accept a few common column name variants."""
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"umur", "umur (bulan)", "umur_bulan", "age_months", "age (months)"}:
            colmap[c] = "Umur (bulan)"
        elif cl in {"jenis kelamin", "gender", "sex"}:
            colmap[c] = "Jenis Kelamin"
        elif cl in {"tinggi badan (cm)", "tinggi badan", "tinggi", "height_cm", "height (cm)"}:
            colmap[c] = "Tinggi Badan (cm)"
    df = df.rename(columns=colmap)
    return df


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {model_path}. Jalankan `python train.py` untuk membuatnya."
        )
    return joblib.load(model_path)


def predict_single(
    umur_bulan: float,
    jenis_kelamin: str,
    tinggi_cm: float,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> PredictionResult:
    model = load_model(model_path)

    df = pd.DataFrame(
        [{
            "Umur (bulan)": float(umur_bulan),
            "Jenis Kelamin": _normalize_gender(jenis_kelamin),
            "Tinggi Badan (cm)": float(tinggi_cm),
        }]
    )

    proba = model.predict_proba(df)[0]
    # We train with y=1 as "stunted"
    proba_stunted = float(proba[1])
    proba_tidak = float(proba[0])

    label = "stunted" if proba_stunted >= 0.5 else "tidak stunted"

    return PredictionResult(
        label=label,
        proba_stunted=proba_stunted,
        proba_tidak_stunted=proba_tidak,
    )


def predict_dataframe(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Predict for a batch DataFrame. Returns a copy with:
    - pred_label: 'stunted' / 'tidak stunted'
    - proba_stunted: probability of stunted
    - proba_tidak_stunted: probability of tidak stunted
    Also returns basic counts.
    """
    model = load_model(model_path)

    df2 = _normalize_columns(df.copy())
    required = {"Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"}
    missing = required - set(df2.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    df2["Jenis Kelamin"] = df2["Jenis Kelamin"].astype(str).map(_normalize_gender)

    proba = model.predict_proba(df2[list(required)])
    proba_tidak = proba[:, 0].astype(float)
    proba_stunted = proba[:, 1].astype(float)

    pred_label = np.where(proba_stunted >= 0.5, "stunted", "tidak stunted")

    out = df.copy()
    out["pred_label"] = pred_label
    out["proba_stunted"] = proba_stunted
    out["proba_tidak_stunted"] = proba_tidak

    counts = pd.Series(pred_label).value_counts().to_dict()
    return out, {k: int(v) for k, v in counts.items()}
