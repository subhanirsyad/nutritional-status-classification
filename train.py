from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "data_balita.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_model() -> Pipeline:
    cat_features = ["Jenis Kelamin"]
    num_features = ["Umur (bulan)", "Tinggi Badan (cm)"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", SimpleImputer(strategy="median"), num_features),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Binarize label: stunted vs tidak stunted (sesuai notebook)
    df["Status Gizi"] = df["Status Gizi"].replace(
        {
            "tinggi": "tidak stunted",
            "normal": "tidak stunted",
            "stunted": "stunted",
            "severely stunted": "stunted",
        }
    )

    X = df[["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]]
    y = (df["Status Gizi"] == "stunted").astype(int)  # 1=stunted

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))

    report = classification_report(
        y_test, y_pred, target_names=["tidak stunted", "stunted"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "train_test_split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "notes": "Label: 1=stunted, 0=tidak stunted. Pipeline: OneHotEncoder(gender) + RandomForest.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model  : {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Accuracy: {acc:.6f} | ROC-AUC: {auc:.6f}")


if __name__ == "__main__":
    main()
