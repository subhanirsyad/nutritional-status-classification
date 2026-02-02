# Nutritional Status Classification (Balita) — Streamlit Inference (Stunting)

Klasifikasi **status gizi balita** berbasis **umur (bulan)**, **jenis kelamin**, dan **tinggi badan (cm)**.
Repo ini sudah disiapkan untuk:
- ✅ **Inference lewat Streamlit** (langsung jalan)
- ✅ **Siap di-push ke GitHub**
- ✅ **Siap deploy ke Streamlit Community Cloud** (nanti dapat link `.streamlit.app`)

📄 Paper/Laporan: [`report/paper_nutritional_status_classification.pdf`](./report/paper_nutritional_status_classification.pdf)  
📓 Notebook eksperimen: [`notebooks/nutritional_status_classification.ipynb`](./notebooks/nutritional_status_classification.ipynb)

---

## Demo Streamlit (lokal)

1) Buat virtual env (opsional tapi disarankan)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Jalankan aplikasi
```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser dan bisa:
- Prediksi 1 data (form input)
- Prediksi batch (upload CSV dan download hasil)

---

## Deploy ke Streamlit Community Cloud (dapat link sendiri)

> Setelah deploy, kamu akan mendapatkan link seperti:  
> `https://<nama-app>.streamlit.app`

Langkah ringkas:
1. Push repo ini ke GitHub (public/private).
2. Buka Streamlit Community Cloud dan pilih **Create app**.
3. Pilih repo + branch.
4. Isi **Main file path**: `streamlit_app.py`
5. Deploy.

---

## Dataset

Dataset ada di: [`data/data_balita.csv`](./data/data_balita.csv)

Fitur:
- **Umur (bulan)**
- **Jenis Kelamin** (`laki-laki` / `perempuan`)
- **Tinggi Badan (cm)**
- **Status Gizi** (asli 4 kelas: `severely stunted`, `stunted`, `normal`, `tinggi`)

Untuk inference, label digabung menjadi 2 kelas (sesuai notebook):
- **stunted**: `stunted` + `severely stunted`
- **tidak stunted**: `normal` + `tinggi`

---

## Model

Model inference default: **RandomForestClassifier** (scikit-learn) dalam pipeline preprocessing:
- `OneHotEncoder` untuk `Jenis Kelamin`
- `Imputer` untuk numerik (median)
- RandomForest (`n_estimators=200`, `class_weight=balanced_subsample`, `random_state=42`)

Metrik (hasil split test 20%, stratify, random_state=42):
- **Accuracy**: `0.999669`
- **ROC-AUC**: `0.999999`

Detail metrik tersimpan di [`models/metrics.json`](./models/metrics.json).  
Model terlatih tersimpan di [`models/model.joblib`](./models/model.joblib).

---

## Re-train model (opsional)

Kalau kamu ingin melatih ulang model dan memperbarui file `models/`:
```bash
pip install -r requirements.txt
python train.py
```

---

## Struktur Folder

```
.
├─ streamlit_app.py          # Streamlit UI (inference)
├─ train.py                  # Training script -> models/model.joblib
├─ src/
│  └─ model.py               # helper load & predict
├─ models/
│  ├─ model.joblib
│  └─ metrics.json
├─ data/
│  ├─ data_balita.csv
│  └─ README.md
├─ notebooks/
├─ report/
├─ requirements.txt          # untuk deploy (minimal)
├─ requirements-dev.txt      # optional (notebook/plotting)
└─ LICENSE
```

---

## Lisensi
Lihat file [`LICENSE`](./LICENSE).
