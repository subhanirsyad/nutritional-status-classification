# Nutritional Status Classification (Balita) — ANN vs Random Forest

Klasifikasi **status gizi balita** dengan membandingkan performa **Artificial Neural Network (MLPClassifier)** dan **Random Forest**.
Fokus utama: hubungan **tinggi badan** dan **umur** terhadap indikasi stunting.

📄 Paper/Laporan: [`report/paper_nutritional_status_classification.pdf`](./report/paper_nutritional_status_classification.pdf)  
📓 Notebook: [`notebooks/nutritional_status_classification.ipynb`](./notebooks/nutritional_status_classification.ipynb)

---

## Dataset
Dataset (CSV) yang digunakan memiliki fitur utama:
- **Umur (bulan)**
- **Jenis kelamin**
- **Tinggi badan (cm)**
- **Status gizi** (4 kelas: *severely stunting*, *stunting*, *normal*, *tinggi*)

> Catatan: File dataset **tidak disertakan** di repo ini.  
> Silakan letakkan dataset sebagai `data/data_balita.csv` atau sesuaikan path di notebook.

---

## Metodologi (ringkas)
1. **Preprocessing**
   - Cek missing value (tidak ada).
   - Deteksi outlier pada tinggi badan (menghapus outlier).
   - Menangani **imbalance** dengan **SMOTE**.
   - Encoding fitur kategorik; *stunted* dan *severely stunted* digabung dalam satu kelompok (sesuai laporan).

2. **Modeling**
   - **Random Forest Classifier**
   - **ANN (MLPClassifier)**

3. **Tuning**
   - GridSearchCV (cross-validation) untuk mencari hyperparameter terbaik.

4. **Evaluasi**
   - Accuracy
   - AUC (ROC)
   - Confusion Matrix

---

## Hasil
Berdasarkan laporan, model mencapai akurasi klasifikasi hingga **~99%** setelah preprocessing, SMOTE, dan tuning.  
Detail metrik & grafik ada di PDF laporan.

---

## Cara Menjalankan (lokal)
1. Clone repo ini
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Letakkan dataset CSV di:
   - `data/data_balita.csv`
4. Jalankan notebook:
   ```bash
   jupyter notebook
   ```

## Cara Menjalankan (Google Colab)
- Upload notebook `notebooks/nutritional_status_classification.ipynb` ke Colab
- Upload dataset CSV saat diminta (notebook versi awal memakai `files.upload()`)

---

## Struktur Folder
```
nutritional-status-classification/
  notebooks/   # notebook eksperimen
  report/      # laporan/paper
  data/        # taruh dataset di sini (tidak termasuk)
  src/         # (opsional) kalau mau pisahin kode dari notebook
```

---

## Referensi
- WHO (2015) — Stunting in a nutshell
- Kaggle dataset (lihat daftar pustaka di laporan PDF)
