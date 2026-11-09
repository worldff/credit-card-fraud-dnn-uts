# Credit Card Fraud Detection using Deep Neural Network

Proyek Mata Kuliah Pembelajaran Mendalam (Semester Ganjil TA 2026/2027): Tugas Proyek 1,
dilanjutkan dan disempurnakan untuk Ujian Tengah Semester (UTS).

Model DNN fully-connected untuk mengklasifikasikan transaksi kartu kredit sebagai **normal** atau
**fraud** pada dataset yang sangat tidak seimbang (fraud hanya 0,173%). Dua fitur unik:

1. **Optimasi threshold keputusan** — threshold dipilih dari data *validation* dengan
   memaksimalkan F1-score, lalu dipakai apa adanya pada data *testing* dan aplikasi.
2. **Kompresi model ke TensorFlow Lite** (dynamic-range int8): 10 KB, keputusan sama dengan
   model Keras pada 42.558 dari 42.559 transaksi uji, dipakai oleh aplikasi Streamlit.

Tiga strategi fungsi loss dibandingkan (BCE, BCE berbobot kelas, Focal Loss; 3 seed) dan
model final dipilih otomatis dari data validation (`select_final.py`).

## Penyempurnaan untuk UTS (dari Tugas Proyek 1)

Model, data, dan angka evaluasi tetap sama dengan Tugas Proyek 1. Penyempurnaan dilakukan pada
sisi sistem agar model dapat dipakai dengan aman di aplikasi:

| Penyempurnaan | Implementasi | Bukti |
|---|---|---|
| Validasi input dan deteksi input di luar domain data latih | `src/input_domain.py`, `outputs/models/input_domain.json` | 6 dari 42.559 transaksi uji (0,014%) ditandai; contoh Amount 60.000 EUR dan V14 = -40 ditandai, Amount negatif ditolak (`outputs/results/input_domain_eval.json`) |
| Penjelasan prediksi per transaksi (occlusion sensitivity) | `src/explain.py` | V14 menjadi fitur paling berpengaruh pada 41 dari 46 fraud yang terdeteksi model TFLite (`outputs/results/explain_eval.json`) |
| Pengaturan threshold interaktif dengan estimasi precision/recall dari data validation | `app/app.py` | Pada threshold optimal: precision 95,0%, recall 80,3% di data validation |
| Aplikasi siap di-deploy ke Streamlit Community Cloud (deploy ditunda sesuai arahan dosen) | `app/app.py`, `app/requirements.txt` (LiteRT, tanpa TensorFlow penuh) | diuji di lingkungan bersih tanpa TensorFlow; lihat bagian *Menjalankan aplikasi* |

## Hasil model final (data testing, threshold 0,1276)

| Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 0,9375 | 0,6338 | 0,7563 | 0,8073 | 0,9724 |

## Dataset

[Credit Card Fraud Detection — ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284.807 transaksi, 31 kolom (`Time`, `V1`–`V28` hasil PCA, `Amount`, `Class`).

Dataset yang dipakai di notebook dan laporan **disertakan di repositori ini** sebagai file asli
Kaggle (`archive.zip`) yang dipecah menjadi tiga bagian agar setiap file < 25 MB:
`data/creditcard_kaggle.zip.part1`, `part2`, `part3`. Gabungkan dan ekstrak dengan:

```bash
python src/get_dataset.py
```

Skrip menghasilkan `data/creditcard.csv` (150.828.752 byte) dan memeriksa checksum MD5
`e90efcb83d69faf99fcab8b0255024de` agar dipastikan sama persis dengan file asli Kaggle.
Notebook menjalankan langkah ini secara otomatis.

## Pemetaan langkah RTM ke implementasi

| Langkah RTM | Implementasi |
|---|---|
| 3. Eksplorasi & praproses (pembersihan, normalisasi, split latih/validasi/uji) | `src/eda.py`, `src/preprocessing.py` |
| 4. Input pipeline `tf.data` + arsitektur MLP | `src/data_pipeline.py`, `src/model.py` |
| 5. Loss function, dropout, batch normalization, Adam/RMSProp, LR scheduling, early stopping | `src/model.py`, `src/train.py` |
| 6. Evaluasi (accuracy, precision, recall, F1, confusion matrix) + iterasi perbaikan | `src/evaluate.py`, `src/select_final.py`, `src/plots.py` |
| 7. Dokumentasi arsitektur, hyperparameter, hasil eksperimen | Laporan PDF, `notebooks/`, `outputs/results/` |
| Luaran (c) notebook + instruksi + anggota | `notebooks/Proyek_Deteksi_Fraud_DNN.ipynb`, README ini |
| Luaran (d) source code lengkap | `src/`, `app/` |

## Anggota kelompok

Kelompok **Blueprints** — Kelas **IF B Sore**

| NIM | Nama | Peran |
|---|---|---|
| 231112189 | Nanda Fachriza Artin | Ketua & ML Engineer |
| 231110155 | Arman Maulana | Data Analyst |
| 231110255 | Nyoto Prasetyo | Data Engineer |
| 231112549 | Mhd. Abi Manyu | Evaluator |
| 231110656 | Celvin | App Developer & Deployment |

## Struktur repositori

```
├── app/app.py              # Aplikasi Streamlit (inferensi, model TFLite)
├── app/requirements.txt    # dependensi aplikasi untuk deploy (LiteRT, tanpa TensorFlow)
├── notebooks/              # notebook Colab yang menjalankan seluruh pipeline
├── data/creditcard_kaggle.zip.part1–3  # dataset asli Kaggle (dipecah < 25 MB)
├── data/contoh_transaksi_uji.csv  # 20 transaksi data uji untuk demo aplikasi
├── outputs/
│   ├── figures/            # grafik EDA & evaluasi
│   ├── models/             # model .keras, scaler, threshold
│   └── results/            # metrik (JSON), riwayat training, log
├── src/
│   ├── config.py           # konfigurasi & hyperparameter
│   ├── get_dataset.py      # menggabungkan & memverifikasi dataset
│   ├── preprocessing.py    # pembersihan, rekayasa fitur, split, scaling
│   ├── data_pipeline.py    # pipeline input tf.data
│   ├── eda.py              # exploratory data analysis
│   ├── model.py            # arsitektur DNN + Binary Focal Loss
│   ├── evaluate.py         # metrik + optimasi threshold (fitur unik #1)
│   ├── train.py            # pelatihan semua skenario
│   ├── select_final.py     # pemilihan model final dari data validation
│   ├── export_tflite.py    # kompresi TFLite (fitur unik #2)
│   ├── error_analysis.py   # analisis TP/FN/FP model final
│   ├── input_domain.py     # UTS: validasi input & deteksi di luar domain
│   ├── explain.py          # UTS: penjelasan prediksi (occlusion)
│   ├── make_sample.py      # membuat contoh data uji untuk demo
│   └── plots.py            # grafik hasil evaluasi
└── requirements.txt
```

## Cara menjalankan

**Cara termudah (Google Colab):**

1. Buka [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook*
   → pilih `notebooks/Proyek_Deteksi_Fraud_DNN.ipynb`.
2. *Runtime → Run all*. Notebook mengambil kode dengan salah satu cara: clone dari GitHub
   (isi `REPO_URL` di sel pertama), file `Source_Code_Credit_Card_Fraud_DNN.zip` di Google Drive,
   atau unggah zip tersebut saat diminta.
3. Dataset otomatis disiapkan dari `data/creditcard_kaggle.zip.part1–3` (salinan asli Kaggle).
   Bila bagian tersebut tidak ada, notebook mencarinya di Google Drive, mencoba mengunduh dari
   Kaggle (`kagglehub`), atau meminta Anda mengunggah `creditcard.csv`.
4. Tidak perlu `pip install`: TensorFlow, scikit-learn, pandas, dan matplotlib sudah ada di Colab.
   Pelatihan 14 run ±20–25 menit di CPU. Set `LATIH_ULANG = False` untuk memakai model yang
   sudah dilatih (±1 menit).

**Lokal (terminal):**

```bash
pip install -r requirements.txt
python src/get_dataset.py   # menyiapkan data/creditcard.csv
cd src
python eda.py      # grafik & ringkasan EDA
python train.py          # melatih 14 run (8 skenario, beberapa seed)
python select_final.py   # pilih model final + threshold
python export_tflite.py  # kompresi TFLite
python error_analysis.py
python plots.py          # grafik evaluasi
python input_domain.py   # UTS: rentang data latih untuk validasi input
python explain.py        # UTS: analisis penjelasan prediksi (occlusion)
cd ..
streamlit run app/app.py
```

## Menjalankan aplikasi

Aplikasi Streamlit dijalankan secara lokal dengan `streamlit run app/app.py`, lalu dibuka di
browser (`http://localhost:8501`). Untuk mencoba, centang **Gunakan contoh data uji** pada tab
*Unggah CSV*, atau pilih salah satu contoh pada tab *Input manual*. Repositori ini juga dapat
di-deploy langsung ke Streamlit Community Cloud dengan main file path `app/app.py`.

Semua angka di laporan berasal dari `outputs/results/all_results.json` (seed 42 untuk model final).
