# Deteksi Komentar Promosi Judi Online dengan Penyamaran Teks

Penelitian ini membandingkan kinerja TF-IDF–Linear SVM dan IndoBERT dalam mendeteksi komentar promosi judi online pada tiga kondisi teks, yaitu normal, penyamaran Unicode (Mathematical Bold), dan substitusi huruf-angka. Kedua model dilatih menggunakan komentar normal, kemudian diuji pada tiga kondisi tersebut tanpa pelatihan ulang. Penyamaran diterapkan pada satu token target dalam setiap komentar promosi yang memenuhi kriteria, sedangkan komentar non-promosi tetap menggunakan teks asli.

## Dataset

*Judi Online: Indonesian Online Gambling Promotion at YouTube* oleh Itsnatt, versi 3, dari Kaggle:
<https://www.kaggle.com/datasets/yaemico/judionline> (lisensi Apache 2.0).

Dataset terdiri atas 10.230 komentar dari 32 video pada 18 channel YouTube. Berkas CSV disimpan di data/raw/deteksi-judi-online.csv tanpa mengubah isi dataset aslinya.

## Struktur

| Berkas | Isi |
|---|---|
| `config.py` | Konfigurasi bersama: seed, blok Unicode, leksikon, pemetaan karakter, parameter model |
| `praproses.py` | Prapemrosesan minimal: huruf A–Z menjadi huruf kecil dan normalisasi spasi |
| `01_persiapan_data.py` | Pemeriksaan kelengkapan, penanganan duplikasi, pemeriksaan penyamaran pada komentar asli, pembagian 70:15:15 |
| `02_kondisi_uji.py` | Leksikon token target dan pembentukan tiga kondisi data uji |
| `03_tfidf_svm.py` | Pelatihan dan prediksi TF-IDF–Linear SVM |
| `notebooks/04_indobert_colab.ipynb` | *Fine-tuning* dan prediksi IndoBERT di Google Colab (GPU T4), lengkap dengan output saat dijalankan ([buka di Colab](https://colab.research.google.com/drive/1zLAverjRDlOrX5zJuET_1pm_UOzFkJSk?usp=sharing)) |
| `05_evaluasi.py` | Precision, recall, F1, confusion matrix, TP→FN, dan Flip Rate |
| `06_analisis_pendukung.py` | Analisis pendukung tanpa pelatihan ulang |
| `data/processed/` | Data hasil persiapan, pembagian, dan kondisi uji |
| `hasil/` | Seluruh keluaran setiap tahap |

## Menjalankan ulang

Python 3.13.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python 01_persiapan_data.py
.venv\Scripts\python 02_kondisi_uji.py
.venv\Scripts\python 03_tfidf_svm.py
```

Langkah 4 dijalankan di Google Colab:
1. Buka [notebook di Colab](https://colab.research.google.com/drive/1zLAverjRDlOrX5zJuET_1pm_UOzFkJSk?usp=sharing), lalu pilih **File → Save a copy in Drive**.
2. Pilih **Runtime → Change runtime type → T4 GPU**, lalu **Runtime → Run all**.
3. Saat diminta, unggah `train.csv`, `val.csv`, dan `test_kondisi.csv` dari `data/processed/`.
4. Simpan tiga berkas yang terunduh ke `hasil/`.

Setelah itu jalankan langkah 5 dan 6:

```powershell
.venv\Scripts\python 05_evaluasi.py
.venv\Scripts\python 06_analisis_pendukung.py
```

Seluruh eksperimen menggunakan seed 42. Proses fine-tuning pada GPU masih memungkinkan adanya sedikit perbedaan hasil saat dijalankan ulang. Namun, pada dua percobaan di Google Colab dengan konfigurasi yang sama, hasil pelatihan dan prediksi yang diperoleh tetap sama.

## Ringkasan hasil

Pembagian data setelah persiapan: 5.154 latih, 1.105 validasi, dan 1.105 uji.
Himpunan evaluasi berpasangan menggunakan 1.072 komentar dari data uji, terdiri atas 170 komentar promosi dan 902 komentar non-promosi.

| Model | Kondisi | Precision | Recall | F1 | TP→FN | Flip Rate |
|---|---|---|---|---|---|---|
| TF-IDF–Linear SVM | Normal | 0,9602 | 0,9941 | 0,9769 | – | – |
| | Penyamaran Unicode | 0,9453 | 0,7118 | 0,8121 | 48 | 28,40% |
| | Substitusi huruf-angka | 0,9453 | 0,7118 | 0,8121 | 48 | 28,40% |
| IndoBERT | Normal | 0,9766 | 0,9824 | 0,9795 | – | – |
| | Penyamaran Unicode | 0,9714 | 0,8000 | 0,8774 | 32 | 19,16% |
| | Substitusi huruf-angka | 0,9726 | 0,8353 | 0,8987 | 25 | 14,97% |

Prediksi pada 902 komentar non-promosi tidak berubah pada kedua kondisi penyamaran.

### IndoBERT sebelum dan setelah fine-tuning

Sebelum fine-tuning, lapisan klasifikasi IndoBERT masih acak, sehingga probabilitas kelas promosi
hanya berkisar 0,33–0,67.

| Tahap | Data | Precision | Recall | F1 |
|---|---|---|---|---|
| Sebelum fine-tuning | Validasi | 0,1535 | 0,6502 | 0,2484 |
| Setelah fine-tuning (epoch 2) | Validasi | 0,9842 | 0,9212 | 0,9517 |
| Sebelum fine-tuning | Uji normal (berpasangan) | 0,1193 | 0,6118 | 0,1996 |
| Setelah fine-tuning | Uji normal (berpasangan) | 0,9766 | 0,9824 | 0,9795 |

Rincian lengkap ada di `hasil/05_*.csv` dan `hasil/06_*.csv`.
