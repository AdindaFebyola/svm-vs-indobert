# Deteksi Komentar Promosi Judi Online dengan Penyamaran Teks

Kode eksperimen skripsi **"Perbandingan Kinerja TF-IDF–Linear SVM dan IndoBERT pada Deteksi Komentar
Promosi Judi Online di YouTube Sebelum dan Setelah Penyamaran Unicode dan Substitusi Huruf-Angka"**.

Adinda Febyola (235150707111055), Program Studi Teknologi Informasi, Fakultas Ilmu Komputer,
Universitas Brawijaya.

Kedua model dilatih pada komentar normal, lalu diuji pada tiga kondisi data uji yang sama:
normal, penyamaran Unicode (huruf Mathematical Bold), dan substitusi huruf-angka.
Pada setiap komentar promosi, satu token target disamarkan pada posisi karakter yang sama untuk
kedua kondisi penyamaran, sedangkan komentar non-promosi tidak diubah.

## Dataset

*Judi Online: Indonesian Online Gambling Promotion at YouTube* oleh Itsnatt, versi 3, dari Kaggle:
<https://www.kaggle.com/datasets/yaemico/judionline> (lisensi Apache 2.0).

Berkas asli `data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv` disimpan tanpa
perubahan isi di `data/raw/deteksi-judi-online.csv`. Data ini berisi 10.230 komentar dari 32 video
pada 18 channel YouTube.

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

Seluruh pengacakan memakai seed 42. Fine-tuning IndoBERT pada GPU tidak sepenuhnya deterministik,
sehingga hasil run ulang dapat sedikit berbeda. Dua kali run di Colab dengan konfigurasi yang sama
menghasilkan nilai loss dan metrik validasi setiap epoch serta prediksi pada himpunan evaluasi
berpasangan yang sama persis.

## Ringkasan hasil

Pembagian data setelah persiapan: 5.154 latih, 1.105 validasi, dan 1.105 uji.
Himpunan evaluasi berpasangan terdiri atas 1.072 komentar uji, yaitu 170 promosi dan 902 non-promosi.

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
