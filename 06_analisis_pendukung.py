"""
Langkah 6 - Analisis pendukung. Model tidak dilatih ulang, hanya membaca
data dan hasil prediksi yang sudah ada.

Keluaran: hasil/06_baseline_seluruh_uji.csv      kinerja kondisi normal pada seluruh data uji
          hasil/06_flip_per_kategori.csv         TP->FN per kategori token target
          hasil/06_hampir_duplikat.csv           komentar uji yang hampir sama dengan komentar latih
          hasil/06_sensitivitas_hampir_duplikat.csv
          hasil/06_pengecualian_unicode_per_blok.csv
          hasil/06_panjang_subword.csv           jumlah subword IndoBERT terpanjang per bagian data
          hasil/06_sebelum_setelah_finetuning.csv  kinerja IndoBERT sebelum dan setelah fine-tuning
"""
import re

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import BLOK_HURUF_BERGAYA, INDOBERT_CHECKPOINT, RANDOM_STATE, path
from praproses import praproses

MODEL = {"svm": "TF-IDF + Linear SVM", "indobert": "IndoBERT"}
KONDISI = {"normal": "Normal", "unicode": "Penyamaran Unicode", "angka": "Substitusi huruf-angka"}


def baca(nama):
    return pd.read_csv(path(nama), dtype={"komentar": str}, keep_default_na=False)


def metrik(d, k):
    tn, fp, fn, tp = confusion_matrix(d.label, d[f"pred_{k}"], labels=[0, 1]).ravel()
    p, r, f, _ = precision_recall_fscore_support(d.label, d[f"pred_{k}"], pos_label=1,
                                                 average="binary", zero_division=0)
    return dict(n=len(d), TN=tn, FP=fp, FN=fn, TP=tp, precision=p, recall=r, f1=f)


def flip(promo, k):
    tp_normal = (promo.pred_normal == 1).sum()
    tp_fn = ((promo.pred_normal == 1) & (promo[f"pred_{k}"] == 0)).sum()
    fn_tp = ((promo.pred_normal == 0) & (promo[f"pred_{k}"] == 1)).sum()
    return dict(TP_normal=tp_normal, TP_ke_FN=tp_fn, FN_ke_TP=fn_tp,
                flip_rate_persen=100 * tp_fn / tp_normal if tp_normal else float("nan"))


def normal_kasar(teks):
    """Normalisasi khusus pemeriksaan kemiripan: huruf kecil, hanya a-z dan 0-9."""
    return re.sub(r"[^a-z0-9]", "", praproses(teks))


def main():
    uji = baca("data/processed/test_kondisi.csv")
    uji["masuk_evaluasi"] = uji.masuk_evaluasi.astype(str) == "True"
    pred = {}
    for kode in MODEL:
        d = pd.read_csv(path(f"hasil/pred_{kode}.csv")).drop(columns="masuk_evaluasi")
        assert (d.id_asli.values == uji.id_asli.values).all() and (d.label.values == uji.label.values).all()
        pred[kode] = d.assign(masuk_evaluasi=uji.masuk_evaluasi.values, kategori_token=uji.kategori_token.values)
    out = {}

    # 1) kondisi normal pada seluruh data uji, himpunan berpasangan, dan promosi di luar himpunan
    baris = []
    for kode, d in pred.items():
        for nama, m in (("Seluruh data uji", d.label >= 0), ("Himpunan evaluasi berpasangan", d.masuk_evaluasi),
                        ("Promosi tanpa token target", ~d.masuk_evaluasi)):
            baris.append(dict(model=MODEL[kode], himpunan=nama, **metrik(d[m], "normal")))
    out["06_baseline_seluruh_uji"] = pd.DataFrame(baris)

    # 2) TP->FN per kategori token target
    baris = []
    for kode, d in pred.items():
        promo = d[d.masuk_evaluasi & (d.label == 1)]
        for kat, g in promo.groupby("kategori_token"):
            for k in ("unicode", "angka"):
                baris.append(dict(model=MODEL[kode], kondisi=KONDISI[k], kategori_token=kat, n=len(g), **flip(g, k)))
    out["06_flip_per_kategori"] = pd.DataFrame(baris)

    # 3) komentar uji yang identik dengan komentar latih setelah normalisasi kasar
    train = baca("data/processed/train.csv")
    latih = {}
    for r in train.itertuples():
        latih.setdefault(normal_kasar(r.komentar), r)
    test = baca("data/processed/test.csv")
    dup = [dict(id_asli=r.id_asli, label=r.label, komentar_uji=r.komentar,
                id_latih=latih[normal_kasar(r.komentar)].id_asli,
                komentar_latih=latih[normal_kasar(r.komentar)].komentar,
                label_latih=latih[normal_kasar(r.komentar)].label)
           for r in test.itertuples() if normal_kasar(r.komentar) in latih]
    dup = pd.DataFrame(dup).merge(uji[["id_asli", "masuk_evaluasi"]], on="id_asli")
    out["06_hampir_duplikat"] = dup

    baris = []
    for kode, d in pred.items():
        for nama, m in (("Himpunan evaluasi berpasangan", d.masuk_evaluasi),
                        ("Tanpa komentar hampir-duplikat", d.masuk_evaluasi & ~d.id_asli.isin(dup.id_asli))):
            x = d[m]
            promo = x[x.label == 1]
            for k, kn in KONDISI.items():
                rec = dict(model=MODEL[kode], himpunan=nama, kondisi=kn, **metrik(x, k))
                if k != "normal":
                    rec.update(flip(promo, k))
                baris.append(rec)
    out["06_sensitivitas_hampir_duplikat"] = pd.DataFrame(baris)

    # 4) komentar yang dikeluarkan karena huruf Unicode bergaya, per blok
    kel = baca("hasil/01_data_dikeluarkan.csv")
    kel = kel[kel.alasan.str.startswith("Memuat huruf")]
    blok = kel.komentar.map(lambda t: {n for c in t for a, b, n in BLOK_HURUF_BERGAYA if a <= ord(c) <= b})
    baris = [dict(blok=n, label=lab,
                  komentar_memuat=sum(n in s for s, l in zip(blok, kel.label) if l == lab),
                  hanya_blok_ini=sum(s == {n} for s, l in zip(blok, kel.label) if l == lab))
             for _, _, n in BLOK_HURUF_BERGAYA for lab in (0, 1)]
    out["06_pengecualian_unicode_per_blok"] = (pd.DataFrame(baris).query("komentar_memuat > 0")
                                               .sort_values("komentar_memuat", ascending=False))

    # 5) panjang subword IndoBERT terpanjang (batas MAX_LEN 128 dikurangi [CLS] dan [SEP])
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(INDOBERT_CHECKPOINT)
        teks = {"Latih": train.komentar.map(praproses), "Validasi": baca("data/processed/val.csv").komentar.map(praproses)}
        teks.update({f"Uji - {kn}": uji[f"teks_{k}"] for k, kn in KONDISI.items()})
        out["06_panjang_subword"] = pd.DataFrame([dict(bagian=n, jumlah_komentar=len(t),
                                                       subword_terpanjang=max(len(tok.tokenize(x)) for x in t),
                                                       batas=126) for n, t in teks.items()])
    except Exception as e:  # tokenizer gagal diunduh
        print("Pemeriksaan panjang subword dilewati:", e)

    # 6) IndoBERT sebelum fine-tuning sebagai pembanding. Bobot encoder dari checkpoint,
    #    lapisan klasifikasi masih acak (diinisialisasi dengan seed 42).
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(RANDOM_STATE)
        tok = AutoTokenizer.from_pretrained(INDOBERT_CHECKPOINT)
        bert = AutoModelForSequenceClassification.from_pretrained(INDOBERT_CHECKPOINT, num_labels=2).eval()

        @torch.no_grad()
        def prob_promosi(teks):
            hasil = []
            for i in range(0, len(teks), 64):
                e = tok(teks[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
                hasil.append(torch.softmax(bert(**e).logits, 1)[:, 1])
            return torch.cat(hasil).numpy()

        val = baca("data/processed/val.csv")
        berpasangan = uji[uji.masuk_evaluasi]
        setelah = pred["indobert"][pred["indobert"].masuk_evaluasi]
        log = pd.read_csv(path("hasil/indobert_log.csv"))
        terbaik = log.loc[log.val_f1.idxmax()]

        baris = []
        for nama, label, teks in (("Validasi", val.label, val.komentar.map(praproses)),
                                  ("Uji normal (himpunan berpasangan)", berpasangan.label, berpasangan.teks_normal)):
            prob = prob_promosi(teks.tolist())
            d = pd.DataFrame({"label": label.values, "pred_normal": (prob >= 0.5).astype(int)})
            baris.append(dict(tahap="Sebelum fine-tuning", data=nama, **metrik(d, "normal"),
                              prob_min=prob.min(), prob_maks=prob.max()))
        baris.insert(1, dict(tahap="Setelah fine-tuning", data=f"Validasi (epoch {int(terbaik.epoch)})",
                             n=len(val), precision=terbaik.val_precision, recall=terbaik.val_recall,
                             f1=terbaik.val_f1))
        baris.append(dict(tahap="Setelah fine-tuning", data="Uji normal (himpunan berpasangan)",
                          **metrik(setelah, "normal"),
                          prob_min=setelah.prob_normal.min(), prob_maks=setelah.prob_normal.max()))
        out["06_sebelum_setelah_finetuning"] = pd.DataFrame(baris).astype(
            {k: "Int64" for k in ("TN", "FP", "FN", "TP")})
    except Exception as e:  # torch atau checkpoint tidak tersedia
        print("Perbandingan sebelum fine-tuning dilewati:", e)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.float_format", "{:.4f}".format)
    for nama, df in out.items():
        df.to_csv(path(f"hasil/{nama}.csv"), index=False, encoding="utf-8")
        tampil = df.drop(columns=["komentar_uji", "komentar_latih"], errors="ignore")
        print(f"\n== {nama} ==\n{tampil.to_string(index=False)}")


if __name__ == "__main__":
    main()
