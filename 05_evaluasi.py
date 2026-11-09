"""
Langkah 5 - Evaluasi kedua model pada tiga kondisi pengujian.

Masukan : hasil/pred_svm.csv, hasil/pred_indobert.csv (dari Colab)
Keluaran: hasil/05_metrik.csv          precision, recall, F1, confusion matrix
          hasil/05_delta.csv           selisih kondisi normal - kondisi penyamaran
          hasil/05_flip.csv            TP->FN, FN->TP, Flip Rate
          hasil/05_flip_per_token.csv  TP->FN per token target
          hasil/05_tokenisasi_indobert.csv
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import INDOBERT_CHECKPOINT, path

KONDISI = {"normal": "Normal", "unicode": "Penyamaran Unicode", "angka": "Substitusi huruf-angka"}
MODEL = {"svm": "TF-IDF + Linear SVM", "indobert": "IndoBERT"}


def muat(uji):
    """Himpunan evaluasi berpasangan ditentukan oleh test_kondisi.csv, bukan oleh berkas prediksi."""
    hasil = {}
    for kode in MODEL:
        f = path(f"hasil/pred_{kode}.csv")
        if f.exists():
            d = pd.read_csv(f).drop(columns="masuk_evaluasi")
            assert (d.id_asli.values == uji.id_asli.values).all() and (d.label.values == uji.label.values).all()
            hasil[kode] = d[uji.masuk_evaluasi.values].reset_index(drop=True)
        else:
            print(f"[lewati] {f.name} belum ada")
    return hasil


def main():
    uji = pd.read_csv(path("data/processed/test_kondisi.csv"), keep_default_na=False)
    uji["masuk_evaluasi"] = uji.masuk_evaluasi.astype(str) == "True"
    data = muat(uji)
    uji = uji[uji.masuk_evaluasi]
    metrik, delta, flip, per_tok = [], [], [], []

    for kode, d in data.items():
        d = d.merge(uji[["id_asli", "token_target", "kategori_token"]], on="id_asli")
        baris = {}
        for k, nama in KONDISI.items():
            tn, fp, fn, tp = confusion_matrix(d.label, d[f"pred_{k}"], labels=[0, 1]).ravel()
            p, r, f, _ = precision_recall_fscore_support(d.label, d[f"pred_{k}"], pos_label=1,
                                                         average="binary", zero_division=0)
            baris[k] = dict(model=MODEL[kode], kondisi=nama, TN=tn, FP=fp, FN=fn, TP=tp,
                            precision=p, recall=r, f1=f)
            metrik.append(baris[k])

        for k in ("unicode", "angka"):
            delta.append(dict(model=MODEL[kode], kondisi=KONDISI[k],
                              **{f"delta_{m}": baris["normal"][m] - baris[k][m] for m in ("precision", "recall", "f1")}))

        promo = d[d.label == 1]
        non = d[d.label == 0]
        tp_normal = (promo.pred_normal == 1).sum()
        for k in ("unicode", "angka"):
            tp_fn = ((promo.pred_normal == 1) & (promo[f"pred_{k}"] == 0)).sum()
            fn_tp = ((promo.pred_normal == 0) & (promo[f"pred_{k}"] == 1)).sum()
            flip_rate = 100 * tp_fn / tp_normal
            # teks non-promosi tidak diubah, jadi prediksinya harus tetap sama
            berubah_non = (non.pred_normal != non[f"pred_{k}"]).sum()
            flip.append(dict(model=MODEL[kode], kondisi=KONDISI[k], TP_normal=tp_normal,
                             TP_ke_FN=tp_fn, FN_ke_TP=fn_tp, flip_rate_persen=flip_rate,
                             perubahan_non_promosi=berubah_non))
            g = promo.assign(flip=(promo.pred_normal == 1) & (promo[f"pred_{k}"] == 0),
                             tp=promo.pred_normal == 1)
            t = g.groupby(["kategori_token", "token_target"]).agg(n=("flip", "size"), TP_normal=("tp", "sum"),
                                                                  TP_ke_FN=("flip", "sum")).reset_index()
            t.insert(0, "kondisi", KONDISI[k])
            t.insert(0, "model", MODEL[kode])
            per_tok.append(t)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    out = {"05_metrik": pd.DataFrame(metrik), "05_delta": pd.DataFrame(delta), "05_flip": pd.DataFrame(flip)}
    if per_tok:
        out["05_flip_per_token"] = pd.concat(per_tok).sort_values(["model", "kondisi", "n"], ascending=[True, True, False])
    for nama, df in out.items():
        df.to_csv(path(f"hasil/{nama}.csv"), index=False, encoding="utf-8")
        print(f"\n== {nama} ==\n{df.to_string(index=False)}")

    # tokenisasi IndoBERT pada token target (tidak memerlukan GPU)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(INDOBERT_CHECKPOINT)
        t = uji[uji.token_target != ""][["token_target", "token_unicode", "token_angka"]].drop_duplicates()
        for k in ("target", "unicode", "angka"):
            t[f"subword_{k}"] = t[f"token_{k}"].map(lambda w: " ".join(tok.tokenize(w)))
            t[f"n_subword_{k}"] = t[f"token_{k}"].map(lambda w: len(tok.tokenize(w)))
        t.to_csv(path("hasil/05_tokenisasi_indobert.csv"), index=False, encoding="utf-8")
        print(f"\n== 05_tokenisasi_indobert ==\n{t.to_string(index=False)}")
    except Exception as e:  # tokenizer gagal diunduh
        print("Tokenisasi IndoBERT dilewati:", e)


if __name__ == "__main__":
    main()