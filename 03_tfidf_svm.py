import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import SVM_PARAMS, TFIDF_PARAMS, path
from praproses import praproses


def baca(nama):
    d = pd.read_csv(path(f"data/processed/{nama}.csv"), dtype={"komentar": str}, keep_default_na=False)
    d["teks"] = d.komentar.map(praproses)
    return d


def main():
    train, val = baca("train"), baca("val")
    uji = pd.read_csv(path("data/processed/test_kondisi.csv"), keep_default_na=False)

    model = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("svm", LinearSVC(**SVM_PARAMS))])
    t0 = time.time()
    model.fit(train.teks, train.label)
    print(f"Waktu pelatihan: {time.time() - t0:.2f} detik")
    joblib.dump(model, path("hasil/model_svm.joblib"))

    vec, svm = model.named_steps["tfidf"], model.named_steps["svm"]
    print(f"Jumlah fitur TF-IDF (unigram+bigram): {len(vec.vocabulary_)}")

    # pemantauan pada data validasi
    pv = model.predict(val.teks)
    p, r, f, _ = precision_recall_fscore_support(val.label, pv, pos_label=1, average="binary")
    pd.DataFrame([dict(precision=p, recall=r, f1=f, n=len(val))]).to_csv(path("hasil/03_svm_validasi.csv"), index=False)
    print("Validasi:\n", classification_report(val.label, pv, digits=4))

    # prediksi tiga kondisi
    out = uji[["id_asli", "label", "masuk_evaluasi"]].copy()
    for k in ("normal", "unicode", "angka"):
        out[f"pred_{k}"] = model.predict(uji[f"teks_{k}"])
        out[f"skor_{k}"] = model.decision_function(uji[f"teks_{k}"])
    out.to_csv(path("hasil/pred_svm.csv"), index=False)

    ev = out[out.masuk_evaluasi.astype(str) == "True"]
    for k in ("normal", "unicode", "angka"):
        print(k, confusion_matrix(ev.label, ev[f"pred_{k}"]).ravel(), "(tn fp fn tp)")

    # bobot fitur dan status token target di kosakata
    fitur = pd.DataFrame({"fitur": vec.get_feature_names_out(), "bobot": svm.coef_[0]})
    fitur.sort_values("bobot", ascending=False).to_csv(path("hasil/03_svm_bobot_fitur.csv"), index=False, encoding="utf-8")
    vocab = vec.vocabulary_
    bobot = dict(zip(fitur.fitur, fitur.bobot))
    tok = uji[uji.token_target != ""][["token_target", "token_unicode", "token_angka"]].drop_duplicates()
    for k in ("target", "unicode", "angka"):
        kol = f"token_{k}"
        tok[f"ada_di_kosakata_{k}"] = tok[kol].isin(vocab)
        tok[f"bobot_{k}"] = tok[kol].map(bobot)
    tok.to_csv(path("hasil/03_svm_token_target_oov.csv"), index=False, encoding="utf-8")
    print(tok.to_string(index=False))


if __name__ == "__main__":
    main()
