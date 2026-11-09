"""
Langkah 2 - Pembentukan kondisi pengujian pada data uji.

Leksikon token target adalah kata pada leksikon kandidat (config.py) yang
muncul pada komentar promosi data latih. Untuk setiap komentar promosi pada
data uji dipilih satu token target:
  1. prioritas pertama nama situs, kedua istilah perjudian;
  2. jika terdapat lebih dari satu kandidat pada prioritas yang sama,
     dipilih token yang muncul paling awal;
  3. token harus memuat minimal satu karakter a, e, i, o, atau s.
Seluruh karakter a/e/i/o/s pada token target diganti dengan Mathematical Bold
(kondisi Unicode) atau angka (kondisi substitusi huruf-angka) pada posisi yang
sama. Komentar non-promosi tidak diubah.

Keluaran: data/processed/test_kondisi.csv
          hasil/02_leksikon.csv
          hasil/02_ringkasan_kondisi.csv
          hasil/02_token_target.csv
"""
import re
from collections import Counter

import pandas as pd

from config import LEKSIKON_ISTILAH, LEKSIKON_SITUS, PEMETAAN_ANGKA, PEMETAAN_UNICODE, path
from praproses import praproses

RE_TOKEN = re.compile(r"[a-z0-9]+")
KARAKTER_TARGET = set(PEMETAAN_UNICODE)


def leksikon_dari_latih():
    """Membatasi leksikon kandidat pada kata yang muncul di komentar promosi data latih."""
    train = pd.read_csv(path("data/processed/train.csv"), dtype={"komentar": str}, keep_default_na=False)
    frek = Counter(t for k in train[train.label == 1].komentar for t in set(RE_TOKEN.findall(praproses(k))))
    tabel = pd.DataFrame([dict(token=t, kategori=kat, frek_promosi_latih=frek[t], dipakai=frek[t] > 0)
                          for kat, lek in (("nama situs", LEKSIKON_SITUS), ("istilah perjudian", LEKSIKON_ISTILAH))
                          for t in sorted(lek)])
    tabel.to_csv(path("hasil/02_leksikon.csv"), index=False, encoding="utf-8")
    pakai = tabel[tabel.dipakai]
    return (set(pakai[pakai.kategori == "nama situs"].token),
            set(pakai[pakai.kategori == "istilah perjudian"].token), tabel)


SITUS, ISTILAH, TABEL_LEKSIKON = leksikon_dari_latih()


def pilih_token_target(teks):
    """Mengembalikan (token, kategori, posisi_awal) atau None."""
    kandidat = [m for m in RE_TOKEN.finditer(teks) if KARAKTER_TARGET & set(m.group())]
    for kategori, leksikon in (("nama situs", SITUS), ("istilah perjudian", ISTILAH)):
        for m in kandidat:  # urut dari kiri ke kanan
            if m.group() in leksikon:
                return m.group(), kategori, m.start()
    return None


def ubah_token(token, pemetaan):
    return "".join(pemetaan.get(c, c) for c in token)


def main():
    test = pd.read_csv(path("data/processed/test.csv"), dtype={"komentar": str}, keep_default_na=False)
    baris = []
    for r in test.itertuples(index=False):
        normal = praproses(r.komentar)
        rec = dict(id_asli=r.id_asli, label=r.label, teks_normal=normal,
                   teks_unicode=normal, teks_angka=normal,
                   token_target="", kategori_token="", posisi=-1, n_karakter_diubah=0,
                   token_unicode="", token_angka="")
        if r.label == 1:
            pilih = pilih_token_target(normal)
            if pilih is None:
                rec["masuk_evaluasi"] = False
            else:
                tok, kat, pos = pilih
                tu, ta = ubah_token(tok, PEMETAAN_UNICODE), ubah_token(tok, PEMETAAN_ANGKA)
                akhir = pos + len(tok)
                rec.update(teks_unicode=normal[:pos] + tu + normal[akhir:],
                           teks_angka=normal[:pos] + ta + normal[akhir:],
                           token_target=tok, kategori_token=kat, posisi=pos,
                           n_karakter_diubah=sum(c in KARAKTER_TARGET for c in tok),
                           token_unicode=tu, token_angka=ta, masuk_evaluasi=True)
        else:
            rec["masuk_evaluasi"] = True
        baris.append(rec)

    df = pd.DataFrame(baris)
    df.to_csv(path("data/processed/test_kondisi.csv"), index=False, encoding="utf-8")

    promo = df[df.label == 1]
    ok = promo[promo.masuk_evaluasi]
    ringkas = pd.DataFrame([
        ("Komentar promosi pada data uji", len(promo)),
        ("Promosi memenuhi kriteria token target", len(ok)),
        ("  - token nama situs", (ok.kategori_token == "nama situs").sum()),
        ("  - token istilah perjudian", (ok.kategori_token == "istilah perjudian").sum()),
        ("Promosi tidak memenuhi kriteria (tidak dievaluasi berpasangan)", len(promo) - len(ok)),
        ("Komentar non-promosi pada data uji (tidak diubah)", (df.label == 0).sum()),
        ("Total himpunan evaluasi berpasangan", df.masuk_evaluasi.sum()),
    ], columns=["keterangan", "jumlah"])
    ringkas.to_csv(path("hasil/02_ringkasan_kondisi.csv"), index=False)

    tok = (ok.groupby(["token_target", "kategori_token", "token_unicode", "token_angka"])
           .size().rename("jumlah").reset_index().sort_values("jumlah", ascending=False))
    tok.to_csv(path("hasil/02_token_target.csv"), index=False, encoding="utf-8")

    # pemeriksaan: hanya token target yang berubah, panjang teks tetap
    for _, r in ok.iterrows():
        assert len(r.teks_angka) == len(r.teks_normal)
        assert r.teks_normal.replace(r.token_target, r.token_angka, 1) == r.teks_angka
        assert r.teks_normal.replace(r.token_target, r.token_unicode, 1) == r.teks_unicode
    non = df[df.label == 0]
    assert (non.teks_normal == non.teks_unicode).all() and (non.teks_normal == non.teks_angka).all()

    tidak = TABEL_LEKSIKON[~TABEL_LEKSIKON.dipakai].token
    print(f"Leksikon kandidat: {len(TABEL_LEKSIKON)} kata, dipakai: {len(SITUS) + len(ISTILAH)}, "
          f"tidak muncul di promosi data latih: {', '.join(tidak)}")
    print()
    print(ringkas.to_string(index=False))
    print()
    print(tok.to_string(index=False))
    print()
    for _, r in ok.head(3).iterrows():
        print(r.teks_normal, "|", r.teks_unicode, "|", r.teks_angka, sep="\n  ")


if __name__ == "__main__":
    main()
