import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (BLOK_HURUF_BERGAYA, KATA_DETEKSI_TAMBAHAN, LEKSIKON_ISTILAH,
                    LEKSIKON_SITUS, PEMETAAN_ANGKA_BALIK, RANDOM_STATE, path)

RAW = path("data/raw/deteksi-judi-online.csv")

# ---------------------------------------------------------------------------
# Fungsi pemeriksaan bentuk penyamaran yang sudah ada pada komentar asli
# ---------------------------------------------------------------------------
ZERO_WIDTH = "​‌‍⁠﻿"
RE_ZW_DALAM_KATA = re.compile(rf"[A-Za-z0-9][{ZERO_WIDTH}]+[A-Za-z0-9]")


def ada_huruf_bergaya(teks):
    """Karakter dari blok Unicode yang memuat huruf/angka bergaya atau homoglyph."""
    return any(a <= ord(c) <= b for c in teks for a, b, _ in BLOK_HURUF_BERGAYA)


def ada_tanda_kombinasi(teks):
    """Combining diacritical marks yang ditempelkan pada huruf (mis. p͎u͎l͎a͎u͎)."""
    return any(unicodedata.category(c) == "Mn"
               and (0x0300 <= ord(c) <= 0x036F or 0x1AB0 <= ord(c) <= 0x1AFF
                    or 0x20D0 <= ord(c) <= 0x20FF)
               for c in teks)


def ada_zero_width_dalam_kata(teks):
    """Karakter zero-width yang disisipkan di antara huruf/angka."""
    return bool(RE_ZW_DALAM_KATA.search(teks))


def _pola_substitusi(kata):
    """Pola regex untuk kata leksikon dengan kemungkinan huruf diganti angka."""
    bagian = []
    for ch in kata:
        alternatif = [ch] + PEMETAAN_ANGKA_BALIK.get(ch, [])
        bagian.append("[" + "".join(alternatif) + "]" if len(alternatif) > 1 else re.escape(ch))
    return re.compile("".join(bagian))


_POLA_SUBSTITUSI = [(k, _pola_substitusi(k)) for k in sorted(LEKSIKON_SITUS | LEKSIKON_ISTILAH | KATA_DETEKSI_TAMBAHAN)]


def substitusi_yang_sudah_ada(teks):
    """Token yang merupakan kata leksikon dengan sebagian huruf diganti angka.

    Contoh: sl0t, s1tus, gac0r. Nama situs yang memang memuat angka (alexis17)
    tidak terdeteksi karena token tersebut sama persis dengan bentuk leksikon.
    """
    hasil = []
    for tok in re.findall(r"[a-z0-9]+", huruf_kecil(teks)):
        if not re.search(r"\d", tok):
            continue
        for kata, pola in _POLA_SUBSTITUSI:
            if tok != kata and pola.fullmatch(tok):
                hasil.append(tok)
                break
            # bentuk berimbuhan, mis. gac0rnya / s1tusnya
            if not tok.startswith(kata) and pola.match(tok) and tok[pola.match(tok).end():] == "nya":
                hasil.append(tok)
                break
    return hasil


def huruf_kecil(teks):
    return teks.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                        "abcdefghijklmnopqrstuvwxyz"))


# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)

    # pemeriksaan kelengkapan data: hanya dicatat, tidak mengubah data
    kosong = df.apply(lambda kolom: (kolom.str.strip() == "").sum())
    kosong.rename_axis("atribut").reset_index(name="jumlah_kosong").to_csv(
        path("hasil/01_pemeriksaan_kelengkapan.csv"), index=False)
    print(f"Atribut: {df.shape[1]}  Nilai kosong: {kosong.sum()}  "
          f"Nilai label: {sorted(df.label.unique())}\n")

    df.insert(0, "id_asli", range(1, len(df) + 1))  # nomor baris pada CSV (tanpa header)
    df["label"] = df["label"].astype(int)
    ringkas = [("Data awal", len(df), (df.label == 0).sum(), (df.label == 1).sum())]
    keluar = []

    def catat(mask, alasan):
        d = df[mask].copy()
        d["alasan"] = alasan
        keluar.append(d[["id_asli", "video_id", "komentar", "label", "alasan"]])

    # 1) teks identik dengan label berbeda -> seluruh kelompok dikeluarkan
    n_label = df.groupby("komentar")["label"].transform("nunique")
    m = n_label > 1
    catat(m, "Teks identik dengan label berbeda")
    df = df[~m]

    # 2) teks identik dengan label sama -> dipertahankan kemunculan pertama
    m = df.duplicated("komentar", keep="first")
    catat(m, "Duplikat (teks dan label identik)")
    df = df[~m]
    ringkas.append(("Setelah penanganan duplikasi", len(df), (df.label == 0).sum(), (df.label == 1).sum()))

    # 3) penyamaran Unicode yang sudah ada pada komentar asli
    m1 = df.komentar.map(ada_huruf_bergaya)
    catat(m1, "Memuat huruf/angka Unicode bergaya atau homoglyph")
    df = df[~m1]
    m2 = df.komentar.map(ada_tanda_kombinasi)
    catat(m2, "Memuat tanda kombinasi (combining mark) pada huruf")
    df = df[~m2]
    m3 = df.komentar.map(ada_zero_width_dalam_kata)
    catat(m3, "Memuat karakter zero-width di dalam kata")
    df = df[~m3]
    ringkas.append(("Setelah pengeluaran penyamaran Unicode", len(df), (df.label == 0).sum(), (df.label == 1).sum()))

    # 4) substitusi huruf-angka yang sudah ada pada kata objek penelitian
    sub = df.komentar.map(substitusi_yang_sudah_ada)
    m = sub.map(len) > 0
    d = df[m].copy()
    d["alasan"] = "Substitusi huruf-angka sudah ada: " + sub[m].map(", ".join)
    keluar.append(d[["id_asli", "video_id", "komentar", "label", "alasan"]])
    df = df[~m]
    ringkas.append(("Setelah pengeluaran substitusi huruf-angka", len(df), (df.label == 0).sum(), (df.label == 1).sum()))

    df = df.reset_index(drop=True)
    df[["id_asli", "video_id", "channel_name", "komentar", "label"]].to_csv(
        path("data/processed/data_bersih.csv"), index=False, encoding="utf-8")

    # 5) pembagian data terstratifikasi 70/15/15
    train, sisa = train_test_split(df, test_size=0.30, stratify=df.label, random_state=RANDOM_STATE)
    val, test = train_test_split(sisa, test_size=0.50, stratify=sisa.label, random_state=RANDOM_STATE)
    kolom = ["id_asli", "komentar", "label"]
    for nama, d in (("train", train), ("val", val), ("test", test)):
        d.sort_values("id_asli")[kolom].to_csv(path(f"data/processed/{nama}.csv"), index=False, encoding="utf-8")

    # ---- ringkasan ---------------------------------------------------------
    r = pd.DataFrame(ringkas, columns=["tahap", "jumlah", "label_0", "label_1"])
    r.to_csv(path("hasil/01_ringkasan_persiapan.csv"), index=False)
    k = pd.concat(keluar, ignore_index=True)
    k.to_csv(path("hasil/01_data_dikeluarkan.csv"), index=False, encoding="utf-8")

    sp = pd.DataFrame([(n, len(d), (d.label == 0).sum(), (d.label == 1).sum())
                       for n, d in (("Latih", train), ("Validasi", val), ("Uji", test))],
                      columns=["bagian", "jumlah", "label_0", "label_1"])
    sp.to_csv(path("hasil/01_distribusi_split.csv"), index=False)

    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    raw["label"] = raw.label.astype(int)
    vid = (raw.groupby(["video_id", "title", "channel_name"])
           .agg(total=("label", "size"), promosi=("label", "sum"))
           .reset_index())
    vid["non_promosi"] = vid.total - vid.promosi
    bersih = df.groupby("video_id").agg(bersih_total=("label", "size"), bersih_promosi=("label", "sum"))
    vid = vid.merge(bersih, on="video_id", how="left").fillna(0)
    vid = vid.sort_values("total", ascending=False)
    vid.to_csv(path("hasil/01_distribusi_video.csv"), index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    print()
    print(k.alasan.str.split(":").str[0].value_counts().to_string())
    print()
    print(sp.to_string(index=False))
    print(f"\nVideo: {vid.video_id.nunique()}  Kanal: {vid.channel_name.nunique()}")


if __name__ == "__main__":
    main()
