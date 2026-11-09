"""Konfigurasi bersama untuk seluruh tahap eksperimen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def path(rel):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Blok Unicode yang memuat huruf/angka bergaya atau homoglyph huruf Latin.
# Komentar asli yang memuat karakter dari blok ini dikeluarkan agar kondisi
# normal tidak mengandung penyamaran Unicode.
# ---------------------------------------------------------------------------
BLOK_HURUF_BERGAYA = [
    (0x0100, 0x024F, "Latin Extended-A dan Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x0370, 0x03FF, "Greek and Coptic"),
    (0x0400, 0x052F, "Cyrillic dan Cyrillic Supplement"),
    (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Unified Canadian Aboriginal Syllabics"),
    (0x1D00, 0x1DBF, "Phonetic Extensions dan Supplement"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0xA000, 0xA4CF, "Yi Syllables dan Yi Radicals"),
    (0xA500, 0xA63F, "Vai"),
    (0xFF10, 0xFF19, "Fullwidth Digit"),
    (0xFF21, 0xFF3A, "Fullwidth Latin Capital Letter"),
    (0xFF41, 0xFF5A, "Fullwidth Latin Small Letter"),
    (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
    (0x1F100, 0x1F1E5, "Enclosed Alphanumeric Supplement (tanpa Regional Indicator)"),
]

# ---------------------------------------------------------------------------
# Leksikon kandidat. Disusun dari kata yang muncul pada komentar promosi
# dataset ini. Pencocokan dilakukan per token utuh (rangkaian a-z dan 0-9)
# setelah huruf A-Z diubah menjadi huruf kecil.
# - Langkah 1 memakai seluruh leksikon kandidat untuk mendeteksi substitusi
#   huruf-angka yang sudah ada (pembersihan data sebelum pembagian).
# - Langkah 2 hanya memakai kata yang muncul pada komentar promosi data latih
#   sebagai token target, agar pemilihan token tidak bergantung pada data uji.
# ---------------------------------------------------------------------------
LEKSIKON_SITUS = {
    "alexis17", "alexis", "mandalika77", "mandalika", "pawpaw4d", "weton88",
    "berkah99", "ggwp88", "dongo88", "betul88", "probet855", "manut88",
    "dora77", "miaw4d", "pulauwin", "pulau", "maengaruda", "garuda", "digaruda",
    "garudahoki",
}

LEKSIKON_ISTILAH = {
    # istilah umum perjudian daring
    "judi", "togel", "slot", "slotnya", "sloter", "situs", "situsnya",
    "gacor", "gacornya", "gacir", "gacirnya", "gacirr", "gacur", "gacoor", "gacorr",
    "gacoorr", "ghacoorr", "ghaacoor", "gicir",
    "maxwin", "makswin", "mekswin", "jackpot", "jakpot",
    "jepe", "jepek", "jepey", "jepee", "wede", "wedey", "wedee",
    "depo", "depoin", "deposit", "depositnya",
    "scatter", "scatternya", "sketer", "skete", "spin",
    # nama permainan slot
    "zeus", "zeusnya", "olympus", "bonanja", "starlight", "mahjong", "gates",
}

# Kata tambahan yang hanya dipakai untuk deteksi substitusi yang sudah ada
# (bukan token target), karena muncul sebagai pecahan "max win" (mis. m4x w1n).
KATA_DETEKSI_TAMBAHAN = {"max", "win"}

# Kemungkinan angka pengganti huruf, dipakai untuk mendeteksi substitusi
# huruf-angka yang SUDAH ada pada komentar asli (mis. sl0t, s1tus, gac0r).
PEMETAAN_ANGKA_BALIK = {"a": ["4"], "e": ["3"], "i": ["1"], "o": ["0"], "s": ["5"], "l": ["1"]}

# ---------------------------------------------------------------------------
# Pemetaan karakter untuk membentuk kondisi pengujian (Tabel 3.4 proposal)
# ---------------------------------------------------------------------------
PEMETAAN_UNICODE = {  # Mathematical Bold Small
    "a": "\U0001D41A", "e": "\U0001D41E", "i": "\U0001D422", "o": "\U0001D428", "s": "\U0001D42C",
}
PEMETAAN_ANGKA = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}

# ---------------------------------------------------------------------------
# Konfigurasi model
# ---------------------------------------------------------------------------
TFIDF_PARAMS = dict(analyzer="word", ngram_range=(1, 2), lowercase=False,
                    token_pattern=r"(?u)\b\w\w+\b")
SVM_PARAMS = dict(C=1.0, random_state=RANDOM_STATE)

INDOBERT_CHECKPOINT = "indobenchmark/indobert-base-p1"
