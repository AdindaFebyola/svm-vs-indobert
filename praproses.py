"""Prapemrosesan minimal yang diterapkan sama pada seluruh data dan kondisi.

Hanya dua operasi: huruf Latin standar A-Z diubah menjadi a-z, dan spasi
berlebih dinormalisasi. Tidak ada stemming, stopword removal, normalisasi
Unicode, maupun pengubahan angka menjadi huruf.
"""
import re

_KECIL = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_SPASI = re.compile(r"\s+")


def praproses(teks: str) -> str:
    return _SPASI.sub(" ", teks.translate(_KECIL)).strip()
