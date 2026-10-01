"""Tombol keyboard jendela OpenCV yang tidak peka Caps Lock.

`cv2.waitKey` mengembalikan kode huruf besar bila Caps Lock menyala, sehingga
`s`, `q`, `d`, … diam-diam tidak bekerja. Semua alat memakai `lower_key`.
"""

from __future__ import annotations


def lower_key(code: int) -> int:
    """Kode ASCII 8-bit dari `waitKey` → huruf kecil bila huruf A–Z; kode lain tidak diubah."""
    char = code & 0xFF
    return char + 32 if ord("A") <= char <= ord("Z") else char
