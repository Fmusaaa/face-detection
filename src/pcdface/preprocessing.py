"""Peningkatan kualitas citra pada kanal luminansi.

Enhancement dikerjakan hanya pada kanal Y dari ruang warna YCrCb, lalu
digabung kembali dengan kanal kroma Cr dan Cb yang tidak diubah. Dengan
begitu kontras membaik tanpa menggeser warna — penting untuk detektor
warna kulit YCbCr yang bekerja di kanal kroma.

`none` berarti citra mentah: tidak ada operasi apa pun, termasuk penghalusan
(PRD §12.2 butir 1).

Referensi: Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram
Equalization. Graphics Gems IV.
"""

from __future__ import annotations

import cv2
import numpy as np

ENHANCEMENTS = ("none", "clahe")


def clahe_luma(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE pada kanal Y; kanal Cr dan Cb tidak disentuh."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    luma, chroma_red, chroma_blue = cv2.split(ycrcb)
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    merged = cv2.merge((operator.apply(luma), chroma_red, chroma_blue))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def enhance(
    image_bgr: np.ndarray,
    method: str,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Terapkan skema enhancement. `none` mengembalikan citra masukan apa adanya."""
    if method not in ENHANCEMENTS:
        raise ValueError(f"enhancement '{method}' tidak dikenal. Pilihan: {ENHANCEMENTS}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("enhance() membutuhkan citra BGR 3 kanal")
    if method == "none":
        return image_bgr
    return clahe_luma(image_bgr, clip_limit=clip_limit, tile_grid=tile_grid)


def mean_luma(image_bgr: np.ndarray) -> float:
    """Rerata kanal Y (0–255) — bukti kuantitatif kondisi pencahayaan."""
    if image_bgr.ndim == 2:
        return float(np.mean(image_bgr))
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return float(np.mean(ycrcb[:, :, 0]))
