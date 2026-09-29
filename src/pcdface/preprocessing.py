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


def motion_blur(image_bgr: np.ndarray, length_px: int, angle_deg: float = 0.0) -> np.ndarray:
    """Blur gerak linear (simulasi E7): rerata `length_px` piksel sepanjang arah `angle_deg`.

    0° = horizontal (kepala/badan bergerak ke samping). `length_px` ≤ 1 → citra tidak diubah.
    Ini degradasi uji, bukan enhancement — tidak pernah dipakai di eksperimen lain.
    """
    if length_px <= 1:
        return image_bgr
    kernel = np.zeros((length_px, length_px), dtype=np.float32)
    kernel[length_px // 2, :] = 1.0
    center = ((length_px - 1) / 2.0, (length_px - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length_px, length_px), flags=cv2.INTER_LINEAR)
    kernel /= kernel.sum()
    return cv2.filter2D(image_bgr, -1, kernel, borderType=cv2.BORDER_REFLECT)
