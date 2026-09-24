"""Versi resolusi kecil dari citra yang sama (PRD §9 E1).

640×360 dibuat dengan memperkecil rekaman 1280×720 pada skala tepat 0,5
memakai `cv2.INTER_AREA`, sehingga rasio aspek 16:9 terjaga dan satu-satunya
perbedaan adalah jumlah piksel. Ground truth ikut diskalakan.
"""

from __future__ import annotations

import cv2
import numpy as np

from pcdface.boxes import Box


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Skalakan citra. Skala 1 mengembalikan masukan apa adanya."""
    if scale == 1.0:
        return image
    if scale <= 0:
        raise ValueError("skala harus positif")
    height, width = image.shape[:2]
    size = (int(round(width * scale)), int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def scale_boxes(boxes: list[Box], scale: float) -> list[Box]:
    """Skalakan kotak (x, y, w, h); ukuran minimal 1 piksel."""
    if scale == 1.0:
        return list(boxes)
    return [
        (
            int(round(x * scale)),
            int(round(y * scale)),
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        for x, y, w, h in boxes
    ]
