"""Detektor warna kulit YCbCr (P1) — pengolahan citra murni, tanpa model.

Tahapan:
1. BGR → YCrCb, ambang kroma Chai & Ngan (1999): 77 ≤ Cb ≤ 127, 133 ≤ Cr ≤ 173
2. Opening (buang bintik) lalu closing (tutup lubang mata/mulut)
3. Connected Component Labeling
4. Saring kandidat: luas, rasio aspek lebar/tinggi, solidity

Ambang bekerja di kanal kroma sehingga relatif tahan perubahan intensitas.
Tidak punya skor, jadi hanya dilaporkan pada titik operasinya (PRD §8.2).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from pcdface.config import YCbCrConfig
from pcdface.detection.base import DetectionResult, Detector


def skin_mask(image_bgr: np.ndarray, params: YCbCrConfig) -> np.ndarray:
    """Citra biner kulit (0/255). Urutan kanal OpenCV: Y, Cr, Cb."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, params.cr_range[0], params.cb_range[0]], dtype=np.uint8)
    upper = np.array([255, params.cr_range[1], params.cb_range[1]], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def clean_mask(mask: np.ndarray, params: YCbCrConfig) -> tuple[np.ndarray, np.ndarray]:
    """(setelah opening, setelah closing) dengan structuring element elips."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.opening_kernel,) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.closing_kernel,) * 2)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=params.opening_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=params.closing_iterations)
    return opened, closed


def filter_candidates(mask: np.ndarray, params: YCbCrConfig) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """CCL + saring geometri. Kembalikan (kotak diterima, semua kandidat + alasan tolak)."""
    image_area = mask.shape[0] * mask.shape[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    accepted, candidates = [], []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if w == 0 or h == 0:
            continue
        aspect, solidity, ratio = w / h, area / float(w * h), area / image_area
        reasons = []
        if ratio < params.min_area_ratio:
            reasons.append("area_terlalu_kecil")
        if ratio > params.max_area_ratio:
            reasons.append("area_terlalu_besar")
        if aspect < params.aspect_range[0]:
            reasons.append("terlalu_sempit")
        if aspect > params.aspect_range[1]:
            reasons.append("terlalu_lebar")
        if solidity < params.min_solidity:
            reasons.append("solidity_rendah")
        candidates.append({"box": (x, y, w, h), "area_ratio": round(ratio, 5), "aspect": round(aspect, 3),
                           "solidity": round(solidity, 3), "rejected_because": reasons})
        if not reasons:
            accepted.append((x, y, w, h))
    return accepted, candidates


class YCbCrDetector(Detector):
    has_scores = False

    def __init__(self, params: YCbCrConfig, name: str = "ycbcr", keep_stages: bool = False) -> None:
        self.name = name
        self.params = params
        self.keep_stages = keep_stages

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        raw = skin_mask(image_bgr, self.params)
        opened, closed = clean_mask(raw, self.params)
        boxes, candidates = filter_candidates(closed, self.params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stages = ({"01_mask_mentah": raw, "02_opening": opened, "03_closing": closed}
                  if self.keep_stages else {})
        return DetectionResult(boxes=boxes, scores=None, elapsed_ms=elapsed_ms, stages=stages,
                               info={"candidates": candidates})
