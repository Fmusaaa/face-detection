"""Detektor tiruan untuk tes dan `run --synthetic` — tanpa model, deterministik.

Menemukan "wajah" pada citra sintetis lewat segmentasi warna kulit, lalu
memutuskan terdeteksi atau tidak dengan peluang yang bergantung ukuran wajah —
meniru perbedaan mekanisme di PRD §2.1:

- profil Haar/YCbCr: batas **absolut dalam piksel** (resolusi kecil merugikan);
- profil MediaPipe: batas **relatif terhadap lebar frame** (resolusi tidak berpengaruh).

Keputusan acak diturunkan dari isi citra + nama detektor, jadi hasilnya sama
setiap dijalankan. Angka dari detektor ini BUKAN hasil penelitian.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from pcdface.boxes import Box, clip_box
from pcdface.detection.base import DetectionResult, Detector

FACE_ASPECT = 1.3  # sama dengan pembangkit sintetis


@dataclass(frozen=True)
class FakeProfile:
    relative: bool          # True: ukuran = lebar/lebar frame; False: piksel
    size50: float           # ukuran dengan peluang deteksi 50%
    slope: float            # kelandaian kurva logistik
    dark_factor: float      # pengali peluang pada wajah gelap (Y < 70)
    distractor_p: float     # peluang benda mirip kulit dianggap wajah
    box_scale: float        # sisi kotak persegi relatif terhadap lebar wajah
    score_range: tuple[float, float]  # skor minimal–maksimal (Haar: levelWeights, bisa negatif)
    scores_at_operating_point: bool


PROFILES: dict[str, FakeProfile] = {
    "haar": FakeProfile(False, 34.0, 4.0, 0.55, 0.06, 1.05, (-3.0, 9.0), False),
    "ycbcr": FakeProfile(False, 45.0, 5.0, 0.35, 0.85, 1.0, (0.0, 1.0), False),
    "mp_short": FakeProfile(True, 0.055, 0.006, 0.9, 0.02, 0.95, (0.0, 1.0), True),
    "mp_full": FakeProfile(True, 0.026, 0.004, 0.9, 0.02, 0.95, (0.0, 1.0), True),
    "mp_sparse": FakeProfile(True, 0.029, 0.004, 0.88, 0.02, 0.95, (0.0, 1.0), True),
}


def profile_for(name: str, kind: str) -> FakeProfile:
    """Profil menurut nama detektor, lalu menurut jenisnya."""
    if name in PROFILES:
        return PROFILES[name]
    return PROFILES["haar"] if kind == "haar" else PROFILES["ycbcr"] if kind == "ycbcr" else PROFILES["mp_short"]


def _components(image_bgr: np.ndarray) -> list[tuple[Box, float, float]]:
    """(kotak wajah, solidity komponen, rerata Y) untuk tiap komponen warna kulit."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 134, 77], np.uint8), np.array([255, 175, 127], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 30 or w < 4:
            continue
        # komponen = wajah + leher; wajah menempati bagian atas setinggi 1,3 × lebar
        face = (x, y, w, min(h, int(round(w * FACE_ASPECT))))
        luma = float(ycrcb[y:y + face[3], x:x + w, 0].mean())
        found.append((face, area / float(w * h), luma))
    return found


class FakeDetector(Detector):
    def __init__(self, name: str, profile: FakeProfile, score_mode: bool = False) -> None:
        self.name = name
        self.profile = profile
        self.score_mode = score_mode
        self.has_scores = score_mode or profile.scores_at_operating_point
        if name == "ycbcr":
            self.has_scores = False

    def _probability(self, box: Box, width: int, solidity: float, luma: float) -> float:
        p = self.profile
        size = box[2] / width if p.relative else float(box[2])
        prob = 1.0 / (1.0 + np.exp(-(size - p.size50) / p.slope))
        if luma < 70:
            prob *= p.dark_factor
        if solidity > 0.97:  # persegi penuh: kardus/kayu, bukan wajah+leher
            prob = p.distractor_p
        return float(prob)

    def _score(self, prob: float, rng: np.random.Generator) -> float:
        low, high = self.profile.score_range
        value = low + (high - low) * float(np.clip(prob + rng.normal(0, 0.08), 0.0, 1.0))
        return round(value, 4)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        height, width = image_bgr.shape[:2]
        seed = zlib.crc32(np.ascontiguousarray(image_bgr[::7, ::7]).tobytes()) ^ zlib.crc32(self.name.encode())
        rng = np.random.default_rng(seed)
        boxes: list[Box] = []
        scores: list[float] = []
        for face, solidity, luma in _components(image_bgr):
            prob = self._probability(face, width, solidity, luma)
            draw = rng.random()
            if not self.score_mode and draw >= prob:
                continue
            if self.score_mode and prob < 0.01:
                continue
            x, y, w, h = face
            side = w * self.profile.box_scale * float(rng.uniform(0.96, 1.04))
            cx = x + w / 2 + rng.normal(0, 0.02 * w)
            cy = y + h * 0.55 + rng.normal(0, 0.02 * w)
            box = clip_box((int(cx - side / 2), int(cy - side / 2), int(side), int(side)), width, height)
            if box is not None:
                boxes.append(box)
                scores.append(self._score(prob, rng))
        # deteksi palsu acak: satu per citra dengan peluang kecil; lebih banyak pada run AP
        extra = int(rng.poisson(1.5)) if self.score_mode else int(rng.random() < 0.04)
        for _ in range(extra):
            side = int(rng.uniform(0.03, 0.12) * width)
            box = clip_box((int(rng.uniform(0, width - side)), int(rng.uniform(0, height - side)), side, side),
                           width, height)
            if box is not None:
                boxes.append(box)
                scores.append(self._score(float(rng.uniform(0.0, 0.35)), rng))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(boxes=boxes, scores=scores if self.has_scores else None, elapsed_ms=elapsed_ms)
