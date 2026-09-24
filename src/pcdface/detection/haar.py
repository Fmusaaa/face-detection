"""Detektor Haar Cascade (Viola & Jones, 2001) via OpenCV.

Dua jalur (PRD §5.1, §12.2 butir 3):

- Titik operasi: `detectMultiScale` dengan parameter config. Tidak ada skor.
- Run AP (`score_mode=True`): `detectMultiScale3(outputRejectLevels=True)`
  dengan `minNeighbors` rendah, lalu NMS. Skor = `levelWeights` — bukan
  probabilitas, tidak di rentang 0–1; hanya urutannya yang dipakai AP.

Jebakan NMS: `cv2.dnn.NMSBoxes` hanya menyimpan skor yang LEBIH BESAR dari
ambang skor, dan diam-diam membuang skor negatif. `levelWeights` bisa negatif,
jadi skor digeser menjadi `skor − min + 1` sebelum NMS (urutan tidak berubah,
semua positif, kandidat terendah tidak ikut terbuang). Skor yang dikembalikan
tetap `levelWeights` asli.

Cara kerja singkat (Landasan Teori): fitur Haar = selisih jumlah intensitas
persegi terang/gelap; integral image membuat jumlah persegi apa pun dihitung
dengan 4 pembacaan; AdaBoost memilih fitur paling diskriminatif; cascade
membuang jendela latar di tahap awal sehingga cepat.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from pcdface.config import HaarConfig
from pcdface.detection.base import DetectionResult, Detector

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"
_cache: dict[str, "cv2.CascadeClassifier"] = {}


def load_cascade(cascade_file: str = DEFAULT_CASCADE) -> "cv2.CascadeClassifier":
    """Muat cascade bawaan OpenCV (di-cache). Galat jelas bila berkas tidak ada."""
    if cascade_file in _cache:
        return _cache[cascade_file]
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(f"OpenCV {cv2.__version__} tidak punya CascadeClassifier. "
                           'Pasang: pip install "opencv-contrib-python>=4.8,<5"')
    path = Path(cv2.data.haarcascades) / cascade_file
    if not path.exists():
        path = Path(cascade_file)
    if not path.exists():
        raise FileNotFoundError(
            f"berkas cascade tidak ditemukan: {cascade_file} (folder bawaan: {cv2.data.haarcascades}). "
            "OpenCV 5 tidak lagi menyertakan haarcascade_*.xml — pastikan opencv-contrib-python<5."
        )
    classifier = cv2.CascadeClassifier(str(path))
    if classifier.empty():
        raise RuntimeError(f"gagal memuat cascade dari {path}")
    _cache[cascade_file] = classifier
    return classifier


def nms_with_shifted_scores(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Indeks kotak yang lolos NMS. Skor digeser ke ≥ 1 agar aman untuk NMSBoxes."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    shifted = scores - scores.min() + 1.0
    keep = cv2.dnn.NMSBoxes(
        [list(map(int, b)) for b in boxes], shifted.astype(float).tolist(), 0.0, float(iou_threshold)
    )
    keep = np.asarray(keep, dtype=int).reshape(-1)
    # urutkan menurut skor supaya keluaran deterministik
    return keep[np.argsort(-scores[keep], kind="stable")]


class HaarDetector(Detector):
    """Haar Cascade `frontalface_default`."""

    def __init__(
        self,
        params: HaarConfig,
        name: str = "haar",
        score_mode: bool = False,
        nms_iou: float = 0.3,
        keep_stages: bool = False,
        cascade_file: str = DEFAULT_CASCADE,
    ) -> None:
        self.name = name
        self.params = params
        self.score_mode = score_mode
        self.has_scores = score_mode
        self.nms_iou = nms_iou
        self.keep_stages = keep_stages
        self.classifier = load_cascade(cascade_file)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        p = self.params
        start = time.perf_counter()
        gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if p.equalize:
            gray = cv2.equalizeHist(gray)

        scores: list[float] | None = None
        info: dict[str, object] = {}
        if not self.score_mode:
            found = self.classifier.detectMultiScale(
                gray, scaleFactor=p.scale_factor, minNeighbors=p.min_neighbors,
                minSize=tuple(p.min_size), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            boxes = [tuple(int(v) for v in b) for b in found]
        else:
            found, _, weights = self.classifier.detectMultiScale3(
                gray, scaleFactor=p.scale_factor, minNeighbors=p.min_neighbors,
                minSize=tuple(p.min_size), flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True,
            )
            found = np.asarray(found, dtype=int).reshape(-1, 4)
            weights = np.asarray(weights, dtype=float).reshape(-1)
            keep = nms_with_shifted_scores(found, weights, self.nms_iou)
            boxes = [tuple(int(v) for v in found[i]) for i in keep]
            scores = [float(weights[i]) for i in keep]
            info["raw_candidates"] = int(len(found))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        stages = {"grayscale_equalized" if p.equalize else "grayscale": gray} if self.keep_stages else {}
        return DetectionResult(boxes=boxes, scores=scores, elapsed_ms=elapsed_ms, stages=stages, info=info)
