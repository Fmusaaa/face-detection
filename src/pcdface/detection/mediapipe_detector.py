"""Detektor MediaPipe BlazeFace lewat Tasks API (PRD §5.2).

Tiga hal yang wajib (CLAUDE.md aturan #1, #2):
- hanya Tasks API `mediapipe.tasks.python.vision.FaceDetector` — `mp.solutions`
  tidak ada di MediaPipe 1.0.1;
- OpenCV membaca BGR, MediaPipe butuh RGB: konversi BGR→RGBA, lalu
  `mp.Image(image_format=SRGBA)`. Salah urutan kanal tidak memunculkan galat,
  deteksi hanya diam-diam memburuk;
- delegate GPU (Metal) di macOS: delegate CPU MediaPipe 1.0.1 crash di macOS, dan
  delegate Metal hanya menerima citra 4 kanal. Di Windows/Linux (laptop anggota
  lain) `delegate: auto` memilih CPU; `SRGBA` juga diterima jalur CPU.
  Variabel lingkungan `PCDFACE_MP_DELEGATE=cpu|gpu` menimpa config.

Kotak dikembalikan dalam piksel dan dipotong ke batas citra.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import clip_box
from pcdface.detection.base import DetectionResult, Detector

RUNNING_MODES = ("image", "video")
DELEGATE_ENV = "PCDFACE_MP_DELEGATE"


def resolve_delegate(requested: str = "auto", platform: str | None = None) -> str:
    """'gpu' atau 'cpu'. Variabel lingkungan menang atas config; 'auto' = GPU hanya di macOS."""
    choice = (os.environ.get(DELEGATE_ENV) or requested or "auto").strip().lower()
    if choice not in ("auto", "gpu", "cpu"):
        raise ValueError(f"delegate MediaPipe harus auto, gpu, atau cpu — dapat {choice!r}")
    if choice == "auto":
        return "gpu" if (platform or sys.platform) == "darwin" else "cpu"
    return choice


def delegate_label(delegate: str, platform: str | None = None) -> str:
    """Nama perangkat untuk tabel dan HUD."""
    if delegate == "cpu":
        return "CPU"
    return "GPU (Metal)" if (platform or sys.platform) == "darwin" else "GPU"


def to_mediapipe_pixels(image_bgr: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) → RGBA kontigu untuk `mp.Image(SRGBA)`."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("butuh citra BGR 3 kanal")
    return np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA))


class MediaPipeDetector(Detector):
    """BlazeFace short-range / full-range / sparse."""

    has_scores = True

    def __init__(
        self,
        model_path: Path,
        name: str = "mp",
        min_detection_confidence: float = 0.5,
        min_suppression_threshold: float = 0.3,
        running_mode: str = "image",
        delegate: str = "auto",
    ) -> None:
        if running_mode not in RUNNING_MODES:
            raise ValueError(f"running_mode harus salah satu dari {RUNNING_MODES}")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model {model_path.name} tidak ada di {model_path.parent}. "
                "Jalankan: python -m pcdface download-models"
            )
        import mediapipe as mp  # impor di sini: berat, dan tidak dibutuhkan perintah lain
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.name = name
        self.running_mode = running_mode
        self.model_path = model_path
        self.delegate = resolve_delegate(delegate)
        mp_delegate = BaseOptions.Delegate.GPU if self.delegate == "gpu" else BaseOptions.Delegate.CPU
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=mp_delegate),

            running_mode=vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE,
            min_detection_confidence=float(min_detection_confidence),
            min_suppression_threshold=float(min_suppression_threshold),
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._last_timestamp = -1

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> DetectionResult:
        height, width = image_bgr.shape[:2]
        start = time.perf_counter()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=to_mediapipe_pixels(image_bgr))
        if self.running_mode == "video":
            # mode VIDEO menuntut cap waktu yang naik tegas
            stamp = int(timestamp_ms if timestamp_ms is not None else time.monotonic() * 1000)
            stamp = max(stamp, self._last_timestamp + 1)
            self._last_timestamp = stamp
            result = self._detector.detect_for_video(image, stamp)
        else:
            result = self._detector.detect(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        boxes, scores, keypoints = [], [], []
        for detection in result.detections:
            bb = detection.bounding_box
            clipped = clip_box((bb.origin_x, bb.origin_y, bb.width, bb.height), width, height)
            if clipped is None:
                continue
            boxes.append(clipped)
            scores.append(float(detection.categories[0].score))
            keypoints.append([(kp.x * width, kp.y * height) for kp in detection.keypoints])
        return DetectionResult(boxes=boxes, scores=scores, elapsed_ms=elapsed_ms, info={"keypoints": keypoints})

    def close(self) -> None:
        if getattr(self, "_detector", None) is not None:
            self._detector.close()
            self._detector = None
