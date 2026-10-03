"""Tes asap MediaPipe (.tflite) dan YOLOv8n-face (.onnx) dengan berkas model asli — `pytest -m models`.

Tanpa citra wajah asli, tes ini hanya memastikan detektor bisa dibuat,
dijalankan (mode IMAGE dan VIDEO), dan ditutup. Bila variabel lingkungan
PCDFACE_SMOKE_IMAGE menunjuk ke foto berisi satu wajah, tes juga memeriksa
bahwa wajah terdeteksi dan bahwa masukan urutan kanal salah memberi skor
lebih rendah (aturan #2).
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from pcdface.config import MediaPipeConfig
from pcdface.detection.registry import build_detector

pytestmark = pytest.mark.models
MP_NAMES = ["mp_short", "mp_full", "mp_sparse"]


@pytest.fixture(autouse=True)
def require_models(cfg):
    missing = [cfg.detector(n).model for n in MP_NAMES
               if not (cfg.paths.models / cfg.detector(n).model).exists()]
    if missing:
        pytest.skip(f"model belum diunduh: {missing} — jalankan python -m pcdface download-models")


@pytest.mark.parametrize("name", MP_NAMES)
@pytest.mark.parametrize("mode", ["operating", "ap"])
def test_mediapipe_runs_on_blank_and_synthetic(cfg, synthetic_paths, name, mode):
    assert isinstance(cfg.detector(name), MediaPipeConfig)
    blank = np.full((720, 1280, 3), 128, dtype=np.uint8)
    synthetic = cv2.imread(str(next((synthetic_paths.raw / "multi").rglob("*.jpg"))))
    with build_detector(name, cfg, mode=mode) as detector:
        empty = detector.detect(blank)
        if mode == "operating":
            assert empty.boxes == [] and empty.scores == []
        else:
            # run AP (ambang 0,05) boleh menemukan kandidat lemah — semuanya harus di bawah ambang titik operasi
            assert all(s < cfg.detector(name).min_detection_confidence for s in empty.scores), empty.scores
        result = detector.detect(synthetic)
        assert len(result.scores) == len(result.boxes)
        assert all(0.0 <= s <= 1.0 for s in result.scores)


def test_mediapipe_video_mode_accepts_repeated_timestamps(cfg):
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    with build_detector("mp_short", cfg, running_mode="video") as detector:
        for _ in range(3):
            detector.detect(frame, timestamp_ms=1000)  # cap waktu sama dinaikkan otomatis


@pytest.mark.skipif(not os.environ.get("PCDFACE_SMOKE_IMAGE"), reason="PCDFACE_SMOKE_IMAGE tidak diset")
def test_real_face_detected_and_channel_order_matters(cfg):
    image = cv2.imread(str(Path(os.environ["PCDFACE_SMOKE_IMAGE"])))
    assert image is not None
    with build_detector("mp_short", cfg) as detector:
        correct = detector.detect(image)
        swapped = detector.detect(image[:, :, ::-1].copy())  # detektor akan "membalik" ke urutan salah
    assert len(correct.boxes) >= 1
    assert max(correct.scores) > max(swapped.scores or [0.0])
    with build_detector("haar", cfg) as haar:
        assert len(haar.detect(image).boxes) >= 1


# ---------------------------------------------------------------------------
# YOLOv8n-face (OpenCV DNN)
# ---------------------------------------------------------------------------
@pytest.fixture
def yolo_model(cfg):
    path = cfg.paths.models / cfg.detector("yolo_n").model
    if not path.exists():
        pytest.skip("yolov8n-face.onnx belum diunduh — jalankan python -m pcdface download-models")
    return path


@pytest.mark.parametrize("mode", ["operating", "ap"])
def test_yolo_runs_on_blank_and_synthetic(cfg, synthetic_paths, yolo_model, mode):
    blank = np.full((720, 1280, 3), 128, dtype=np.uint8)
    synthetic = cv2.imread(str(next((synthetic_paths.raw / "multi").rglob("*.jpg"))))
    with build_detector("yolo_n", cfg, mode=mode) as detector:
        assert detector.layout == "raw"                      # head mentah derronqi/hpc203
        expected = cfg.evaluation.ap_run.yolo_conf_threshold if mode == "ap" else cfg.detector("yolo_n").conf_threshold
        assert detector.conf_threshold == expected
        assert detector.detect(blank).boxes == []
        result = detector.detect(synthetic)
        assert len(result.scores) == len(result.boxes) and all(0.0 < s <= 1.0 for s in result.scores)


@pytest.mark.skipif(not os.environ.get("PCDFACE_SMOKE_IMAGE"), reason="PCDFACE_SMOKE_IMAGE tidak diset")
def test_yolo_finds_real_face_with_landmarks(cfg, yolo_model):
    image = cv2.imread(str(Path(os.environ["PCDFACE_SMOKE_IMAGE"])))
    with build_detector("yolo_n", cfg) as detector:
        result = detector.detect(image)
    assert len(result.boxes) >= 1 and max(result.scores) > 0.5
    assert len(result.info["keypoints"]) == len(result.boxes) and len(result.info["keypoints"][0]) == 5
