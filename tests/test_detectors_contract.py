"""Semua detektor mematuhi DetectionResult; registry; jebakan NMS dan urutan kanal."""

import numpy as np
import pytest

from pcdface.config import ConfigError
from pcdface.dataset.loader import load_samples
from pcdface.dataset.resize import scale_image
from pcdface.detection.base import DetectionResult, Detector
from pcdface.detection.fake import PROFILES, FakeDetector
from pcdface.detection.haar import nms_with_shifted_scores
from pcdface.detection.mediapipe_detector import to_mediapipe_pixels
from pcdface.detection.registry import build_detector, supports_scores


def check_contract(result: DetectionResult, image: np.ndarray, has_scores: bool) -> None:
    assert isinstance(result, DetectionResult)
    assert result.elapsed_ms >= 0
    height, width = image.shape[:2]
    for x, y, w, h in result.boxes:
        assert all(isinstance(v, int) for v in (x, y, w, h))
        assert w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height
    if has_scores:
        assert result.scores is not None and len(result.scores) == len(result.boxes)
    else:
        assert result.scores is None


def test_detection_result_rejects_score_length_mismatch():
    with pytest.raises(ValueError):
        DetectionResult(boxes=[(0, 0, 1, 1)], scores=[], elapsed_ms=1.0)
    with pytest.raises(TypeError):
        DetectionResult([(0, 0, 1, 1)], None, 1.0)  # kw_only


@pytest.fixture(scope="module")
def sample_image(synthetic_paths):
    samples = load_samples(synthetic_paths, sets=("multi",)).samples
    return samples[0].load()


@pytest.mark.parametrize("name", ["haar", "ycbcr"])
@pytest.mark.parametrize("mode", ["operating", "ap"])
def test_real_classical_detectors_contract(cfg, sample_image, name, mode):
    if mode == "ap" and not supports_scores(name, cfg):
        with pytest.raises(ValueError):
            build_detector(name, cfg, mode=mode)
        return
    with build_detector(name, cfg, mode=mode) as detector:
        assert isinstance(detector, Detector)
        result = detector.detect(sample_image)
        check_contract(result, sample_image, detector.has_scores)
        assert detector.has_scores == (mode == "ap")


@pytest.mark.parametrize("name", ["haar", "mp_short", "mp_full", "mp_sparse", "ycbcr", "yolo_n", "yolo_m"])
@pytest.mark.parametrize("mode", ["operating", "ap"])
def test_fake_detectors_contract_and_determinism(cfg, sample_image, name, mode):
    if mode == "ap" and not supports_scores(name, cfg):
        return
    detector = build_detector(name, cfg, mode=mode, synthetic=True)
    assert isinstance(detector, FakeDetector)
    first, second = detector.detect(sample_image), detector.detect(sample_image)
    check_contract(first, sample_image, detector.has_scores)
    assert first.boxes == second.boxes and first.scores == second.scores


def test_fake_profiles_mimic_absolute_vs_relative_limits(synthetic_paths):
    # wajah 50 px di 1280 → 25 px di 640: profil Haar turun, profil MediaPipe tetap
    haar, mp = FakeDetector("haar", PROFILES["haar"]), FakeDetector("mp_full", PROFILES["mp_full"])
    box_full, box_half = (0, 0, 50, 65), (0, 0, 25, 32)
    assert haar._probability(box_half, 640, 0.8, 150) < haar._probability(box_full, 1280, 0.8, 150) - 0.3
    assert mp._probability(box_half, 640, 0.8, 150) == pytest.approx(mp._probability(box_full, 1280, 0.8, 150))


def test_fake_yolo_is_relative_and_beats_mediapipe_on_small_faces():
    yolo, mp = FakeDetector("yolo_n", PROFILES["yolo_n"]), FakeDetector("mp_full", PROFILES["mp_full"])
    small = (0, 0, 26, 34)                                     # wajah ±300 cm pada 1280 px
    assert yolo._probability(small, 1280, 0.8, 150) > mp._probability(small, 1280, 0.8, 150)
    assert yolo._probability((0, 0, 13, 17), 640, 0.8, 150) == pytest.approx(yolo._probability(small, 1280, 0.8, 150))


def test_registry_model_helpers(cfg):
    from pcdface.detection.registry import model_file

    assert model_file("haar", cfg) is None and model_file("ycbcr", cfg) is None
    assert model_file("yolo_n", cfg) == cfg.paths.models / "yolov8n-face.onnx"
    assert model_file("mp_short", cfg).suffix == ".tflite"


def test_registry_overrides_and_errors(cfg):
    detector = build_detector("haar", cfg, overrides={"equalize": False})
    assert detector.params.equalize is False
    ap = build_detector("haar", cfg, mode="ap")
    assert ap.params.min_neighbors == cfg.evaluation.ap_run.haar_min_neighbors
    with pytest.raises(ConfigError):
        build_detector("haar", cfg, overrides={"tidak_ada": 1})
    with pytest.raises(ConfigError):
        build_detector("yunet", cfg)
    with pytest.raises(ValueError):
        build_detector("haar", cfg, mode="lain")


def test_mediapipe_missing_model_message(cfg, tmp_path):
    from pcdface.detection.mediapipe_detector import MediaPipeDetector

    with pytest.raises(FileNotFoundError, match="download-models"):
        MediaPipeDetector(tmp_path / "tidak_ada.tflite")


# ---------------------------------------------------------------------------
# Jebakan yang dikunci tes
# ---------------------------------------------------------------------------
def test_nms_keeps_single_and_negative_score_candidates():
    boxes = np.array([[0, 0, 10, 10]])
    assert nms_with_shifted_scores(boxes, np.array([-2.5]), 0.3).tolist() == [0]
    boxes = np.array([[0, 0, 10, 10], [1, 0, 10, 10], [100, 0, 10, 10]])
    keep = nms_with_shifted_scores(boxes, np.array([-1.0, -3.0, -2.0]), 0.3)
    assert keep.tolist() == [0, 2]            # kotak 1 tertindih kotak 0; urut skor menurun
    assert nms_with_shifted_scores(np.zeros((0, 4)), np.zeros(0), 0.3).size == 0


def test_mediapipe_pixels_are_rgba_not_bgr():
    blue_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255                    # biru murni dalam urutan BGR
    rgba = to_mediapipe_pixels(blue_bgr)
    assert rgba.shape == (2, 3, 4) and rgba.flags["C_CONTIGUOUS"]
    assert rgba[0, 0].tolist() == [0, 0, 255, 255]  # R=0, G=0, B=255, A=255
    with pytest.raises(ValueError):
        to_mediapipe_pixels(np.zeros((2, 2), dtype=np.uint8))


def test_haar_detects_smaller_faces_less_after_downscale(cfg, synthetic_paths):
    """Sanity: jalur Haar asli berjalan pada kedua resolusi tanpa galat."""
    image = load_samples(synthetic_paths, sets=("jarak",)).samples[0].load()
    with build_detector("haar", cfg) as detector:
        for scale in (1.0, 0.5):
            small = scale_image(image, scale)
            check_contract(detector.detect(small), small, False)
