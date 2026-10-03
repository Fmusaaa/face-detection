"""YOLO-face lewat OpenCV DNN: letterbox, dua tata letak keluaran, NMS, config, pengunduh — tanpa model."""

import copy

import numpy as np
import pytest
import yaml

from pcdface.config import ConfigError, YoloConfig, parse_config
from pcdface.detection.yolo import (
    Letterbox,
    YoloDetector,
    decode_raw,
    decode_ultralytics,
    detect_layout,
    letterbox,
    postprocess,
)
from pcdface.paths import CONFIG_PATH
from pcdface.selftest import yolo_raw_outputs
from pcdface.tools.download_models import MODELS, is_onnx


@pytest.mark.parametrize("shape, scale, pad", [
    ((720, 1280), 0.5, (0, 140)),       # 16:9 → pita atas-bawah
    ((360, 640), 1.0, (0, 140)),        # tidak diubah ukurannya
    ((1080, 810), 640 / 1080, (80, 0)),  # potret → pita kiri-kanan
    ((512, 512), 1.25, (0, 0)),         # diperbesar
])
def test_letterbox_geometry(shape, scale, pad):
    padded, lb = letterbox(np.full((*shape, 3), 7, np.uint8), 640)
    assert padded.shape == (640, 640, 3)
    assert lb.scale == pytest.approx(scale) and (lb.pad_x, lb.pad_y) == pad
    if pad[1]:
        assert (padded[0, 320] == 114).all() and (padded[320, 320] == 7).all()   # isi abu-abu 114 seperti ultralytics


def test_raw_head_decodes_dfl_box_back_to_image():
    raw = yolo_raw_outputs((16, 10, 15), bins=2)
    assert detect_layout(raw) == "raw"
    boxes, scores, points = decode_raw(raw, 640)
    assert boxes.shape == (80 * 80 + 40 * 40 + 20 * 20, 4) and points is not None and points.shape[1:] == (5, 2)
    lb = Letterbox(0.5, 0, 140)
    found, found_scores, found_points = postprocess(boxes, scores, points, lb, (1280, 720), 0.25, 0.7, 300)
    assert found == [(272, 152, 128, 128)]
    assert found_scores[0] == pytest.approx(1 / (1 + np.exp(-4.0)))
    assert len(found_points) == 1 and len(found_points[0]) == 5


@pytest.mark.parametrize("stride, gx, gy", [(8, 3, 70), (32, 19, 0)])
def test_raw_head_cell_center_per_stride(stride, gx, gy):
    """Pusat kotak = (sel + 0,5) × stride, di semua skala."""
    boxes, scores, _ = decode_raw(yolo_raw_outputs((stride, gx, gy), bins=1), 640)
    best = boxes[int(np.argmax(scores))]
    assert ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2) == pytest.approx(((gx + 0.5) * stride, (gy + 0.5) * stride))
    assert best[2] - best[0] == pytest.approx(2 * stride, abs=1e-3)


def test_ultralytics_layout_both_orientations_and_keypoints():
    pred = np.zeros((1, 5, 8400), np.float32)
    pred[0, :, 7] = [320, 320, 100, 50, 0.6]
    for tensor in (pred, pred.transpose(0, 2, 1)):            # (1, C, N) dan (1, N, C)
        boxes, scores, points = decode_ultralytics(tensor)
        assert scores[7] == pytest.approx(0.6) and boxes[7].tolist() == [270, 295, 370, 345] and points is None
    pose = np.zeros((1, 20, 8400), np.float32)                # model pose: 5 titik (x, y, keyakinan)
    pose[0, 5:8, 3] = [11, 22, 0.9]
    assert decode_ultralytics(pose)[2][3, 0].tolist() == [11, 22]


def test_postprocess_threshold_nms_cap_and_clip():
    lb = Letterbox(1.0, 0, 0)
    boxes = np.array([[10, 10, 50, 50], [12, 10, 52, 50], [100, 100, 140, 140], [-20, -20, 5, 5], [0, 0, 9, 9]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.25])              # 0,25 = ambang → dibuang (harus lebih besar)
    found, found_scores, _ = postprocess(boxes, scores, None, lb, (120, 120), 0.25, 0.7, 300)
    assert found == [(10, 10, 40, 40), (100, 100, 20, 20), (0, 0, 5, 5)]   # duplikat dibuang, kotak dipotong
    assert found_scores == pytest.approx([0.9, 0.7, 0.6])
    assert len(postprocess(boxes, scores, None, lb, (120, 120), 0.25, 0.7, 2)[0]) == 2
    assert postprocess(boxes, scores, None, lb, (120, 120), 0.95, 0.7, 300) == ([], [], [])


def test_unknown_layout_rejected():
    with pytest.raises(ValueError, match="tidak dikenal"):
        detect_layout([np.zeros((1, 3, 8400), np.float32)])


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-models"):
        YoloDetector(tmp_path / "tidak_ada.onnx")


def test_yolo_config_parsed_and_validated(cfg):
    spec = cfg.detector("yolo_n")
    assert isinstance(spec, YoloConfig) and spec.model.endswith(".onnx") and spec.input_size == 640
    assert cfg.evaluation.ap_run.yolo_conf_threshold < spec.conf_threshold
    for key in ("e1", "e2", "e3", "e4", "e6", "e7"):
        assert "yolo_n" in getattr(cfg.experiments, key).detectors
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    broken = copy.deepcopy(raw)
    broken["detectors"]["yolo_n"]["model"] = "yolov8n-face.pt"
    broken["detectors"]["yolo_n"]["input_size"] = 600
    broken["experiments"]["e8"]["detector"] = "tidak_ada"
    with pytest.raises(ConfigError) as error:
        parse_config(broken)
    message = str(error.value)
    assert "detectors.yolo_n.model" in message and "detectors.yolo_n.input_size" in message
    assert "experiments.e8.detector" in message


def test_download_list_has_pinned_yolo_and_rejects_html(tmp_path):
    spec = next(m for m in MODELS if m.filename == "yolov8n-face.onnx")
    assert spec.kind == "onnx" and "/main/" not in spec.url          # dipatok ke satu commit
    html = tmp_path / "galat.onnx"
    html.write_text("<!DOCTYPE html>")
    assert not is_onnx(html)
    proto = tmp_path / "model.onnx"
    proto.write_bytes(b"\x08\x07\x12\x07pytorch")
    assert is_onnx(proto)
