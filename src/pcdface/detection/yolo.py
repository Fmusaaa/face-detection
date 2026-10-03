"""Detektor YOLO-face (YOLOv8) lewat OpenCV DNN — tanpa PyTorch (PRD §5.3).

Mengapa OpenCV DNN, bukan paket `ultralytics`: `ultralytics` menarik
`opencv-python`, yang menimpa modul `cv2` milik `opencv-contrib-python`
(CLAUDE.md aturan #3). Model `.onnx` dibaca `cv2.dnn.readNetFromONNX` dan
dijalankan di CPU. Model `.pt` diekspor ke ONNX di venv terpisah (`export-yolo`).

Dua tata letak keluaran ONNX dikenali otomatis saat detektor dibuat:

- ``ultralytics`` — satu tensor (1, 4+1[+3k], N) dari `YOLO(...).export(format="onnx")`.
  Kotak (cx, cy, w, h) sudah di ruang masukan, skor sudah sigmoid. Dipakai model
  akanametov/yolo-face (mis. `yolov8m-face` seperti repo referensi MariyaSha).
- ``raw`` — tiga tensor (1, 4·16+1+3k, H, W), satu per stride 8/16/32: keluaran head
  mentah derronqi/yolov8-face (ONNX dari hpc203). Kotak didekode dengan DFL (softmax
  16 bin → jarak ke tepi), skor dan landmark diaktifkan dengan sigmoid.

Prapemrosesan sama dengan ultralytics: letterbox (rasio aspek dijaga, sisa diisi
abu-abu 114 di tengah), BGR→RGB, skala 1/255. Karena seluruh frame diperkecil ke
640×640, batas ukuran wajahnya **relatif terhadap lebar frame** seperti BlazeFace,
tetapi resolusi masukannya 5× lebih besar (640 vs 128) — hipotesis H1/H2 di PRD §2.2.

NMS memakai `cv2.dnn.NMSBoxes`. Skor sigmoid selalu > 0 dan kandidat sudah disaring
> ambang keyakinan sebelum NMS, jadi jebakan skor ≤ 0 (CLAUDE.md aturan #8) tidak terjadi.

Cara kerja singkat (Landasan Teori): YOLO memprediksi kotak dan skor untuk setiap sel
grid pada tiga skala sekaligus dalam satu kali inferensi (*single-stage*); YOLOv8
*anchor-free* — setiap sel memprediksi jarak ke empat tepi kotak sebagai distribusi
diskret (DFL), lalu kotak yang tumpang tindih disaring dengan NMS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box, clip_box
from pcdface.detection.base import DetectionResult, Detector

LAYOUTS = ("ultralytics", "raw")
PAD_VALUE = 114          # abu-abu letterbox ultralytics
REG_MAX = 16             # bin DFL YOLOv8


@dataclass(frozen=True)
class Letterbox:
    """Parameter letterbox untuk mengembalikan kotak ke koordinat citra asli."""

    scale: float
    pad_x: int
    pad_y: int


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Perkecil/perbesar dengan rasio aspek tetap lalu isi tepi sampai size×size (seperti ultralytics)."""
    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    if (new_w, new_h) != (width, height):
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    top, left = int(round(dh / 2 - 0.1)), int(round(dw / 2 - 0.1))
    padded = cv2.copyMakeBorder(image_bgr, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT,
                                value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    return padded, Letterbox(scale, left, top)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def detect_layout(outputs: list[np.ndarray]) -> str:
    """'ultralytics' atau 'raw' menurut bentuk tensor keluaran; ValueError bila tidak dikenal."""
    if len(outputs) == 1 and outputs[0].ndim == 3 and min(outputs[0].shape[1:]) >= 5:
        return "ultralytics"
    if len(outputs) >= 2 and all(o.ndim == 4 and o.shape[1] > 4 * REG_MAX for o in outputs):
        return "raw"
    shapes = [tuple(o.shape) for o in outputs]
    raise ValueError(
        f"tata letak keluaran YOLO tidak dikenal: {shapes}. Didukung: satu tensor (1, 5+, N) hasil "
        "ekspor ultralytics, atau tiga tensor (1, 65+, H, W) head mentah YOLOv8-face."
    )


def decode_ultralytics(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Satu tensor (1, C, N) atau (1, N, C) → (kotak xyxy, skor, landmark (N, K, 2) atau None)."""
    pred = output[0]
    if pred.shape[0] > pred.shape[1]:   # (N, C) → (C, N); C selalu jauh lebih kecil dari N
        pred = pred.T
    channels = pred.shape[0]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = pred[4].astype(np.float64)
    keypoints = None
    extra = channels - 5
    if extra > 0 and extra % 3 == 0:    # model pose: (x, y, keyakinan) per titik
        keypoints = pred[5:].T.reshape(-1, extra // 3, 3)[..., :2]
    return boxes, scores, keypoints


def decode_raw(outputs: list[np.ndarray], input_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Tiga tensor head mentah (1, 64+1+3k, H, W) → (kotak xyxy, skor, landmark atau None)."""
    all_boxes, all_scores, all_points = [], [], []
    bins = np.arange(REG_MAX, dtype=np.float64)
    for out in sorted(outputs, key=lambda o: -o.shape[2]):          # stride kecil dulu
        _, channels, grid_h, grid_w = out.shape
        stride = input_size / grid_h
        pred = out[0].reshape(channels, grid_h * grid_w).T.astype(np.float64)   # (H·W, C), baris demi baris
        logits = pred[:, :4 * REG_MAX].reshape(-1, 4, REG_MAX)
        logits = logits - logits.max(axis=-1, keepdims=True)
        prob = np.exp(logits)
        dist = (prob / prob.sum(axis=-1, keepdims=True)) @ bins           # (H·W, 4): kiri, atas, kanan, bawah
        gx, gy = np.meshgrid(np.arange(grid_w) + 0.5, np.arange(grid_h) + 0.5)
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        all_boxes.append(np.stack([gx - dist[:, 0], gy - dist[:, 1], gx + dist[:, 2], gy + dist[:, 3]], axis=1) * stride)
        all_scores.append(_sigmoid(pred[:, 4 * REG_MAX]))
        extra = channels - 4 * REG_MAX - 1
        if extra > 0 and extra % 3 == 0:
            points = pred[:, 4 * REG_MAX + 1:].reshape(-1, extra // 3, 3)
            px = (points[..., 0] * 2.0 + (gx[:, None] - 0.5)) * stride
            py = (points[..., 1] * 2.0 + (gy[:, None] - 0.5)) * stride
            all_points.append(np.stack([px, py], axis=-1))
    keypoints = np.concatenate(all_points) if len(all_points) == len(all_boxes) and all_points else None
    return np.concatenate(all_boxes), np.concatenate(all_scores), keypoints


def postprocess(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray | None,
    lb: Letterbox,
    image_size: tuple[int, int],
    conf_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> tuple[list[Box], list[float], list[list[tuple[float, float]]]]:
    """Saring skor, NMS, kembalikan ke koordinat citra asli, potong ke batas citra.

    Kotak di ruang letterbox; IoU tidak berubah oleh skala + geser seragam, jadi NMS
    di ruang itu sama dengan NMS di ruang citra asli.
    """
    width, height = image_size
    keep = np.flatnonzero(scores > conf_threshold)
    if keep.size == 0:
        return [], [], []
    boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]
    xywh = np.column_stack([boxes_xyxy[:, :2], boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]])
    # semua skor > conf_threshold > 0 (aturan #8 aman); ambang skor NMS 0 agar tidak ada yang dibuang ulang
    picked = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, float(nms_iou))
    picked = np.asarray(picked, dtype=int).reshape(-1)
    picked = picked[np.argsort(-scores[picked], kind="stable")][:max_detections]

    out_boxes: list[Box] = []
    out_scores: list[float] = []
    out_points: list[list[tuple[float, float]]] = []
    for index in picked:
        x1, y1, x2, y2 = (boxes_xyxy[index] - [lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y]) / lb.scale
        left, top = int(round(x1)), int(round(y1))
        box = clip_box((left, top, int(round(x2)) - left, int(round(y2)) - top), width, height)
        if box is None:
            continue
        out_boxes.append(box)
        out_scores.append(float(scores[index]))
        if keypoints is not None:
            points = (keypoints[keep[index]] - [lb.pad_x, lb.pad_y]) / lb.scale
            out_points.append([(float(px), float(py)) for px, py in points])
    return out_boxes, out_scores, out_points


class YoloDetector(Detector):
    """YOLOv8-face lewat OpenCV DNN (CPU)."""

    has_scores = True

    def __init__(
        self,
        model_path: Path,
        name: str = "yolo",
        input_size: int = 640,
        conf_threshold: float = 0.25,
        nms_iou: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model {model_path.name} tidak ada di {model_path.parent}. Jalankan: python -m pcdface "
                "download-models (yolov8n-face.onnx), atau python -m pcdface export-yolo untuk model .pt "
                "akanametov (mis. yolov8m-face seperti repo referensi)"
            )
        self.name = name
        self.model_path = model_path
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._outputs = list(self._net.getUnconnectedOutLayersNames())
        # satu inferensi pada citra kosong: memastikan model terbaca dan tata letaknya dikenali
        self.layout = detect_layout(self._forward(np.full((self.input_size, self.input_size, 3), PAD_VALUE, np.uint8)))

    def _forward(self, padded_bgr: np.ndarray) -> list[np.ndarray]:
        blob = cv2.dnn.blobFromImage(padded_bgr, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self._net.setInput(blob)
        return [np.asarray(o) for o in self._net.forward(self._outputs)]

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("butuh citra BGR 3 kanal")
        height, width = image_bgr.shape[:2]
        start = time.perf_counter()
        padded, lb = letterbox(image_bgr, self.input_size)
        outputs = self._forward(padded)
        if self.layout == "ultralytics":
            boxes, scores, keypoints = decode_ultralytics(outputs[0])
        else:
            boxes, scores, keypoints = decode_raw(outputs, self.input_size)
        out_boxes, out_scores, out_points = postprocess(
            boxes, scores, keypoints, lb, (width, height), self.conf_threshold, self.nms_iou, self.max_detections)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        info: dict[str, object] = {"layout": self.layout}
        if out_points:
            info["keypoints"] = out_points
        return DetectionResult(boxes=out_boxes, scores=out_scores, elapsed_ms=elapsed_ms, info=info)

    def close(self) -> None:
        self._net = None
