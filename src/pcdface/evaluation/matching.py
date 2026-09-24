"""IoU dan pencocokan prediksi ke ground truth.

Dua cara pencocokan dipakai, sesuai kegunaannya:

- `match_greedy` — untuk titik operasi (PRD §8.1). Semua pasangan diurutkan
  menurun menurut IoU, lalu diambil selama kedua anggotanya belum terpakai.
  Tidak butuh skor, jadi berlaku juga untuk Haar mode bawaan dan YCbCr.
- `match_by_score` — untuk AP (PRD §8.2), mengikuti PASCAL VOC: prediksi
  diproses dari skor tertinggi; masing-masing dicocokkan ke ground truth
  dengan IoU terbesar. Bila ground truth itu sudah terpakai, prediksi
  dihitung FP (duplikat).

IoU = luas(P ∩ G) / luas(P ∪ G).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pcdface.boxes import Box


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over Union dua kotak (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(ground_truth: Sequence[Box], predictions: Sequence[Box]) -> np.ndarray:
    """Matriks IoU berukuran (jumlah GT, jumlah prediksi)."""
    matrix = np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predictions):
            matrix[i, j] = iou(gt, pred)
    return matrix


@dataclass
class MatchResult:
    """Hasil pencocokan satu citra."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    matched_ious: list[float] = field(default_factory=list)
    pred_matched: list[bool] = field(default_factory=list)   # per prediksi, urutan asli
    gt_matched: list[bool] = field(default_factory=list)     # per GT, urutan asli
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (gt, pred, iou)


def match_greedy(
    ground_truth: Sequence[Box],
    predictions: Sequence[Box],
    iou_threshold: float,
) -> MatchResult:
    """Pencocokan greedy berdasarkan IoU tertinggi (titik operasi)."""
    matrix = iou_matrix(ground_truth, predictions)
    candidates = [
        (matrix[i, j], i, j)
        for i in range(len(ground_truth))
        for j in range(len(predictions))
        if matrix[i, j] >= iou_threshold
    ]
    # IoU menurun; bila seri, indeks lebih kecil dulu supaya deterministik
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    gt_used = [False] * len(ground_truth)
    pred_used = [False] * len(predictions)
    result = MatchResult()
    for value, i, j in candidates:
        if gt_used[i] or pred_used[j]:
            continue
        gt_used[i] = pred_used[j] = True
        result.pairs.append((i, j, float(value)))
        result.matched_ious.append(float(value))

    result.true_positive = len(result.pairs)
    result.false_positive = len(predictions) - result.true_positive
    result.false_negative = len(ground_truth) - result.true_positive
    result.pred_matched = pred_used
    result.gt_matched = gt_used
    return result


def match_by_score(
    ground_truth: Sequence[Box],
    predictions: Sequence[Box],
    scores: Sequence[float],
    iou_threshold: float,
) -> list[bool]:
    """Tandai tiap prediksi TP/FP gaya PASCAL VOC. Hasil dalam urutan asli."""
    if len(scores) != len(predictions):
        raise ValueError("jumlah skor harus sama dengan jumlah prediksi")
    matrix = iou_matrix(ground_truth, predictions)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    gt_used = [False] * len(ground_truth)
    is_tp = [False] * len(predictions)
    for j in order:
        if not ground_truth:
            break
        i = int(np.argmax(matrix[:, j]))
        if matrix[i, j] >= iou_threshold and not gt_used[i]:
            gt_used[i] = True
            is_tp[j] = True
    return is_tp
