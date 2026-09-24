"""Metrik pada titik operasi (PRD §8.1): TP/FP/FN, P/R/F1, rerata IoU, FPPI.

Nilai yang tidak terdefinisi dikembalikan NaN, bukan 0 — mis. precision
tanpa satu pun deteksi — supaya tabel tidak menyiratkan kinerja nol.
F1 dihitung sebagai 2TP / (2TP + FP + FN), setara dengan rerata harmonik
P dan R, tetapi tetap terdefinisi bila salah satunya tidak.

Akurasi tidak dipakai karena true negative tidak terdefinisi pada deteksi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from pcdface.boxes import Box
from pcdface.evaluation.matching import match_greedy


@dataclass
class ImageEval:
    """Hasil satu citra pada satu ambang IoU."""

    key: str
    group: str
    n_gt: int
    tp: int
    fp: int
    fn: int
    ious: list[float] = field(default_factory=list)
    gt_matched: list[bool] = field(default_factory=list)


def evaluate_image(key: str, group: str, gt: Sequence[Box], boxes: Sequence[Box], iou_threshold: float) -> ImageEval:
    match = match_greedy(gt, boxes, iou_threshold)
    return ImageEval(
        key=key, group=group, n_gt=len(gt),
        tp=match.true_positive, fp=match.false_positive, fn=match.false_negative,
        ious=list(match.matched_ious), gt_matched=list(match.gt_matched),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


@dataclass
class OperatingPoint:
    images: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def mean_iou(self) -> float:
        return _ratio(self.iou_sum, self.iou_count)

    @property
    def fppi(self) -> float:
        """Deteksi palsu per citra."""
        return _ratio(self.fp, self.images)

    @property
    def n_gt(self) -> int:
        return self.tp + self.fn


def aggregate(evals: Iterable[ImageEval]) -> OperatingPoint:
    total = OperatingPoint()
    for e in evals:
        total.images += 1
        total.tp += e.tp
        total.fp += e.fp
        total.fn += e.fn
        total.iou_sum += sum(e.ious)
        total.iou_count += len(e.ious)
    return total
