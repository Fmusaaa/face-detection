"""Kurva precision–recall dan Average Precision (PRD §8.2).

Seluruh deteksi dari semua citra diurutkan menurun menurut skor. AP memakai
interpolasi *all-point* PASCAL VOC 2010+ (Everingham dkk., 2010): precision
dibuat tidak naik dari kanan ke kiri, lalu dijumlahkan luas di bawahnya pada
setiap perubahan recall. Karena hanya urutan skor yang berpengaruh, skor Haar
(`levelWeights`) dan skor MediaPipe (0–1) bisa dibandingkan secara adil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pcdface.boxes import Box
from pcdface.evaluation.matching import match_by_score


@dataclass
class ScoredImage:
    """Deteksi satu citra yang sudah ditandai TP/FP pada satu ambang IoU."""

    key: str
    group: str
    n_gt: int
    scores: np.ndarray
    is_tp: np.ndarray


@dataclass
class PRCurve:
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray
    n_gt: int
    ap: float


def score_image(key: str, group: str, gt: Sequence[Box], boxes: Sequence[Box],
                scores: Sequence[float], iou_threshold: float) -> ScoredImage:
    flags = match_by_score(gt, boxes, scores, iou_threshold)
    return ScoredImage(key, group, len(gt), np.asarray(scores, dtype=float), np.asarray(flags, dtype=bool))


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP all-point dari recall (naik) dan precision berpasangan."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # selubung: tidak naik dari kanan ke kiri
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def pr_curve(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> PRCurve:
    """Kurva PR dari deteksi gabungan. `n_gt == 0` → AP NaN."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order].astype(float))
    fp = np.cumsum((~is_tp[order]).astype(float))
    if n_gt == 0:
        return PRCurve(np.array([]), np.array([]), scores[order], 0, math.nan)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    return PRCurve(precision, recall, scores[order], n_gt, average_precision(recall, precision))


def pool(images: Sequence[ScoredImage]) -> PRCurve:
    """Gabungkan citra (boleh berulang, untuk bootstrap) lalu hitung kurva PR."""
    if not images:
        return PRCurve(np.array([]), np.array([]), np.array([]), 0, math.nan)
    scores = np.concatenate([im.scores for im in images]) if images else np.array([])
    is_tp = np.concatenate([im.is_tp for im in images]) if images else np.array([], dtype=bool)
    return pr_curve(scores, is_tp, sum(im.n_gt for im in images))
