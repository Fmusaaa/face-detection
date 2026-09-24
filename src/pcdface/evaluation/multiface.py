"""Metrik multi-wajah (PRD §8.3) dan pemasangan kotak ke posisi jarak (§6.6).

Jarak setiap wajah pada citra multi-wajah ditentukan dengan mengurutkan
kotak manual dari kiri ke kanan lalu memasangkannya dengan `positions_cm`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pcdface.boxes import Box, center_x


def count_accuracy(predicted: Sequence[int], actual: Sequence[int]) -> tuple[int, int]:
    """(jumlah citra dengan hitungan tepat, jumlah citra)."""
    if len(predicted) != len(actual):
        raise ValueError("panjang predicted dan actual harus sama")
    return sum(int(p == a) for p, a in zip(predicted, actual)), len(actual)


def count_mae(predicted: Sequence[int], actual: Sequence[int]) -> float:
    """Rerata |deteksi − sebenarnya|."""
    if len(predicted) != len(actual):
        raise ValueError("panjang predicted dan actual harus sama")
    if not actual:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def positions_of(gt: Sequence[Box], positions_cm: Sequence[int]) -> list[int]:
    """Jarak tiap kotak manual (urutan asli `gt`) dari urutan kiri → kanan."""
    if len(gt) != len(positions_cm):
        raise ValueError(f"{len(gt)} kotak tetapi {len(positions_cm)} posisi")
    order = sorted(range(len(gt)), key=lambda i: center_x(gt[i]))
    distance = [0] * len(gt)
    for rank, index in enumerate(order):
        distance[index] = int(positions_cm[rank])
    return distance


def position_hits(gt: Sequence[Box], gt_matched: Sequence[bool], positions_cm: Sequence[int]) -> list[tuple[int, bool]]:
    """[(jarak, terdeteksi)] per wajah untuk recall per posisi jarak."""
    return list(zip(positions_of(gt, positions_cm), (bool(m) for m in gt_matched)))
