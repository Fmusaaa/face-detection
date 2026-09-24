"""Analisis jarak (PRD §8.4, §12.2 butir 6).

1. Validasi model kamera: regresi log(w_px) terhadap log(Z). Kemiringan ≈ −1
   membuktikan w ∝ 1/Z; jauh dari −1 menandakan Center Stage masih aktif.
2. Jarak optimal: jarak dengan estimasi titik recall ≥ target.
3. Jarak maksimum efektif: jarak terbesar yang memenuhi target, dan versi
   "berturut-turut" (semua jarak dari yang terdekat sampai jarak itu memenuhi).
4. Ukuran wajah minimum: batas bawah bin terkecil sehingga bin itu dan semua
   bin di atasnya memenuhi target (bin kosong diabaikan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LogLogFit:
    slope: float
    intercept: float
    r2: float
    n: int
    slope_se: float

    @property
    def slope_ci95(self) -> tuple[float, float]:
        """Interval 95% aproksimasi normal (n besar)."""
        return self.slope - 1.96 * self.slope_se, self.slope + 1.96 * self.slope_se

    @property
    def equation(self) -> str:
        return f"w = {math.exp(self.intercept):.0f} · Z^{self.slope:.3f}"


def loglog_fit(distances: Sequence[float], widths: Sequence[float]) -> LogLogFit:
    """Regresi kuadrat terkecil log(w) = a + b·log(Z)."""
    x = np.log(np.asarray(distances, dtype=float))
    y = np.log(np.asarray(widths, dtype=float))
    n = len(x)
    if n < 3 or np.unique(x).size < 2:
        raise ValueError("butuh minimal 3 titik dan 2 jarak berbeda")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 and sxx > 0 else math.nan
    return LogLogFit(float(slope), float(intercept), r2, n, se)


@dataclass(frozen=True)
class RangeResult:
    qualifying: tuple[int, ...]          # jarak dengan recall ≥ target
    max_effective: int | None            # jarak terbesar yang memenuhi
    max_contiguous: int | None           # terbesar dengan semua jarak lebih dekat juga memenuhi


def distance_range(recall_by_distance: Mapping[int, float], target: float) -> RangeResult:
    ordered = sorted(recall_by_distance)
    qualifying = tuple(d for d in ordered if not math.isnan(recall_by_distance[d]) and recall_by_distance[d] >= target)
    contiguous = None
    for d in ordered:
        if d in qualifying:
            contiguous = d
        else:
            break
    return RangeResult(qualifying, max(qualifying) if qualifying else None, contiguous)


@dataclass(frozen=True)
class SizeBin:
    low: float
    high: float
    n: int
    detected: int

    @property
    def recall(self) -> float:
        return self.detected / self.n if self.n else math.nan


@dataclass(frozen=True)
class MinSizeResult:
    threshold: float | None
    bins: tuple[SizeBin, ...]


def bin_detections(sizes: Sequence[float], detected: Sequence[bool], edges: Sequence[float]) -> list[SizeBin]:
    """Kelompokkan wajah ke bin [low, high)."""
    sizes_arr = np.asarray(sizes, dtype=float)
    hits = np.asarray(detected, dtype=bool)
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (sizes_arr >= low) & (sizes_arr < high)
        bins.append(SizeBin(float(low), float(high), int(inside.sum()), int(hits[inside].sum())))
    return bins


def min_face_size(sizes: Sequence[float], detected: Sequence[bool], edges: Sequence[float],
                  target: float) -> MinSizeResult:
    bins = bin_detections(sizes, detected, edges)
    threshold = None
    for index in range(len(bins) - 1, -1, -1):
        if bins[index].n == 0:
            continue
        if bins[index].recall >= target:
            threshold = bins[index].low
        else:
            break
    return MinSizeResult(threshold, tuple(bins))
