"""Interval kepercayaan (PRD §8.6, §12.2 butir 5).

- `wilson`: interval Wilson untuk proporsi (recall, precision, akurasi hitung).
- `cluster_bootstrap`: bootstrap persentil dengan resampel **kelompok** (subjek
  untuk set satu wajah, citra untuk multi-wajah), karena frame dari subjek
  yang sama saling berkorelasi.
- `paired_difference`: bootstrap berpasangan untuk selisih dua detektor pada
  resampel kelompok yang sama — dasar pernyataan "lebih baik".

Referensi: Wilson, E. B. (1927). JASA 22(158), 209–212.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """Nilai titik beserta interval kepercayaannya."""

    value: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """True bila interval tidak memuat 0 (untuk selisih)."""
        return (self.low > 0 or self.high < 0) and not math.isnan(self.low)

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {prefix: self.value, f"{prefix}_low": self.low, f"{prefix}_high": self.high}


NAN_ESTIMATE = Estimate(math.nan, math.nan, math.nan)


def _z(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def wilson(successes: int, trials: int, level: float = 0.95) -> Estimate:
    """Interval skor Wilson. `trials == 0` → NaN (tidak terdefinisi)."""
    if trials == 0:
        return NAN_ESTIMATE
    if not 0 <= successes <= trials:
        raise ValueError("successes harus di antara 0 dan trials")
    z = _z(level)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return Estimate(p, max(0.0, center - half), min(1.0, center + half))


def _resample_indices(n_groups: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=(n_resamples, n_groups))


def cluster_bootstrap(
    groups: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> Estimate:
    """Bootstrap persentil dengan resampel kelompok berpengembalian.

    `statistic` menerima daftar kode kelompok (boleh berulang) dan
    mengembalikan satu angka. Resampel yang menghasilkan NaN diabaikan.
    """
    unique = sorted(set(groups))
    if not unique:
        return NAN_ESTIMATE
    point = statistic(unique)
    values = np.array([
        statistic([unique[i] for i in row]) for row in _resample_indices(len(unique), n_resamples, seed)
    ], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return Estimate(point, math.nan, math.nan)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return Estimate(point, float(low), float(high))


def paired_difference(
    groups: Sequence[str],
    statistic_a: Callable[[Sequence[str]], float],
    statistic_b: Callable[[Sequence[str]], float],
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> Estimate:
    """Selisih A − B dengan resampel kelompok yang sama untuk keduanya."""
    return cluster_bootstrap(
        groups, lambda keys: statistic_a(keys) - statistic_b(keys),
        n_resamples=n_resamples, level=level, seed=seed,
    )
