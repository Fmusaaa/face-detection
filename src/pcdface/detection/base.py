"""Kontrak bersama semua detektor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcdface.boxes import Box


@dataclass(kw_only=True)
class DetectionResult:
    """Hasil deteksi satu citra.

    boxes      : kotak (x, y, w, h) dalam piksel citra masukan
    scores     : skor per kotak, atau None bila detektor tidak punya skor.
                 Skor Haar (`levelWeights`) bukan probabilitas — hanya urutannya
                 yang bermakna (PRD §5.1).
    elapsed_ms : waktu deteksi, termasuk konversi warna yang dibutuhkan
                 detektor, tanpa baca berkas (PRD §12.2 butir 8)
    stages     : citra antar-tahap untuk laporan (opsional)
    info       : keterangan khusus detektor (mis. kandidat YCbCr yang ditolak)
    """

    boxes: list[Box]
    scores: list[float] | None = None
    elapsed_ms: float
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scores is not None and len(self.scores) != len(self.boxes):
            raise ValueError(
                f"jumlah skor ({len(self.scores)}) ≠ jumlah kotak ({len(self.boxes)})"
            )


class Detector(ABC):
    """Detektor wajah. Pakai sebagai context manager agar sumber dayanya dilepas."""

    #: nama pendaftaran, mis. "haar", "mp_short"
    name: str = ""
    #: True bila `detect` mengisi `scores`
    has_scores: bool = False

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        """Deteksi wajah pada citra BGR uint8."""

    def close(self) -> None:
        """Lepas sumber daya (model, GPU). Bawaan: tidak ada yang dilepas."""

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
