"""Gabungkan metadata dan anotasi menjadi sampel siap dievaluasi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box
from pcdface.dataset.annotations import load_annotations
from pcdface.dataset.metadata import MetadataRow, read_metadata
from pcdface.paths import ProjectPaths


@dataclass
class Sample:
    """Satu citra berlabel."""

    meta: MetadataRow
    gt: list[Box]
    path: Path

    @property
    def key(self) -> str:
        return self.meta.file

    def load(self) -> np.ndarray:
        image = cv2.imread(str(self.path))
        if image is None:
            raise FileNotFoundError(f"gagal membaca citra: {self.path}")
        return image


@dataclass
class LoadReport:
    samples: list[Sample]
    unannotated: list[str]
    missing_files: list[str]


def load_samples(paths: ProjectPaths, sets: tuple[str, ...] | None = None) -> LoadReport:
    """Sampel dari metadata yang sudah dianotasi dan berkasnya ada.

    Citra tanpa anotasi atau tanpa berkas tidak diikutkan, tetapi dicatat
    supaya eksperimen bisa memperingatkan (dan `validate` menangkapnya lebih dulu).
    """
    annotations = load_annotations(paths.annotations)
    samples: list[Sample] = []
    unannotated: list[str] = []
    missing: list[str] = []
    for row in read_metadata(paths.metadata):
        if sets is not None and row.set not in sets:
            continue
        path = paths.raw / row.file
        if not path.exists():
            missing.append(row.file)
            continue
        if row.file not in annotations:
            unannotated.append(row.file)
            continue
        samples.append(Sample(meta=row, gt=list(annotations[row.file]), path=path))
    return LoadReport(samples=samples, unannotated=unannotated, missing_files=missing)
