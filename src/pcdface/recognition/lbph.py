"""Pengenalan identitas LBPH lewat `cv2.face` (PRD v4 §5.4).

Alurnya sama dengan repo referensi MariyaSha/FaceRecognition: crop kotak wajah →
abu-abu → ubah ukuran 200×200 → `LBPHFaceRecognizer`. Prediksi = label galeri dengan
jarak histogram terkecil; jarak > `max_distance` → ``unknown``.

Privasi (PRD §11, CLAUDE.md aturan #5):

- label hanya kode pseudonim (`S01`, `S02`, …) — nama ditolak;
- model terlatih menyimpan histogram LBP wajah, yaitu **templat biometrik**: disimpan di
  `data/recognition/` (di-gitignore) dan dihapus oleh `forget`;
- hanya peserta dengan `consent_research` **dan** `consent_recognition` = ya yang boleh
  masuk galeri atau diuji (`consenting_subjects`).

Cara kerja singkat (Landasan Teori): LBP membandingkan setiap piksel dengan 8 tetangganya
pada radius 1 → kode 8-bit tekstur lokal yang tidak berubah oleh perubahan kecerahan
monoton. Wajah dibagi grid 8×8; histogram LBP tiap sel digabung menjadi vektor ciri, dan
dua wajah dibandingkan dengan jarak chi-kuadrat antar histogram (tetangga terdekat).
Ahonen, Hadid & Pietikäinen (2006).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from pcdface.boxes import Box, clip_box
from pcdface.config import RecognitionConfig
from pcdface.dataset.metadata import SINGLE_FACE_SETS, SUBJECT_ID, MetadataRow, SubjectRow
from pcdface.pose import REFERENCE_EXPRESSION, REFERENCE_OCCLUSION, REFERENCE_POSE

UNKNOWN = "unknown"
MODEL_FILE = "lbph.yml"
LABELS_FILE = "labels.json"


def consenting_subjects(subjects: Mapping[str, SubjectRow]) -> set[str]:
    """Peserta yang mengizinkan penelitian DAN pengenalan identitas."""
    return {sid for sid, s in subjects.items() if s.consent_research and s.consent_recognition}


def is_reference_condition(meta: MetadataRow, reference_distance_cm: int) -> bool:
    """Foto satu wajah pada kondisi acuan: jarak acuan, cahaya normal, depan, netral, tanpa penutup.

    Dipakai `enroll` untuk galeri operasional (±14 foto per peserta: set jarak 100 cm,
    pose `depan` 100 cm, ekspresi `netral`, oklusi `tanpa`).
    """
    return (
        meta.set in SINGLE_FACE_SETS
        and bool(meta.subject_id)
        and meta.distance_cm == reference_distance_cm
        and meta.lighting == "normal"
        and meta.pose in ("", REFERENCE_POSE)
        and meta.expression in ("", REFERENCE_EXPRESSION)
        and meta.occlusion in ("", REFERENCE_OCCLUSION)
    )


def is_e8_gallery(meta: MetadataRow, reference_distance_cm: int) -> bool:
    """Galeri E8 (PRD §12.2 butir 13): hanya set jarak di jarak acuan, cahaya normal (5 foto/peserta).

    Lebih sempit dari `is_reference_condition` supaya foto kondisi acuan lain (pose `depan`,
    `netral`, `tanpa`) tetap tersisa sebagai foto uji acuan pembanding.
    """
    return (meta.set == "jarak" and bool(meta.subject_id)
            and meta.distance_cm == reference_distance_cm and meta.lighting == "normal")


def face_patch(image: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """Crop kotak → abu-abu → ukuran `size` (lebar, tinggi). ValueError bila kotak di luar citra."""
    height, width = image.shape[:2]
    clipped = clip_box(tuple(int(v) for v in box), width, height)  # type: ignore[arg-type]
    if clipped is None:
        raise ValueError(f"kotak {box} di luar citra {width}×{height}")
    x, y, w, h = clipped
    crop = image[y:y + h, x:x + w]
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # INTER_AREA saat memperkecil (anti-aliasing), INTER_LINEAR saat memperbesar wajah kecil
    shrinking = w >= size[0] and h >= size[1]
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


@dataclass(frozen=True)
class Prediction:
    nearest: str        # label galeri terdekat, tanpa memperhitungkan ambang
    distance: float     # jarak LBPH (chi-kuadrat) ke tetangga terdekat
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.distance <= self.threshold

    @property
    def label(self) -> str:
        """Kode peserta bila jaraknya ≤ ambang, selain itu ``unknown``."""
        return self.nearest if self.accepted else UNKNOWN


def check_label(label: str) -> str:
    """Hanya kode pseudonim yang boleh menjadi label (CLAUDE.md aturan #5)."""
    if not SUBJECT_ID.match(label):
        raise ValueError(f"label pengenalan harus kode peserta seperti S01, bukan {label!r} — nama asli dilarang")
    return label


class LBPHRecognizer:
    """Pembungkus `cv2.face.LBPHFaceRecognizer` dengan label kode peserta."""

    def __init__(self, params: RecognitionConfig) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face tidak ada — butuh opencv-contrib-python, bukan opencv-python")
        self.params = params
        self._model = cv2.face.LBPHFaceRecognizer_create(
            radius=params.radius, neighbors=params.neighbors, grid_x=params.grid_x, grid_y=params.grid_y)
        self.labels: list[str] = []
        self.meta: dict[str, object] = {}

    @property
    def trained(self) -> bool:
        return bool(self.labels)

    def train(self, patches: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        if not patches or len(patches) != len(labels):
            raise ValueError("butuh minimal satu wajah dan jumlah label yang sama")
        size = (self.params.face_size[1], self.params.face_size[0])
        for patch in patches:
            if patch.shape != size:
                raise ValueError(f"wajah harus abu-abu {self.params.face_size[0]}×{self.params.face_size[1]}")
        self.labels = sorted({check_label(label) for label in labels})
        index = {label: i for i, label in enumerate(self.labels)}
        self._model.train(list(patches), np.array([index[label] for label in labels], dtype=np.int32))

    def predict(self, patch: np.ndarray) -> Prediction:
        if not self.trained:
            raise RuntimeError("model pengenalan belum dilatih")
        label_index, distance = self._model.predict(patch)
        return Prediction(self.labels[int(label_index)], float(distance), self.params.max_distance)

    def predict_box(self, image: np.ndarray, box: Box) -> Prediction:
        return self.predict(face_patch(image, box, self.params.face_size))

    # ------------------------------------------------------------------ simpan / muat
    def save(self, folder: Path, **meta: object) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        self._model.write(str(folder / MODEL_FILE))
        self.meta = {"labels": self.labels, "params": asdict(self.params),
                     "dilatih": datetime.now().astimezone().isoformat(timespec="seconds"), **meta}
        (folder / LABELS_FILE).write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return folder / MODEL_FILE

    @classmethod
    def load(cls, folder: Path, params: RecognitionConfig) -> "LBPHRecognizer":
        """Muat model. Parameter LBPH dan ukuran wajah diambil dari saat model dilatih;
        ambang `max_distance` dari config sekarang."""
        model_path, labels_path = folder / MODEL_FILE, folder / LABELS_FILE
        if not model_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"model pengenalan belum ada di {folder}. Jalankan: python -m pcdface enroll")
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        trained = dict(meta.get("params", {}))
        trained["face_size"] = tuple(trained.get("face_size", params.face_size))
        trained["max_distance"] = params.max_distance
        recognizer = cls(RecognitionConfig(**{**asdict(params), **trained}))
        recognizer._model.read(str(model_path))
        recognizer.labels = [check_label(label) for label in meta["labels"]]
        recognizer.meta = meta
        return recognizer


def model_exists(folder: Path) -> bool:
    return (folder / MODEL_FILE).exists() and (folder / LABELS_FILE).exists()


def delete_model(folder: Path) -> list[Path]:
    """Hapus model terlatih (templat biometrik). Kembalikan berkas yang dihapus."""
    removed = []
    for name in (MODEL_FILE, LABELS_FILE):
        path = folder / name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def model_subjects(folder: Path) -> list[str]:
    """Kode peserta di model tersimpan, tanpa memuat histogramnya."""
    path = folder / LABELS_FILE
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("labels", []))


def train_from(items: Iterable[tuple[np.ndarray, Box, str]], params: RecognitionConfig) -> LBPHRecognizer:
    """Latih dari (citra, kotak, kode peserta)."""
    patches, labels = [], []
    for image, box, label in items:
        patches.append(face_patch(image, box, params.face_size))
        labels.append(label)
    recognizer = LBPHRecognizer(params)
    recognizer.train(patches, labels)
    return recognizer
