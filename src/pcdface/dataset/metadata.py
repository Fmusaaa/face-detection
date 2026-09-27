"""Skema `metadata.csv` dan `subjects.csv`, serta aturan penamaan berkas.

Satu baris metadata per foto (PRD §6.6, ditambah kolom `subjects`, `pose`,
`expression`, dan `session` — PRD §12.2 butir 7, §6.2). `subjects.csv` hanya memuat kode peserta dan izinnya;
nama asli tidak pernah disimpan di folder proyek.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

COLUMNS = (
    "file", "set", "session", "subject_id", "subjects", "formation", "positions_cm",
    "distance_cm", "lighting", "pose", "expression", "expected_faces", "luma_mean",
    "width", "height", "captured_at",
)
# Kolom yang boleh tidak ada di berkas lama; dianggap kosong saat dibaca
OPTIONAL_COLUMNS = frozenset({"session", "expression"})
SINGLE_FACE_SETS = ("jarak", "cahaya", "pose", "ekspresi")
SUBJECT_COLUMNS = ("subject_id", "consent_research", "consent_publication", "consent_date")

SUBJECT_ID = re.compile(r"^S\d{2,3}$")
FORMATION_ID = re.compile(r"^F\d+$")
_LIST_SEP = ";"


# ---------------------------------------------------------------------------
# metadata.csv
# ---------------------------------------------------------------------------
@dataclass
class MetadataRow:
    """Satu foto beserta kondisi perekamannya."""

    file: str                                   # relatif terhadap data/raw/
    set: str
    expected_faces: int
    width: int
    height: int
    lighting: str = "normal"
    session: str = ""                           # sesi pengambilan, mis. 2026-09-30 atau 2026-09-30-sore
    subject_id: str = ""                        # set satu wajah
    subjects: tuple[str, ...] = ()              # multi-wajah, kiri → kanan
    formation: str = ""
    positions_cm: tuple[int, ...] = ()
    distance_cm: int | None = None
    pose: str = ""
    expression: str = ""                        # set ekspresi: netral, senyum, marah, ...
    luma_mean: float | None = None
    captured_at: str = ""

    @property
    def people(self) -> tuple[str, ...]:
        """Semua kode peserta yang tampak di foto ini."""
        if self.subject_id:
            return (self.subject_id,)
        return self.subjects

    @property
    def group(self) -> str:
        """Kelompok untuk bootstrap: subjek untuk set satu wajah, berkas untuk lainnya."""
        return self.subject_id or self.file

    def to_csv(self) -> dict[str, str]:
        return {
            "file": self.file,
            "set": self.set,
            "session": self.session,
            "subject_id": self.subject_id,
            "subjects": _LIST_SEP.join(self.subjects),
            "formation": self.formation,
            "positions_cm": _LIST_SEP.join(str(p) for p in self.positions_cm),
            "distance_cm": "" if self.distance_cm is None else str(self.distance_cm),
            "lighting": self.lighting,
            "pose": self.pose,
            "expression": self.expression,
            "expected_faces": str(self.expected_faces),
            "luma_mean": "" if self.luma_mean is None else f"{self.luma_mean:.1f}",
            "width": str(self.width),
            "height": str(self.height),
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> "MetadataRow":
        def split(value: str) -> list[str]:
            return [item for item in (value or "").split(_LIST_SEP) if item]

        return cls(
            file=row["file"],
            set=row["set"],
            session=row.get("session", "") or "",
            subject_id=row.get("subject_id", "") or "",
            subjects=tuple(split(row.get("subjects", ""))),
            formation=row.get("formation", "") or "",
            positions_cm=tuple(int(p) for p in split(row.get("positions_cm", ""))),
            distance_cm=int(row["distance_cm"]) if row.get("distance_cm") else None,
            lighting=row.get("lighting", "") or "normal",
            pose=row.get("pose", "") or "",
            expression=row.get("expression", "") or "",
            expected_faces=int(row["expected_faces"]),
            luma_mean=float(row["luma_mean"]) if row.get("luma_mean") else None,
            width=int(row["width"]),
            height=int(row["height"]),
            captured_at=row.get("captured_at", "") or "",
        )


def read_metadata(path: Path) -> list[MetadataRow]:
    """Baca metadata. Berkas yang belum ada dianggap kosong."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - OPTIONAL_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: kolom hilang {sorted(missing)}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(MetadataRow.from_csv(row))
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line_no}: baris tidak valid ({error})") from error
        return rows


def _atomic_write(path: Path, writer_fn) -> None:
    """Tulis ke berkas sementara lalu ganti, agar metadata tidak rusak bila terputus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer_fn(stream)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_metadata(path: Path, rows: Iterable[MetadataRow]) -> None:
    rows = list(rows)

    def write(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())

    _atomic_write(path, write)


def append_metadata(path: Path, row: MetadataRow) -> None:
    rows = read_metadata(path)
    if any(existing.file == row.file for existing in rows):
        raise ValueError(f"metadata untuk {row.file} sudah ada")
    rows.append(row)
    write_metadata(path, rows)


# ---------------------------------------------------------------------------
# subjects.csv
# ---------------------------------------------------------------------------
@dataclass
class SubjectRow:
    subject_id: str
    consent_research: bool
    consent_publication: bool
    consent_date: str = ""

    def to_csv(self) -> dict[str, str]:
        return {
            "subject_id": self.subject_id,
            "consent_research": "ya" if self.consent_research else "tidak",
            "consent_publication": "ya" if self.consent_publication else "tidak",
            "consent_date": self.consent_date,
        }


def _yes(value: str, where: str) -> bool:
    value = (value or "").strip().lower()
    if value in ("ya", "yes", "y", "true", "1"):
        return True
    if value in ("tidak", "no", "n", "false", "0", ""):
        return False
    raise ValueError(f"{where}: isi 'ya' atau 'tidak', dapat {value!r}")


def read_subjects(path: Path) -> dict[str, SubjectRow]:
    if not path.exists():
        return {}
    subjects: dict[str, SubjectRow] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(SUBJECT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: kolom hilang {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            where = f"{path.name}:{line_no}"
            subject_id = (row["subject_id"] or "").strip()
            if not SUBJECT_ID.match(subject_id):
                raise ValueError(f"{where}: kode peserta harus seperti S01, dapat {subject_id!r}")
            if subject_id in subjects:
                raise ValueError(f"{where}: kode {subject_id} muncul dua kali")
            subjects[subject_id] = SubjectRow(
                subject_id=subject_id,
                consent_research=_yes(row["consent_research"], f"{where} consent_research"),
                consent_publication=_yes(row["consent_publication"], f"{where} consent_publication"),
                consent_date=(row["consent_date"] or "").strip(),
            )
    return subjects


def write_subjects(path: Path, subjects: Iterable[SubjectRow]) -> None:
    rows = sorted(subjects, key=lambda s: s.subject_id)

    def write(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=SUBJECT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())

    _atomic_write(path, write)


# ---------------------------------------------------------------------------
# Penamaan berkas (PRD §6.6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureSpec:
    """Kondisi satu sesi rekam; menentukan folder dan nama berkas."""

    set: str
    subject_id: str = ""
    distance_cm: int | None = None
    lighting: str = "normal"
    pose: str = ""
    formation: str = ""
    subjects: tuple[str, ...] = field(default=())
    expression: str = ""

    def folder(self) -> str:
        if self.set in SINGLE_FACE_SETS:
            return f"{self.set}/{self.subject_id}"
        if self.set == "multi":
            return f"multi/{self.formation}"
        return "kosong"

    def stem_prefix(self) -> str:
        if self.set in ("jarak", "cahaya"):
            return f"{self.set}_{self.subject_id}_{self.distance_cm}cm_{self.lighting}"
        if self.set == "pose":
            return f"pose_{self.subject_id}_{self.distance_cm}cm_{self.pose}"
        if self.set == "ekspresi":
            return f"ekspresi_{self.subject_id}_{self.distance_cm}cm_{self.expression}"

        if self.set == "multi":
            return f"multi_{self.formation}"
        return "kosong"

    def relative_path(self, index: int) -> str:
        return f"{self.folder()}/{self.stem_prefix()}_{index:02d}.jpg"


def next_index(raw_dir: Path, spec: CaptureSpec, taken: Iterable[str] = ()) -> int:
    """Nomor urut berikutnya — dari berkas di disk dan dari metadata."""
    prefix = spec.stem_prefix() + "_"
    numbers = [0]
    candidates = [p.name for p in (raw_dir / spec.folder()).glob(prefix + "*.jpg")]
    candidates += [Path(t).name for t in taken]
    for name in candidates:
        if name.startswith(prefix):
            tail = Path(name).stem[len(prefix):]
            if tail.isdigit():
                numbers.append(int(tail))
    return max(numbers) + 1
