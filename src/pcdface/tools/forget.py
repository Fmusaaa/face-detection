"""Hak menarik diri (PRD §11): hapus seluruh data satu peserta.

`forget S03` menghapus:
- semua foto set satu wajah milik S03,
- semua foto multi-wajah yang memuat S03 (dari kolom `subjects`),
- crop dari foto-foto itu,
- baris metadata dan anotasinya,
- baris S03 di subjects.csv.

Tanpa `--yes`, rencana penghapusan ditampilkan dulu dan harus dikonfirmasi
dengan mengetik ulang kode peserta. Hasil eksperimen di `results/` tidak
dihapus otomatis — jalankan ulang `run all` dan `report` sesudahnya.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pcdface.config import Config
from pcdface.dataset.annotations import load_annotations, save_annotations
from pcdface.dataset.metadata import (
    SUBJECT_ID,
    read_metadata,
    read_subjects,
    write_metadata,
    write_subjects,
)
from pcdface.paths import ProjectPaths


@dataclass
class ForgetPlan:
    subject_id: str
    photos: list[str] = field(default_factory=list)          # relatif terhadap raw/
    crops: list[Path] = field(default_factory=list)
    annotation_keys: list[str] = field(default_factory=list)
    in_subjects_csv: bool = False

    @property
    def empty(self) -> bool:
        return not (self.photos or self.crops or self.annotation_keys or self.in_subjects_csv)


def crop_paths_for(crops_dir: Path, relative_photo: str) -> list[Path]:
    """Crop yang berasal dari satu foto: crops/<folder>/<stem>_fN.jpg."""
    rel = Path(relative_photo)
    folder = crops_dir / rel.parent
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{rel.stem}_f*.jpg"))


def plan_forget(paths: ProjectPaths, subject_id: str) -> ForgetPlan:
    if not SUBJECT_ID.match(subject_id):
        raise ValueError(f"kode peserta harus seperti S03, dapat {subject_id!r}")
    plan = ForgetPlan(subject_id=subject_id)
    rows = read_metadata(paths.metadata)
    plan.photos = [row.file for row in rows if subject_id in row.people]
    # foto di folder subjek yang tidak tercatat di metadata tetap ikut dihapus
    for set_name in ("jarak", "cahaya", "pose"):
        folder = paths.raw / set_name / subject_id
        if folder.exists():
            for path in sorted(folder.glob("*")):
                rel = path.relative_to(paths.raw).as_posix()
                if path.is_file() and rel not in plan.photos:
                    plan.photos.append(rel)
    annotations = load_annotations(paths.annotations)
    plan.annotation_keys = [key for key in plan.photos if key in annotations]
    plan.crops = [crop for photo in plan.photos for crop in crop_paths_for(paths.crops, photo)]
    plan.in_subjects_csv = subject_id in read_subjects(paths.subjects)
    return plan


def _remove_empty_dirs(start: Path, stop: Path) -> None:
    """Hapus folder kosong dari `start` ke atas, berhenti di `stop`."""
    current = start
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def execute_forget(paths: ProjectPaths, plan: ForgetPlan) -> None:
    photos = set(plan.photos)
    rows = read_metadata(paths.metadata)
    write_metadata(paths.metadata, [row for row in rows if row.file not in photos])

    annotations = load_annotations(paths.annotations)
    if any(key in annotations for key in photos):
        save_annotations(paths.annotations, {k: v for k, v in annotations.items() if k not in photos})

    subjects = read_subjects(paths.subjects)
    if plan.subject_id in subjects:
        write_subjects(paths.subjects, [s for sid, s in subjects.items() if sid != plan.subject_id])

    for crop in plan.crops:
        crop.unlink(missing_ok=True)
        _remove_empty_dirs(crop.parent, paths.crops)
    for rel in plan.photos:
        path = paths.raw / rel
        path.unlink(missing_ok=True)
        _remove_empty_dirs(path.parent, paths.raw)


def describe(plan: ForgetPlan) -> str:
    multi = [p for p in plan.photos if p.startswith("multi/")]
    lines = [
        f"Data peserta {plan.subject_id} yang akan DIHAPUS PERMANEN:",
        f"  foto                : {len(plan.photos)} (termasuk {len(multi)} foto multi-wajah bersama peserta lain)",
        f"  crop                : {len(plan.crops)}",
        f"  anotasi             : {len(plan.annotation_keys)}",
        f"  baris subjects.csv  : {'ya' if plan.in_subjects_csv else 'tidak ada'}",
    ]
    if multi:
        lines.append("  Foto multi-wajah ikut dihapus karena memuat wajah peserta ini (PRD §11).")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("subject_id", help="Kode peserta, mis. S03")
    parser.add_argument("--yes", action="store_true", help="Hapus tanpa konfirmasi interaktif")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan rencana saja, jangan hapus")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    try:
        plan = plan_forget(paths, args.subject_id)
    except ValueError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 2
    if plan.empty:
        print(f"Tidak ada data untuk {args.subject_id}.")
        return 0
    print(describe(plan))
    if args.dry_run:
        print("\n(--dry-run: tidak ada yang dihapus)")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("\n[GAGAL] Butuh konfirmasi: jalankan di terminal atau tambahkan --yes.", file=sys.stderr)
            return 2
        answer = input(f"\nKetik {plan.subject_id} untuk menghapus: ").strip()
        if answer != plan.subject_id:
            print("Dibatalkan.")
            return 1
    execute_forget(paths, plan)
    print(f"\nSelesai. Data {plan.subject_id} sudah dihapus.")
    print("Hasil lama di results/ masih memuat angka dari data ini — jalankan ulang:")
    print("  python -m pcdface run all && python -m pcdface report")
    return 0
