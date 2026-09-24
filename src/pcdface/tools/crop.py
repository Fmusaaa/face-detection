"""Ekspor isi kotak manual ke `data/crops/` (PRD §7).

Kegunaan: memeriksa konsistensi anotasi secara visual, mengukur lebar wajah
per jarak, dan menyediakan contoh gambar (hanya peserta yang mengizinkan
publikasi — pakai `--publishable-only`).

Nama crop: `crops/<folder foto>/<nama foto>_f<N>.jpg`, N urut kiri → kanan.
Ringkasan lebar wajah ditulis ke `crops/face_widths.csv`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import clip_box, sort_left_to_right
from pcdface.config import Config
from pcdface.dataset.loader import load_samples
from pcdface.dataset.metadata import read_subjects
from pcdface.paths import ProjectPaths

WIDTH_COLUMNS = ("file", "set", "subject_id", "distance_cm", "lighting", "face", "x", "y", "w", "h",
                 "w_ratio", "position_cm")


def export_crops(cfg: Config, paths: ProjectPaths, publishable_only: bool = False,
                 clean: bool = False) -> tuple[int, list[dict[str, object]]]:
    """Tulis crop dan kembalikan (jumlah crop, baris lebar wajah)."""
    if clean and paths.crops.exists():
        shutil.rmtree(paths.crops)
    subjects = read_subjects(paths.subjects)
    samples = load_samples(paths).samples
    written = 0
    widths: list[dict[str, object]] = []
    for sample in samples:
        if not sample.gt:
            continue
        people = sample.meta.people
        allowed = all(subjects.get(p) is not None and subjects[p].consent_publication for p in people)
        if publishable_only and not allowed:
            continue
        image = sample.load()
        height, width = image.shape[:2]
        rel = Path(sample.key)
        target_dir = paths.crops / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        positions = list(sample.meta.positions_cm)
        for index, box in enumerate(sort_left_to_right(sample.gt), start=1):
            clipped = clip_box(box, width, height)
            if clipped is None:
                continue
            x, y, w, h = clipped
            cv2.imwrite(str(target_dir / f"{rel.stem}_f{index}.jpg"), image[y:y + h, x:x + w])
            written += 1
            widths.append({
                "file": sample.key,
                "set": sample.meta.set,
                "subject_id": sample.meta.subject_id,
                "distance_cm": sample.meta.distance_cm if sample.meta.distance_cm is not None else "",
                "lighting": sample.meta.lighting,
                "face": index,
                "x": x, "y": y, "w": w, "h": h,
                "w_ratio": round(w / width, 5),
                "position_cm": positions[index - 1] if index - 1 < len(positions) else "",
            })
    paths.crops.mkdir(parents=True, exist_ok=True)
    with (paths.crops / "face_widths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WIDTH_COLUMNS)
        writer.writeheader()
        writer.writerows(widths)
    return written, widths


def width_summary(widths: list[dict[str, object]]) -> list[tuple[int, int, float, float, float]]:
    """(jarak, n, rerata px, simpangan baku px, rerata proporsi) untuk set jarak."""
    by_distance: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in widths:
        if row["set"] == "jarak" and row["distance_cm"] != "":
            by_distance[int(row["distance_cm"])].append((float(row["w"]), float(row["w_ratio"])))
    summary = []
    for distance in sorted(by_distance):
        px = np.array([w for w, _ in by_distance[distance]])
        ratio = np.array([r for _, r in by_distance[distance]])
        sd = float(px.std(ddof=1)) if len(px) > 1 else 0.0
        summary.append((distance, len(px), float(px.mean()), sd, float(ratio.mean())))
    return summary


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publishable-only", action="store_true",
                        help="Hanya foto yang semua pesertanya mengizinkan publikasi")
    parser.add_argument("--clean", action="store_true", help="Hapus isi crops/ lama dulu")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    written, widths = export_crops(cfg, paths, publishable_only=args.publishable_only, clean=args.clean)
    print(f"{written} crop ditulis ke {paths.crops}")
    print(f"Lebar wajah per kotak: {paths.crops / 'face_widths.csv'}")
    summary = width_summary(widths)
    if summary:
        print("\nLebar wajah set jarak (dari kotak manual):")
        print(f"  {'jarak':>6} {'n':>4} {'rerata px':>10} {'sb px':>7} {'proporsi':>9}")
        for distance, n, mean, sd, ratio in summary:
            print(f"  {distance:>4}cm {n:>4} {mean:>10.1f} {sd:>7.1f} {ratio:>9.4f}")
    return 0
