"""Latih pengenal identitas LBPH (PRD v4 §5.4) — `python -m pcdface enroll`.

    python -m pcdface enroll                      # galeri = foto kondisi acuan dari dataset, kotak manual
    python -m pcdface enroll --source yolo_n      # crop dari deteksi YOLO, bukan kotak manual
    python -m pcdface enroll --folder faces/      # gaya repo referensi: faces/S01/*.jpg, faces/S02/*.jpg

Hanya peserta dengan `consent_research` dan `consent_recognition` = ya di `subjects.csv`
yang dipakai; peserta lain dilewati dan disebutkan. Nama folder di `--folder` harus kode
peserta (S01, …) — nama asli ditolak (CLAUDE.md aturan #5).

Galeri dari dataset = foto kondisi acuan (`is_reference_condition`): jarak acuan, cahaya
normal, pose `depan`, ekspresi `netral`, tanpa penutup — ±14 foto per peserta. Model ditulis
ke `data/recognition/` (templat biometrik: di-gitignore, dihapus oleh `forget`).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box
from pcdface.config import Config
from pcdface.dataset.loader import load_samples
from pcdface.dataset.metadata import SINGLE_FACE_SETS, SUBJECT_ID, read_subjects
from pcdface.detection.base import Detector
from pcdface.detection.registry import build_detector
from pcdface.paths import ProjectPaths
from pcdface.recognition.lbph import LBPHRecognizer, consenting_subjects, face_patch, is_reference_condition

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
MANUAL = "manual"


def largest_detection(detector: Detector, image: np.ndarray) -> Box | None:
    """Kotak terbesar — foto pendaftaran berisi satu wajah utama (seperti repo referensi)."""
    boxes = detector.detect(image).boxes
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def dataset_items(cfg: Config, paths: ProjectPaths, allowed: set[str], detector: Detector | None
                  ) -> tuple[list[tuple[np.ndarray, str]], Counter]:
    """(wajah abu-abu, kode) dari foto kondisi acuan peserta yang mengizinkan."""
    items, skipped = [], Counter()
    for sample in load_samples(paths, SINGLE_FACE_SETS).samples:
        meta = sample.meta
        if not is_reference_condition(meta, cfg.dataset.reference_distance_cm):
            continue
        if meta.subject_id not in allowed:
            skipped["tanpa izin pengenalan"] += 1
            continue
        image = sample.load()
        box = sample.gt[0] if detector is None else largest_detection(detector, image)
        if box is None:
            skipped["wajah tidak terdeteksi"] += 1
            continue
        items.append((face_patch(image, box, cfg.recognition.face_size), meta.subject_id))
    return items, skipped


def folder_items(cfg: Config, folder: Path, allowed: set[str], detector: Detector
                 ) -> tuple[list[tuple[np.ndarray, str]], Counter]:
    """(wajah abu-abu, kode) dari folder per peserta, crop dengan detektor."""
    items, skipped = [], Counter()
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        if not SUBJECT_ID.match(person.name):
            raise ValueError(f"nama folder harus kode peserta seperti S01, bukan {person.name!r} "
                             "(nama asli dilarang — CLAUDE.md aturan #5)")
        if person.name not in allowed:
            skipped[f"{person.name} tanpa izin pengenalan"] += 1
            continue
        for path in sorted(person.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            box = largest_detection(detector, image) if image is not None else None
            if box is None:
                skipped["wajah tidak terdeteksi / berkas rusak"] += 1
                continue
            items.append((face_patch(image, box, cfg.recognition.face_size), person.name))
    return items, skipped


def enroll(cfg: Config, paths: ProjectPaths, source: str = MANUAL, folder: Path | None = None,
           synthetic: bool = False) -> tuple[LBPHRecognizer, Counter, Counter]:
    """Latih dan simpan model. Kembalikan (model, jumlah wajah per peserta, alasan dilewati)."""
    allowed = consenting_subjects(read_subjects(paths.subjects))
    if not allowed:
        raise ValueError("tidak ada peserta dengan consent_recognition = ya di subjects.csv "
                         "(izin pengenalan identitas wajib, PRD §11)")
    detector = None
    if folder is not None or source != MANUAL:
        name = cfg.experiments.e8.detector if source == MANUAL else source
        detector = build_detector(name, cfg, synthetic=synthetic)
    try:
        if folder is not None:
            items, skipped = folder_items(cfg, folder, allowed, detector)
        else:
            items, skipped = dataset_items(cfg, paths, allowed, detector)
    finally:
        if detector is not None:
            detector.close()
    if not items:
        raise ValueError("tidak ada wajah untuk galeri — sudah dianotasi? (annotate) atau folder kosong?")
    recognizer = LBPHRecognizer(cfg.recognition)
    recognizer.train([patch for patch, _ in items], [label for _, label in items])
    counts = Counter(label for _, label in items)
    recognizer.save(paths.recognition, sumber=str(folder) if folder else source, wajah_per_peserta=dict(counts))
    return recognizer, counts, skipped


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default=MANUAL,
                        help="Kotak galeri: 'manual' (anotasi, bawaan) atau nama detektor, mis. yolo_n")
    parser.add_argument("--folder", default=None,
                        help="Folder berisi subfolder per kode peserta (faces/S01/*.jpg), crop dengan detektor")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    if args.source != MANUAL:
        cfg.detector(args.source)   # galat jelas bila nama salah
    folder = Path(args.folder) if args.folder else None
    if folder is not None and not folder.is_dir():
        print(f"[GAGAL] folder tidak ada: {folder}", file=sys.stderr)
        return 2
    try:
        recognizer, counts, skipped = enroll(cfg, paths, args.source, folder)
    except (ValueError, FileNotFoundError) as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1
    print(f"Model LBPH: {paths.recognition} ({len(recognizer.labels)} peserta, {sum(counts.values())} wajah)")
    for label in recognizer.labels:
        print(f"  {label}: {counts[label]} wajah")
    for reason, n in sorted(skipped.items()):
        print(f"  dilewati — {reason}: {n}")
    if len(recognizer.labels) < 2:
        print("PERINGATAN: hanya satu peserta — semua wajah dikenal akan diberi kode yang sama.", file=sys.stderr)
    print("Model ini templat biometrik: jangan dibagikan atau di-commit. `forget` menghapusnya.")
    return 0
