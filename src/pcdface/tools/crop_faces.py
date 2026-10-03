"""Deteksi lalu crop setiap wajah di setiap foto — `python -m pcdface crop-faces`.

    python -m pcdface crop-faces                              # semua foto data/raw/, detektor yolo_n
    python -m pcdface crop-faces --input foto_kelas/          # folder lain (rekursif) atau satu berkas
    python -m pcdface crop-faces --recognize                  # kelompokkan per kode peserta (butuh enroll)
    python -m pcdface crop-faces --detector mp_full --margin 0.2

Keluaran di `results/crop_wajah/` (di-gitignore — berisi wajah):

- tanpa `--recognize`: `<folder asal>/<nama foto>_f<N>.jpg`, N urut kiri → kanan;
- dengan `--recognize`: `<S01|S02|…|unknown>/<nama foto>_f<N>.jpg` — kode dari model LBPH
  (`enroll`), ``unknown`` bila jarak LBPH > `recognition.max_distance`;
- `crop_wajah.csv`: satu baris per wajah (foto, urutan, kotak, skor, identitas, jarak LBPH).

Beda dengan `crop`: `crop` mengekspor kotak **manual** (ground truth, PRD §7); `crop-faces`
mengekspor kotak **hasil deteksi** dan bisa dipakai pada foto apa pun. Pengenalan hanya
mencocokkan wajah dengan peserta yang sudah mengizinkan dan didaftarkan; wajah lain
selalu berlabel ``unknown`` — sistem tidak pernah menebak nama.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box, clip_box, sort_left_to_right
from pcdface.config import Config
from pcdface.detection.base import Detector
from pcdface.detection.registry import build_detector, model_missing
from pcdface.recognition.lbph import LBPHRecognizer

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
MANIFEST = "crop_wajah.csv"
COLUMNS = ("foto", "wajah", "x", "y", "w", "h", "skor", "identitas", "jarak_lbph", "crop")


@dataclass
class FaceCrop:
    source: str             # foto asal, relatif terhadap folder masukan
    index: int              # urutan kiri → kanan, mulai 1
    box: Box                # kotak deteksi (sebelum margin)
    score: float | None
    identity: str = ""      # kode peserta / unknown; kosong bila tanpa --recognize
    distance: float | None = None
    path: Path | None = None

    def row(self, root: Path) -> dict[str, object]:
        x, y, w, h = self.box
        return {"foto": self.source, "wajah": self.index, "x": x, "y": y, "w": w, "h": h,
                "skor": "" if self.score is None else round(self.score, 4), "identitas": self.identity,
                "jarak_lbph": "" if self.distance is None else round(self.distance, 2),
                "crop": self.path.relative_to(root).as_posix() if self.path else ""}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def expand(box: Box, margin: float, width: int, height: int) -> Box | None:
    """Lebarkan kotak `margin` × lebar/tinggi di tiap sisi, lalu potong ke batas citra."""
    x, y, w, h = box
    dx, dy = int(round(w * margin)), int(round(h * margin))
    return clip_box((x - dx, y - dy, w + 2 * dx, h + 2 * dy), width, height)


def crop_image(image: np.ndarray, source: str, detector: Detector, margin: float,
               recognizer: LBPHRecognizer | None) -> list[tuple[FaceCrop, np.ndarray]]:
    """Wajah dalam satu foto, urut kiri → kanan, beserta potongan citranya."""
    height, width = image.shape[:2]
    result = detector.detect(image)
    score_of = dict(zip(result.boxes, result.scores)) if result.scores is not None else {}
    out = []
    for index, box in enumerate(sort_left_to_right(result.boxes), start=1):
        region = expand(box, margin, width, height)
        if region is None:
            continue
        face = FaceCrop(source, index, box, score_of.get(box))
        if recognizer is not None:
            prediction = recognizer.predict_box(image, box)   # LBPH selalu pada kotak tanpa margin
            face.identity, face.distance = prediction.label, prediction.distance
        x, y, w, h = region
        out.append((face, image[y:y + h, x:x + w]))
    return out


def crop_faces(cfg: Config, source: Path, out_dir: Path, detector_name: str, margin: float = 0.0,
               recognize: bool = False, synthetic: bool = False) -> list[FaceCrop]:
    """Proses semua foto di `source`; tulis crop dan manifest ke `out_dir`."""
    recognizer = LBPHRecognizer.load(cfg.paths.recognition, cfg.recognition) if recognize else None
    images = list_images(source)
    base = source if source.is_dir() else source.parent
    faces: list[FaceCrop] = []
    with build_detector(detector_name, cfg, synthetic=synthetic) as detector:
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                print(f"  dilewati (tidak bisa dibaca): {path}", file=sys.stderr)
                continue
            rel = path.relative_to(base)
            for face, pixels in crop_image(image, rel.as_posix(), detector, margin, recognizer):
                folder = out_dir / (face.identity if recognizer is not None else rel.parent)
                folder.mkdir(parents=True, exist_ok=True)
                face.path = folder / f"{rel.stem}_f{face.index}.jpg"
                cv2.imwrite(str(face.path), pixels)
                faces.append(face)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / MANIFEST).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(face.row(out_dir) for face in faces)
    return faces


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None, help="Foto atau folder masukan (bawaan: data/raw)")
    parser.add_argument("--output", default=None, help="Folder keluaran (bawaan: results/crop_wajah)")
    parser.add_argument("--detector", default="yolo_n", help="Detektor (bawaan: yolo_n)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Pelebaran kotak per sisi, proporsi lebar/tinggi (bawaan 0 = kotak deteksi apa adanya)")
    parser.add_argument("--recognize", action="store_true",
                        help="Kelompokkan crop per kode peserta dengan model LBPH (jalankan enroll dulu)")


def run(args: argparse.Namespace, cfg: Config) -> int:
    cfg.detector(args.detector)   # galat jelas bila nama salah
    if model_missing(args.detector, cfg):
        print(f"[GAGAL] model {args.detector} belum ada — python -m pcdface download-models", file=sys.stderr)
        return 1
    if not 0.0 <= args.margin <= 1.0:
        print("[GAGAL] --margin harus 0–1", file=sys.stderr)
        return 2
    source = Path(args.input) if args.input else cfg.paths.raw
    if not source.exists():
        print(f"[GAGAL] masukan tidak ada: {source}", file=sys.stderr)
        return 2
    out_dir = Path(args.output) if args.output else cfg.paths.results / "crop_wajah"
    try:
        faces = crop_faces(cfg, source, out_dir, args.detector, args.margin, args.recognize)
    except FileNotFoundError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1
    photos = len({face.source for face in faces})
    print(f"{len(faces)} wajah dari {photos} foto → {out_dir} (manifest: {MANIFEST})")
    if args.recognize:
        tally: dict[str, int] = {}
        for face in faces:
            tally[face.identity] = tally.get(face.identity, 0) + 1
        for identity in sorted(tally):
            print(f"  {identity}: {tally[identity]}")
    print("Crop berisi wajah: jangan dibagikan tanpa izin publikasi peserta.")
    return 0
