"""`python -m pcdface report` — bangun ulang grafik dan ringkasan dari hasil tersimpan.

Tidak menjalankan detektor untuk tabel/grafik: semua dibaca dari CSV di
`results/<eN>/`. Menulis `results/RINGKASAN.md` yang menggabungkan semua tabel
dan tautan grafik, siap dijadikan bahan bab Hasil.

`--examples N` membuat N contoh gambar deteksi di `results/contoh/`
(detektor dijalankan pada citra asli). Wajah peserta yang tidak mencentang izin
publikasi di subjects.csv DIBURAMKAN (PRD §10, §11).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from pcdface.boxes import Box, clip_box, sort_left_to_right
from pcdface.config import Config
from pcdface.dataset.loader import Sample, load_samples
from pcdface.dataset.metadata import read_subjects
from pcdface.detection.registry import build_detector
from pcdface.paths import ProjectPaths
from pcdface.reporting.plots import LABEL, plot_experiment

EXPERIMENT_TITLES = {
    "e1": "E1 — Jarak × resolusi",
    "e2": "E2 — Multi-wajah",
    "e3": "E3 — Pencahayaan × enhancement",
    "e4": "E4 — Kecepatan",
    "e5": "E5 — Sensitivitas parameter Haar",
    "e6": "E6 — Pose, ekspresi, dan oklusi",
    "e7": "E7 — Blur gerak (simulasi)",
    "e8": "E8 — Pengenalan identitas LBPH",
}


GT_COLOR = (0, 190, 0)
DET_COLOR = (0, 140, 255)


def build_summary(results_root: Path) -> Path | None:
    """Bangun ulang grafik dan tulis RINGKASAN.md. None bila belum ada hasil."""
    sections = []
    for name, title in EXPERIMENT_TITLES.items():
        out_dir = results_root / name
        snapshot_path = out_dir / "config_snapshot.yaml"
        if not snapshot_path.exists():
            continue
        snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        figures = [p for p in plot_experiment(name, out_dir) if p.suffix == ".png"]
        versions = snapshot.get("versi", {})
        lines = [f"## {title}", ""]
        if snapshot.get("sintetis"):
            lines += ["> **Data sintetis + FakeDetector — angka BUKAN hasil penelitian.**", ""]
        lines += [
            f"- Dijalankan: {snapshot.get('waktu')} ({snapshot.get('durasi_detik')} detik)",
            f"- Sampel: {snapshot.get('sampel', {})}",
            f"- Python {versions.get('python')}, OpenCV {versions.get('opencv')}, "
            f"MediaPipe {versions.get('mediapipe')}, {versions.get('platform')}",
            "",
        ]
        for figure in figures:
            lines += [f"![{figure.stem}]({name}/{figure.name})", ""]
        for stem in snapshot.get("tabel", []):
            md = out_dir / f"{stem}.md"
            if md.exists() and not stem.endswith(("_kurva_pr", "_ambang", "_prediksi")):
                lines += [md.read_text(encoding="utf-8").strip(), ""]
        sections.append("\n".join(lines))
    if not sections:
        return None
    path = results_root / "RINGKASAN.md"
    header = ("# Ringkasan hasil eksperimen\n\nDibangun ulang oleh `python -m pcdface report` dari tabel CSV. "
              "Interval: Wilson 95% untuk proporsi, bootstrap kelompok untuk F1/AP (PRD §8.6).\n")
    path.write_text(header + "\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Contoh gambar dengan pemburaman wajah tanpa izin
# ---------------------------------------------------------------------------
def private_boxes(sample: Sample, publishable: dict[str, bool]) -> list[Box]:
    """Kotak manual milik peserta yang TIDAK mengizinkan publikasi (atau tidak diketahui)."""
    people = sample.meta.people
    boxes = sort_left_to_right(sample.gt)
    if len(people) != len(boxes):
        return boxes  # tidak bisa dipasangkan dengan pasti → buramkan semua
    return [box for person, box in zip(people, boxes) if not publishable.get(person, False)]


def blur_faces(image: np.ndarray, boxes: list[Box], margin: float = 0.25) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for x, y, w, h in boxes:
        grown = clip_box((int(x - margin * w), int(y - margin * h), int(w * (1 + 2 * margin)), int(h * (1 + 2 * margin))),
                         width, height)
        if grown is None:
            continue
        gx, gy, gw, gh = grown
        region = out[gy:gy + gh, gx:gx + gw]
        small = cv2.resize(region, (max(1, gw // 16), max(1, gh // 16)), interpolation=cv2.INTER_AREA)
        out[gy:gy + gh, gx:gx + gw] = cv2.GaussianBlur(
            cv2.resize(small, (gw, gh), interpolation=cv2.INTER_NEAREST), (0, 0), max(gw, gh) / 12)
    return out


def _panel(image: np.ndarray, gt: list[Box], boxes: list[Box], title: str) -> np.ndarray:
    canvas = image.copy()
    for x, y, w, h in gt:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), GT_COLOR, 2)
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), DET_COLOR, 3)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(canvas, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_examples(cfg: Config, paths: ProjectPaths, count: int, detectors: list[str],
                   synthetic: bool = False) -> list[Path]:
    subjects = read_subjects(paths.subjects)
    publishable = {sid: s.consent_publication for sid, s in subjects.items()}
    samples = load_samples(paths, sets=("multi", "jarak")).samples
    samples.sort(key=lambda s: (s.meta.set != "multi", s.key))
    rng = np.random.default_rng(cfg.seed)
    chosen = [samples[i] for i in sorted(rng.choice(len(samples), size=min(count, len(samples)), replace=False))] \
        if samples else []
    out_dir = paths.results / "contoh"
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {name: build_detector(name, cfg, synthetic=synthetic) for name in detectors}
    written = []
    try:
        for index, sample in enumerate(chosen, start=1):
            image = sample.load()
            hidden = private_boxes(sample, publishable)
            safe = blur_faces(image, hidden)
            panels = [_panel(safe, sample.gt, built[n].detect(image).boxes, LABEL.get(n, n)) for n in detectors]
            sheet = np.vstack([np.hstack(panels[i:i + 2]) if i + 1 < len(panels)
                               else np.hstack([panels[i], np.full_like(panels[i], 252)])
                               for i in range(0, len(panels), 2)])
            sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
            path = out_dir / f"contoh_{index:02d}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written.append(path)
            print(f"  {path.name}: {sample.key} ({len(hidden)} wajah diburamkan)")
    finally:
        for detector in built.values():
            detector.close()
    return written


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthetic", action="store_true", help="Pakai hasil run sintetis (results/synthetic)")
    parser.add_argument("--examples", type=int, default=0, metavar="N",
                        help="Buat N contoh gambar deteksi (wajah tanpa izin publikasi diburamkan)")
    parser.add_argument("--detectors", default=None,
                        help="Detektor untuk contoh, dipisah koma (bawaan: detektor E2)")


def run(args: argparse.Namespace, cfg: Config) -> int:
    results_root = cfg.paths.results / "synthetic" if args.synthetic else cfg.paths.results
    summary = build_summary(results_root)
    if summary is None:
        print(f"Belum ada hasil di {results_root}. Jalankan dulu: python -m pcdface run all"
              + (" --synthetic" if args.synthetic else ""), file=sys.stderr)
        return 1
    print(f"Ringkasan: {summary}")
    if args.examples > 0:
        if args.synthetic:
            print("Contoh gambar hanya untuk data asli (run sintetis memakai folder sementara).", file=sys.stderr)
            return 2
        detectors = args.detectors.split(",") if args.detectors else list(cfg.experiments.e2.detectors)
        written = write_examples(cfg, cfg.paths, args.examples, detectors)
        print(f"{len(written)} contoh gambar di {cfg.paths.results / 'contoh'}")
    return 0
