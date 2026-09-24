"""Uji mandiri tanpa webcam dan tanpa model.

Memeriksa (1) metrik terhadap nilai acuan hitungan tangan dan (2) jalur data
dan detektor pada dataset sintetis. Angka dari citra sintetis bukan hasil
penelitian — hanya bukti bahwa kode berjalan dan metrik terhitung benar.
"""

from __future__ import annotations

import math
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from pcdface.config import Config


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


# ---------------------------------------------------------------------------
# Pemeriksaan metrik (nilai acuan hitungan tangan)
# ---------------------------------------------------------------------------
def check_matching(_: Config, __: Path) -> str:
    from pcdface.evaluation.matching import iou, match_by_score, match_greedy

    # Dua kotak 10×10 bergeser 5 px: irisan 50, gabungan 150
    assert _close(iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    # GT A dan B; prediksi p0 menimpa A dan B, p1 tepat di A.
    gt = [(0, 0, 10, 10), (9, 0, 10, 10)]
    preds = [(4, 0, 10, 10), (0, 0, 10, 10)]
    m = match_greedy(gt, preds, 0.3)
    # p1–A IoU 1,0 diambil dulu; lalu p0–B IoU 50/150 = 0,333 ≥ 0,3
    assert (m.true_positive, m.false_positive, m.false_negative) == (2, 0, 0), m
    # Urut skor: p0 (0,9) mengambil A (IoU 60/140 = 0,429 > 0,333), p1 lalu duplikat di A → FP
    assert match_by_score(gt, preds, [0.9, 0.8], 0.3) == [True, False]
    return "IoU 1/3, greedy 2 TP, VOC duplikat → FP"


def check_preprocessing(_: Config, __: Path) -> str:
    from pcdface.preprocessing import enhance, mean_luma

    rng = np.random.default_rng(0)
    image = rng.integers(20, 90, (120, 160, 3), dtype=np.uint8)
    assert enhance(image, "none") is image, "enhancement none harus citra mentah"
    out = enhance(image, "clahe")
    before = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(int)
    after = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb).astype(int)
    chroma_shift = np.abs(before[:, :, 1:] - after[:, :, 1:]).mean()
    assert chroma_shift < 3.0, f"CLAHE menggeser kroma {chroma_shift:.2f}"
    assert mean_luma(out) > mean_luma(image)
    return f"none = mentah; CLAHE hanya kanal Y (geser kroma {chroma_shift:.2f})"


# ---------------------------------------------------------------------------
# Pemeriksaan dataset sintetis
# ---------------------------------------------------------------------------
def check_synthetic_dataset(cfg: Config, workdir: Path) -> str:
    from pcdface.dataset.annotations import load_annotations
    from pcdface.dataset.metadata import read_metadata, read_subjects
    from pcdface.synthetic import generate_dataset

    paths = generate_dataset(cfg, workdir / "data")
    rows = read_metadata(paths.metadata)
    boxes = load_annotations(paths.annotations)
    assert rows and len(rows) == len(boxes)
    for row in rows:
        assert (paths.raw / row.file).exists(), row.file
        assert row.expected_faces == len(boxes[row.file]), row.file
        assert (row.width, row.height) == (cfg.capture.width, cfg.capture.height)
    assert read_subjects(paths.subjects)

    # Model lubang jarum: kemiringan log(w) terhadap log(Z) ≈ −1
    pairs = [(row.distance_cm, boxes[row.file][0][2]) for row in rows if row.set == "jarak"]
    slope = np.polyfit(np.log([d for d, _ in pairs]), np.log([w for _, w in pairs]), 1)[0]
    assert -1.1 < slope < -0.9, f"kemiringan log-log {slope:.3f}"
    per_set: dict[str, int] = {}
    for row in rows:
        per_set[row.set] = per_set.get(row.set, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(per_set.items()))
    return f"{len(rows)} citra ({summary}); kemiringan log-log {slope:.3f}"


CHECKS: list[tuple[str, Callable[[Config, Path], str]]] = [
    ("matching", check_matching),
    ("preprocessing", check_preprocessing),
    ("dataset sintetis", check_synthetic_dataset),
]


def run_selftest(cfg: Config, verbose: bool = False) -> int:
    """Jalankan semua pemeriksaan. Kode keluar 0 bila semua lulus."""
    print("=== UJI MANDIRI pcdface (tanpa webcam, tanpa model) ===\n")
    results: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="pcdface_selftest_") as temp:
        workdir = Path(temp)
        for name, check in CHECKS:
            try:
                detail = check(cfg, workdir)
                results.append(CheckResult(name, True, detail))
            except Exception as error:  # laporkan semua kegagalan, jangan berhenti di yang pertama
                message = f"{type(error).__name__}: {error}"
                if verbose:
                    message += "\n" + traceback.format_exc()
                results.append(CheckResult(name, False, message))
            status = "OK   " if results[-1].ok else "GAGAL"
            print(f"  [{status}] {name:<22} {results[-1].detail}")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"Uji mandiri: {len(failed)} dari {len(results)} pemeriksaan GAGAL.")
        return 1
    print(f"Uji mandiri: LULUS ({len(results)} pemeriksaan).")
    return 0
