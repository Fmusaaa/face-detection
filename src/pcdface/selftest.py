"""Uji mandiri tanpa webcam dan tanpa model.

Memeriksa (1) metrik terhadap nilai acuan hitungan tangan dan (2) jalur data
dan detektor pada dataset sintetis. Angka dari citra sintetis bukan hasil
penelitian — hanya bukti bahwa kode berjalan dan metrik terhitung benar.
"""

from __future__ import annotations

import dataclasses
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


def check_metrics(_: Config, __: Path) -> str:
    from pcdface.evaluation.average_precision import pr_curve
    from pcdface.evaluation.distance_analysis import loglog_fit, min_face_size
    from pcdface.evaluation.operating_point import aggregate, evaluate_image
    from pcdface.evaluation.stats import wilson

    ap = pr_curve(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, False, True, False]), 3).ap
    assert _close(ap, 5 / 9), ap
    est = wilson(8, 10)
    assert _close(est.low, 0.4902, 1e-4) and _close(est.high, 0.9433, 1e-4), est
    op = aggregate([evaluate_image("a", "g", [(0, 0, 10, 10), (50, 0, 10, 10)], [(0, 0, 10, 10), (200, 200, 5, 5)], 0.5)])
    assert _close(op.f1, 0.5) and _close(op.fppi, 1.0), op
    fit = loglog_fit([50, 100, 200, 300], [300, 150, 75, 50])
    assert _close(fit.slope, -1.0, 1e-9), fit
    assert min_face_size([10, 30, 50], [False, True, True], [0, 20, 40, 1000], 0.9).threshold == 20
    return "AP 5/9, Wilson 8/10 [0,490; 0,943], F1 0,5, log-log −1, ukuran minimum"


def check_detectors(cfg: Config, workdir: Path) -> str:
    from pcdface.dataset.loader import load_samples
    from pcdface.detection.haar import nms_with_shifted_scores
    from pcdface.detection.registry import build_detector, supports_scores

    assert nms_with_shifted_scores(np.array([[0, 0, 10, 10]]), np.array([-3.0]), 0.3).tolist() == [0], \
        "NMS membuang kandidat tunggal berskor negatif"
    paths = cfg.paths.with_data_root(workdir / "data")
    samples = load_samples(paths, sets=("multi",)).samples[:3]
    assert samples, "dataset sintetis belum dibuat"
    parts = []
    for name, synthetic in (("haar", False), ("ycbcr", False), ("haar", True), ("mp_short", True), ("mp_full", True),
                            ("yolo_n", True)):
        modes = ("operating", "ap") if supports_scores(name, cfg) else ("operating",)
        for mode in modes:
            with build_detector(name, cfg, mode=mode, synthetic=synthetic) as detector:
                for sample in samples:
                    result = detector.detect(sample.load())
                    if detector.has_scores:
                        assert result.scores is not None and len(result.scores) == len(result.boxes)
        parts.append(f"{'fake:' if synthetic else ''}{name}")
    return "kontrak DetectionResult OK untuk " + ", ".join(parts)


def yolo_raw_outputs(cell: tuple[int, int, int], bins: int, score_logit: float = 4.0,
                     input_size: int = 640) -> list[np.ndarray]:
    """Tiga tensor head mentah YOLOv8-face buatan: satu sel (stride, gx, gy) berisi wajah dengan jarak
    ke keempat tepi = `bins` × stride (DFL hampir satu-panas), sel lain berskor ≈ 0."""
    outputs = []
    for stride in (32, 8, 16):                              # urutan acak seperti keluaran OpenCV
        grid = input_size // stride
        out = np.zeros((1, 80, grid, grid), dtype=np.float32)
        out[0, 64] = -20.0                                  # logit skor sangat rendah
        if stride == cell[0]:
            gx, gy = cell[1], cell[2]
            out[0, :64, gy, gx] = -10.0
            for side in range(4):
                out[0, side * 16 + bins, gy, gx] = 10.0     # softmax ≈ satu-panas di bin `bins`
            out[0, 64, gy, gx] = score_logit
        outputs.append(out)
    return outputs


def check_yolo_decode(_: Config, __: Path) -> str:
    from pcdface.detection.yolo import (decode_raw, decode_ultralytics, detect_layout, letterbox,
                                        postprocess)

    # 1280×720 → letterbox 640: skala 0,5, pita atas 140 px
    padded, lb = letterbox(np.zeros((720, 1280, 3), np.uint8), 640)
    assert padded.shape == (640, 640, 3) and (lb.scale, lb.pad_x, lb.pad_y) == (0.5, 0, 140), lb
    # sel stride 16 (gx 10, gy 15): pusat (168, 248), jarak tepi 2 bin × 16 = 32 px
    raw = yolo_raw_outputs((16, 10, 15), bins=2)
    assert detect_layout(raw) == "raw"
    boxes, scores, _ = decode_raw(raw, 640)
    found, found_scores, _ = postprocess(boxes, scores, None, lb, (1280, 720), 0.25, 0.7, 300)
    assert found == [(272, 152, 128, 128)], found
    assert abs(found_scores[0] - 1 / (1 + np.exp(-4.0))) < 1e-6
    # tata letak ultralytics: kotak yang sama + duplikat bergeser 2 px yang harus dibuang NMS
    pred = np.zeros((1, 5, 8400), dtype=np.float32)
    pred[0, :, 0] = [168, 248, 64, 64, 0.9]
    pred[0, :, 1] = [170, 248, 64, 64, 0.8]
    assert detect_layout([pred]) == "ultralytics"
    found, found_scores, _ = postprocess(*decode_ultralytics(pred), lb, (1280, 720), 0.25, 0.7, 300)
    assert found == [(272, 152, 128, 128)] and found_scores == [np.float32(0.9).item()], (found, found_scores)
    return "letterbox 1280×720→640, decode DFL & ultralytics → kotak (272, 152, 128, 128), NMS buang duplikat"


def check_recognition(cfg: Config, _: Path) -> str:
    from pcdface.recognition.lbph import UNKNOWN, LBPHRecognizer, check_label

    rng = np.random.default_rng(cfg.seed)
    size = cfg.recognition.face_size
    bases = {label: rng.integers(0, 256, (size[1], size[0])).astype(np.uint8) for label in ("S01", "S02")}

    def noisy(image: np.ndarray) -> np.ndarray:
        return np.clip(image.astype(int) + rng.integers(-12, 13, image.shape), 0, 255).astype(np.uint8)

    recognizer = LBPHRecognizer(cfg.recognition)
    recognizer.train([noisy(b) for b in bases.values() for _ in range(3)], [k for k in bases for _ in range(3)])
    for label, base in bases.items():
        prediction = recognizer.predict(noisy(base))
        assert prediction.nearest == label, prediction
    strict = LBPHRecognizer(dataclasses.replace(cfg.recognition, max_distance=1e-6))
    strict.train([bases["S01"]], ["S01"])
    assert strict.predict(noisy(bases["S02"])).label == UNKNOWN
    try:
        check_label("Budi")
    except ValueError:
        pass
    else:
        raise AssertionError("nama asli harus ditolak sebagai label")
    return "LBPH membedakan 2 tekstur, ambang → unknown, label nama ditolak"


CHECKS: list[tuple[str, Callable[[Config, Path], str]]] = [
    ("matching", check_matching),
    ("preprocessing", check_preprocessing),
    ("metrik", check_metrics),
    ("dataset sintetis", check_synthetic_dataset),
    ("detektor", check_detectors),
    ("yolo (dekode)", check_yolo_decode),
    ("pengenalan LBPH", check_recognition),
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
