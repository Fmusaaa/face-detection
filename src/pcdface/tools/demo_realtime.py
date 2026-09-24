"""Demo deteksi wajah realtime dari webcam (PRD §12.1).

    python -m pcdface demo                         # mulai dengan mp_short
    python -m pcdface demo --detector haar
    python -m pcdface demo --selftest              # uji tanpa kamera dan tanpa jendela

Tombol: 1/2/3… pilih detektor, d = detektor berikutnya, e = ganti enhancement
(none ↔ clahe), s = simpan tangkapan layar ke results/demo/, q/ESC = keluar.

RUANG LINGKUP — sebutkan saat presentasi: setiap frame dideteksi ulang dari nol.
Tidak ada tracking dan tidak ada pengenalan identitas; nomor #1, #2 hanya
urutan kiri → kanan di frame itu, bukan identitas.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pcdface.config import Config, MediaPipeConfig
from pcdface.detection.base import DetectionResult, Detector
from pcdface.detection.registry import build_detector
from pcdface.preprocessing import ENHANCEMENTS, enhance

WINDOW = "Demo deteksi wajah - pcdface"
BOX = (0, 220, 60)
HUD_BG = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class DetectorCache:
    """Buat detektor saat pertama dipilih, simpan untuk dipakai ulang, tutup semua di akhir."""

    def __init__(self, cfg: Config, names: list[str]) -> None:
        self.cfg = cfg
        self.names = names
        self._built: dict[str, Detector] = {}

    def get(self, name: str) -> Detector:
        if name not in self._built:
            mode = "video" if isinstance(self.cfg.detector(name), MediaPipeConfig) else "image"
            self._built[name] = build_detector(name, self.cfg, running_mode=mode)
        return self._built[name]

    def device(self, name: str) -> str:
        return "GPU Metal" if isinstance(self.cfg.detector(name), MediaPipeConfig) else "CPU"

    def close(self) -> None:
        for detector in self._built.values():
            detector.close()
        self._built.clear()


def available_detectors(cfg: Config, requested: list[str]) -> tuple[list[str], list[str]]:
    """(dipakai, dilewati) — MediaPipe dilewati bila berkas modelnya belum ada."""
    usable, skipped = [], []
    for name in requested:
        spec = cfg.detector(name)
        if isinstance(spec, MediaPipeConfig) and not (cfg.paths.models / spec.model).exists():
            skipped.append(name)
        else:
            usable.append(name)
    return usable, skipped


def draw_detections(canvas: np.ndarray, result: DetectionResult) -> None:
    width = canvas.shape[1]
    order = sorted(range(len(result.boxes)), key=lambda i: result.boxes[i][0])
    keypoints = result.info.get("keypoints", [])
    for number, index in enumerate(order, start=1):
        x, y, w, h = result.boxes[index]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BOX, 2)
        label = f"#{number} {w}x{h}px {100 * w / width:.1f}%"
        if result.scores is not None:
            label += f" s={result.scores[index]:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y - th - base - 4, 0)
        cv2.rectangle(canvas, (x, top), (x + tw + 6, top + th + base + 4), BOX, -1)
        cv2.putText(canvas, label, (x + 3, top + th + 2), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if index < len(keypoints):
            for kx, ky in keypoints[index]:
                cv2.circle(canvas, (int(kx), int(ky)), 3, (0, 200, 255), -1)


def draw_hud(canvas: np.ndarray, name: str, device: str, enhancement: str, faces: int,
             detect_ms: float, fps: float) -> None:
    lines = [f"Detektor : {name} ({device})", f"Enhance  : {enhancement}", f"Wajah    : {faces}",
             f"Deteksi  : {detect_ms:.1f} ms | {fps:.1f} FPS"]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (330, 16 + 22 * len(lines)), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 28 + 22 * i), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    height = canvas.shape[0]
    cv2.putText(canvas, "1/2/3 atau d = detektor  e = enhancement  s = simpan  q = keluar",
                (10, height - 34), FONT, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "deteksi per frame - tanpa tracking, tanpa pengenalan identitas",
                (10, height - 12), FONT, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


def process_frame(frame: np.ndarray, detector: Detector, enhancement: str, cfg: Config,
                  timestamp_ms: int | None = None) -> tuple[np.ndarray, DetectionResult]:
    pre = cfg.preprocessing
    image = enhance(frame, enhancement, pre.clahe_clip_limit, pre.clahe_tile_grid)
    if timestamp_ms is not None and hasattr(detector, "running_mode") and detector.running_mode == "video":
        result = detector.detect(image, timestamp_ms=timestamp_ms)
    else:
        result = detector.detect(image)
    canvas = image.copy()
    draw_detections(canvas, result)
    return canvas, result


def run_selftest(cfg: Config, names: list[str]) -> int:
    """Jalankan tiap detektor pada frame sintetis bergerak, tanpa kamera dan jendela."""
    from pcdface.synthetic import FaceSpec, render_scene

    usable, skipped = available_detectors(cfg, names)
    print("=== UJI MANDIRI DEMO (frame sintetis, tanpa kamera) ===")
    if skipped:
        print(f"  dilewati (model belum diunduh): {', '.join(skipped)}")
    rng = np.random.default_rng(cfg.seed)
    width, height = cfg.capture.width, cfg.capture.height
    cache = DetectorCache(cfg, usable)
    failures = 0
    try:
        for name in usable:
            detector = cache.get(name)
            shapes = set()
            for i in range(8):
                face = FaceSpec(int(width / 2 + 200 * np.sin(i / 8 * 2 * np.pi)), int(height * 0.45), 150,
                                (150, 175, 220))
                frame, _ = render_scene(width, height, [face], "normal", rng)
                canvas, result = process_frame(frame, detector, ENHANCEMENTS[i % 2], cfg, timestamp_ms=i * 33)
                draw_hud(canvas, name, cache.device(name), ENHANCEMENTS[i % 2], len(result.boxes),
                         result.elapsed_ms, 30.0)
                shapes.add(canvas.shape)
            ok = shapes == {(height, width, 3)}
            failures += not ok
            print(f"  {name:<9} {'OK' if ok else 'GAGAL'}  (8 frame, ganti enhancement tiap frame)")
    finally:
        cache.close()
    print("Uji mandiri demo: " + ("LULUS" if failures == 0 and usable else "GAGAL"))
    return 0 if failures == 0 and usable else 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", default="mp_short", help="Detektor awal (bawaan: mp_short)")
    parser.add_argument("--detectors", default=None,
                        help="Daftar detektor yang bisa dipilih, dipisah koma (bawaan: detektor E1)")
    parser.add_argument("--enhance", default="none", choices=ENHANCEMENTS)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true", help="Jangan cerminkan tampilan")
    parser.add_argument("--selftest", action="store_true", help="Uji tanpa kamera dan tanpa jendela")


def run(args: argparse.Namespace, cfg: Config) -> int:
    requested = args.detectors.split(",") if args.detectors else list(cfg.experiments.e1.detectors)
    if args.detector not in requested:
        requested.insert(0, args.detector)
    for name in requested:
        cfg.detector(name)  # galat jelas bila nama salah
    if args.selftest:
        return run_selftest(cfg, requested)

    names, skipped = available_detectors(cfg, requested)
    if skipped:
        print(f"[PERINGATAN] model belum diunduh, dilewati: {', '.join(skipped)} "
              "(python -m pcdface download-models)", file=sys.stderr)
    if not names:
        return 1
    current = args.detector if args.detector in names else names[0]
    enhancement = args.enhance

    camera_index = cfg.capture.camera_index if args.camera is None else args.camera
    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture.height)
    if not camera.isOpened():
        print(f"[GAGAL] Kamera {camera_index} tidak bisa dibuka. Beri izin kamera untuk Terminal di "
              "System Settings → Privacy & Security → Camera.", file=sys.stderr)
        return 1

    cache = DetectorCache(cfg, names)
    print("Detektor: " + ", ".join(f"{i}={n}" for i, n in enumerate(names, start=1)))
    print("Catatan: deteksi per frame, tanpa tracking dan tanpa pengenalan identitas.")
    fps, last, start = 0.0, time.perf_counter(), time.monotonic()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[GAGAL] Tidak bisa membaca frame dari kamera.", file=sys.stderr)
                return 1
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            detector = cache.get(current)
            canvas, result = process_frame(frame, detector, enhancement, cfg,
                                           timestamp_ms=int((time.monotonic() - start) * 1000))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            draw_hud(canvas, current, cache.device(current), enhancement, len(result.boxes), result.elapsed_ms, fps)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9") and key - ord("1") < len(names):
                current = names[key - ord("1")]
            elif key == ord("d"):
                current = names[(names.index(current) + 1) % len(names)]
            elif key == ord("e"):
                enhancement = ENHANCEMENTS[(ENHANCEMENTS.index(enhancement) + 1) % len(ENHANCEMENTS)]
            elif key == ord("s"):
                folder = cfg.paths.results / "demo"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"demo_{current}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"  tersimpan {path} — berisi wajah: jangan dibagikan tanpa izin")
            if key != 255:
                print(f"  detektor={current} enhancement={enhancement}")
    finally:
        cache.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0
