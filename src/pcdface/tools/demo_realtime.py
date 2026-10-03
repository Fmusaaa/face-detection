"""Demo deteksi wajah realtime dari webcam (PRD §12.1).

    python -m pcdface demo                         # mulai dengan mp_short
    python -m pcdface demo --detector haar
    python -m pcdface demo --selftest              # uji tanpa kamera dan tanpa jendela

Tombol: 1/2/3… pilih detektor, d = detektor berikutnya, e = ganti enhancement
(none ↔ clahe), r = pengenalan identitas LBPH aktif/nonaktif (butuh `enroll`),
s = simpan tangkapan layar ke results/demo/, q/ESC = keluar.

RUANG LINGKUP — sebutkan saat presentasi: setiap frame dideteksi ulang dari nol,
tanpa tracking. Nomor #1, #2 hanya urutan kiri → kanan di frame itu. Bila pengenalan
aktif, wajah diberi kode peserta (S01, …) dari model LBPH yang hanya memuat peserta
berizin; wajah lain berlabel "unknown" — sistem tidak pernah menampilkan nama.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pcdface.config import Config, MediaPipeConfig, YoloConfig
from pcdface.detection.base import DetectionResult, Detector
from pcdface.detection.registry import build_detector, model_missing
from pcdface.preprocessing import ENHANCEMENTS, enhance
from pcdface.recognition.lbph import LBPHRecognizer, Prediction, model_exists
from pcdface.tools.keys import lower_key


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
        spec = self.cfg.detector(name)
        if isinstance(spec, YoloConfig):
            return "CPU, OpenCV DNN"
        if not isinstance(spec, MediaPipeConfig):
            return "CPU"
        from pcdface.detection.mediapipe_detector import delegate_label, resolve_delegate

        return delegate_label(resolve_delegate(spec.delegate))


    def close(self) -> None:
        for detector in self._built.values():
            detector.close()
        self._built.clear()


def available_detectors(cfg: Config, requested: list[str]) -> tuple[list[str], list[str]]:
    """(dipakai, dilewati) — MediaPipe/YOLO dilewati bila berkas modelnya belum ada."""
    usable, skipped = [], []
    for name in requested:
        if model_missing(name, cfg):
            skipped.append(name)
        else:
            usable.append(name)
    return usable, skipped


def draw_detections(canvas: np.ndarray, result: DetectionResult,
                    identities: list[Prediction] | None = None) -> None:
    width = canvas.shape[1]
    order = sorted(range(len(result.boxes)), key=lambda i: result.boxes[i][0])
    keypoints = result.info.get("keypoints", [])
    for number, index in enumerate(order, start=1):
        x, y, w, h = result.boxes[index]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BOX, 2)
        label = f"#{number} {w}x{h}px {100 * w / width:.1f}%"
        if result.scores is not None:
            label += f" s={result.scores[index]:.2f}"
        if identities is not None and index < len(identities):
            label = f"{identities[index].label} ({identities[index].distance:.0f}) " + label
        (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y - th - base - 4, 0)
        cv2.rectangle(canvas, (x, top), (x + tw + 6, top + th + base + 4), BOX, -1)
        cv2.putText(canvas, label, (x + 3, top + th + 2), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if index < len(keypoints):
            for kx, ky in keypoints[index]:
                cv2.circle(canvas, (int(kx), int(ky)), 3, (0, 200, 255), -1)


def draw_hud(canvas: np.ndarray, name: str, device: str, enhancement: str, faces: int,
             detect_ms: float, fps: float, recognizing: bool = False) -> None:
    lines = [f"Detektor : {name} ({device})", f"Enhance  : {enhancement}", f"Wajah    : {faces}",
             f"Deteksi  : {detect_ms:.1f} ms | {fps:.1f} FPS", f"Kenali   : {'LBPH aktif' if recognizing else 'mati'}"]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (330, 16 + 22 * len(lines)), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 28 + 22 * i), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    height = canvas.shape[0]
    cv2.putText(canvas, "1/2/3 atau d = detektor  e = enhancement  r = kenali  s = simpan  q = keluar",
                (10, height - 34), FONT, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    scope = ("kode peserta berizin dari LBPH, lainnya 'unknown' - tanpa nama" if recognizing
             else "deteksi per frame - tanpa tracking, tanpa pengenalan identitas")
    cv2.putText(canvas, scope, (10, height - 12), FONT, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


def recognize(frame: np.ndarray, boxes: list, recognizer: LBPHRecognizer, mirrored: bool) -> list[Prediction]:
    """LBPH pada frame asli tanpa enhancement, seperti galeri dilatih.

    Bila tampilan dicerminkan, kotak dipetakan balik ke frame asli: pola LBP wajah cermin
    berbeda dari wajah aslinya, jadi LBPH pada wajah cermin diam-diam memburuk.
    """
    source = cv2.flip(frame, 1) if mirrored else frame
    width = frame.shape[1]
    return [recognizer.predict_box(source, (width - x - w, y, w, h) if mirrored else (x, y, w, h))
            for x, y, w, h in boxes]


def process_frame(frame: np.ndarray, detector: Detector, enhancement: str, cfg: Config,
                  timestamp_ms: int | None = None, recognizer: LBPHRecognizer | None = None,
                  mirrored: bool = False) -> tuple[np.ndarray, DetectionResult]:
    pre = cfg.preprocessing
    image = enhance(frame, enhancement, pre.clahe_clip_limit, pre.clahe_tile_grid)
    if timestamp_ms is not None and hasattr(detector, "running_mode") and detector.running_mode == "video":
        result = detector.detect(image, timestamp_ms=timestamp_ms)
    else:
        result = detector.detect(image)
    identities = recognize(frame, result.boxes, recognizer, mirrored) if recognizer is not None else None
    canvas = image.copy()
    draw_detections(canvas, result, identities)
    return canvas, result


def load_recognizer(cfg: Config) -> LBPHRecognizer | None:
    """Model LBPH bila sudah dilatih (`enroll`), selain itu None."""
    if not model_exists(cfg.paths.recognition):
        return None
    return LBPHRecognizer.load(cfg.paths.recognition, cfg.recognition)


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
        if usable:
            failures += not _selftest_recognition(cfg, cache.get(usable[0]), rng)
    finally:
        cache.close()
    print("Uji mandiri demo: " + ("LULUS" if failures == 0 and usable else "GAGAL"))
    return 0 if failures == 0 and usable else 1


def _selftest_recognition(cfg: Config, detector: Detector, rng: np.random.Generator) -> bool:
    """LBPH sementara (di memori, tidak disimpan) dari dua wajah sintetis; uji jalur kenali + cermin."""
    from pcdface.recognition.lbph import train_from
    from pcdface.synthetic import FaceSpec, render_scene

    width, height = cfg.capture.width, cfg.capture.height
    items = []
    for trait, label in ((0, "S01"), (1, "S02")):
        frame, boxes = render_scene(width, height, [FaceSpec(width // 2, int(height * 0.45), 220, (150, 175, 220),
                                                             trait=trait)], "normal", rng)
        items.append((frame, boxes[0], label))
    recognizer = train_from(items, cfg.recognition)
    frame, box = items[1][0], items[1][1]
    direct = recognize(frame, [box], recognizer, mirrored=False)[0]
    flipped = cv2.flip(frame, 1)
    mirrored_box = (width - box[0] - box[2], box[1], box[2], box[3])
    via_mirror = recognize(flipped, [mirrored_box], recognizer, mirrored=True)[0]
    canvas, _ = process_frame(flipped, detector, "none", cfg, recognizer=recognizer, mirrored=True)
    ok = (direct.nearest == "S02" and via_mirror.nearest == "S02"
          and abs(direct.distance - via_mirror.distance) < 1e-6 and canvas.shape == (height, width, 3))
    print(f"  {'kenali':<9} {'OK' if ok else 'GAGAL'}  (LBPH sementara, frame cermin dipetakan balik)")
    return ok


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
    recognizer: LBPHRecognizer | None = None
    recognizing = False
    print("Detektor: " + ", ".join(f"{i}={n}" for i, n in enumerate(names, start=1)))
    print("Catatan: deteksi per frame, tanpa tracking. Tombol r = pengenalan LBPH (kode peserta berizin).")
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
                                           timestamp_ms=int((time.monotonic() - start) * 1000),
                                           recognizer=recognizer if recognizing else None,
                                           mirrored=not args.no_mirror)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            draw_hud(canvas, current, cache.device(current), enhancement, len(result.boxes), result.elapsed_ms, fps,
                     recognizing)
            cv2.imshow(WINDOW, canvas)

            key = lower_key(cv2.waitKey(1))
            before = (current, enhancement)
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9") and key - ord("1") < len(names):
                current = names[key - ord("1")]
            elif key == ord("d"):
                current = names[(names.index(current) + 1) % len(names)]
            elif key == ord("e"):
                enhancement = ENHANCEMENTS[(ENHANCEMENTS.index(enhancement) + 1) % len(ENHANCEMENTS)]
            elif key == ord("r"):
                if recognizer is None:
                    recognizer = load_recognizer(cfg)
                if recognizer is None:
                    print("  model pengenalan belum ada — jalankan dulu: python -m pcdface enroll")
                else:
                    recognizing = not recognizing
                    print(f"  pengenalan {'aktif' if recognizing else 'mati'} ({', '.join(recognizer.labels)})")
            elif key == ord("s"):
                folder = cfg.paths.results / "demo"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"demo_{current}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"  tersimpan {path} — berisi wajah: jangan dibagikan tanpa izin")
            if (current, enhancement) != before:
                print(f"  detektor={current} enhancement={enhancement}")
    finally:
        cache.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0
