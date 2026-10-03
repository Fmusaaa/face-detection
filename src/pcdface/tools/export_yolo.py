"""Ekspor model YOLO-face `.pt` (ultralytics) ke ONNX untuk OpenCV DNN — di venv TERPISAH.

    python -m pcdface export-yolo                          # yolov8m-face, model repo referensi
    python -m pcdface export-yolo --weights yolov8n-face   # varian lain dari akanametov/yolo-face
    python -m pcdface export-yolo --weights path/ke/model.pt

Mengapa venv terpisah: paket `ultralytics` menarik `opencv-python` (saat diuji: versi 5.0),
yang menimpa modul `cv2` milik `opencv-contrib-python<5` bila dipasang di `.venv` yang sama
(CLAUDE.md aturan #3). Alat ini membuat `.venv-yolo-export/` sekali, memasang ultralytics di
sana, menjalankan ekspor sebagai proses terpisah, lalu memeriksa hasilnya dengan OpenCV DNN
di `.venv` utama. `.venv` utama tidak pernah tersentuh.

Langkah:
1. Berkas `.pt` diunduh dari rilis akanametov/yolo-face ke `models/` bila belum ada.
2. venv ekspor dibuat bila belum ada, lalu diisi `ultralytics onnx onnxslim` (butuh internet, ±1 GB).
3. `YOLO(pt).export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)` —
   ukuran masukan tetap dan opset 12 supaya terbaca OpenCV DNN.
4. ONNX dibuka dengan `YoloDetector` (tata letak keluaran harus dikenali) dan SHA-256 berkas
   `.pt` serta `.onnx` dicatat di `models/checksums.txt`.

URL rilis diambil dari README akanametov/yolo-face; tag `v0.0.0` ada, tetapi isi rilisnya tidak
bisa diperiksa dari lingkungan pengembangan. Bila unduhan gagal, unduh manual dari halaman
Releases repo itu, simpan ke `models/`, lalu jalankan ulang dengan `--weights models/<nama>.pt`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pcdface.config import Config
from pcdface.paths import PROJECT_ROOT
from pcdface.tools.download_models import read_checksums, sha256_of, write_checksums

DEFAULT_VENV = PROJECT_ROOT / ".venv-yolo-export"
RELEASE_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/{name}.pt"
KNOWN_WEIGHTS = ("yolov8n-face", "yolov8m-face", "yolov8l-face")
EXPORT_PACKAGES = ("ultralytics", "onnx", "onnxslim")
EXPORT_SCRIPT = """
import sys
from ultralytics import YOLO
path = YOLO(sys.argv[1]).export(format="onnx", imgsz=int(sys.argv[2]), opset=12, simplify=True, dynamic=False)
print("ONNX:", path)
"""


def venv_python(venv: Path) -> Path:
    """Interpreter di dalam venv (Windows: Scripts/python.exe)."""
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def resolve_weights(weights: str, models_dir: Path) -> tuple[Path, str | None]:
    """(path .pt lokal, URL unduhan atau None). Nama pendek → models/<nama>.pt dari rilis akanametov."""
    candidate = Path(weights)
    if candidate.suffix == ".pt":
        return candidate.resolve(), None
    if weights not in KNOWN_WEIGHTS:
        raise ValueError(f"bobot tidak dikenal: {weights}. Pilihan: {', '.join(KNOWN_WEIGHTS)} atau path .pt")
    return models_dir / f"{weights}.pt", RELEASE_URL.format(name=weights)


def download_weights(url: str, target: Path, timeout: float = 120.0) -> None:
    partial = target.with_suffix(".pt.part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
            for chunk in iter(lambda: response.read(1 << 16), b""):
                handle.write(chunk)
        with partial.open("rb") as handle:
            head = handle.read(2)
        if head != b"PK":   # checkpoint PyTorch adalah arsip zip
            raise RuntimeError(f"berkas dari {url} bukan checkpoint PyTorch (.pt)")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def ensure_export_venv(venv: Path, python: str, install: bool = True) -> Path:
    """Buat venv ekspor dan pasang ultralytics bila belum ada. Kembalikan interpreter-nya."""
    interpreter = venv_python(venv)
    if not interpreter.exists():
        print(f"Membuat venv ekspor terpisah: {venv}")
        subprocess.run([python, "-m", "venv", str(venv)], check=True)
    probe = subprocess.run([str(interpreter), "-c", "import ultralytics, onnx"], capture_output=True)
    if probe.returncode != 0:
        if not install:
            raise RuntimeError(f"ultralytics belum terpasang di {venv} (jalankan tanpa --no-install)")
        print(f"Memasang {' '.join(EXPORT_PACKAGES)} di venv ekspor (bukan di .venv utama)…")
        subprocess.run([str(interpreter), "-m", "pip", "install", "--upgrade", *EXPORT_PACKAGES], check=True)
    return interpreter


def export_onnx(interpreter: Path, weights: Path, input_size: int) -> Path:
    """Jalankan ekspor ultralytics sebagai proses terpisah. Kembalikan path .onnx."""
    subprocess.run([str(interpreter), "-c", EXPORT_SCRIPT, str(weights), str(input_size)],
                   check=True, cwd=weights.parent)
    onnx = weights.with_suffix(".onnx")
    if not onnx.exists():
        raise RuntimeError(f"ekspor selesai tetapi {onnx} tidak ditemukan")
    return onnx


def verify_onnx(onnx: Path, input_size: int) -> str:
    """Buka dengan OpenCV DNN di proses ini (.venv utama). Kembalikan tata letak keluaran."""
    import numpy as np

    from pcdface.detection.yolo import YoloDetector

    with YoloDetector(onnx, name="verifikasi", input_size=input_size) as detector:
        result = detector.detect(np.full((720, 1280, 3), 128, dtype=np.uint8))
    if len(result.boxes) != len(result.scores or []):
        raise RuntimeError("keluaran detektor tidak konsisten")
    return detector.layout


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", default="yolov8m-face",
                        help=f"Nama bobot akanametov ({', '.join(KNOWN_WEIGHTS)}) atau path .pt (bawaan: yolov8m-face)")
    parser.add_argument("--venv", default=str(DEFAULT_VENV), help="Folder venv ekspor (bawaan: .venv-yolo-export)")
    parser.add_argument("--python", default=sys.executable, help="Interpreter untuk membuat venv ekspor")
    parser.add_argument("--input-size", type=int, default=640, help="Sisi masukan ONNX (bawaan: 640)")
    parser.add_argument("--no-install", action="store_true", help="Jangan memasang paket; gagal bila belum ada")


def run(args: argparse.Namespace, cfg: Config) -> int:
    models_dir = cfg.paths.models
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        weights, url = resolve_weights(args.weights, models_dir)
        if not weights.exists():
            if url is None:
                raise FileNotFoundError(f"berkas bobot tidak ada: {weights}")
            print(f"Mengunduh {url}")
            try:
                download_weights(url, weights)
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"unduhan gagal ({error}). Unduh manual {weights.name} dari "
                    "https://github.com/akanametov/yolo-face (Releases), simpan ke models/, "
                    f"lalu jalankan: python -m pcdface export-yolo --weights {weights}"
                ) from error
        interpreter = ensure_export_venv(Path(args.venv), args.python, install=not args.no_install)
        onnx = export_onnx(interpreter, weights, args.input_size)
        if onnx.parent.resolve() != models_dir.resolve():   # config mencari model di models/
            onnx = Path(shutil.copy2(onnx, models_dir / onnx.name))
        layout = verify_onnx(onnx, args.input_size)
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1

    checksums = read_checksums(models_dir)
    for path in (weights, onnx):
        if path.parent.resolve() == models_dir.resolve():
            checksums[path.name] = sha256_of(path)
    write_checksums(models_dir, checksums)
    print(f"[OK] {onnx.name}: tata letak '{layout}', SHA-256 {sha256_of(onnx)}")
    names = [n for n, spec in cfg.detectors.items() if getattr(spec, "model", None) == onnx.name]
    if names:
        print(f"Dipakai detektor: {', '.join(names)} (configs/experiment.yaml)")
    else:
        print(f"Belum ada detektor di config dengan model: {onnx.name} — tambahkan entri type: yolo.")
    return 0
