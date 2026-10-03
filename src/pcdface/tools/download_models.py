"""
Pengunduh Model Detektor Wajah
==============================

Mengunduh berkas model BlazeFace (.tflite) dan YOLOv8n-face (.onnx) ke folder
`models/` dan mencatat checksum SHA-256-nya di `models/checksums.txt` (format
`shasum -a 256`), sehingga bisa diperiksa ulang dengan:

    cd models && shasum -a 256 -c checksums.txt

Bila berkas sudah ada, isinya diverifikasi terhadap checksum yang tercatat
dan tidak diunduh ulang. Bila checksum tidak cocok, skrip berhenti dengan
galat — berkas tidak ditimpa diam-diam kecuali memakai `--force`.

URL diambil dari PRD §5.2 dan §5.3. Bila salah satu gagal diunduh, galatnya
dilaporkan apa adanya; skrip ini tidak mencari URL pengganti.

YOLOv8n-face adalah model derronqi/yolov8-face (dilatih pada WIDER FACE) dalam
bentuk ONNX dari repo hpc203/yolov8-face-landmarks-opencv-dnn, dipatok ke satu
commit. Model `.pt` lain (mis. `yolov8m-face` akanametov seperti repo referensi)
harus diekspor dulu: `python -m pcdface export-yolo`.

Pemakaian
---------
    python -m pcdface download-models
    python -m pcdface download-models --only short_range
    python -m pcdface download-models --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pcdface.paths import PROJECT_ROOT

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
CHECKSUM_FILE = "checksums.txt"

_BASE_URL = "https://storage.googleapis.com/mediapipe-models/face_detector"
_YOLO_COMMIT = "1f91851f7d8d9475e5b4f0c1d6e5e385aa9bf0f4"
_YOLO_URL = (f"https://raw.githubusercontent.com/hpc203/yolov8-face-landmarks-opencv-dnn/"
             f"{_YOLO_COMMIT}/weights/yolov8n-face.onnx")

# Berkas TFLite adalah FlatBuffer dengan penanda "TFL3" pada byte 4-7.
# Dipakai untuk menolak halaman galat HTML yang tersimpan sebagai .tflite.
_TFLITE_MAGIC = b"TFL3"
_CHUNK_SIZE = 1 << 16


@dataclass(frozen=True)
class ModelSpec:
    """Satu berkas model yang akan diunduh."""

    key: str
    filename: str
    url: str
    kind: str = "tflite"            # tflite | onnx — menentukan pemeriksaan isi berkas


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="short_range",
        filename="blaze_face_short_range.tflite",
        url=f"{_BASE_URL}/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    ),
    ModelSpec(
        key="full_range",
        filename="blaze_face_full_range.tflite",
        url=f"{_BASE_URL}/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
    ),
    ModelSpec(
        key="full_range_sparse",
        filename="blaze_face_full_range_sparse.tflite",
        url=f"{_BASE_URL}/blaze_face_full_range/float16/latest/blaze_face_full_range_sparse.tflite",
    ),
    ModelSpec(
        key="yolov8n_face",
        filename="yolov8n-face.onnx",
        url=_YOLO_URL,
        kind="onnx",
    ),
)


def sha256_of(path: Path) -> str:
    """Hitung SHA-256 sebuah berkas secara bertahap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(models_dir: Path) -> dict[str, str]:
    """Baca `checksums.txt` menjadi {nama_berkas: sha256}."""
    path = models_dir / CHECKSUM_FILE
    if not path.exists():
        return {}
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            checksums[parts[1]] = parts[0]
    return checksums


def write_checksums(models_dir: Path, checksums: dict[str, str]) -> None:
    """Tulis checksum dalam format yang bisa dibaca `shasum -a 256 -c`."""
    lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
    (models_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_tflite(path: Path) -> bool:
    """Periksa penanda FlatBuffer TFLite pada awal berkas."""
    with path.open("rb") as handle:
        header = handle.read(8)
    return len(header) == 8 and header[4:8] == _TFLITE_MAGIC


def is_onnx(path: Path) -> bool:
    """Periksa awal berkas ONNX (protobuf ModelProto: medan 1 `ir_version`, byte 0x08).

    Pemeriksaan longgar, cukup untuk menolak halaman galat HTML/teks yang tersimpan sebagai .onnx.
    """
    with path.open("rb") as handle:
        header = handle.read(2)
    return len(header) == 2 and header[0] == 0x08


def looks_valid(path: Path, kind: str) -> bool:
    return is_onnx(path) if kind == "onnx" else is_tflite(path)


def download(spec: ModelSpec, target: Path, timeout: float = 60.0) -> None:
    """Unduh ke berkas sementara lalu ganti nama, agar unduhan yang terputus
    tidak meninggalkan berkas model yang rusak."""
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=timeout) as response:
            with partial.open("wb") as handle:
                for chunk in iter(lambda: response.read(_CHUNK_SIZE), b""):
                    handle.write(chunk)
        if not looks_valid(partial, spec.kind):
            raise RuntimeError(
                f"Berkas dari {spec.url} bukan model {spec.kind.upper()} (isi awal berkas tidak cocok)."
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def fetch_model(
    spec: ModelSpec,
    models_dir: Path,
    checksums: dict[str, str],
    force: bool = False,
) -> str:
    """Pastikan satu model tersedia dan cocok dengan checksum-nya.

    Mengembalikan keterangan status singkat untuk dicetak. Checksum baru
    langsung dicatat ke `checksums`.
    """
    target = models_dir / spec.filename
    recorded = checksums.get(spec.filename)

    if target.exists() and not force:
        actual = sha256_of(target)
        if recorded is None:
            checksums[spec.filename] = actual
            return "sudah ada, checksum baru dicatat"
        if actual != recorded:
            raise RuntimeError(
                f"Checksum {spec.filename} tidak cocok.\n"
                f"  tercatat: {recorded}\n  aktual  : {actual}\n"
                "Hapus berkasnya atau jalankan ulang dengan --force."
            )
        return "sudah ada, checksum cocok"

    download(spec, target)
    actual = sha256_of(target)
    if recorded is not None and actual != recorded and not force:
        raise RuntimeError(
            f"Berkas {spec.filename} yang baru diunduh tidak cocok dengan checksum "
            f"tercatat ({recorded}). Model di server mungkin berubah; "
            "periksa lalu jalankan ulang dengan --force."
        )
    checksums[spec.filename] = actual
    return "terunduh"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-dir", default=None, help="Folder model (bawaan: paths.models di config)")
    parser.add_argument(
        "--only",
        choices=[spec.key for spec in MODELS],
        action="append",
        help="Unduh model tertentu saja (boleh diulang)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Unduh ulang dan timpa checksum yang tercatat"
    )


def run(args: argparse.Namespace, models_dir: Path = DEFAULT_MODELS_DIR) -> int:
    """Unduh model terpilih. Kode keluar 1 bila ada yang gagal."""
    models_dir = Path(args.models_dir) if args.models_dir else models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    selected = [spec for spec in MODELS if not args.only or spec.key in args.only]
    checksums = read_checksums(models_dir)
    failures: list[str] = []

    for spec in selected:
        try:
            status = fetch_model(spec, models_dir, checksums, force=args.force)
        except (urllib.error.URLError, OSError, RuntimeError) as error:
            failures.append(spec.key)
            print(f"[GAGAL] {spec.key}: {spec.url}\n  {error}", file=sys.stderr)
            continue
        size = (models_dir / spec.filename).stat().st_size
        print(f"[OK]    {spec.key:<18} {size:>9,} B  {checksums[spec.filename]}  ({status})")

    if checksums:
        write_checksums(models_dir, checksums)
        print(f"\nChecksum: {models_dir / CHECKSUM_FILE}")

    if failures:
        print(f"{len(failures)} model gagal: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unduh model detektor wajah (BlazeFace, YOLOv8n-face) dan catat SHA-256-nya."
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
