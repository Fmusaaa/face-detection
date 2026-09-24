"""CLI `python -m pcdface <perintah>` (PRD §12.1).

Modul perintah diimpor saat dipakai saja, supaya perintah ringan seperti
`validate` tidak ikut memuat MediaPipe.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from pcdface import __version__
from pcdface.config import Config, ConfigError, load_config

Handler = Callable[[argparse.Namespace, Config], int]


def _pending(phase: int) -> Handler:
    def handler(args: argparse.Namespace, cfg: Config) -> int:
        print(f"'{args.command}' belum diimplementasi (Fase {phase}).", file=sys.stderr)
        return 2

    return handler


# ---------------------------------------------------------------------------
# Perintah
# ---------------------------------------------------------------------------
def _download_models_args(parser: argparse.ArgumentParser) -> None:
    from pcdface.tools import download_models

    download_models.add_arguments(parser)


def _download_models(args: argparse.Namespace, cfg: Config) -> int:
    from pcdface.tools import download_models

    return download_models.run(args, models_dir=cfg.paths.models)


def _selftest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="Tampilkan traceback galat")


def _selftest(args: argparse.Namespace, cfg: Config) -> int:
    from pcdface.selftest import run_selftest

    return run_selftest(cfg, verbose=args.verbose)


def _no_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)


def _tool(module: str) -> tuple[Callable[[argparse.ArgumentParser], None], Handler]:
    """Perintah dari modul `pcdface.tools.<module>` yang punya add_arguments() dan run()."""

    def load():
        import importlib

        return importlib.import_module(f"pcdface.tools.{module}")

    def add_args(parser: argparse.ArgumentParser) -> None:
        load().add_arguments(parser)

    def handler(args: argparse.Namespace, cfg: Config) -> int:
        return load().run(args, cfg)

    return add_args, handler


COMMANDS: list[tuple[str, str, Callable[[argparse.ArgumentParser], None], Handler]] = [
    ("download-models", "Unduh model .tflite, tulis dan verifikasi SHA-256", _download_models_args, _download_models),
    ("capture", "Rekam foto dari webcam dan tulis metadata", *_tool("capture")),
    ("annotate", "Gambar kotak wajah manual (ground truth)", *_tool("annotate")),
    ("crop", "Ekspor isi kotak manual ke data/crops/", *_tool("crop")),
    ("validate", "Periksa konsistensi data, metadata, dan anotasi", *_tool("validate")),
    ("forget", "Hapus seluruh data satu subjek", *_tool("forget")),
    ("run", "Jalankan eksperimen e1..e5 atau all", _no_args, _pending(4)),
    ("report", "Bangun ulang tabel dan grafik dari hasil tersimpan", _no_args, _pending(5)),
    ("demo", "Demo deteksi realtime dari webcam", _no_args, _pending(5)),
    ("selftest", "Uji metrik dan jalur sintetis tanpa webcam/model", _selftest_args, _selftest),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pcdface",
        description="Deteksi wajah Haar Cascade vs MediaPipe BlazeFace — tanpa pengenalan identitas.",
    )
    parser.add_argument("--version", action="version", version=f"pcdface {__version__}")
    parser.add_argument("--config", default=None, help="Berkas config (bawaan: configs/experiment.yaml)")
    sub = parser.add_subparsers(dest="command", metavar="<perintah>")
    sub.required = True
    for name, help_text, add_args, handler in COMMANDS:
        command = sub.add_parser(name, help=help_text, description=help_text)
        add_args(command)
        command.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args, cfg) or 0)
    except KeyboardInterrupt:
        print("\nDihentikan.", file=sys.stderr)
        return 130
