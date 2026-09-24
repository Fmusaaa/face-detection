"""Jalankan eksperimen, simpan snapshot config + versi, dan catat deteksi.

    python -m pcdface run e1
    python -m pcdface run all
    python -m pcdface run all --synthetic     # citra sintetis + FakeDetector

Setiap eksperimen menulis ke `results/<eN>/` (run sintetis ke
`results/synthetic/<eN>/`): tabel `.csv` + `.md`, `config_snapshot.yaml`,
dan `deteksi.jsonl` (kotak/skor per citra, untuk audit). Folder eksperimen
dikosongkan dulu supaya tidak ada berkas basi dari run sebelumnya.

Sebelum data asli dipakai, `validate` dijalankan; eksperimen berhenti bila
ada GALAT (lewati dengan `--skip-validate`).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pcdface import __version__
from pcdface.config import Config
from pcdface.dataset.loader import Sample, load_samples
from pcdface.detection.base import Detector
from pcdface.detection.registry import build_detector
from pcdface.experiments.common import Record, append_jsonl
from pcdface.paths import ProjectPaths
from pcdface.reporting.tables import write_table

EXPERIMENTS: dict[str, str] = {
    "e1": "pcdface.experiments.e1_distance_resolution",
    "e2": "pcdface.experiments.e2_multiface",
    "e3": "pcdface.experiments.e3_lighting",
    "e4": "pcdface.experiments.e4_speed",
    "e5": "pcdface.experiments.e5_sensitivity",
}


class ExperimentError(RuntimeError):
    """Eksperimen tidak bisa dijalankan (mis. data yang dibutuhkan kosong)."""


@dataclass
class RunContext:
    cfg: Config
    paths: ProjectPaths
    name: str
    out_dir: Path
    synthetic: bool = False
    tables: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    _samples: dict[tuple[str, ...], list[Sample]] = field(default_factory=dict)

    def log(self, message: str) -> None:
        print(f"  [{self.name}] {message}", flush=True)

    def samples(self, sets: tuple[str, ...]) -> list[Sample]:
        if sets not in self._samples:
            report = load_samples(self.paths, sets)
            if report.unannotated:
                self.log(f"PERINGATAN: {len(report.unannotated)} citra belum dianotasi, tidak diikutkan")
            if report.missing_files:
                self.log(f"PERINGATAN: {len(report.missing_files)} berkas foto hilang, tidak diikutkan")
            self._samples[sets] = report.samples
            counts = pd.Series([s.meta.set for s in report.samples]).value_counts().to_dict()
            self.notes.setdefault("sampel", {}).update({k: int(v) for k, v in counts.items()})
            sessions = pd.Series([s.meta.session or "(tanpa sesi)" for s in report.samples]).value_counts().to_dict()
            self.notes.setdefault("sampel_per_sesi", {}).update({str(k): int(v) for k, v in sessions.items()})
        return self._samples[sets]

    def require(self, samples: list[Sample], set_name: str) -> list[Sample]:
        chosen = [s for s in samples if s.meta.set == set_name]
        if not chosen:
            raise ExperimentError(f"set '{set_name}' tidak punya citra beranotasi")
        return chosen

    def detector(self, name: str, mode: str = "operating", overrides: dict[str, Any] | None = None) -> Detector:
        return build_detector(name, self.cfg, mode=mode, synthetic=self.synthetic, overrides=overrides)

    def device(self, name: str) -> str:
        if self.synthetic:
            return "tiruan (FakeDetector)"
        return "GPU (Metal)" if self.cfg.detector(name).type == "mediapipe" else "CPU"

    def table(self, df: pd.DataFrame, stem: str, title: str = "", notes: str = "") -> None:
        write_table(df, self.out_dir, stem, title, notes)
        self.tables.append(stem)

    def dump(self, records: list[Record]) -> None:
        append_jsonl(self.out_dir / "deteksi.jsonl", records)


def software_versions() -> dict[str, str]:
    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "tidak terpasang"

    import cv2

    return {
        "pcdface": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "opencv": cv2.__version__,
        "opencv-contrib-python": version("opencv-contrib-python"),
        "mediapipe": version("mediapipe"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "matplotlib": version("matplotlib"),
    }


def write_snapshot(ctx: RunContext, seconds: float) -> None:
    snapshot = {
        "eksperimen": ctx.name,
        "waktu": datetime.now().astimezone().isoformat(timespec="seconds"),
        "durasi_detik": round(seconds, 1),
        "sintetis": ctx.synthetic,
        "catatan": ("Citra sintetis + FakeDetector — angka BUKAN hasil penelitian."
                    if ctx.synthetic else "Data asli."),
        "versi": software_versions(),
        "tabel": ctx.tables,
        **ctx.notes,
        "config": ctx.cfg.raw,
    }
    with (ctx.out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)


def _selected(names: list[str], cfg: Config) -> list[str]:
    if names == ["all"]:
        chosen = ["e1", "e2", "e3", "e4"]
        if cfg.experiments.e5.enabled:
            chosen.append("e5")
        return chosen
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise ExperimentError(f"eksperimen tidak dikenal: {unknown}. Pilihan: {list(EXPERIMENTS)} atau all")
    return names


def run_experiments(
    cfg: Config,
    names: list[str],
    synthetic: bool = False,
    results_root: Path | None = None,
    skip_validate: bool = False,
    plots: bool = True,
) -> dict[str, Path]:
    """Jalankan eksperimen terpilih. Kembalikan {nama: folder hasil}."""
    names = _selected(names, cfg)
    outputs: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="pcdface_synthetic_") as temp:
        if synthetic:
            from pcdface.synthetic import generate_dataset

            paths = generate_dataset(cfg, Path(temp) / "data")
        else:
            paths = cfg.paths
        if results_root is not None:
            paths = ProjectPaths(**{**paths.__dict__, "results": results_root})

        if not skip_validate:
            from pcdface.tools.validate import ERROR, validate_dataset

            errors = [i for i in validate_dataset(cfg, paths, check_images=False)[0] if i.level == ERROR]
            if errors:
                for issue in errors[:10]:
                    print(issue, file=sys.stderr)
                raise ExperimentError(f"validate menemukan {len(errors)} galat — perbaiki dulu "
                                      "(python -m pcdface validate) atau pakai --skip-validate")

        for name in names:
            out_dir = paths.results / name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            ctx = RunContext(cfg=cfg, paths=paths, name=name, out_dir=out_dir, synthetic=synthetic)
            label = " (sintetis)" if synthetic else ""
            print(f"== {name}{label} → {out_dir}", flush=True)
            start = time.perf_counter()
            try:
                importlib.import_module(EXPERIMENTS[name]).run(ctx)
            except BaseException:
                shutil.rmtree(out_dir, ignore_errors=True)  # hasil setengah jadi menyesatkan
                raise
            if plots:
                try:
                    from pcdface.reporting.plots import plot_experiment
                except ModuleNotFoundError:
                    ctx.log("grafik belum tersedia (Fase 5)")
                else:
                    figures = [p for p in plot_experiment(name, out_dir) if p.suffix == ".png"]
                    ctx.log(f"{len(figures)} grafik (PNG 300 dpi + PDF)")
            write_snapshot(ctx, time.perf_counter() - start)
            ctx.log(f"selesai dalam {time.perf_counter() - start:.1f} detik, {len(ctx.tables)} tabel")
            outputs[name] = out_dir
    return outputs


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiments", nargs="+", help="e1 e2 e3 e4 e5 atau all")
    parser.add_argument("--synthetic", action="store_true", help="Pakai citra sintetis + FakeDetector (uji jalur)")
    parser.add_argument("--skip-validate", action="store_true", help="Jangan hentikan run bila validate menemukan galat")
    parser.add_argument("--no-plots", action="store_true", help="Tabel saja, tanpa grafik")


def run(args: argparse.Namespace, cfg: Config) -> int:
    try:
        outputs = run_experiments(cfg, args.experiments, synthetic=args.synthetic,
                                  skip_validate=args.skip_validate, plots=not args.no_plots)
    except ExperimentError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1
    print("\nHasil:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    if args.synthetic:
        print("\nCatatan: run sintetis hanya menguji jalur; angkanya BUKAN hasil penelitian.")
    return 0
