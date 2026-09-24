"""Fixture bersama: config proyek dan dataset sintetis sekali per sesi tes."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pcdface.config import Config, load_config
from pcdface.paths import ProjectPaths
from pcdface.synthetic import generate_dataset


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(cfg: Config) -> Config:
    """Config dengan dataset sintetis kecil dan bootstrap ringan, untuk tes cepat."""
    synthetic = dataclasses.replace(cfg.synthetic, subjects=4, frames_per_condition=1, empty_images=3)
    stats = dataclasses.replace(cfg.stats, bootstrap_resamples=200)
    e4 = dataclasses.replace(cfg.experiments.e4, repeats=3, sample_images=2)
    experiments = dataclasses.replace(cfg.experiments, e4=e4)
    return dataclasses.replace(cfg, synthetic=synthetic, stats=stats, experiments=experiments)


@pytest.fixture(scope="session")
def synthetic_paths(small_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Dataset sintetis hanya-baca yang dipakai bersama banyak tes."""
    root = tmp_path_factory.mktemp("synthetic")
    return generate_dataset(small_cfg, root / "data")


@pytest.fixture
def fresh_synthetic(small_cfg: Config, tmp_path: Path) -> ProjectPaths:
    """Dataset sintetis baru untuk tes yang mengubah data (validate, forget)."""
    return generate_dataset(small_cfg, tmp_path / "data")
