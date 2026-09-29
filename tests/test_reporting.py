"""Tabel Markdown, dua belas grafik, ringkasan, contoh gambar dengan pemburaman, demo."""

import argparse

import numpy as np
import pandas as pd
import pytest

from pcdface.dataset.loader import load_samples
from pcdface.dataset.metadata import read_subjects
from pcdface.experiments.runner import run_experiments
from pcdface.reporting.report import blur_faces, build_summary, private_boxes, write_examples
from pcdface.reporting.tables import display_frame, to_markdown
from pcdface.tools import demo_realtime

FIGURES = {
    "e1": ["grafik1_recall_jarak", "grafik2_lebar_loglog", "grafik3_recall_ukuran", "grafik4_resolusi",
           "grafik5_kurva_pr_e1"],
    "e2": ["grafik5_kurva_pr_e2", "grafik6_akurasi_hitung"],
    "e3": ["grafik7_f1_cahaya"],
    "e4": ["grafik8_kecepatan"],
    "e6": ["grafik9_recall_pose", "grafik10_recall_ekspresi", "grafik11_recall_oklusi"],
    "e7": ["grafik12_recall_blur"],
}


def test_markdown_merges_intervals_and_uses_decimal_comma():
    df = pd.DataFrame({"detektor": ["haar"], "citra": [12], "recall": [0.8], "recall_low": [0.4902],
                       "recall_high": [0.9433], "ap": [float("nan")]})
    shown = display_frame(df)
    assert list(shown.columns) == ["detektor", "citra", "recall", "ap"]
    assert shown.loc[0, "recall"] == "0,800 [0,490; 0,943]"
    assert shown.loc[0, "ap"] == "–" and shown.loc[0, "citra"] == "12"
    assert to_markdown(df).splitlines()[1] == "|---|---|---|---|"


@pytest.fixture(scope="module")
def results(small_cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("results_plots")
    run_experiments(small_cfg, ["e1", "e2", "e3", "e4", "e6", "e7"], synthetic=True, results_root=root, plots=True)

    return root


@pytest.mark.parametrize("name", list(FIGURES))
def test_figures_written(results, name):

    for stem in FIGURES[name]:
        for suffix in (".png", ".pdf"):
            path = results / name / f"{stem}{suffix}"
            assert path.exists() and path.stat().st_size > 5000, path


def test_summary_links_figures_and_tables(results):
    summary = build_summary(results).read_text(encoding="utf-8")
    assert "BUKAN hasil penelitian" in summary
    for name, stems in FIGURES.items():
        for stem in stems:
            assert f"({name}/{stem}.png)" in summary
    assert "E1 — ringkasan per detektor dan resolusi" in summary
    assert build_summary(results / "tidak_ada") is None


def test_private_faces_are_blurred(synthetic_paths):
    publishable = {sid: s.consent_publication for sid, s in read_subjects(synthetic_paths.subjects).items()}
    assert set(publishable.values()) == {True, False}
    multi = [s for s in load_samples(synthetic_paths, sets=("multi",)).samples]
    for sample in multi:
        hidden = private_boxes(sample, publishable)
        assert len(hidden) == sum(not publishable[p] for p in sample.meta.subjects)
    sample = multi[0]
    hidden = private_boxes(sample, {})                      # izin tidak diketahui → semua diburamkan
    assert len(hidden) == len(sample.gt)
    image = sample.load()
    blurred = blur_faces(image, hidden[:1])
    x, y, w, h = hidden[0]
    assert not np.array_equal(blurred[y:y + h, x:x + w], image[y:y + h, x:x + w])
    assert np.array_equal(blurred[:5, :5], image[:5, :5])   # di luar wajah tidak berubah


def test_examples_written(small_cfg, synthetic_paths, tmp_path):
    import dataclasses

    paths = dataclasses.replace(synthetic_paths, results=tmp_path)
    written = write_examples(small_cfg, paths, 2, ["haar", "mp_short", "mp_full"], synthetic=True)
    assert len(written) == 2 and all(p.exists() for p in written)


def test_demo_selftest_without_models(cfg):
    args = argparse.Namespace(detector="haar", detectors="haar,ycbcr", enhance="none", camera=None,
                              no_mirror=False, selftest=True)
    assert demo_realtime.run(args, cfg) == 0


@pytest.mark.models
def test_demo_selftest_with_mediapipe(cfg):
    args = argparse.Namespace(detector="mp_short", detectors=None, enhance="none", camera=None,
                              no_mirror=False, selftest=True)
    assert demo_realtime.run(args, cfg) == 0
