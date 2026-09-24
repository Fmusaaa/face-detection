"""Jalur penuh eksperimen dengan dataset sintetis + FakeDetector (PRD §14 Fase 4)."""

import json

import pandas as pd
import pytest
import yaml

from pcdface.experiments.runner import ExperimentError, run_experiments

EXPECTED_TABLES = {
    "e1": ["e1_ringkasan", "e1_recall_per_jarak", "e1_jangkauan", "e1_ukuran_px", "e1_ukuran_proporsi",
           "e1_ukuran_minimum", "e1_loglog", "e1_klaim_dokumentasi", "e1_resolusi", "e1_perbandingan", "e1_kurva_pr"],
    "e2": ["e2_ringkasan", "e2_per_formasi", "e2_recall_per_posisi", "e2_kurva_pr", "e2_perbandingan"],
    "e3": ["e3_f1", "e3_ringkasan", "e3_luminansi", "e3_efek_clahe", "e3_perbandingan"],
    "e4": ["e4_kecepatan"],
    "e5": ["e5_sensitivitas_haar"],
}


@pytest.fixture(scope="module")
def outputs(small_cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    return run_experiments(small_cfg, ["e1", "e2", "e3", "e4", "e5"], synthetic=True, results_root=root, plots=False)


@pytest.mark.parametrize("name", list(EXPECTED_TABLES))
def test_all_tables_written(outputs, name):
    out_dir = outputs[name]
    for stem in EXPECTED_TABLES[name]:
        df = pd.read_csv(out_dir / f"{stem}.csv")
        assert len(df) > 0, stem
        assert (out_dir / f"{stem}.md").read_text(encoding="utf-8").count("|") > 4
    snapshot = yaml.safe_load((out_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["sintetis"] is True
    assert snapshot["versi"]["mediapipe"] and snapshot["versi"]["opencv"].startswith("4.")
    assert set(EXPECTED_TABLES[name]) <= set(snapshot["tabel"])
    assert snapshot["config"]["seed"] == 42


def test_e1_contents_are_coherent(outputs, small_cfg):
    out = outputs["e1"]
    summary = pd.read_csv(out / "e1_ringkasan.csv")
    assert set(summary["iou"]) == set(small_cfg.evaluation.iou_thresholds)
    primary = summary[summary["iou"] == small_cfg.evaluation.iou_primary]
    assert primary["ap"].notna().all()                      # semua detektor E1 punya skor
    assert summary[summary["iou"] != small_cfg.evaluation.iou_primary]["ap"].isna().all()
    for column in ("precision", "recall", "f1"):
        valid = summary.dropna(subset=[f"{column}_low"])
        assert ((valid[f"{column}_low"] <= valid[column] + 1e-9) & (valid[column] <= valid[f"{column}_high"] + 1e-9)).all()
    loglog = pd.read_csv(out / "e1_loglog.csv")
    assert loglog["kemiringan"].between(-1.1, -0.9).all()
    per_distance = pd.read_csv(out / "e1_recall_per_jarak.csv")
    assert set(per_distance["jarak_cm"]) == set(small_cfg.dataset.distances_cm)
    # H2 pada profil tiruan: Haar dirugikan resolusi kecil, MediaPipe tidak
    h2 = pd.read_csv(out / "e1_resolusi.csv").set_index(["detektor", "cakupan"])
    assert h2.loc[("haar", "≥200 cm"), "selisih_recall"] > 0.3
    assert abs(h2.loc[("mp_full", "semua jarak"), "selisih_recall"]) < 0.15


def test_e2_counts_and_positions(outputs, small_cfg):
    formations = pd.read_csv(outputs["e2"] / "e2_per_formasi.csv")
    usable = {f for f, p in small_cfg.dataset.formations.items() if len(p) <= small_cfg.synthetic.subjects}
    assert set(formations["formasi"]) == usable
    positions = pd.read_csv(outputs["e2"] / "e2_recall_per_posisi.csv")
    expected = {d for f in usable for d in small_cfg.dataset.formations[f]}
    assert set(positions["jarak_cm"]) == expected


def test_e3_variants_cover_equalize(outputs, small_cfg):
    f1 = pd.read_csv(outputs["e3"] / "e3_f1.csv")
    assert {"haar_eq", "haar_noeq", "mp_short", "mp_full"} == set(f1["varian"])
    assert set(f1["cahaya"]) == set(small_cfg.dataset.lightings)
    assert set(f1["enhancement"]) == {"none", "clahe"}


def test_e4_counts_calls(outputs, small_cfg):
    speed = pd.read_csv(outputs["e4"] / "e4_kecepatan.csv")
    assert (speed["panggilan"] == small_cfg.experiments.e4.repeats).all()
    assert (speed["p95_ms"] >= speed["median_ms"]).all()
    assert set(speed["perangkat"]) == {"tiruan (FakeDetector)"}


def test_detections_logged(outputs):
    lines = (outputs["e2"] / "deteksi.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert {"key", "group", "gt", "boxes", "scores", "elapsed_ms", "labels"} <= set(record)


def test_unknown_experiment_rejected(small_cfg, tmp_path):
    with pytest.raises(ExperimentError, match="tidak dikenal"):
        run_experiments(small_cfg, ["e9"], synthetic=True, results_root=tmp_path, plots=False)
