"""Pengambilan data dalam beberapa sesi: metadata, capture, validate, E1/E3 per sesi."""

import argparse
import dataclasses
from datetime import datetime

import pandas as pd
import pytest

from pcdface.dataset.metadata import MetadataRow, read_metadata, write_metadata
from pcdface.experiments.runner import run_experiments
from pcdface.tools.capture import build_plan
from pcdface.tools.validate import WARNING, validate_dataset


def warnings_of(cfg, paths):
    return [i for i in validate_dataset(cfg, paths, check_images=False)[0] if i.level == WARNING]


def test_legacy_metadata_without_session_column_still_reads(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("file,set,subject_id,subjects,formation,positions_cm,distance_cm,lighting,pose,"
                    "expected_faces,luma_mean,width,height,captured_at\n"
                    "kosong/kosong_01.jpg,kosong,,,,,,normal,,0,100.0,1280,720,\n", encoding="utf-8")
    rows = read_metadata(path)
    assert rows[0].session == ""
    write_metadata(path, rows)                 # ditulis ulang dengan kolom session
    assert "session" in path.read_text(encoding="utf-8").splitlines()[0]


def test_session_roundtrip(tmp_path):
    path = tmp_path / "metadata.csv"
    write_metadata(path, [MetadataRow(file="kosong/kosong_01.jpg", set="kosong", expected_faces=0,
                                      width=1280, height=720, session="2026-10-02-sore")])
    assert read_metadata(path)[0].session == "2026-10-02-sore"


def capture_args(**overrides):
    base = dict(set="jarak", subject="S01", distance=150, lighting="normal", pose=None, expression=None, occlusion=None,
                formation=None, subjects=None, count=None, camera=None, session=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_capture_session_defaults_to_today_and_is_validated(small_cfg, fresh_synthetic):
    plan = build_plan(capture_args(), small_cfg, fresh_synthetic)
    assert plan.session == datetime.now().strftime("%Y-%m-%d")
    assert build_plan(capture_args(session="sesi2_sore"), small_cfg, fresh_synthetic).session == "sesi2_sore"
    with pytest.raises(ValueError, match="--session"):
        build_plan(capture_args(session="sesi dua"), small_cfg, fresh_synthetic)


def test_validate_warns_when_sessions_confound_factors(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    assert not [w for w in warnings_of(small_cfg, paths) if "sesi" in w.message]  # satu sesi: aman
    rows = read_metadata(paths.metadata)
    for row in rows:
        if row.subject_id == "S01" and row.set == "cahaya":
            row.session = "sesi-B"                                   # cahaya S01 di sesi lain
        if row.subject_id == "S02" and row.set == "jarak" and row.distance_cm >= 200:
            row.session = "sesi-B"                                   # jarak S02 terpecah
            row.luma_mean = (row.luma_mean or 0) + 90                # sesi B jauh lebih terang
        if row.set == "kosong":
            row.session = ""
    write_metadata(paths.metadata, rows)
    messages = {(w.where, w.message.split(" ")[0]) for w in warnings_of(small_cfg, paths)}
    text = " | ".join(f"{w.where}: {w.message}" for w in warnings_of(small_cfg, paths))
    assert any(where == "S01" for where, _ in messages) and "tercampur efek sesi" in text
    assert "S02: set jarak terpecah" in text
    assert "rerata Y set jarak berbeda" in text
    assert "tanpa kode sesi" in text


def test_e1_and_e3_report_per_session(small_cfg, fresh_synthetic, tmp_path):
    rows = read_metadata(fresh_synthetic.metadata)
    for row in rows:
        if row.subject_id in ("S03", "S04"):
            row.session = "sesi-B"
    write_metadata(fresh_synthetic.metadata, rows)
    experiments = dataclasses.replace(
        small_cfg.experiments,
        e1=dataclasses.replace(small_cfg.experiments.e1, detectors=("haar",)),
        e3=dataclasses.replace(small_cfg.experiments.e3, detectors=("haar",)),
    )
    cfg = dataclasses.replace(small_cfg, experiments=experiments,
                              paths=dataclasses.replace(fresh_synthetic, results=tmp_path))
    out = run_experiments(cfg, ["e1", "e3"], plots=False)
    per_session = pd.read_csv(out["e1"] / "e1_loglog_per_sesi.csv")
    assert set(per_session["sesi"]) == {"2026-09-30", "sesi-B"}
    assert per_session["kemiringan"].between(-1.1, -0.9).all()
    luma = pd.read_csv(out["e3"] / "e3_luminansi.csv")
    assert {"semua", "2026-09-30", "sesi-B"} <= set(luma["sesi"])
