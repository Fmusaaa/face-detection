"""Set pose, ekspresi, dan oklusi (E6), blur gerak (E7), dan delegate MediaPipe."""

import argparse
import copy
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from pcdface.config import ConfigError, parse_config
from pcdface.dataset.metadata import MetadataRow, read_metadata, write_metadata
from pcdface.detection.mediapipe_detector import DELEGATE_ENV, delegate_label, resolve_delegate
from pcdface.experiments.e6_pose_expression import angle_limit
from pcdface.experiments.runner import run_experiments
from pcdface.paths import CONFIG_PATH
from pcdface.pose import parse_pose
from pcdface.preprocessing import motion_blur
from pcdface.synthetic import SYNTHETIC_SESSION
from pcdface.tools.capture import build_plan, build_plans
from pcdface.tools.validate import ERROR, WARNING, validate_dataset


def test_parse_pose():
    assert parse_pose("depan").axis == "depan" and parse_pose("depan").angle == 0
    kiri = parse_pose("kiri60")
    assert (kiri.direction, kiri.axis, kiri.angle) == ("kiri", "menoleh", 60)
    assert parse_pose("menunduk30").axis == "mengangguk"
    assert parse_pose("miringkanan30").axis == "miring"
    for bad in ("kiri", "depan30", "kiri0", "kiri120", "samping30", ""):
        with pytest.raises(ValueError):
            parse_pose(bad)


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key, value, message", [
    ("poses", ["kiri30", "kanan30"], "depan"),
    ("poses", ["depan", "samping30"], "tidak dikenal"),
    ("pose_distances_cm", [100, 175], "pose_distances_cm"),
    ("expressions", ["senyum", "marah"], "netral"),
    ("occlusions", ["masker"], "tanpa"),
])
def test_config_rejects_bad_pose_setup(raw, key, value, message):
    broken = copy.deepcopy(raw)
    broken["dataset"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(broken)


def capture_args(**overrides):
    base = dict(set="pose", subject="S01", distance=100, lighting="normal", pose="kiri30", expression=None,
                occlusion=None, formation=None, subjects=None, count=None, camera=None, session=SYNTHETIC_SESSION)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_capture_pose_and_expression_plans(small_cfg, fresh_synthetic):
    ds = small_cfg.dataset
    plan = build_plan(capture_args(distance=200), small_cfg, fresh_synthetic)
    assert plan.spec.relative_path(1) == "pose/S01/pose_S01_200cm_kiri30_01.jpg"
    assert plan.count == ds.pose_frames_per_condition
    assert "KIRI" in plan.instruction()

    plans = build_plans(capture_args(pose="semua"), small_cfg, fresh_synthetic)
    assert [p.spec.pose for p in plans] == list(ds.poses)

    plans = build_plans(capture_args(set="ekspresi", pose=None, distance=None, expression="semua"),
                        small_cfg, fresh_synthetic)
    assert [p.spec.expression for p in plans] == list(ds.expressions)
    assert plans[0].spec.relative_path(2) == f"ekspresi/S01/ekspresi_S01_{ds.reference_distance_cm}cm_netral_02.jpg"

    plans = build_plans(capture_args(set="oklusi", pose=None, distance=None, occlusion="semua"),
                        small_cfg, fresh_synthetic)
    assert [p.spec.occlusion for p in plans] == list(ds.occlusions)
    assert plans[1].spec.relative_path(1) == f"oklusi/S01/oklusi_S01_{ds.reference_distance_cm}cm_masker_01.jpg"
    assert "MASKER" in plans[1].instruction() and plans[1].count == ds.pose_frames_per_condition

    bad = [
        capture_args(distance=None),                                    # pose butuh jarak
        capture_args(distance=150),                                     # bukan jarak pose
        capture_args(pose="kiri45"),                                    # bukan pose di config
        capture_args(lighting="redup"),                                 # pose selalu cahaya normal
        capture_args(set="ekspresi", pose=None, expression="sedih"),    # bukan ekspresi di config
        capture_args(set="ekspresi", pose=None, expression="marah", distance=200),  # ekspresi di jarak acuan
        capture_args(set="oklusi", pose=None, occlusion="helm", distance=None),     # bukan oklusi di config
    ]
    for args in bad:
        with pytest.raises(ValueError):
            build_plan(args, small_cfg, fresh_synthetic)


def issues_of(cfg, paths, level):
    return [i for i in validate_dataset(cfg, paths, check_images=False)[0] if i.level == level]


def test_validate_pose_and_expression_rows(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    assert issues_of(small_cfg, paths, ERROR) == [] and issues_of(small_cfg, paths, WARNING) == []
    rows = read_metadata(paths.metadata)
    pose_row = next(r for r in rows if r.set == "pose" and r.pose == "kiri30")
    expression_row = next(r for r in rows if r.set == "ekspresi" and r.expression == "marah")
    occlusion_row = next(r for r in rows if r.set == "oklusi" and r.occlusion == "masker")
    pose_row.distance_cm = 150
    expression_row.expression = "sedih"
    occlusion_row.occlusion = "helm"
    rows = [r for r in rows if not (r.subject_id == "S02" and (r.pose == "depan" or r.expression == "netral"
                                                               or r.occlusion == "tanpa"))]
    write_metadata(paths.metadata, rows)
    errors = " | ".join(i.message for i in issues_of(small_cfg, paths, ERROR))
    assert "set pose harus di salah satu jarak" in errors
    assert "ekspresi 'sedih'" in errors
    assert "oklusi 'helm'" in errors
    warnings = [(i.where, i.message) for i in issues_of(small_cfg, paths, WARNING)]
    assert any(where == "S02" and "tanpa pose 'depan'" in message for where, message in warnings)
    assert any(where == "S02" and "tanpa 'netral'" in message for where, message in warnings)
    assert any(where == "S02" and "set oklusi tanpa 'tanpa'" in message for where, message in warnings)


def test_metadata_expression_roundtrip(tmp_path):
    path = tmp_path / "metadata.csv"
    write_metadata(path, [MetadataRow(file="ekspresi/S01/ekspresi_S01_100cm_kaget_01.jpg", set="ekspresi",
                                      subject_id="S01", distance_cm=100, expression="kaget", expected_faces=1,
                                      width=1280, height=720)])
    assert read_metadata(path)[0].expression == "kaget"
    write_metadata(path, [MetadataRow(file="oklusi/S01/oklusi_S01_100cm_masker_01.jpg", set="oklusi",
                                      subject_id="S01", distance_cm=100, occlusion="masker", expected_faces=1,
                                      width=1280, height=720)])
    assert read_metadata(path)[0].occlusion == "masker"


def test_angle_limit():
    recalls = {0: (1.0, 0.92), 30: (0.95, 0.91), 60: (0.9, 0.7), 90: (0.1, 0.0)}
    assert angle_limit(recalls, 0.9) == (60, 30)
    assert angle_limit({0: (0.5, 0.2), 30: (1.0, 1.0)}, 0.9) == (None, None)
    assert angle_limit({0: (math.nan, math.nan)}, 0.9) == (None, None)


def test_e6_tables(small_cfg, tmp_path):
    experiments = dataclasses.replace(small_cfg.experiments,
                                      e6=dataclasses.replace(small_cfg.experiments.e6, detectors=("haar", "mp_full")))
    cfg = dataclasses.replace(small_cfg, experiments=experiments)
    out = run_experiments(cfg, ["e6"], synthetic=True, results_root=tmp_path, plots=False)["e6"]
    ds = cfg.dataset
    pose = pd.read_csv(out / "e6_pose.csv")
    assert len(pose) == 2 * len(ds.pose_distances_cm) * len(ds.poses)
    assert set(pose["sumbu"]) == {"depan", "menoleh", "mengangguk", "miring"}
    assert pose.loc[pose["detektor"] == "haar", "rerata_skor"].isna().all()   # Haar tanpa skor
    per_axis = pd.read_csv(out / "e6_per_sumbu.csv")
    yaw = per_axis[(per_axis["detektor"] == "haar") & (per_axis["jarak_cm"] == 100) & (per_axis["sumbu"] == "menoleh")]
    assert list(yaw["sudut"]) == [0, 30, 60, 90]
    assert (yaw.loc[yaw["sudut"] > 0, "citra"] == 2 * cfg.synthetic.subjects).all()  # kiri + kanan digabung
    versus = pd.read_csv(out / "e6_pose_vs_depan.csv")
    assert "depan" not in set(versus["pose"])
    assert set(versus["kesimpulan"]) <= {"lebih rendah dari acuan", "lebih tinggi dari acuan",
                                          "tidak dapat disimpulkan", "tidak dapat dihitung"}
    expression = pd.read_csv(out / "e6_ekspresi.csv")
    assert list(expression.loc[expression["detektor"] == "haar", "ekspresi"]) == list(ds.expressions)
    occlusion = pd.read_csv(out / "e6_oklusi.csv")
    assert list(occlusion.loc[occlusion["detektor"] == "haar", "oklusi"]) == list(ds.occlusions)
    assert "tanpa" not in set(pd.read_csv(out / "e6_oklusi_vs_tanpa.csv")["oklusi"])
    summary = pd.read_csv(out / "e6_ringkasan.csv")
    assert {"recall_pose", "recall_ekspresi", "recall_oklusi"} <= set(summary.columns)
    compare = pd.read_csv(out / "e6_perbandingan.csv")
    assert len(compare) == len(ds.pose_distances_cm) * len(ds.poses) + len(ds.expressions) + len(ds.occlusions)


# ---------------------------------------------------------------------------
# E7 — blur gerak
# ---------------------------------------------------------------------------
def test_motion_blur_spreads_along_direction_and_keeps_brightness():
    image = np.zeros((41, 41, 3), np.uint8)
    image[:, 20] = 200                                           # garis vertikal
    assert motion_blur(image, 0) is image and motion_blur(image, 1) is image
    horizontal = motion_blur(image, 9)
    assert np.count_nonzero(horizontal[20, :, 0]) == 9           # menyebar 9 kolom
    assert abs(int(horizontal.sum()) - int(image.sum())) / image.sum() < 0.02
    vertical = motion_blur(image, 9, angle_deg=90)
    assert np.array_equal(vertical[5:36, 20], image[5:36, 20])   # searah garis: tidak berubah


def test_e7_tables(small_cfg, tmp_path):
    e7 = dataclasses.replace(small_cfg.experiments.e7, detectors=("haar", "mp_short"), blur_px=(0, 11))
    cfg = dataclasses.replace(small_cfg, experiments=dataclasses.replace(small_cfg.experiments, e7=e7))
    out = run_experiments(cfg, ["e7"], synthetic=True, results_root=tmp_path, plots=False)["e7"]
    summary = pd.read_csv(out / "e7_ringkasan.csv")
    assert set(zip(summary["detektor"], summary["blur_px"])) == {("haar", 0), ("haar", 11), ("mp_short", 0),
                                                                  ("mp_short", 11)}
    per_distance = pd.read_csv(out / "e7_recall_per_jarak.csv")
    assert set(per_distance["jarak_cm"]) == set(cfg.dataset.distances_cm)
    versus = pd.read_csv(out / "e7_vs_asli.csv")
    assert list(versus["blur_px"]) == [11, 11]
    assert len(pd.read_csv(out / "e7_perbandingan.csv")) == 2


# ---------------------------------------------------------------------------
# Delegate MediaPipe — laptop Windows/Linux memakai CPU
# ---------------------------------------------------------------------------
def test_resolve_delegate(monkeypatch):
    monkeypatch.delenv(DELEGATE_ENV, raising=False)
    assert resolve_delegate("auto", platform="darwin") == "gpu"
    assert resolve_delegate("auto", platform="win32") == "cpu"
    assert resolve_delegate("auto", platform="linux") == "cpu"
    assert resolve_delegate("gpu", platform="linux") == "gpu"
    monkeypatch.setenv(DELEGATE_ENV, "cpu")
    assert resolve_delegate("gpu", platform="darwin") == "cpu"          # variabel lingkungan menang
    monkeypatch.setenv(DELEGATE_ENV, "tpu")
    with pytest.raises(ValueError):
        resolve_delegate("auto")
    assert delegate_label("gpu", platform="darwin") == "GPU (Metal)"
    assert delegate_label("cpu", platform="win32") == "CPU"


def test_config_rejects_bad_delegate(raw):
    broken = copy.deepcopy(raw)
    broken["detectors"]["mp_short"]["delegate"] = "tpu"
    with pytest.raises(ConfigError, match="delegate"):
        parse_config(broken)

