"""Pengenalan LBPH: model, izin, enroll, forget, validate, crop-faces, dan E8 — pada data sintetis."""

import argparse
import csv
import dataclasses

import numpy as np
import pandas as pd
import pytest

from pcdface.dataset.metadata import MetadataRow, SubjectRow, read_subjects, write_subjects
from pcdface.experiments.runner import ExperimentError, run_experiments
from pcdface.recognition.lbph import (
    UNKNOWN,
    LBPHRecognizer,
    consenting_subjects,
    face_patch,
    is_e8_gallery,
    is_reference_condition,
    model_exists,
    model_subjects,
)
from pcdface.tools import crop_faces, enroll, forget
from pcdface.tools.validate import ERROR, WARNING, validate_dataset


def textures(cfg, seed=0):
    rng = np.random.default_rng(seed)
    w, h = cfg.recognition.face_size
    return {label: rng.integers(0, 256, (h, w)).astype(np.uint8) for label in ("S01", "S02", "S03")}


def jitter(image, rng):
    return np.clip(image.astype(int) + rng.integers(-10, 11, image.shape), 0, 255).astype(np.uint8)


def test_lbph_train_predict_save_load(cfg, tmp_path):
    rng = np.random.default_rng(1)
    bases = textures(cfg)
    recognizer = LBPHRecognizer(cfg.recognition)
    recognizer.train([jitter(b, rng) for b in bases.values() for _ in range(2)], [k for k in bases for _ in range(2)])
    probes = {label: jitter(base, rng) for label, base in bases.items()}
    before = {label: recognizer.predict(p) for label, p in probes.items()}
    assert all(p.nearest == label for label, p in before.items())
    recognizer.save(tmp_path / "rec", sumber="tes")
    assert model_exists(tmp_path / "rec") and model_subjects(tmp_path / "rec") == ["S01", "S02", "S03"]
    loaded = LBPHRecognizer.load(tmp_path / "rec", cfg.recognition)
    after = {label: loaded.predict(p) for label, p in probes.items()}
    assert {k: (v.nearest, round(v.distance, 6)) for k, v in after.items()} == \
           {k: (v.nearest, round(v.distance, 6)) for k, v in before.items()}
    # ambang diambil dari config saat ini, bukan dari saat dilatih
    strict = LBPHRecognizer.load(tmp_path / "rec", dataclasses.replace(cfg.recognition, max_distance=1e-9))
    assert strict.predict(probes["S01"]).label == UNKNOWN and not strict.predict(probes["S01"]).accepted


def test_lbph_rejects_real_names_and_bad_patches(cfg):
    recognizer = LBPHRecognizer(cfg.recognition)
    patch = textures(cfg)["S01"]
    with pytest.raises(ValueError, match="nama asli dilarang"):
        recognizer.train([patch], ["Budi"])
    with pytest.raises(ValueError, match="abu-abu"):
        recognizer.train([patch[:10]], ["S01"])
    with pytest.raises(RuntimeError, match="belum dilatih"):
        recognizer.predict(patch)


def test_face_patch_crops_resizes_and_checks_bounds(cfg):
    image = np.zeros((100, 200, 3), np.uint8)
    image[20:60, 50:90] = 200
    patch = face_patch(image, (50, 20, 40, 40), cfg.recognition.face_size)
    assert patch.shape == (cfg.recognition.face_size[1], cfg.recognition.face_size[0]) and patch.min() > 150
    with pytest.raises(ValueError, match="di luar citra"):
        face_patch(image, (500, 500, 10, 10), (64, 64))


def test_gallery_rules():
    def row(**kw):
        base = dict(file="x.jpg", set="jarak", expected_faces=1, width=1280, height=720, subject_id="S01",
                    distance_cm=100, lighting="normal")
        return MetadataRow(**{**base, **kw})

    assert is_e8_gallery(row(), 100) and is_reference_condition(row(), 100)
    assert not is_e8_gallery(row(distance_cm=150), 100)
    for meta in (row(set="pose", pose="depan"), row(set="ekspresi", expression="netral"),
                 row(set="oklusi", occlusion="tanpa")):
        assert is_reference_condition(meta, 100) and not is_e8_gallery(meta, 100)   # acuan tersisa sebagai foto uji
    for meta in (row(set="pose", pose="kiri30"), row(set="cahaya", lighting="redup"), row(set="pose", pose="depan",
                 distance_cm=200), row(set="oklusi", occlusion="masker"), row(set="multi", subject_id="")):
        assert not is_reference_condition(meta, 100)


def test_subjects_without_recognition_column_default_to_no(tmp_path):
    path = tmp_path / "subjects.csv"
    path.write_text("subject_id,consent_research,consent_publication,consent_date\nS01,ya,tidak,\n", encoding="utf-8")
    subjects = read_subjects(path)
    assert subjects["S01"].consent_recognition is False and consenting_subjects(subjects) == set()
    write_subjects(path, [SubjectRow("S01", True, False, "", consent_recognition=True),
                          SubjectRow("S02", False, False, "", consent_recognition=True)])
    assert consenting_subjects(read_subjects(path)) == {"S01"}   # izin penelitian tetap wajib


def revoke(paths, subject_id):
    subjects = read_subjects(paths.subjects)
    subjects[subject_id] = dataclasses.replace(subjects[subject_id], consent_recognition=False)
    write_subjects(paths.subjects, subjects.values())


def test_enroll_uses_only_consenting_subjects(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    revoke(paths, "S02")
    recognizer, counts, skipped = enroll.enroll(small_cfg, paths)
    assert recognizer.labels == ["S01", "S03", "S04"] and "S02" not in counts
    assert skipped["tanpa izin pengenalan"] > 0
    assert model_subjects(paths.recognition) == ["S01", "S03", "S04"]
    # galeri = foto kondisi acuan: jarak 100 cm, depan 100 cm, netral, tanpa (1 frame per kondisi di data sintetis)
    assert set(counts.values()) == {4}


def test_enroll_folder_requires_subject_codes(small_cfg, fresh_synthetic, tmp_path):
    (tmp_path / "faces" / "Budi").mkdir(parents=True)
    with pytest.raises(ValueError, match="kode peserta"):
        enroll.enroll(small_cfg, fresh_synthetic, folder=tmp_path / "faces", synthetic=True)


def test_enroll_without_any_consent_fails(small_cfg, fresh_synthetic):
    for sid in ("S01", "S02", "S03", "S04"):
        revoke(fresh_synthetic, sid)
    with pytest.raises(ValueError, match="consent_recognition"):
        enroll.enroll(small_cfg, fresh_synthetic)


def test_forget_deletes_templates_and_validate_guards_consent(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    enroll.enroll(small_cfg, paths)
    revoke(paths, "S03")                                       # izin dicabut setelah model dilatih
    issues, _ = validate_dataset(small_cfg, paths, check_images=False)
    assert any(i.level == ERROR and "model pengenalan memuat S03" in i.message for i in issues)
    assert any(i.level == WARNING and "belum mengizinkan pengenalan" in i.message for i in issues)

    plan = forget.plan_forget(paths, "S03")
    assert plan.recognition_files and "model pengenalan    : ya" in forget.describe(plan)
    forget.execute_forget(paths, plan)
    assert not model_exists(paths.recognition)
    issues, _ = validate_dataset(small_cfg, paths, check_images=False)
    assert not any("model pengenalan" in i.message for i in issues)


def test_crop_faces_writes_crops_manifest_and_identities(small_cfg, fresh_synthetic, tmp_path):
    paths = fresh_synthetic
    cfg = dataclasses.replace(small_cfg, paths=paths)
    source = paths.raw / "multi"
    faces = crop_faces.crop_faces(cfg, source, tmp_path / "plain", "yolo_n", margin=0.1, synthetic=True)
    assert faces and all(f.path.exists() and f.identity == "" for f in faces)
    for photo in {f.source for f in faces}:                   # urutan kiri → kanan per foto
        xs = [f.box[0] for f in faces if f.source == photo]
        assert xs == sorted(xs) and [f.index for f in faces if f.source == photo] == list(range(1, len(xs) + 1))
    with (tmp_path / "plain" / crop_faces.MANIFEST).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(faces) and set(rows[0]) == set(crop_faces.COLUMNS)

    with pytest.raises(FileNotFoundError, match="enroll"):
        crop_faces.crop_faces(cfg, source, tmp_path / "rec", "yolo_n", recognize=True, synthetic=True)
    enroll.enroll(cfg, paths)
    faces = crop_faces.crop_faces(cfg, source, tmp_path / "rec", "yolo_n", recognize=True, synthetic=True)
    identities = {f.identity for f in faces}
    assert identities and identities <= {"S01", "S02", "S03", "S04", UNKNOWN}
    assert all(f.path.parent.name == f.identity for f in faces)


def test_crop_faces_cli_validates_arguments(cfg, tmp_path):
    def args(**kw):
        return argparse.Namespace(**{"input": str(tmp_path), "output": str(tmp_path / "o"), "detector": "haar",
                                     "margin": 0.0, "recognize": False, **kw})

    assert crop_faces.run(args(margin=2.0), cfg) == 2
    assert crop_faces.run(args(input=str(tmp_path / "tidak_ada")), cfg) == 2
    assert crop_faces.run(args(), cfg) == 0                     # folder kosong → 0 wajah, manifest tetap ditulis
    assert (tmp_path / "o" / crop_faces.MANIFEST).exists()


@pytest.fixture(scope="module")
def e8_out(small_cfg, tmp_path_factory):
    return run_experiments(small_cfg, ["e8"], synthetic=True, results_root=tmp_path_factory.mktemp("e8"),
                           plots=True)["e8"]


def test_e8_tables_and_figures(e8_out, small_cfg):
    summary = pd.read_csv(e8_out / "e8_ringkasan.csv").set_index("sumber")
    assert set(summary.index) == {"manual", small_cfg.experiments.e8.detector}
    for column in ("akurasi", "dir", "frr", "far"):
        valid = summary.dropna(subset=[f"{column}_low"])
        assert ((valid[f"{column}_low"] <= valid[column] + 1e-9) & (valid[column] <= valid[f"{column}_high"] + 1e-9)).all()
    # label tersambung benar: akurasi jauh di atas tebakan acak (1/4 peserta)
    assert summary.loc["manual", "akurasi"] > 0.5
    assert summary.loc["manual", "tak_terdeteksi"] == 0 and summary.loc["manual", "galeri"] == 4
    conditions = pd.read_csv(e8_out / "e8_per_kondisi.csv")
    assert {"acuan", "jarak", "cahaya", "pose", "ekspresi", "oklusi"} <= set(conditions["faktor"])
    distances = conditions[(conditions["sumber"] == "manual") & (conditions["faktor"] == "jarak")]["jarak_cm"].tolist()
    assert distances == sorted(distances) and small_cfg.dataset.reference_distance_cm not in distances  # 100 cm = galeri
    curve = pd.read_csv(e8_out / "e8_ambang.csv")
    for _, part in curve.groupby("sumber"):
        assert part["dir"].is_monotonic_increasing and part["far"].is_monotonic_increasing
        assert small_cfg.recognition.max_distance in set(part["ambang"])
    for stem in ("grafik13_pengenalan_jarak", "grafik14_pengenalan_kondisi", "grafik15_pengenalan_ambang"):
        assert (e8_out / f"{stem}.png").stat().st_size > 5000


def test_e8_requires_recognition_consent(small_cfg, tmp_path, monkeypatch):
    import pcdface.experiments.e8_recognition as e8

    monkeypatch.setattr(e8, "consenting_subjects", lambda subjects: {"S01"})
    with pytest.raises(ExperimentError, match="≥ 2 peserta"):
        run_experiments(small_cfg, ["e8"], synthetic=True, results_root=tmp_path, plots=False)
