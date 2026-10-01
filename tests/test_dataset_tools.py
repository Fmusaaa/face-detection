"""validate, forget, crop, logika capture dan annotate — pada data sintetis."""

import argparse
import dataclasses

import numpy as np
import pytest

from pcdface.dataset.annotations import load_annotations, save_annotations
from pcdface.dataset.metadata import (
    SubjectRow,
    read_metadata,
    read_subjects,
    write_metadata,
    write_subjects,
)
from pcdface.synthetic import SYNTHETIC_SESSION
from pcdface.tools import crop, forget
from pcdface.tools.annotate import AnnotationState, Viewport
from pcdface.tools.capture import build_plan, save_frame
from pcdface.tools.validate import ERROR, WARNING, validate_dataset


def errors_of(cfg, paths):
    issues, _ = validate_dataset(cfg, paths)
    return [i for i in issues if i.level == ERROR], [i for i in issues if i.level == WARNING]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_validate_clean_synthetic_dataset(small_cfg, synthetic_paths):
    errors, warnings = errors_of(small_cfg, synthetic_paths)
    assert errors == [] and warnings == []


def test_validate_catches_box_count_and_orphans(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    annotations = load_annotations(paths.annotations)
    multi = next(k for k in annotations if k.startswith("multi/"))
    annotations[multi] = annotations[multi][:-1]              # satu wajah terlewat
    annotations["jarak/S99/tidak_ada.jpg"] = [(1, 1, 10, 10)]  # anotasi yatim
    unannotated = next(k for k in annotations if k.startswith("cahaya/"))
    del annotations[unannotated]
    save_annotations(paths.annotations, annotations)
    (paths.raw / "kosong" / "yatim.jpg").write_bytes((paths.raw / "kosong" / "kosong_01.jpg").read_bytes())

    errors, warnings = errors_of(small_cfg, paths)
    messages = [str(e) for e in errors]
    assert any(multi in m and "expected_faces" in m for m in messages)
    assert any("anotasi yatim" in m for m in messages)
    assert any("kosong/yatim.jpg" in m and "foto yatim" in m for m in messages)
    assert any(w.where == unannotated and "belum dianotasi" in w.message for w in warnings)


def test_validate_catches_metadata_and_consent_errors(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    rows = read_metadata(paths.metadata)
    multi = next(r for r in rows if r.set == "multi")
    multi.positions_cm = tuple(reversed(multi.positions_cm)) + (999,)
    jarak = next(r for r in rows if r.set == "jarak")
    jarak.distance_cm = 175
    write_metadata(paths.metadata, rows)
    subjects = read_subjects(paths.subjects)
    subjects["S02"] = SubjectRow("S02", consent_research=False, consent_publication=False)
    write_subjects(paths.subjects, subjects.values())

    messages = [str(e) for e in errors_of(small_cfg, paths)[0]]
    assert any("positions_cm" in m for m in messages)
    assert any("175 cm" in m for m in messages)          # jarak tidak dikenal
    assert any("nama berkas tidak sesuai" in m for m in messages)  # nama berkas memuat 50cm, metadata 175
    assert any("S02 tidak punya persetujuan" in m for m in messages)


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------
def test_forget_removes_everything_including_multi(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    crop.export_crops(small_cfg, paths)
    before = read_metadata(paths.metadata)
    with_s01 = [r.file for r in before if "S01" in r.people]
    multi_with_s01 = [f for f in with_s01 if f.startswith("multi/")]
    assert multi_with_s01, "dataset sintetis harus punya foto multi yang memuat S01"

    plan = forget.plan_forget(paths, "S01")
    assert sorted(plan.photos) == sorted(with_s01)
    assert plan.crops and plan.in_subjects_csv
    forget.execute_forget(paths, plan)

    after = read_metadata(paths.metadata)
    assert all("S01" not in r.people for r in after)
    assert len(after) == len(before) - len(with_s01)
    assert not any((paths.raw / f).exists() for f in with_s01)
    assert not (paths.raw / "jarak" / "S01").exists()
    assert "S01" not in read_subjects(paths.subjects)
    assert not set(with_s01) & set(load_annotations(paths.annotations))
    assert not any(c.exists() for c in plan.crops)
    # peserta lain utuh dan dataset tetap konsisten
    assert any("S02" in r.people for r in after)
    assert errors_of(small_cfg, paths)[0] == []


def test_forget_rejects_bad_id_and_handles_unknown(fresh_synthetic):
    with pytest.raises(ValueError):
        forget.plan_forget(fresh_synthetic, "Budi")
    assert forget.plan_forget(fresh_synthetic, "S77").empty


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------
def test_crop_exports_one_file_per_box(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    total = sum(len(v) for v in load_annotations(paths.annotations).values())
    written, widths = crop.export_crops(small_cfg, paths)
    assert written == total == len(widths)
    assert (paths.crops / "face_widths.csv").exists()
    summary = crop.width_summary(widths)
    means = [mean for _, _, mean, _, _ in summary]
    assert means == sorted(means, reverse=True), "wajah mengecil seiring jarak"

    subjects = read_subjects(paths.subjects)
    written_pub, widths_pub = crop.export_crops(small_cfg, paths, publishable_only=True, clean=True)
    assert 0 < written_pub < written
    for row in widths_pub:
        if row["subject_id"]:
            assert subjects[row["subject_id"]].consent_publication


# ---------------------------------------------------------------------------
# capture (tanpa kamera)
# ---------------------------------------------------------------------------
def capture_args(**overrides):
    base = dict(set="jarak", subject="S01", distance=150, lighting="normal", pose=None, expression=None, occlusion=None,
                formation=None, subjects=None, count=None, camera=None, session=SYNTHETIC_SESSION)

    base.update(overrides)
    return argparse.Namespace(**base)


def test_capture_plan_validation(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    plan = build_plan(capture_args(), small_cfg, paths)
    assert plan.spec.relative_path(1) == "jarak/S01/jarak_S01_150cm_normal_01.jpg"
    assert plan.count == small_cfg.dataset.frames_per_condition

    plan = build_plan(capture_args(set="multi", formation="F4", subjects="S02,S01"), small_cfg, paths)
    assert plan.positions_cm == (100, 200) and plan.expected_faces == 2

    bad = [
        capture_args(distance=175),                                   # jarak tidak ada di config
        capture_args(subject="S99"),                                  # belum terdaftar
        capture_args(subject="Budi"),                                 # bukan kode
        capture_args(lighting="redup"),                               # jarak harus normal
        capture_args(set="cahaya", lighting="normal", distance=None),  # normal milik set jarak
        capture_args(set="multi", formation="F5", subjects="S01,S02"),  # kurang satu orang
        capture_args(set="multi", formation="F4", subjects="S01,S01"),  # kode ganda
        capture_args(set="pose", pose="miring", distance=None),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            build_plan(args, small_cfg, paths)

    subjects = read_subjects(paths.subjects)
    subjects["S03"] = dataclasses.replace(subjects["S03"], consent_research=False)
    write_subjects(paths.subjects, subjects.values())
    with pytest.raises(ValueError, match="tidak memberi persetujuan"):
        build_plan(capture_args(subject="S03"), small_cfg, paths)


def test_capture_save_frame_never_overwrites(small_cfg, fresh_synthetic):
    paths = fresh_synthetic
    plan = build_plan(capture_args(distance=250), small_cfg, paths)
    frame = np.full((small_cfg.capture.height, small_cfg.capture.width, 3), 120, dtype=np.uint8)
    existing = {r.file for r in read_metadata(paths.metadata)}
    first = save_frame(plan, frame, paths)
    second = save_frame(plan, frame, paths)
    assert first.file not in existing and second.file != first.file
    assert (paths.raw / second.file).exists()
    assert second.luma_mean == pytest.approx(120, abs=1)
    errors, warnings = errors_of(small_cfg, paths)
    assert errors == []
    assert {w.where for w in warnings} == {first.file, second.file}  # belum dianotasi


# ---------------------------------------------------------------------------
# annotate (logika tanpa jendela)
# ---------------------------------------------------------------------------
def test_viewport_roundtrip_with_zoom_and_display_scale():
    view = Viewport(1280, 720, display_scale=0.8)
    assert view.to_image(*view.to_window(400, 300)) == (400, 300)
    view.cycle_zoom((1200, 700))            # zoom 2× di dekat pojok kanan bawah
    assert view.zoom == 2
    assert view.origin == (640.0, 360.0)    # dijepit ke batas citra
    assert view.to_image(0, 0) == (640, 360)
    assert view.to_image(*view.to_window(1000, 500)) == (1000, 500)
    assert view.render(np.zeros((720, 1280, 3), np.uint8)).shape == (576, 1024, 3)


def test_annotation_state_drag_and_delete():
    state = AnnotationState()
    state.drag_start = (100, 80)
    assert state.finish_drag((40, 200)) == (40, 80, 60, 120)
    state.drag_start = (10, 10)
    assert state.finish_drag((11, 12)) is None  # klik tak sengaja
    state.boxes.append((50, 90, 20, 20))
    assert state.delete_at((55, 95))            # kotak terkecil yang memuat titik
    assert state.boxes == [(40, 80, 60, 120)]
    assert not state.delete_at((500, 500))


def test_keys_ignore_caps_lock():
    from pcdface.tools.annotate import KEY_LEFT, KEY_RIGHT
    from pcdface.tools.keys import lower_key

    assert lower_key(ord("S")) == ord("s") and lower_key(ord("s")) == ord("s")
    assert lower_key(ord("Q")) == ord("q") and lower_key(ord("1")) == ord("1")
    assert lower_key(27) == 27 and lower_key(0x100000 | ord("S")) == ord("s")   # bit modifier GTK
    assert lower_key(-1) & 0xFF == 255                                             # tanpa tombol
    assert not ({ord("Q"), ord("S")} & (KEY_LEFT | KEY_RIGHT))
