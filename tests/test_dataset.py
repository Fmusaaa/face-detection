"""Skema metadata, subjects, anotasi, penamaan, perkecilan, dan dataset sintetis."""

import numpy as np
import pytest

from pcdface.dataset.annotations import load_annotations, save_annotations
from pcdface.dataset.loader import load_samples
from pcdface.dataset.metadata import (
    CaptureSpec,
    MetadataRow,
    SubjectRow,
    append_metadata,
    next_index,
    read_metadata,
    read_subjects,
    write_metadata,
    write_subjects,
)
from pcdface.dataset.resize import scale_boxes, scale_image
from pcdface.preprocessing import enhance


def test_filenames_follow_prd():
    assert CaptureSpec("jarak", "S03", 150, "normal").relative_path(4) == "jarak/S03/jarak_S03_150cm_normal_04.jpg"
    assert CaptureSpec("cahaya", "S03", 100, "redup").relative_path(2) == "cahaya/S03/cahaya_S03_100cm_redup_02.jpg"
    assert CaptureSpec("multi", formation="F5").relative_path(3) == "multi/F5/multi_F5_03.jpg"
    assert CaptureSpec("kosong").relative_path(7) == "kosong/kosong_07.jpg"
    assert CaptureSpec("pose", "S01", 100, pose="kiri").relative_path(1) == "pose/S01/pose_S01_100cm_kiri_01.jpg"


def test_next_index_uses_disk_and_metadata(tmp_path):
    spec = CaptureSpec("jarak", "S01", 50)
    assert next_index(tmp_path, spec) == 1
    (tmp_path / spec.folder()).mkdir(parents=True)
    (tmp_path / spec.relative_path(3)).write_bytes(b"x")
    assert next_index(tmp_path, spec) == 4
    assert next_index(tmp_path, spec, taken=[spec.relative_path(9)]) == 10


def test_metadata_roundtrip(tmp_path):
    path = tmp_path / "metadata.csv"
    rows = [
        MetadataRow(file="multi/F5/multi_F5_01.jpg", set="multi", expected_faces=3, width=1280, height=720,
                    subjects=("S02", "S05", "S01"), formation="F5", positions_cm=(80, 150, 250), luma_mean=112.44),
        MetadataRow(file="jarak/S01/jarak_S01_50cm_normal_01.jpg", set="jarak", expected_faces=1,
                    width=1280, height=720, subject_id="S01", distance_cm=50),
    ]
    write_metadata(path, rows)
    back = read_metadata(path)
    assert back[0].subjects == ("S02", "S05", "S01") and back[0].positions_cm == (80, 150, 250)
    assert back[0].people == ("S02", "S05", "S01") and back[0].group == "multi/F5/multi_F5_01.jpg"
    assert back[0].luma_mean == pytest.approx(112.4)
    assert back[1].distance_cm == 50 and back[1].people == ("S01",) and back[1].group == "S01"
    with pytest.raises(ValueError, match="sudah ada"):
        append_metadata(path, rows[1])


def test_subjects_roundtrip_and_validation(tmp_path):
    path = tmp_path / "subjects.csv"
    write_subjects(path, [SubjectRow("S02", True, False, "2026-09-30"), SubjectRow("S01", True, True)])
    subjects = read_subjects(path)
    assert list(subjects) == ["S01", "S02"]
    assert subjects["S02"].consent_publication is False
    path.write_text("subject_id,consent_research,consent_publication,consent_date\nBudi,ya,ya,\n")
    with pytest.raises(ValueError, match="kode peserta"):
        read_subjects(path)
    path.write_text("subject_id,consent_research,consent_publication,consent_date\nS01,mungkin,ya,\n")
    with pytest.raises(ValueError, match="'ya' atau 'tidak'"):
        read_subjects(path)


def test_annotations_roundtrip_keeps_empty_lists(tmp_path):
    path = tmp_path / "annotations" / "boxes.json"
    assert load_annotations(path) == {}
    save_annotations(path, {"b.jpg": [(1, 2, 3, 4)], "kosong/kosong_01.jpg": []})
    assert load_annotations(path) == {"b.jpg": [(1, 2, 3, 4)], "kosong/kosong_01.jpg": []}


def test_scale_image_and_boxes():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert scale_image(image, 0.5).shape == (360, 640, 3)
    assert scale_image(image, 1.0) is image
    assert scale_boxes([(101, 50, 31, 41)], 0.5) == [(50, 25, 16, 20)]
    assert scale_boxes([(0, 0, 1, 1)], 0.1) == [(0, 0, 1, 1)]


def test_enhance_rejects_unknown_method():
    with pytest.raises(ValueError):
        enhance(np.zeros((4, 4, 3), dtype=np.uint8), "he")


def test_synthetic_dataset_is_consistent(synthetic_paths, small_cfg):
    report = load_samples(synthetic_paths)
    assert not report.unannotated and not report.missing_files
    sets = {s.meta.set for s in report.samples}
    assert sets == {"jarak", "cahaya", "multi", "kosong", "pose"}
    for sample in report.samples:
        assert len(sample.gt) == sample.meta.expected_faces
        if sample.meta.set == "multi":
            assert len(sample.meta.subjects) == len(sample.meta.positions_cm) == len(sample.gt)
            assert len(set(sample.meta.subjects)) == len(sample.meta.subjects)
    image = report.samples[0].load()
    assert image.shape == (small_cfg.capture.height, small_cfg.capture.width, 3)
    only_multi = load_samples(synthetic_paths, sets=("multi",)).samples
    assert only_multi and all(s.meta.set == "multi" for s in only_multi)
