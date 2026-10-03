"""Periksa konsistensi data, metadata, anotasi, dan persetujuan peserta.

Dijalankan sebelum eksperimen. Galat (GALAT) membuat kode keluar 1;
peringatan (PERINGATAN) — mis. citra belum dianotasi — tidak.

Yang diperiksa:
- setiap baris metadata punya berkas, nama berkas sesuai PRD §6.6, resolusi
  sesuai config, dan nilai set/jarak/cahaya/formasi/pose valid
- setiap peserta di foto terdaftar di subjects.csv dengan persetujuan penelitian
- jumlah kotak anotasi = expected_faces, kotak berada di dalam citra
- berkas yatim: foto tanpa metadata, anotasi tanpa metadata
- izin pengenalan (PRD v4 §11): model LBPH tersimpan hanya boleh memuat peserta dengan
  `consent_recognition`; peserta tanpa izin itu disebutkan (tidak ikut `enroll`/E8)
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from pcdface.config import Config
from pcdface.dataset.annotations import load_annotations
from pcdface.dataset.metadata import (
    SINGLE_FACE_SETS,
    SUBJECT_ID,
    CaptureSpec,
    MetadataRow,
    SubjectRow,
    read_metadata,
    read_subjects,
)
from pcdface.paths import ProjectPaths
from pcdface.pose import REFERENCE_EXPRESSION, REFERENCE_OCCLUSION, REFERENCE_POSE
from pcdface.recognition.lbph import consenting_subjects, model_subjects

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_BOX_PX = 4

ERROR = "GALAT"
WARNING = "PERINGATAN"


@dataclass
class Issue:
    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


def _spec_of(row: MetadataRow) -> CaptureSpec:
    return CaptureSpec(
        set=row.set,
        subject_id=row.subject_id,
        distance_cm=row.distance_cm,
        lighting=row.lighting,
        pose=row.pose,
        formation=row.formation,
        subjects=row.subjects,
        expression=row.expression,
        occlusion=row.occlusion,
    )


def _check_row(row: MetadataRow, cfg: Config, subjects: dict[str, SubjectRow]) -> list[Issue]:
    issues: list[Issue] = []

    def error(message: str) -> None:
        issues.append(Issue(ERROR, row.file, message))

    ds = cfg.dataset
    if row.set not in ds.sets:
        error(f"set '{row.set}' tidak dikenal")
        return issues
    if (row.width, row.height) != (cfg.capture.width, cfg.capture.height):
        error(f"resolusi metadata {row.width}×{row.height} ≠ {cfg.capture.width}×{cfg.capture.height}")
    if row.lighting not in ds.lightings:
        error(f"cahaya '{row.lighting}' tidak dikenal")

    if row.pose and row.set != "pose":
        error("kolom pose hanya untuk set pose")
    if row.expression and row.set != "ekspresi":
        error("kolom expression hanya untuk set ekspresi")
    if row.occlusion and row.set != "oklusi":
        error("kolom occlusion hanya untuk set oklusi")

    if row.set in SINGLE_FACE_SETS:
        if not SUBJECT_ID.match(row.subject_id):
            error(f"subject_id '{row.subject_id}' harus seperti S01")
        if row.subjects or row.formation or row.positions_cm:
            error("set satu wajah tidak boleh punya subjects/formation/positions_cm")
        if row.expected_faces != 1:
            error(f"expected_faces {row.expected_faces}, seharusnya 1")
        if row.set == "jarak":
            if row.distance_cm not in ds.distances_cm:
                error(f"jarak {row.distance_cm} cm bukan salah satu dari {list(ds.distances_cm)}")
            if row.lighting != "normal":
                error("set jarak harus cahaya normal (kondisi lain masuk set cahaya)")
        elif row.set == "pose":
            if row.distance_cm not in ds.pose_distances_cm:
                error(f"set pose harus di salah satu jarak {list(ds.pose_distances_cm)} cm")
        elif row.distance_cm != ds.reference_distance_cm:
            error(f"set {row.set} harus di jarak acuan {ds.reference_distance_cm} cm")
        if row.set == "cahaya" and row.lighting == "normal":
            error("cahaya normal diambil dari set jarak, bukan set cahaya")
        if row.set in ("pose", "ekspresi", "oklusi") and row.lighting != "normal":
            error(f"set {row.set} harus cahaya normal")
        if row.set == "pose" and row.pose not in ds.poses:
            error(f"pose '{row.pose}' bukan salah satu dari {list(ds.poses)}")
        if row.set == "ekspresi" and row.expression not in ds.expressions:
            error(f"ekspresi '{row.expression}' bukan salah satu dari {list(ds.expressions)}")
        if row.set == "oklusi" and row.occlusion not in ds.occlusions:
            error(f"oklusi '{row.occlusion}' bukan salah satu dari {list(ds.occlusions)}")
    elif row.set == "multi":
        expected = ds.formations.get(row.formation)
        if expected is None:
            error(f"formasi '{row.formation}' tidak ada di config")
        else:
            if row.positions_cm != expected:
                error(f"positions_cm {list(row.positions_cm)} ≠ formasi config {list(expected)}")
            if row.expected_faces != len(expected):
                error(f"expected_faces {row.expected_faces} ≠ {len(expected)} posisi")
            if len(row.subjects) != len(expected):
                error(f"subjects berisi {len(row.subjects)} kode, formasi punya {len(expected)} posisi")
        if len(set(row.subjects)) != len(row.subjects):
            error("kode peserta ganda di subjects")
        if row.subject_id:
            error("set multi memakai kolom subjects, bukan subject_id")
    elif row.set == "kosong":
        if row.expected_faces != 0 or row.people:
            error("set kosong harus expected_faces 0 tanpa peserta")

    for person in row.people:
        subject = subjects.get(person)
        if subject is None:
            error(f"peserta {person} tidak terdaftar di subjects.csv")
        elif not subject.consent_research:
            error(f"peserta {person} tidak punya persetujuan penelitian — foto harus dihapus (forget {person})")

    spec = _spec_of(row)
    folder, prefix = spec.folder(), spec.stem_prefix()
    name = Path(row.file)
    if name.parent.as_posix() != folder or not name.stem.startswith(prefix + "_"):
        error(f"nama berkas tidak sesuai metadata; seharusnya {folder}/{prefix}_NN.jpg")
    return issues


def _check_boxes(row: MetadataRow, boxes: list, image_size: tuple[int, int] | None) -> list[Issue]:
    issues: list[Issue] = []
    if len(boxes) != row.expected_faces:
        issues.append(Issue(ERROR, row.file, f"{len(boxes)} kotak anotasi, expected_faces {row.expected_faces}"))
    width, height = image_size or (row.width, row.height)
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            issues.append(Issue(ERROR, row.file, f"kotak #{index} terlalu kecil ({w}×{h})"))
        if x < 0 or y < 0 or x + w > width or y + h > height:
            issues.append(Issue(ERROR, row.file, f"kotak #{index} keluar dari citra {width}×{height}"))
    return issues


def _reference_checks(rows: list[MetadataRow]) -> list[Issue]:
    """E6 membandingkan tiap pose/ekspresi dengan acuan peserta yang sama — acuannya harus ada."""
    issues: list[Issue] = []
    poses: dict[tuple[str, int | None], set[str]] = {}
    expressions: dict[str, set[str]] = {}
    occlusions: dict[str, set[str]] = {}
    for row in rows:
        if row.set == "pose":
            poses.setdefault((row.subject_id, row.distance_cm), set()).add(row.pose)
        elif row.set == "ekspresi":
            expressions.setdefault(row.subject_id, set()).add(row.expression)
        elif row.set == "oklusi":
            occlusions.setdefault(row.subject_id, set()).add(row.occlusion)
    for (subject_id, distance), found in sorted(poses.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        if REFERENCE_POSE not in found:
            issues.append(Issue(WARNING, subject_id, f"set pose {distance} cm tanpa pose '{REFERENCE_POSE}' — "
                                                     "perbandingan terhadap acuan E6 tidak bisa dihitung"))
    for set_name, levels, reference in (("ekspresi", expressions, REFERENCE_EXPRESSION),
                                        ("oklusi", occlusions, REFERENCE_OCCLUSION)):
        for subject_id, found in sorted(levels.items()):
            if reference not in found:
                issues.append(Issue(WARNING, subject_id, f"set {set_name} tanpa '{reference}' — "
                                                         "perbandingan terhadap acuan E6 tidak bisa dihitung"))
    return issues


NORMAL_LUMA_SPREAD = 25.0  # selisih rerata Y "normal" antar sesi yang dianggap mencurigakan


def _session_checks(rows: list[MetadataRow], reference_cm: int) -> list[Issue]:
    """Peringatan bila sesi berbeda bisa tercampur dengan faktor eksperimen (PRD §6.2)."""
    issues: list[Issue] = []
    without = [r.file for r in rows if not r.session]
    if without:
        issues.append(Issue(WARNING, f"{len(without)} foto", "tanpa kode sesi (kolom session kosong)"))

    by_subject: dict[str, list[MetadataRow]] = {}
    for row in rows:
        if row.subject_id and row.session:
            by_subject.setdefault(row.subject_id, []).append(row)
    for subject_id, own in sorted(by_subject.items()):
        sessions = sorted({r.session for r in own})
        if len(sessions) <= 1:
            continue
        normal = {r.session for r in own if r.set == "jarak" and r.distance_cm == reference_cm}
        lighting = {r.session for r in own if r.set == "cahaya"}
        if normal and lighting and not normal & lighting:
            issues.append(Issue(WARNING, subject_id,
                                f"kondisi normal ({reference_cm} cm, set jarak) di sesi {sorted(normal)} tetapi set "
                                f"cahaya di sesi {sorted(lighting)} — perbandingan cahaya E3 tercampur efek sesi"))
        distance_sessions = {r.session for r in own if r.set == "jarak"}
        if len(distance_sessions) > 1:
            issues.append(Issue(WARNING, subject_id,
                                f"set jarak terpecah di sesi {sorted(distance_sessions)} — pastikan posisi kamera dan "
                                "tanda lakban identik (periksa e1_loglog_per_sesi)"))
        for set_name in ("pose", "ekspresi", "oklusi"):
            set_sessions = {r.session for r in own if r.set == set_name}
            if len(set_sessions) > 1:
                issues.append(Issue(WARNING, subject_id,
                                    f"set {set_name} terpecah di sesi {sorted(set_sessions)} — acuan "
                                    f"({REFERENCE_POSE}/{REFERENCE_EXPRESSION}/{REFERENCE_OCCLUSION}) "
                                    "sebaiknya satu sesi dengan kondisi lain"))


    luma: dict[str, list[float]] = {}
    for row in rows:
        if row.set == "jarak" and row.session and row.luma_mean is not None:
            luma.setdefault(row.session, []).append(row.luma_mean)
    if len(luma) > 1:
        means = {s: sum(v) / len(v) for s, v in luma.items()}
        spread = max(means.values()) - min(means.values())
        if spread > NORMAL_LUMA_SPREAD:
            detail = ", ".join(f"{s}: {m:.0f}" for s, m in sorted(means.items()))
            issues.append(Issue(WARNING, "cahaya normal",
                                f"rerata Y set jarak berbeda {spread:.0f} antar sesi ({detail}) — kondisi 'normal' "
                                "tidak sama; samakan lampu/tirai atau bahas sebagai keterbatasan"))
    return issues


def _recognition_checks(paths: ProjectPaths, subjects: dict[str, SubjectRow], used: set[str]) -> list[Issue]:
    """Model LBPH hanya boleh memuat peserta yang mengizinkan pengenalan (templat biometrik)."""
    issues = []
    allowed = consenting_subjects(subjects)
    for subject_id in model_subjects(paths.recognition):
        if subject_id not in allowed:
            issues.append(Issue(ERROR, f"{paths.recognition.name}/", f"model pengenalan memuat {subject_id} yang tidak "
                                "mengizinkan pengenalan — latih ulang: python -m pcdface enroll"))
    without = sorted(p for p in used if p in subjects and p not in allowed)
    if without:
        issues.append(Issue(WARNING, "subjects.csv", f"{len(without)} peserta belum mengizinkan pengenalan identitas "
                            f"({', '.join(without)}) — tidak ikut enroll/E8. Isi consent_recognition setelah "
                            "formulir v4 bagian 7 ditandatangani."))
    return issues


def validate_dataset(cfg: Config, paths: ProjectPaths, check_images: bool = True) -> tuple[list[Issue], dict[str, int]]:
    """Kembalikan (daftar masalah, ringkasan jumlah citra per set)."""
    issues: list[Issue] = []
    try:
        subjects = read_subjects(paths.subjects)
        rows = read_metadata(paths.metadata)
        annotations = load_annotations(paths.annotations)
    except (ValueError, OSError) as error:
        return [Issue(ERROR, "berkas data", str(error))], {}

    counts = Counter(row.file for row in rows)
    for file, count in counts.items():
        if count > 1:
            issues.append(Issue(ERROR, file, f"muncul {count} kali di metadata"))

    for row in rows:
        issues += _check_row(row, cfg, subjects)
        path = paths.raw / row.file
        size = None
        if not path.exists():
            issues.append(Issue(ERROR, row.file, "berkas foto tidak ada"))
        elif check_images:
            image = cv2.imread(str(path))
            if image is None:
                issues.append(Issue(ERROR, row.file, "berkas foto tidak bisa dibaca"))
            else:
                size = (image.shape[1], image.shape[0])
                if size != (row.width, row.height):
                    issues.append(Issue(ERROR, row.file, f"ukuran foto {size[0]}×{size[1]} ≠ metadata {row.width}×{row.height}"))
        if row.file in annotations:
            issues += _check_boxes(row, annotations[row.file], size)
        else:
            issues.append(Issue(WARNING, row.file, "belum dianotasi"))

    known = set(counts)
    if paths.raw.exists():
        for path in sorted(paths.raw.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rel = path.relative_to(paths.raw).as_posix()
                if rel not in known:
                    issues.append(Issue(ERROR, rel, "foto yatim: ada di disk tetapi tidak ada di metadata"))
    for key in sorted(set(annotations) - known):
        issues.append(Issue(ERROR, key, "anotasi yatim: tidak ada di metadata"))

    issues += _session_checks(rows, cfg.dataset.reference_distance_cm)
    issues += _reference_checks(rows)


    used = {person for row in rows for person in row.people}
    for subject_id in sorted(set(subjects) - used):
        issues.append(Issue(WARNING, subject_id, "terdaftar di subjects.csv tetapi belum punya foto"))
    issues += _recognition_checks(paths, subjects, used)

    summary = dict(Counter(row.set for row in rows))
    summary["dianotasi"] = sum(1 for row in rows if row.file in annotations)
    summary["peserta"] = len(used)
    summary["sesi"] = len({row.session for row in rows if row.session})
    return issues, summary


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-images", action="store_true", help="Lewati pembacaan berkas foto (lebih cepat)")
    parser.add_argument("--quiet", action="store_true", help="Sembunyikan PERINGATAN, tampilkan GALAT saja")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    issues, summary = validate_dataset(cfg, paths, check_images=not args.no_images)
    errors = [i for i in issues if i.level == ERROR]
    warnings = [i for i in issues if i.level == WARNING]
    for issue in errors + ([] if args.quiet else warnings):
        print(issue)
    if summary:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(summary.items()))
        print(f"\nRingkasan: {parts}")
    print(f"{len(errors)} galat, {len(warnings)} peringatan.")
    return 1 if errors else 0
