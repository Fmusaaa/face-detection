"""Rekam foto dari webcam dan tulis metadata (PRD §6).

Contoh:
    python -m pcdface capture --set jarak --subject S03 --distance 150
    python -m pcdface capture --set cahaya --subject S03 --lighting redup
    python -m pcdface capture --set pose --subject S03 --distance 100 --pose kiri30
    python -m pcdface capture --set pose --subject S03 --distance 200 --pose semua
    python -m pcdface capture --set ekspresi --subject S03 --expression semua
    python -m pcdface capture --set multi --formation F5 --subjects S02,S05,S01
    python -m pcdface capture --set kosong

`--pose semua` / `--expression semua` merekam semua level dari config
berurutan dalam satu jendela; instruksi kondisi berikutnya tampil di layar.

Tombol: SPASI simpan frame, q/ESC keluar.

Aturan yang dijaga alat ini:
- peserta harus terdaftar di data/subjects.csv dengan persetujuan penelitian
  (formulir sudah ditandatangani) — kalau belum, perekaman ditolak;
- frame yang resolusinya bukan 1280×720 ditolak, tidak disimpan;
- pratinjau TIDAK dicerminkan, supaya "kiri → kanan di citra" untuk formasi
  multi-wajah sama dengan yang terlihat di layar;
- foto disimpan apa adanya, tanpa enhancement.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from pcdface.config import Config, DatasetConfig
from pcdface.dataset.metadata import (
    SINGLE_FACE_SETS,
    SUBJECT_ID,
    CaptureSpec,
    MetadataRow,
    SubjectRow,
    append_metadata,
    next_index,
    read_metadata,
    read_subjects,
)
from pcdface.paths import ProjectPaths
from pcdface.pose import parse_pose
from pcdface.preprocessing import mean_luma

CHECKLIST = (
    "Sebelum merekam, pastikan (Control Center → Video Effects saat kamera aktif):",
    "  • Center Stage MATI — kalau aktif, ukuran wajah tidak mengecil sesuai jarak",
    "  • Studio Light dan Portrait MATI",
    "  • Kamera tetap di meja/tripod, lensa setinggi mata; latar dinding polos",
)


SESSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,39}$")
ALL_LEVELS = "semua"


@dataclass(frozen=True)
class CapturePlan:
    spec: CaptureSpec
    positions_cm: tuple[int, ...]
    expected_faces: int
    count: int
    session: str = ""

    def describe(self) -> str:
        spec = self.spec
        if spec.set in ("jarak", "cahaya"):
            return f"{spec.set} {spec.subject_id} {spec.distance_cm} cm {spec.lighting}"
        if spec.set == "pose":
            return f"pose {spec.subject_id} {spec.distance_cm} cm {spec.pose}"
        if spec.set == "ekspresi":
            return f"ekspresi {spec.subject_id} {spec.distance_cm} cm {spec.expression}"
        if spec.set == "multi":
            layout = ", ".join(f"{s}@{p}cm" for s, p in zip(spec.subjects, self.positions_cm))
            return f"multi {spec.formation}: {layout} (kiri → kanan)"
        return "kosong (tanpa wajah)"

    def instruction(self) -> str:
        """Arahan untuk peserta pada set pose dan ekspresi; kosong untuk set lain."""
        if self.spec.set == "pose":
            return parse_pose(self.spec.pose).describe()
        if self.spec.set == "ekspresi":
            return f"ekspresi {self.spec.expression.upper()}, tatap lensa"
        return ""


def build_plan(args: argparse.Namespace, cfg: Config, paths: ProjectPaths) -> CapturePlan:
    """Validasi argumen terhadap config dan persetujuan peserta. ValueError bila salah."""
    ds = cfg.dataset
    subjects = read_subjects(paths.subjects)
    default_count = ds.pose_frames_per_condition if args.set in ("pose", "ekspresi") else ds.frames_per_condition
    count = args.count if args.count is not None else default_count
    if count < 1:
        raise ValueError("--count minimal 1")

    def require_consent(subject_id: str) -> None:
        if not SUBJECT_ID.match(subject_id or ""):
            raise ValueError(f"kode peserta harus seperti S03, dapat {subject_id!r}")
        subject = subjects.get(subject_id)
        if subject is None:
            raise ValueError(f"{subject_id} belum terdaftar di {paths.subjects} — isi dulu dari formulir persetujuan")
        if not subject.consent_research:
            raise ValueError(f"{subject_id} tidak memberi persetujuan penelitian — tidak boleh difoto")

    if args.set not in ds.sets:
        raise ValueError(f"set '{args.set}' tidak dikenal. Pilihan: {', '.join(ds.sets)}")
    session = getattr(args, "session", None) or datetime.now().strftime("%Y-%m-%d")
    if not SESSION_PATTERN.match(session):
        raise ValueError(f"--session hanya huruf, angka, '-' atau '_' (maks. 40), dapat {session!r}")
    plan = _build_spec(args, cfg, ds, subjects, count, require_consent)
    return replace(plan, session=session)


def build_plans(args: argparse.Namespace, cfg: Config, paths: ProjectPaths) -> list[CapturePlan]:
    """Satu rencana, atau satu per level bila `--pose semua` / `--expression semua`."""
    if args.set == "pose" and getattr(args, "pose", None) == ALL_LEVELS:
        return [build_plan(argparse.Namespace(**{**vars(args), "pose": pose}), cfg, paths)
                for pose in cfg.dataset.poses]
    if args.set == "ekspresi" and getattr(args, "expression", None) == ALL_LEVELS:
        return [build_plan(argparse.Namespace(**{**vars(args), "expression": expression}), cfg, paths)
                for expression in cfg.dataset.expressions]
    return [build_plan(args, cfg, paths)]


def _build_spec(
    args: argparse.Namespace,
    cfg: Config,
    ds: DatasetConfig,
    subjects: dict[str, SubjectRow],
    count: int,
    require_consent: Callable[[str], None],
) -> CapturePlan:
    """Rencana rekam per set, tanpa sesi (sesi ditambahkan oleh build_plan)."""
    if args.set in SINGLE_FACE_SETS:
        require_consent(args.subject)
        if args.set in ("pose", "ekspresi") and args.lighting != "normal":
            raise ValueError(f"set {args.set} selalu cahaya normal")
        if args.set == "jarak":
            if args.distance not in ds.distances_cm:
                raise ValueError(f"--distance harus salah satu dari {list(ds.distances_cm)}")
            if args.lighting != "normal":
                raise ValueError("set jarak selalu cahaya normal; kondisi lain pakai --set cahaya")
            spec = CaptureSpec("jarak", args.subject, args.distance, "normal")
        elif args.set == "cahaya":
            options = [l for l in ds.lightings if l != "normal"]
            if args.lighting not in options:
                raise ValueError(f"--lighting untuk set cahaya harus salah satu dari {options} "
                                 "(normal sudah tercakup set jarak)")
            spec = CaptureSpec("cahaya", args.subject, ds.reference_distance_cm, args.lighting)
        elif args.set == "pose":
            if args.pose not in ds.poses:
                raise ValueError(f"--pose harus salah satu dari {list(ds.poses)} atau '{ALL_LEVELS}'")
            if args.distance not in ds.pose_distances_cm:
                raise ValueError(f"set pose butuh --distance salah satu dari {list(ds.pose_distances_cm)}")
            spec = CaptureSpec("pose", args.subject, args.distance, "normal", pose=args.pose)
        else:
            expression = getattr(args, "expression", None)
            if expression not in ds.expressions:
                raise ValueError(f"--expression harus salah satu dari {list(ds.expressions)} atau '{ALL_LEVELS}'")
            spec = CaptureSpec("ekspresi", args.subject, ds.reference_distance_cm, "normal", expression=expression)
        if args.set in ("cahaya", "ekspresi") and args.distance not in (None, ds.reference_distance_cm):
            raise ValueError(f"set {args.set} selalu di jarak acuan {ds.reference_distance_cm} cm")
        return CapturePlan(spec, (), 1, count)

    if args.set == "multi":
        positions = ds.formations.get(args.formation or "")
        if positions is None:
            raise ValueError(f"--formation harus salah satu dari {list(ds.formations)}")
        people = tuple(s.strip() for s in (args.subjects or "").split(",") if s.strip())
        if len(people) != len(positions):
            raise ValueError(f"formasi {args.formation} butuh {len(positions)} peserta di --subjects "
                             f"(kiri → kanan di citra), dapat {len(people)}")
        if len(set(people)) != len(people):
            raise ValueError("--subjects berisi kode ganda")
        for person in people:
            require_consent(person)
        spec = CaptureSpec("multi", formation=args.formation, subjects=people)
        return CapturePlan(spec, tuple(positions), len(positions), count)

    return CapturePlan(CaptureSpec("kosong"), (), 0, count)


def make_row(plan: CapturePlan, relative_path: str, frame: np.ndarray) -> MetadataRow:
    spec = plan.spec
    return MetadataRow(
        file=relative_path,
        set=spec.set,
        session=plan.session,
        subject_id=spec.subject_id,
        subjects=spec.subjects,
        formation=spec.formation,
        positions_cm=plan.positions_cm,
        distance_cm=spec.distance_cm,
        lighting=spec.lighting,
        pose=spec.pose,
        expression=spec.expression,
        expected_faces=plan.expected_faces,
        luma_mean=mean_luma(frame),
        width=frame.shape[1],
        height=frame.shape[0],
        captured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def save_frame(plan: CapturePlan, frame: np.ndarray, paths: ProjectPaths) -> MetadataRow:
    """Simpan satu frame + satu baris metadata. Nomor urut tidak pernah menimpa."""
    taken = [row.file for row in read_metadata(paths.metadata)]
    rel = plan.spec.relative_path(next_index(paths.raw, plan.spec, taken))
    target = paths.raw / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"gagal menulis {target}")
    row = make_row(plan, rel, frame)
    append_metadata(paths.metadata, row)
    return row


def _draw_overlay(frame: np.ndarray, plan: CapturePlan, saved: int, step: str = "") -> np.ndarray:
    preview = frame.copy()
    height, width = preview.shape[:2]
    if plan.spec.set == "multi":
        slot = width / len(plan.positions_cm)
        for k in range(1, len(plan.positions_cm)):
            x = int(slot * k)
            cv2.line(preview, (x, 0), (x, height), (0, 200, 255), 1)
        for k, (person, distance) in enumerate(zip(plan.spec.subjects, plan.positions_cm)):
            cv2.putText(preview, f"{person} {distance}cm", (int(slot * k) + 10, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        cv2.drawMarker(preview, (width // 2, int(height * 0.45)), (0, 200, 255), cv2.MARKER_CROSS, 40, 1)
    status = f"{step}sesi {plan.session} | {plan.describe()} | tersimpan {saved}/{plan.count} | Y {mean_luma(frame):.0f}"
    cv2.rectangle(preview, (0, 0), (width, 34), (20, 20, 20), -1)
    cv2.putText(preview, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    if plan.instruction():
        cv2.rectangle(preview, (0, 34), (width, 80), (20, 20, 20), -1)
        cv2.putText(preview, plan.instruction(), (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "SPASI=simpan  q=keluar  |  Center Stage harus MATI", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return preview


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", required=True, help="jarak | cahaya | multi | kosong | pose | ekspresi")
    parser.add_argument("--subject", help="Kode peserta untuk set satu wajah, mis. S03")
    parser.add_argument("--distance", type=int, help="Jarak cm (set jarak dan pose)")
    parser.add_argument("--lighting", default="normal", help="normal | terang | redup | backlight")
    parser.add_argument("--pose", help="Pose (set pose), mis. depan, kiri30, menunduk30, atau 'semua'")
    parser.add_argument("--expression", help="Ekspresi (set ekspresi), mis. netral, marah, atau 'semua'")

    parser.add_argument("--formation", help="Formasi multi-wajah, mis. F5")
    parser.add_argument("--subjects", help="Peserta multi-wajah kiri → kanan DI CITRA, mis. S02,S05,S01")
    parser.add_argument("--count", type=int, default=None, help="Jumlah frame (bawaan: dataset.frames_per_condition)")
    parser.add_argument("--camera", type=int, default=None, help="Indeks kamera (bawaan: capture.camera_index)")
    parser.add_argument("--session", default=None,
                        help="Kode sesi pengambilan (bawaan: tanggal hari ini, mis. 2026-09-30). "
                             "Pakai kode sama untuk seluruh pertemuan, mis. 2026-09-30-sore")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    try:
        plans = build_plans(args, cfg, paths)
    except ValueError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 2

    print("\n".join(CHECKLIST))
    print(f"\nSesi {plans[0].session} — {len(plans)} kondisi:")
    for plan in plans:
        print(f"  • {plan.describe()} — {plan.count} frame")
    print()

    camera_index = cfg.capture.camera_index if args.camera is None else args.camera
    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture.height)
    if not camera.isOpened():
        print(f"[GAGAL] Kamera {camera_index} tidak bisa dibuka. Beri izin kamera untuk Terminal di "
              "System Settings → Privacy & Security → Camera.", file=sys.stderr)
        return 1

    total = 0
    window = "Perekam - pcdface"
    try:
        for number, plan in enumerate(plans, start=1):
            step = f"[{number}/{len(plans)}] " if len(plans) > 1 else ""
            print(f"{step}{plan.describe()}" + (f" — {plan.instruction()}" if plan.instruction() else ""))

            saved = 0
            while saved < plan.count:
                ok, frame = camera.read()
                if not ok:
                    print("[GAGAL] Tidak bisa membaca frame dari kamera.", file=sys.stderr)
                    return 1
                if (frame.shape[1], frame.shape[0]) != (cfg.capture.width, cfg.capture.height):
                    print(f"[GAGAL] Kamera memberi {frame.shape[1]}×{frame.shape[0]}, bukan "
                          f"{cfg.capture.width}×{cfg.capture.height}. Tidak ada yang disimpan.", file=sys.stderr)
                    return 1
                cv2.imshow(window, _draw_overlay(frame, plan, saved, step))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print(f"\nDihentikan. {total} foto tersimpan.")
                    return 0
                if key == ord(" "):
                    row = save_frame(plan, frame, paths)
                    saved += 1
                    total += 1
                    print(f"  tersimpan {row.file}  (Y rata-rata {row.luma_mean:.1f})")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n{total} foto tersimpan. Berikutnya: python -m pcdface annotate")
    return 0

