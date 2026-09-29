"""Dataset sintetis untuk uji jalur tanpa webcam dan tanpa data asli.

Membangun folder data lengkap — `raw/`, `metadata.csv`, `subjects.csv`,
`annotations/boxes.json` — dengan skema dan penamaan yang sama persis dengan
data asli (PRD §6.6), sehingga seluruh jalur eksperimen, `validate`, dan
`forget` bisa diuji ujung ke ujung.

"Wajah" berupa elips berwarna kulit dengan rambut, alis, mata, hidung, mulut,
dan leher. Lebarnya mengikuti model lubang jarum `w = f × W / Z`, jadi
eksperimen jarak menghasilkan kurva yang masuk akal.

Angka dari data sintetis BUKAN hasil penelitian — hanya bukti bahwa kode
berjalan dan metrik terhitung.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box
from pcdface.config import Config
from pcdface.dataset.annotations import save_annotations
from pcdface.dataset.metadata import (
    CaptureSpec,
    MetadataRow,
    SubjectRow,
    write_metadata,
    write_subjects,
)
from pcdface.paths import ProjectPaths
from pcdface.pose import REFERENCE_POSE, parse_pose
from pcdface.preprocessing import mean_luma

# Warna kulit BGR — kromanya jatuh di dalam ambang Chai & Ngan (1999)
SKIN_TONES: tuple[tuple[int, int, int], ...] = (
    (150, 175, 220),
    (120, 150, 200),
    (95, 125, 175),
    (135, 160, 210),
    (110, 140, 190),
)
WALL_BGR = (172, 170, 164)          # dinding polos abu-abu, di luar ambang kulit
HAIR_BGR = (35, 38, 45)
SHIRT_BGR = (140, 90, 60)
CARDBOARD_BGR = (110, 150, 190)     # berwarna mirip kulit — pengecoh set kosong
WOOD_BGR = (70, 105, 150)
FACE_ASPECT = 1.3                   # tinggi / lebar kotak manual (garis rambut – dagu)

_LIGHTING = {
    # (pengali latar, pengali orang, sigma derau)
    "normal": (1.0, 1.0, 4.0),
    "terang": (1.3, 1.3, 3.0),
    "redup": (0.38, 0.38, 6.0),
    "backlight": (1.75, 0.55, 5.0),
}


@dataclass(frozen=True)
class FaceSpec:
    center_x: int
    center_y: int
    width: int
    tone: tuple[int, int, int]
    pose: str = ""
    expression: str = ""
    occlusion: str = ""


def face_width_px(distance_cm: float, focal_px: float, face_width_cm: float) -> int:
    """Model lubang jarum: w_px = f_px × W_wajah / Z."""
    return max(4, int(round(focal_px * face_width_cm / distance_cm)))


def _pose_geometry(pose: str, w: int, h: int) -> tuple[float, float, float, float]:
    """(faktor lebar tampak, geser fitur x, geser fitur y, beda tinggi mata) untuk pose sintetis.

    Menoleh menyempitkan wajah tampak (≈ 0,55 + 0,45·cos θ) dan menggeser fitur ke arah
    toleh; kiri peserta tampak di kanan citra karena citra tidak dicerminkan.
    """
    info = parse_pose(pose or REFERENCE_POSE)
    t = np.radians(info.angle)
    if info.axis == "menoleh":
        sign = 1.0 if info.direction == "kiri" else -1.0
        return 0.55 + 0.45 * float(np.cos(t)), sign * 0.2 * float(np.sin(t)) * w, 0.0, 0.0
    if info.axis == "mengangguk":
        sign = 1.0 if info.direction == "menunduk" else -1.0
        return 1.0, 0.0, sign * 0.16 * float(np.sin(t)) * h, 0.0
    if info.axis == "miring":
        sign = 1.0 if info.direction == "miringkiri" else -1.0
        return 1.0, 0.0, 0.0, sign * 0.3 * float(np.sin(t)) * h
    return 1.0, 0.0, 0.0, 0.0


def _draw_person(canvas: np.ndarray, mask: np.ndarray, face: FaceSpec) -> Box:
    """Gambar kepala + leher + bahu. Kembalikan kotak manual (garis rambut – dagu)."""
    cx, cy = face.center_x, face.center_y
    h = int(round(face.width * FACE_ASPECT))
    width_factor, dx, dy, eye_tilt = _pose_geometry(face.pose, face.width, h)
    w = max(4, int(round(face.width * width_factor)))
    half_w, half_h = w // 2, h // 2
    top = cy - half_h

    def fill_ellipse(center, axes, color):
        cv2.ellipse(canvas, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    def fill_rect(p1, p2, color):
        cv2.rectangle(canvas, p1, p2, color, -1)
        cv2.rectangle(mask, p1, p2, 255, -1)

    # bahu dan leher dulu, lalu rambut, lalu wajah di atasnya
    fill_rect((cx - int(1.3 * w), cy + half_h + int(0.35 * w)), (cx + int(1.3 * w), cy + half_h + 3 * w), SHIRT_BGR)
    fill_rect((cx - int(0.22 * w), cy + int(0.3 * h)), (cx + int(0.22 * w), cy + half_h + int(0.4 * w)), face.tone)
    fill_ellipse((cx, top + int(0.12 * h)), (int(0.56 * w), int(0.3 * h)), HAIR_BGR)
    fill_ellipse((cx, cy), (half_w, half_h), face.tone)

    # posisi fitur bergeser untuk pose; ekspresi mengubah alis dan mulut
    eye_dx = int(0.2 * w)
    eye_axes = (max(int(0.08 * w), 1), max(int(0.04 * w), 1))
    dark = tuple(int(c * 0.35) for c in face.tone)
    brow = {"marah": 0.03, "kaget": -0.03}.get(face.expression, 0.0) * h
    for sign in (-1, 1):
        ex = int(cx + sign * eye_dx + dx)
        eye_y = int(cy - 0.08 * h + dy + sign * eye_tilt / 2)
        cv2.ellipse(canvas, (ex, eye_y), eye_axes, 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        inner = int(brow * 1.5) if face.expression == "marah" else int(brow)
        cv2.line(canvas, (ex - sign * eye_axes[0], eye_y - int(0.07 * h + brow)),
                 (ex + sign * eye_axes[0], eye_y - int(0.08 * h + brow) + inner),
                 HAIR_BGR, max(1, w // 30), cv2.LINE_AA)
    nose_top = (int(cx + dx), int(cy - 0.02 * h + dy))
    nose_bottom = (int(cx + dx * 1.3), int(cy + 0.12 * h + dy))
    cv2.line(canvas, nose_top, nose_bottom, dark, max(1, w // 40), cv2.LINE_AA)
    mouth = (int(cx + dx), int(cy + 0.27 * h + dy))
    if face.expression == "kaget":
        cv2.ellipse(canvas, mouth, (max(int(0.1 * w), 1), max(int(0.07 * h), 1)), 0, 0, 360, (40, 30, 70), -1, cv2.LINE_AA)
    else:
        scale = 1.4 if face.expression == "senyum" else 1.0
        start, end = (180, 360) if face.expression == "marah" else (0, 180)
        cv2.ellipse(canvas, mouth, (max(int(0.18 * w * scale), 1), max(int(0.035 * h * scale), 1)),
                    0, start, end, (45, 45, 110), -1, cv2.LINE_AA)
    _draw_occlusion(canvas, face, (cx, cy, w, h), (dx, dy))
    return (cx - half_w, top, w, h)


MASK_BGR = (225, 205, 170)          # masker biru muda
SUNGLASS_BGR = (25, 25, 25)


def _draw_occlusion(canvas: np.ndarray, face: FaceSpec, geometry: tuple[int, int, int, int],
                    shift: tuple[float, float]) -> None:
    """Masker, tangan, atau kacamata hitam di atas wajah (set oklusi). Kotak manual tidak berubah."""
    cx, cy, w, h = geometry
    dx, dy = shift
    if face.occlusion == "masker":
        cv2.ellipse(canvas, (int(cx + dx), int(cy + 0.22 * h + dy)), (max(int(0.48 * w), 1), max(int(0.26 * h), 1)),
                    0, 0, 360, MASK_BGR, -1, cv2.LINE_AA)
    elif face.occlusion == "tangan":
        palm = tuple(int(min(255, c * 1.08)) for c in face.tone)
        cv2.ellipse(canvas, (int(cx + dx), int(cy + 0.3 * h + dy)), (max(int(0.4 * w), 1), max(int(0.2 * h), 1)),
                    0, 0, 360, palm, -1, cv2.LINE_AA)
        for k in range(-2, 3):
            x = int(cx + dx + k * 0.12 * w)
            cv2.line(canvas, (x, int(cy + 0.15 * h + dy)), (x, int(cy + 0.45 * h + dy)),
                     tuple(int(c * 0.8) for c in face.tone), max(1, w // 60), cv2.LINE_AA)
    elif face.occlusion == "kacamata_hitam":
        eye_y = int(cy - 0.08 * h + dy)
        for sign in (-1, 1):
            cv2.ellipse(canvas, (int(cx + sign * 0.2 * w + dx), eye_y), (max(int(0.15 * w), 1), max(int(0.08 * h), 1)),
                        0, 0, 360, SUNGLASS_BGR, -1, cv2.LINE_AA)
        cv2.line(canvas, (int(cx - 0.06 * w + dx), eye_y), (int(cx + 0.06 * w + dx), eye_y),
                 SUNGLASS_BGR, max(1, w // 40), cv2.LINE_AA)


def _compose(background: np.ndarray, people: np.ndarray, mask: np.ndarray,
             lighting: str, rng: np.random.Generator) -> np.ndarray:
    bg_gain, person_gain, sigma = _LIGHTING[lighting]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    lit_bg = background.astype(np.float32) * bg_gain
    if lighting == "backlight":
        lit_bg += 40.0
    image = lit_bg * (1.0 - alpha) + people.astype(np.float32) * person_gain * alpha
    image += rng.normal(0.0, sigma, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full((height, width, 3), WALL_BGR, dtype=np.float32)
    gradient = np.linspace(-12, 12, width, dtype=np.float32)[None, :, None]
    base += gradient + rng.normal(0, 2.0, (1, 1, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def render_scene(
    width: int,
    height: int,
    faces: list[FaceSpec],
    lighting: str,
    rng: np.random.Generator,
    distractors: bool = False,
) -> tuple[np.ndarray, list[Box]]:
    """Render satu citra. Kembalikan (citra BGR, kotak manual)."""
    background = _background(width, height, rng)
    if distractors:
        cv2.rectangle(background, (0, int(height * 0.82)), (width, height), WOOD_BGR, -1)
        for _ in range(int(rng.integers(2, 5))):
            x = int(rng.integers(0, width - 200))
            y = int(rng.integers(height // 3, height - 120))
            w = int(rng.integers(90, 260))
            h = int(rng.integers(70, 200))
            color = CARDBOARD_BGR if rng.random() < 0.6 else WOOD_BGR
            cv2.rectangle(background, (x, y), (x + w, y + h), color, -1)
    people = background.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = [_draw_person(people, mask, face) for face in faces]
    return _compose(background, people, mask, lighting, rng), boxes


SYNTHETIC_SESSION = "2026-09-30"


def _row(spec: CaptureSpec, rel: str, image: np.ndarray, expected: int,
         positions: tuple[int, ...] = ()) -> MetadataRow:
    return MetadataRow(
        file=rel,
        set=spec.set,
        session=SYNTHETIC_SESSION,
        subject_id=spec.subject_id,
        subjects=spec.subjects,
        formation=spec.formation,
        positions_cm=positions,
        distance_cm=spec.distance_cm,
        lighting=spec.lighting,
        pose=spec.pose,
        expression=spec.expression,
        occlusion=spec.occlusion,
        expected_faces=expected,
        luma_mean=mean_luma(image),
        width=image.shape[1],
        height=image.shape[0],
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def generate_dataset(cfg: Config, data_root: Path, seed: int | None = None) -> ProjectPaths:
    """Bangun dataset sintetis di `data_root`. Kembalikan path-nya."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    paths = cfg.paths.with_data_root(Path(data_root))
    width, height = cfg.capture.width, cfg.capture.height
    syn, ds = cfg.synthetic, cfg.dataset

    subject_ids = [f"S{i:02d}" for i in range(1, syn.subjects + 1)]
    tones = {sid: SKIN_TONES[i % len(SKIN_TONES)] for i, sid in enumerate(subject_ids)}
    write_subjects(paths.subjects, [
        SubjectRow(sid, consent_research=True, consent_publication=(i % 2 == 0), consent_date="2026-09-30")
        for i, sid in enumerate(subject_ids)
    ])

    rows: list[MetadataRow] = []
    annotations: dict[str, list[Box]] = {}

    def save(spec: CaptureSpec, index: int, image: np.ndarray, boxes: list[Box],
             positions: tuple[int, ...] = ()) -> None:
        rel = spec.relative_path(index)
        target = paths.raw / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(_row(spec, rel, image, len(boxes), positions))
        annotations[rel] = boxes

    def single_face(sid: str, distance: int, pose: str = "", expression: str = "", occlusion: str = "") -> FaceSpec:
        w = face_width_px(distance * float(rng.uniform(0.97, 1.03)), syn.focal_px, syn.face_width_cm)
        return FaceSpec(
            center_x=int(width / 2 + rng.integers(-60, 61)),
            center_y=int(height * 0.45 + rng.integers(-20, 21)),
            width=w,
            tone=tones[sid],
            pose=pose,
            expression=expression,
            occlusion=occlusion,
        )

    for sid in subject_ids:
        for distance in ds.distances_cm:
            spec = CaptureSpec(set="jarak", subject_id=sid, distance_cm=distance, lighting="normal")
            for index in range(1, syn.frames_per_condition + 1):
                image, boxes = render_scene(width, height, [single_face(sid, distance)], "normal", rng)
                save(spec, index, image, boxes)
        for lighting in ds.lightings:
            if lighting == "normal":
                continue  # kondisi normal diambil dari set jarak di jarak acuan
            spec = CaptureSpec(set="cahaya", subject_id=sid, distance_cm=ds.reference_distance_cm, lighting=lighting)
            for index in range(1, syn.frames_per_condition + 1):
                image, boxes = render_scene(width, height, [single_face(sid, ds.reference_distance_cm)], lighting, rng)
                save(spec, index, image, boxes)
        # set pose dan ekspresi: satu frame per kondisi supaya dataset sintetis tetap kecil
        for distance in ds.pose_distances_cm:
            for pose in ds.poses:
                spec = CaptureSpec(set="pose", subject_id=sid, distance_cm=distance, pose=pose)
                image, boxes = render_scene(width, height, [single_face(sid, distance, pose)], "normal", rng)
                save(spec, 1, image, boxes)
        for expression in ds.expressions:
            spec = CaptureSpec(set="ekspresi", subject_id=sid, distance_cm=ds.reference_distance_cm, expression=expression)
            image, boxes = render_scene(width, height, [single_face(sid, ds.reference_distance_cm, expression=expression)],
                                        "normal", rng)
            save(spec, 1, image, boxes)
        for occlusion in ds.occlusions:
            spec = CaptureSpec(set="oklusi", subject_id=sid, distance_cm=ds.reference_distance_cm, occlusion=occlusion)
            image, boxes = render_scene(width, height, [single_face(sid, ds.reference_distance_cm, occlusion=occlusion)],
                                        "normal", rng)
            save(spec, 1, image, boxes)



    rotation = 0
    for formation, positions in ds.formations.items():
        if len(positions) > len(subject_ids):
            continue  # butuh peserta berbeda sebanyak posisi
        for index in range(1, syn.frames_per_condition + 1):
            people = tuple(subject_ids[(rotation + k) % len(subject_ids)] for k in range(len(positions)))
            rotation += 1
            slot = width / len(positions)
            faces = [
                FaceSpec(
                    center_x=int(slot * (k + 0.5) + rng.integers(-15, 16)),
                    center_y=int(height * 0.42 + rng.integers(-15, 16)),
                    width=min(face_width_px(distance, syn.focal_px, syn.face_width_cm), int(slot * 0.8)),
                    tone=tones[sid],
                )
                for k, (sid, distance) in enumerate(zip(people, positions))
            ]
            spec = CaptureSpec(set="multi", formation=formation, subjects=people)
            image, boxes = render_scene(width, height, faces, "normal", rng)
            save(spec, index, image, boxes, positions=tuple(positions))

    for index in range(1, syn.empty_images + 1):
        image, _ = render_scene(width, height, [], "normal", rng, distractors=True)
        save(CaptureSpec(set="kosong"), index, image, [])

    write_metadata(paths.metadata, rows)
    save_annotations(paths.annotations, annotations)
    paths.crops.mkdir(parents=True, exist_ok=True)
    return paths
