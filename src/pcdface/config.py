"""Muat dan validasi `configs/experiment.yaml`.

Galat konfigurasi dilaporkan sekaligus (bukan berhenti di galat pertama)
dengan path kunci yang jelas, mis. `detectors.haar.scale_factor`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from pcdface.paths import CONFIG_PATH, PROJECT_ROOT, ProjectPaths
from pcdface.pose import REFERENCE_EXPRESSION, REFERENCE_OCCLUSION, REFERENCE_POSE, parse_pose

SETS = ("jarak", "cahaya", "multi", "kosong", "pose", "ekspresi", "oklusi")
ENHANCEMENTS = ("none", "clahe")
DETECTOR_TYPES = ("haar", "mediapipe", "ycbcr", "yolo")
DELEGATES = ("auto", "gpu", "cpu")



class ConfigError(ValueError):
    """Konfigurasi tidak valid. Pesan memuat semua galat yang ditemukan."""


# ---------------------------------------------------------------------------
# Struktur konfigurasi
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureConfig:
    width: int
    height: int
    camera_index: int


@dataclass(frozen=True)
class DatasetConfig:
    sets: tuple[str, ...]
    distances_cm: tuple[int, ...]
    lightings: tuple[str, ...]
    poses: tuple[str, ...]
    reference_distance_cm: int
    frames_per_condition: int
    empty_images: int
    formations: dict[str, tuple[int, ...]]
    pose_distances_cm: tuple[int, ...] = (100,)
    expressions: tuple[str, ...] = (REFERENCE_EXPRESSION,)
    occlusions: tuple[str, ...] = (REFERENCE_OCCLUSION,)
    pose_frames_per_condition: int = 3


@dataclass(frozen=True)
class PreprocessingConfig:
    clahe_clip_limit: float
    clahe_tile_grid: tuple[int, int]


@dataclass(frozen=True)
class HaarConfig:
    scale_factor: float
    min_neighbors: int
    min_size: tuple[int, int]
    equalize: bool
    type: str = "haar"


@dataclass(frozen=True)
class MediaPipeConfig:
    model: str
    min_detection_confidence: float
    min_suppression_threshold: float
    delegate: str = "auto"          # auto | gpu | cpu — auto: GPU (Metal) di macOS, CPU di sistem lain
    type: str = "mediapipe"


@dataclass(frozen=True)
class YCbCrConfig:
    cb_range: tuple[int, int]
    cr_range: tuple[int, int]
    opening_kernel: int
    closing_kernel: int
    opening_iterations: int
    closing_iterations: int
    min_area_ratio: float
    max_area_ratio: float
    aspect_range: tuple[float, float]
    min_solidity: float
    type: str = "ycbcr"


@dataclass(frozen=True)
class YoloConfig:
    model: str                      # berkas .onnx di folder model
    input_size: int                 # sisi masukan letterbox (640 untuk YOLOv8-face)
    conf_threshold: float           # ambang keyakinan titik operasi (bawaan ultralytics 0,25)
    nms_iou: float                  # ambang IoU NMS (bawaan ultralytics 0,7)
    max_detections: int = 300       # bawaan ultralytics max_det
    type: str = "yolo"


DetectorConfig = HaarConfig | MediaPipeConfig | YCbCrConfig | YoloConfig


@dataclass(frozen=True)
class APRunConfig:
    mp_min_detection_confidence: float
    haar_min_neighbors: int
    haar_nms_iou: float
    yolo_conf_threshold: float = 0.05


@dataclass(frozen=True)
class EvaluationConfig:
    iou_primary: float
    iou_sensitivity: tuple[float, ...]
    recall_target: float
    ap_run: APRunConfig
    size_bins_px: tuple[float, ...]
    size_bins_ratio: tuple[float, ...]

    @property
    def iou_thresholds(self) -> tuple[float, ...]:
        """Ambang utama diikuti ambang sensitivitas, tanpa duplikat, terurut."""
        return tuple(sorted({self.iou_primary, *self.iou_sensitivity}))


@dataclass(frozen=True)
class E1Config:
    detectors: tuple[str, ...]
    resolutions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class E2Config:
    detectors: tuple[str, ...]


@dataclass(frozen=True)
class E3Config:
    detectors: tuple[str, ...]
    haar_equalize: tuple[bool, ...]
    enhancements: tuple[str, ...]


@dataclass(frozen=True)
class E4Config:
    detectors: tuple[str, ...]
    resolutions: tuple[tuple[int, int], ...]
    warmup_runs: int
    repeats: int
    sample_images: int


@dataclass(frozen=True)
class E5Config:
    enabled: bool
    scale_factors: tuple[float, ...]
    min_neighbors: tuple[int, ...]


@dataclass(frozen=True)
class E6Config:
    detectors: tuple[str, ...]


@dataclass(frozen=True)
class E7Config:
    detectors: tuple[str, ...]
    blur_px: tuple[int, ...]
    blur_angle_deg: float


@dataclass(frozen=True)
class E8Config:
    detector: str                   # detektor jalur ujung-ke-ujung (crop dari deteksi)


@dataclass(frozen=True)
class ExperimentsConfig:
    e1: E1Config
    e2: E2Config
    e3: E3Config
    e4: E4Config
    e5: E5Config
    e6: E6Config
    e7: E7Config
    e8: E8Config


@dataclass(frozen=True)
class RecognitionConfig:
    """Pengenalan identitas LBPH (PRD v4 §5.4)."""

    face_size: tuple[int, int]      # (lebar, tinggi) crop abu-abu
    max_distance: float             # jarak LBPH di atas ambang → "unknown"
    radius: int
    neighbors: int
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class StatsConfig:
    ci_level: float
    bootstrap_resamples: int


@dataclass(frozen=True)
class SyntheticConfig:
    subjects: int
    frames_per_condition: int
    empty_images: int
    focal_px: float
    face_width_cm: float


@dataclass(frozen=True)
class Config:
    seed: int
    paths: ProjectPaths
    capture: CaptureConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    detectors: dict[str, DetectorConfig]
    evaluation: EvaluationConfig
    experiments: ExperimentsConfig
    stats: StatsConfig
    synthetic: SyntheticConfig
    recognition: RecognitionConfig
    raw: dict[str, Any]
    source: Path | None = None

    def detector(self, name: str) -> DetectorConfig:
        try:
            return self.detectors[name]
        except KeyError:
            raise ConfigError(
                f"detektor '{name}' tidak ada di config. Pilihan: {', '.join(self.detectors)}"
            ) from None

    def scale_for(self, resolution: tuple[int, int]) -> float:
        """Faktor skala dari resolusi rekaman ke resolusi `resolution`."""
        return resolution[0] / self.capture.width


# ---------------------------------------------------------------------------
# Pembaca dengan pengumpul galat
# ---------------------------------------------------------------------------
class _Reader:
    """Membaca nilai bertipe dari mapping sambil mengumpulkan galat."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def section(self, mapping: Any, key: str, where: str) -> dict[str, Any]:
        value = mapping.get(key) if isinstance(mapping, Mapping) else None
        if not isinstance(value, Mapping):
            self.errors.append(f"{where}{key}: harus berupa bagian (mapping)")
            return {}
        return dict(value)

    def get(
        self,
        mapping: Mapping[str, Any],
        key: str,
        where: str,
        convert: Callable[[Any], Any],
        check: Callable[[Any], bool] | None = None,
        rule: str = "",
    ) -> Any:
        name = f"{where}{key}"
        if key not in mapping:
            self.errors.append(f"{name}: wajib ada")
            return None
        try:
            value = convert(mapping[key])
        except (TypeError, ValueError) as error:
            self.errors.append(f"{name}: nilai {mapping[key]!r} tidak valid ({error})")
            return None
        if check is not None and not check(value):
            self.errors.append(f"{name}: nilai {mapping[key]!r} tidak memenuhi syarat {rule}")
            return None
        return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("harus true/false")


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise TypeError("harus bilangan bulat")
    return int(value)


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("harus angka")
    return float(value)


def _str(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("harus teks tidak kosong")
    return value


def _list(convert: Callable[[Any], Any], length: int | None = None) -> Callable[[Any], tuple]:
    def inner(value: Any) -> tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError("harus daftar")
        if length is not None and len(value) != length:
            raise ValueError(f"harus {length} elemen")
        return tuple(convert(item) for item in value)

    return inner


def _pair_int(value: Any) -> tuple[int, int]:
    return _list(_int, 2)(value)  # type: ignore[return-value]


def _is_prob(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _is_open_prob(value: float) -> bool:
    return 0.0 < value <= 1.0


# ---------------------------------------------------------------------------
# Parser per bagian
# ---------------------------------------------------------------------------
def _parse_detector(r: _Reader, name: str, spec: Mapping[str, Any]) -> DetectorConfig | None:
    where = f"detectors.{name}."
    kind = spec.get("type")
    if kind not in DETECTOR_TYPES:
        r.errors.append(f"{where}type: harus salah satu dari {DETECTOR_TYPES}, dapat {kind!r}")
        return None

    if kind == "haar":
        values = dict(
            scale_factor=r.get(spec, "scale_factor", where, _float, lambda v: v > 1.0, "> 1"),
            min_neighbors=r.get(spec, "min_neighbors", where, _int, lambda v: v >= 0, ">= 0"),
            min_size=r.get(spec, "min_size", where, _pair_int, lambda v: min(v) > 0, "> 0"),
            equalize=r.get(spec, "equalize", where, _bool),
        )
        return None if None in values.values() else HaarConfig(**values)

    if kind == "mediapipe":
        values = dict(
            model=r.get(spec, "model", where, _str, lambda v: v.endswith(".tflite"), "berakhiran .tflite"),
            min_detection_confidence=r.get(spec, "min_detection_confidence", where, _float, _is_prob, "0–1"),
            min_suppression_threshold=r.get(spec, "min_suppression_threshold", where, _float, _is_prob, "0–1"),
        )
        if "delegate" in spec:
            values["delegate"] = r.get(spec, "delegate", where, _str, lambda v: v in DELEGATES, f"salah satu dari {DELEGATES}")
        return None if None in values.values() else MediaPipeConfig(**values)

    if kind == "yolo":
        values = dict(
            model=r.get(spec, "model", where, _str, lambda v: v.endswith(".onnx"), "berakhiran .onnx"),
            input_size=r.get(spec, "input_size", where, _int, lambda v: v >= 32 and v % 32 == 0, "kelipatan 32"),
            conf_threshold=r.get(spec, "conf_threshold", where, _float, _is_open_prob, "0 < x ≤ 1"),
            nms_iou=r.get(spec, "nms_iou", where, _float, _is_open_prob, "0 < x ≤ 1"),
        )
        if "max_detections" in spec:
            values["max_detections"] = r.get(spec, "max_detections", where, _int, lambda v: v >= 1, ">= 1")
        return None if None in values.values() else YoloConfig(**values)

    values = dict(
        cb_range=r.get(spec, "cb_range", where, _pair_int, lambda v: 0 <= v[0] < v[1] <= 255, "0 ≤ min < max ≤ 255"),
        cr_range=r.get(spec, "cr_range", where, _pair_int, lambda v: 0 <= v[0] < v[1] <= 255, "0 ≤ min < max ≤ 255"),
        opening_kernel=r.get(spec, "opening_kernel", where, _int, lambda v: v >= 1, ">= 1"),
        closing_kernel=r.get(spec, "closing_kernel", where, _int, lambda v: v >= 1, ">= 1"),
        opening_iterations=r.get(spec, "opening_iterations", where, _int, lambda v: v >= 0, ">= 0"),
        closing_iterations=r.get(spec, "closing_iterations", where, _int, lambda v: v >= 0, ">= 0"),
        min_area_ratio=r.get(spec, "min_area_ratio", where, _float, _is_prob, "0–1"),
        max_area_ratio=r.get(spec, "max_area_ratio", where, _float, _is_prob, "0–1"),
        aspect_range=r.get(spec, "aspect_range", where, _list(_float, 2), lambda v: 0 < v[0] < v[1], "0 < min < max"),
        min_solidity=r.get(spec, "min_solidity", where, _float, _is_prob, "0–1"),
    )
    return None if None in values.values() else YCbCrConfig(**values)


def _check_bins(r: _Reader, name: str, bins: tuple[float, ...] | None) -> None:
    if bins is None:
        return
    if len(bins) < 2 or any(later <= earlier for earlier, later in zip(bins, bins[1:])):
        r.errors.append(f"evaluation.{name}: minimal 2 batas dan harus naik tegas")


def parse_config(raw: Mapping[str, Any], root: Path = PROJECT_ROOT, source: Path | None = None) -> Config:
    """Validasi mapping mentah menjadi `Config`. Melempar ConfigError."""
    r = _Reader()
    if not isinstance(raw, Mapping):
        raise ConfigError("isi config harus berupa mapping YAML")

    seed = r.get(raw, "seed", "", _int, lambda v: v >= 0, ">= 0")

    paths_raw = r.section(raw, "paths", "")
    paths = None
    try:
        paths = ProjectPaths.from_mapping({k: str(v) for k, v in paths_raw.items()}, root)
    except KeyError as error:
        r.errors.append(f"paths: {error.args[0]}")

    cap = r.section(raw, "capture", "")
    capture = CaptureConfig(
        width=r.get(cap, "width", "capture.", _int, lambda v: v > 0, "> 0"),
        height=r.get(cap, "height", "capture.", _int, lambda v: v > 0, "> 0"),
        camera_index=r.get(cap, "camera_index", "capture.", _int, lambda v: v >= 0, ">= 0"),
    )

    ds = r.section(raw, "dataset", "")
    formations_raw = ds.get("formations")
    formations: dict[str, tuple[int, ...]] = {}
    if not isinstance(formations_raw, Mapping) or not formations_raw:
        r.errors.append("dataset.formations: harus mapping F1: [jarak, ...]")
    else:
        for key, value in formations_raw.items():
            positions = r.get(formations_raw, key, "dataset.formations.", _list(_int),
                              lambda v: 2 <= len(v) <= 4 and min(v) > 0, "2–4 jarak positif")
            if positions is not None:
                formations[str(key)] = positions
    dataset = DatasetConfig(
        sets=r.get(ds, "sets", "dataset.", _list(_str), lambda v: set(v) <= set(SETS), f"subset dari {SETS}"),
        distances_cm=r.get(ds, "distances_cm", "dataset.", _list(_int), lambda v: len(v) > 0 and min(v) > 0 and list(v) == sorted(set(v)), "positif, naik, unik"),
        lightings=r.get(ds, "lightings", "dataset.", _list(_str), lambda v: "normal" in v, "memuat 'normal'"),
        poses=r.get(ds, "poses", "dataset.", _list(_str)),
        reference_distance_cm=r.get(ds, "reference_distance_cm", "dataset.", _int, lambda v: v > 0, "> 0"),
        frames_per_condition=r.get(ds, "frames_per_condition", "dataset.", _int, lambda v: v > 0, "> 0"),
        empty_images=r.get(ds, "empty_images", "dataset.", _int, lambda v: v >= 0, ">= 0"),
        formations=formations,
        pose_distances_cm=r.get(ds, "pose_distances_cm", "dataset.", _list(_int),
                                lambda v: len(v) > 0 and list(v) == sorted(set(v)), "naik, unik"),
        expressions=r.get(ds, "expressions", "dataset.", _list(_str),
                          lambda v: REFERENCE_EXPRESSION in v, f"memuat '{REFERENCE_EXPRESSION}' sebagai acuan"),
        occlusions=r.get(ds, "occlusions", "dataset.", _list(_str),
                         lambda v: REFERENCE_OCCLUSION in v, f"memuat '{REFERENCE_OCCLUSION}' sebagai acuan"),
        pose_frames_per_condition=r.get(ds, "pose_frames_per_condition", "dataset.", _int, lambda v: v > 0, "> 0"),
    )
    for pose in dataset.poses or ():
        try:
            parse_pose(pose)
        except ValueError as error:
            r.errors.append(f"dataset.poses: {error}")
    if dataset.poses and REFERENCE_POSE not in dataset.poses:
        r.errors.append(f"dataset.poses: harus memuat '{REFERENCE_POSE}' sebagai acuan perbandingan E6")
    if dataset.distances_cm and dataset.pose_distances_cm and not set(dataset.pose_distances_cm) <= set(dataset.distances_cm):
        r.errors.append("dataset.pose_distances_cm: harus bagian dari distances_cm (tanda lakban yang sama)")
    if dataset.distances_cm and dataset.reference_distance_cm and dataset.reference_distance_cm not in dataset.distances_cm:
        r.errors.append("dataset.reference_distance_cm: harus salah satu dari distances_cm "
                        "(kondisi normal E3 diambil dari set jarak)")

    pre = r.section(raw, "preprocessing", "")
    preprocessing = PreprocessingConfig(
        clahe_clip_limit=r.get(pre, "clahe_clip_limit", "preprocessing.", _float, lambda v: v > 0, "> 0"),
        clahe_tile_grid=r.get(pre, "clahe_tile_grid", "preprocessing.", _pair_int, lambda v: min(v) > 0, "> 0"),
    )

    det_raw = r.section(raw, "detectors", "")
    detectors: dict[str, DetectorConfig] = {}
    for name, spec in det_raw.items():
        if not isinstance(spec, Mapping):
            r.errors.append(f"detectors.{name}: harus mapping berisi 'type'")
            continue
        parsed = _parse_detector(r, str(name), spec)
        if parsed is not None:
            detectors[str(name)] = parsed

    ev = r.section(raw, "evaluation", "")
    ap = r.section(ev, "ap_run", "evaluation.")
    ap_run = APRunConfig(
        mp_min_detection_confidence=r.get(ap, "mp_min_detection_confidence", "evaluation.ap_run.", _float, _is_prob, "0–1"),
        haar_min_neighbors=r.get(ap, "haar_min_neighbors", "evaluation.ap_run.", _int, lambda v: v >= 0, ">= 0"),
        haar_nms_iou=r.get(ap, "haar_nms_iou", "evaluation.ap_run.", _float, _is_open_prob, "0 < x ≤ 1"),
        yolo_conf_threshold=r.get(ap, "yolo_conf_threshold", "evaluation.ap_run.", _float, _is_open_prob, "0 < x ≤ 1"),
    )
    evaluation = EvaluationConfig(
        iou_primary=r.get(ev, "iou_primary", "evaluation.", _float, _is_open_prob, "0 < x ≤ 1"),
        iou_sensitivity=r.get(ev, "iou_sensitivity", "evaluation.", _list(_float), lambda v: all(map(_is_open_prob, v)), "0 < x ≤ 1"),
        recall_target=r.get(ev, "recall_target", "evaluation.", _float, _is_open_prob, "0 < x ≤ 1"),
        ap_run=ap_run,
        size_bins_px=r.get(ev, "size_bins_px", "evaluation.", _list(_float)),
        size_bins_ratio=r.get(ev, "size_bins_ratio", "evaluation.", _list(_float)),
    )
    _check_bins(r, "size_bins_px", evaluation.size_bins_px)
    _check_bins(r, "size_bins_ratio", evaluation.size_bins_ratio)

    ex = r.section(raw, "experiments", "")
    names = _list(_str)
    resolutions = _list(_pair_int)

    def section(key: str) -> dict[str, Any]:
        return r.section(ex, key, "experiments.")

    e1, e2, e3, e4, e5, e6, e7, e8 = (section(k) for k in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"))
    experiments = ExperimentsConfig(
        e1=E1Config(
            detectors=r.get(e1, "detectors", "experiments.e1.", names),
            resolutions=r.get(e1, "resolutions", "experiments.e1.", resolutions),
        ),
        e2=E2Config(detectors=r.get(e2, "detectors", "experiments.e2.", names)),
        e3=E3Config(
            detectors=r.get(e3, "detectors", "experiments.e3.", names),
            haar_equalize=r.get(e3, "haar_equalize", "experiments.e3.", _list(_bool)),
            enhancements=r.get(e3, "enhancements", "experiments.e3.", _list(_str),
                               lambda v: set(v) <= set(ENHANCEMENTS), f"subset dari {ENHANCEMENTS}"),
        ),
        e4=E4Config(
            detectors=r.get(e4, "detectors", "experiments.e4.", names),
            resolutions=r.get(e4, "resolutions", "experiments.e4.", resolutions),
            warmup_runs=r.get(e4, "warmup_runs", "experiments.e4.", _int, lambda v: v >= 1, ">= 1 (PRD §8.5)"),
            repeats=r.get(e4, "repeats", "experiments.e4.", _int, lambda v: v >= 1, ">= 1"),
            sample_images=r.get(e4, "sample_images", "experiments.e4.", _int, lambda v: v >= 1, ">= 1"),
        ),
        e5=E5Config(
            enabled=r.get(e5, "enabled", "experiments.e5.", _bool),
            scale_factors=r.get(e5, "scale_factors", "experiments.e5.", _list(_float), lambda v: len(v) > 0 and min(v) > 1, "> 1"),
            min_neighbors=r.get(e5, "min_neighbors", "experiments.e5.", _list(_int), lambda v: len(v) > 0 and min(v) >= 0, ">= 0"),
        ),
        e6=E6Config(detectors=r.get(e6, "detectors", "experiments.e6.", names)),
        e7=E7Config(
            detectors=r.get(e7, "detectors", "experiments.e7.", names),
            blur_px=r.get(e7, "blur_px", "experiments.e7.", _list(_int),
                          lambda v: len(v) > 0 and min(v) >= 0 and 0 in v and list(v) == sorted(set(v)),
                          "naik, unik, ≥ 0, memuat 0 sebagai acuan"),
            blur_angle_deg=r.get(e7, "blur_angle_deg", "experiments.e7.", _float),
        ),
        e8=E8Config(detector=r.get(e8, "detector", "experiments.e8.", _str)),
    )

    for key, spec in (("e1", experiments.e1), ("e2", experiments.e2), ("e3", experiments.e3), ("e4", experiments.e4),
                      ("e6", experiments.e6), ("e7", experiments.e7)):


        for name in spec.detectors or ():
            if name not in det_raw:
                r.errors.append(f"experiments.{key}.detectors: '{name}' tidak ada di bagian detectors")
    if experiments.e8.detector and experiments.e8.detector not in det_raw:
        r.errors.append(f"experiments.e8.detector: '{experiments.e8.detector}' tidak ada di bagian detectors")
    if experiments.e3.detectors and "haar" in experiments.e3.detectors and not experiments.e3.haar_equalize:
        r.errors.append("experiments.e3.haar_equalize: tidak boleh kosong bila haar diuji")
    if capture.width and capture.height:
        for key, res in (("e1", experiments.e1.resolutions), ("e4", experiments.e4.resolutions)):
            for w, h in res or ():
                if w * capture.height != h * capture.width:
                    r.errors.append(
                        f"experiments.{key}.resolutions: {w}×{h} tidak sebanding dengan "
                        f"{capture.width}×{capture.height} — perkecilan harus menjaga rasio aspek (PRD §9 E1)"
                    )

    st = r.section(raw, "stats", "")
    stats = StatsConfig(
        ci_level=r.get(st, "ci_level", "stats.", _float, lambda v: 0 < v < 1, "0 < x < 1"),
        bootstrap_resamples=r.get(st, "bootstrap_resamples", "stats.", _int, lambda v: v >= 100, ">= 100"),
    )

    sy = r.section(raw, "synthetic", "")
    synthetic = SyntheticConfig(
        subjects=r.get(sy, "subjects", "synthetic.", _int, lambda v: v >= 1, ">= 1"),
        frames_per_condition=r.get(sy, "frames_per_condition", "synthetic.", _int, lambda v: v >= 1, ">= 1"),
        empty_images=r.get(sy, "empty_images", "synthetic.", _int, lambda v: v >= 0, ">= 0"),
        focal_px=r.get(sy, "focal_px", "synthetic.", _float, lambda v: v > 0, "> 0"),
        face_width_cm=r.get(sy, "face_width_cm", "synthetic.", _float, lambda v: v > 0, "> 0"),
    )

    rc = r.section(raw, "recognition", "")
    recognition = RecognitionConfig(
        face_size=r.get(rc, "face_size", "recognition.", _pair_int, lambda v: min(v) >= 16, ">= 16"),
        max_distance=r.get(rc, "max_distance", "recognition.", _float, lambda v: v > 0, "> 0"),
        radius=r.get(rc, "radius", "recognition.", _int, lambda v: v >= 1, ">= 1"),
        neighbors=r.get(rc, "neighbors", "recognition.", _int, lambda v: 1 <= v <= 32, "1–32"),
        grid_x=r.get(rc, "grid_x", "recognition.", _int, lambda v: v >= 1, ">= 1"),
        grid_y=r.get(rc, "grid_y", "recognition.", _int, lambda v: v >= 1, ">= 1"),
    )

    if r.errors:
        where = f" ({source})" if source else ""
        raise ConfigError(f"config tidak valid{where}:\n  - " + "\n  - ".join(r.errors))

    assert paths is not None and seed is not None
    return Config(
        seed=seed,
        paths=paths,
        capture=capture,
        dataset=dataset,
        preprocessing=preprocessing,
        detectors=detectors,
        evaluation=evaluation,
        experiments=experiments,
        stats=stats,
        synthetic=synthetic,
        recognition=recognition,
        raw=copy.deepcopy(dict(raw)),
        source=source,
    )


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    """Baca YAML lalu validasi. Bawaan: `configs/experiment.yaml`."""
    source = Path(path) if path is not None else CONFIG_PATH
    if not source.exists():
        raise ConfigError(f"berkas config tidak ditemukan: {source}")
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw, root=root, source=source)
