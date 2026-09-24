"""Nama detektor → objek detektor, dibaca dari config.

Mode:
- "operating": parameter bawaan config — untuk P/R/F1, IoU, FPPI (PRD §8.1)
- "ap": ambang skor rendah dari `evaluation.ap_run` — untuk kurva PR dan AP
  (PRD §8.2). Detektor tanpa skor (YCbCr) tidak punya mode ini.

`synthetic=True` mengganti semua detektor dengan FakeDetector (tanpa model).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pcdface.config import Config, ConfigError, HaarConfig, MediaPipeConfig, YCbCrConfig
from pcdface.detection.base import Detector

MODES = ("operating", "ap")


def supports_scores(name: str, cfg: Config) -> bool:
    """True bila detektor punya skor sehingga bisa dihitung AP-nya."""
    return not isinstance(cfg.detector(name), YCbCrConfig)


def build_detector(
    name: str,
    cfg: Config,
    mode: str = "operating",
    synthetic: bool = False,
    overrides: dict[str, Any] | None = None,
    running_mode: str = "image",
    keep_stages: bool = False,
) -> Detector:
    """Bangun detektor `name`. `overrides` mengganti parameter config (mis. equalize untuk E3)."""
    if mode not in MODES:
        raise ValueError(f"mode harus salah satu dari {MODES}")
    spec = cfg.detector(name)
    if overrides:
        unknown = set(overrides) - {f.name for f in dataclasses.fields(spec)}
        if unknown:
            raise ConfigError(f"parameter {sorted(unknown)} tidak dikenal untuk detektor {name}")
        spec = dataclasses.replace(spec, **overrides)
    if mode == "ap" and not supports_scores(name, cfg):
        raise ValueError(f"detektor {name} tidak punya skor, jadi tidak punya run AP")

    if synthetic:
        from pcdface.detection.fake import FakeDetector, profile_for

        return FakeDetector(name, profile_for(name, spec.type), score_mode=(mode == "ap"))

    ap = cfg.evaluation.ap_run
    if isinstance(spec, HaarConfig):
        from pcdface.detection.haar import HaarDetector

        if mode == "ap":
            spec = dataclasses.replace(spec, min_neighbors=ap.haar_min_neighbors)
        return HaarDetector(spec, name=name, score_mode=(mode == "ap"), nms_iou=ap.haar_nms_iou,
                            keep_stages=keep_stages)

    if isinstance(spec, MediaPipeConfig):
        from pcdface.detection.mediapipe_detector import MediaPipeDetector

        confidence = ap.mp_min_detection_confidence if mode == "ap" else spec.min_detection_confidence
        return MediaPipeDetector(
            cfg.paths.models / spec.model,
            name=name,
            min_detection_confidence=confidence,
            min_suppression_threshold=spec.min_suppression_threshold,
            running_mode=running_mode,
        )

    from pcdface.detection.ycbcr import YCbCrDetector

    return YCbCrDetector(spec, name=name, keep_stages=keep_stages)
