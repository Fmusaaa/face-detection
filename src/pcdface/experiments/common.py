"""Lapisan bersama eksperimen: jalankan detektor, catat deteksi, hitung metrik + interval.

Alur tiap eksperimen:
1. `run_records` — jalankan satu detektor pada sampel (setelah enhancement dan
   perkecilan) dan simpan kotak, skor, waktu per citra.
2. `OperatingSummary` / `APSummary` — metrik titik operasi dan AP dari
   catatan itu, dengan Wilson untuk proporsi dan bootstrap kelompok untuk F1/AP.
3. `paired` — selisih dua catatan pada citra yang sama (bootstrap berpasangan);
   `paired_conditions` — selisih dua kondisi (citra berbeda) pada peserta yang sama.

Kelompok bootstrap diambil dari `MetadataRow.group`: kode subjek untuk set satu
wajah, nama berkas untuk multi-wajah dan kosong (PRD §8.6).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from pcdface.boxes import Box
from pcdface.config import Config
from pcdface.dataset.loader import Sample
from pcdface.dataset.resize import scale_boxes, scale_image
from pcdface.detection.base import Detector
from pcdface.evaluation.average_precision import ScoredImage, pool, score_image
from pcdface.evaluation.operating_point import ImageEval, aggregate, evaluate_image
from pcdface.evaluation.stats import NAN_ESTIMATE, Estimate, cluster_bootstrap, paired_difference, wilson
from pcdface.preprocessing import enhance


@dataclass
class Record:
    """Hasil satu detektor pada satu citra."""

    key: str
    group: str
    set: str
    gt: list[Box]
    boxes: list[Box]
    scores: list[float] | None
    elapsed_ms: float
    width: int
    height: int
    distance_cm: int | None = None
    lighting: str = "normal"
    formation: str = ""
    positions_cm: tuple[int, ...] = ()
    luma_mean: float | None = None
    session: str = ""
    pose: str = ""
    expression: str = ""
    occlusion: str = ""
    labels: dict[str, str] = field(default_factory=dict)  # detektor, mode, resolusi, enhancement

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def run_records(
    detector: Detector,
    samples: Sequence[Sample],
    cfg: Config,
    scale: float = 1.0,
    enhancement: str = "none",
    labels: dict[str, str] | None = None,
    degrade: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Record]:
    """Jalankan detektor pada sampel. Enhancement, degradasi (mis. blur E7), dan
    perkecilan tidak ikut diukur waktunya."""
    pre = cfg.preprocessing
    records = []
    for sample in samples:
        image = enhance(sample.load(), enhancement, pre.clahe_clip_limit, pre.clahe_tile_grid)
        if degrade is not None:
            image = degrade(image)
        image = scale_image(image, scale)
        result = detector.detect(image)
        meta = sample.meta
        records.append(Record(
            key=sample.key, group=meta.group, set=meta.set,
            gt=scale_boxes(sample.gt, scale), boxes=list(result.boxes),
            scores=list(result.scores) if result.scores is not None else None,
            elapsed_ms=result.elapsed_ms, width=image.shape[1], height=image.shape[0],
            distance_cm=meta.distance_cm, lighting=meta.lighting, formation=meta.formation,
            positions_cm=tuple(meta.positions_cm), luma_mean=meta.luma_mean, session=meta.session,
            pose=meta.pose, expression=meta.expression, occlusion=meta.occlusion,

            labels=dict(labels or {}),
        ))
    return records


def append_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


# ---------------------------------------------------------------------------
# Titik operasi
# ---------------------------------------------------------------------------
def evaluate(records: Sequence[Record], iou: float) -> list[ImageEval]:
    return [evaluate_image(r.key, r.group, r.gt, r.boxes, iou) for r in records]


class GroupedCounts:
    """TP/FP/FN per kelompok supaya statistik bootstrap cepat dihitung."""

    def __init__(self, evals: Sequence[ImageEval]) -> None:
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for e in evals:
            c = counts[e.group]
            c[0] += e.tp
            c[1] += e.fp
            c[2] += e.fn
            c[3] += 1
        self.counts = dict(counts)
        self.groups = sorted(self.counts)

    def _sum(self, keys: Sequence[str]) -> tuple[int, int, int, int]:
        tp = fp = fn = n = 0
        for key in keys:
            c = self.counts.get(key)
            if c:
                tp, fp, fn, n = tp + c[0], fp + c[1], fn + c[2], n + c[3]
        return tp, fp, fn, n

    def f1(self, keys: Sequence[str]) -> float:
        tp, fp, fn, _ = self._sum(keys)
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom else math.nan

    def recall(self, keys: Sequence[str]) -> float:
        tp, _, fn, _ = self._sum(keys)
        return tp / (tp + fn) if tp + fn else math.nan

    def precision(self, keys: Sequence[str]) -> float:
        tp, fp, _, _ = self._sum(keys)
        return tp / (tp + fp) if tp + fp else math.nan


@dataclass
class OperatingSummary:
    images: int
    faces: int
    tp: int
    fp: int
    fn: int
    precision: Estimate
    recall: Estimate
    f1: Estimate
    mean_iou: float
    fppi: float

    def row(self) -> dict[str, float]:
        out: dict[str, float] = {"citra": self.images, "wajah": self.faces, "TP": self.tp, "FP": self.fp, "FN": self.fn}
        out.update(self.precision.as_dict("precision"))
        out.update(self.recall.as_dict("recall"))
        out.update(self.f1.as_dict("f1"))
        out["rerata_iou"] = self.mean_iou
        out["fppi_berwajah"] = self.fppi  # deteksi palsu per citra pada citra yang dievaluasi (berwajah)
        return out


def operating_summary(records: Sequence[Record], iou: float, cfg: Config, bootstrap: bool = True) -> OperatingSummary:
    """P/R (Wilson), F1 (bootstrap kelompok), rerata IoU, FPPI."""
    evals = evaluate(records, iou)
    op = aggregate(evals)
    level = cfg.stats.ci_level
    f1 = Estimate(op.f1, math.nan, math.nan)
    if bootstrap and evals:
        grouped = GroupedCounts(evals)
        f1 = cluster_bootstrap(grouped.groups, grouped.f1, cfg.stats.bootstrap_resamples, level, cfg.seed)
    return OperatingSummary(
        images=op.images, faces=op.n_gt, tp=op.tp, fp=op.fp, fn=op.fn,
        precision=wilson(op.tp, op.tp + op.fp, level),
        recall=wilson(op.tp, op.tp + op.fn, level),
        f1=f1, mean_iou=op.mean_iou, fppi=op.fppi,
    )


def fppi(records: Sequence[Record], iou: float) -> float:
    return aggregate(evaluate(records, iou)).fppi


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------
def scored(records: Sequence[Record], iou: float) -> list[ScoredImage]:
    out = []
    for r in records:
        if r.scores is None:
            raise ValueError(f"{r.key}: detektor tanpa skor tidak bisa dihitung AP-nya")
        out.append(score_image(r.key, r.group, r.gt, r.boxes, r.scores, iou))
    return out


class GroupedScored:
    def __init__(self, images: Sequence[ScoredImage]) -> None:
        by_group: dict[str, list[ScoredImage]] = defaultdict(list)
        for image in images:
            by_group[image.group].append(image)
        self.by_group = dict(by_group)
        self.groups = sorted(self.by_group)

    def ap(self, keys: Sequence[str]) -> float:
        return pool([image for key in keys for image in self.by_group.get(key, [])]).ap


def ap_summary(records: Sequence[Record], iou: float, cfg: Config) -> tuple[Estimate, np.ndarray, np.ndarray]:
    """AP + interval bootstrap kelompok, serta kurva PR (recall, precision)."""
    images = scored(records, iou)
    curve = pool(images)
    grouped = GroupedScored(images)
    estimate = cluster_bootstrap(grouped.groups, grouped.ap, cfg.stats.bootstrap_resamples,
                                 cfg.stats.ci_level, cfg.seed) if images else NAN_ESTIMATE
    return estimate, curve.recall, curve.precision


def thin_curve(recall: np.ndarray, precision: np.ndarray, max_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Kurangi titik kurva PR untuk disimpan, tetap menyertakan titik ujung."""
    if recall.size <= max_points:
        return recall, precision
    index = np.unique(np.linspace(0, recall.size - 1, max_points).astype(int))
    return recall[index], precision[index]


# ---------------------------------------------------------------------------
# Perbandingan berpasangan
# ---------------------------------------------------------------------------
def paired(
    records_a: Sequence[Record],
    records_b: Sequence[Record],
    iou: float,
    cfg: Config,
    metric: str = "f1",
) -> Estimate:
    """Selisih metrik A − B pada citra yang sama, bootstrap kelompok berpasangan."""
    keys_a, keys_b = [r.key for r in records_a], [r.key for r in records_b]
    if sorted(keys_a) != sorted(keys_b):
        raise ValueError("perbandingan berpasangan butuh citra yang sama")
    if metric == "ap":
        a, b = GroupedScored(scored(records_a, iou)), GroupedScored(scored(records_b, iou))
        stat_a: Callable[[Sequence[str]], float] = a.ap
        stat_b: Callable[[Sequence[str]], float] = b.ap
        groups = a.groups
    else:
        a, b = GroupedCounts(evaluate(records_a, iou)), GroupedCounts(evaluate(records_b, iou))
        stat_a, stat_b = getattr(a, metric), getattr(b, metric)
        groups = a.groups
    return paired_difference(groups, stat_a, stat_b, cfg.stats.bootstrap_resamples, cfg.stats.ci_level, cfg.seed)


def paired_conditions(
    records_a: Sequence[Record],
    records_b: Sequence[Record],
    iou: float,
    cfg: Config,
    metric: str = "recall",
) -> Estimate:
    """Selisih metrik kondisi A − kondisi B untuk detektor yang sama.

    Citranya berbeda (mis. pose kiri60 vs depan), jadi pasangannya adalah
    **kelompok** (peserta): hanya peserta yang punya kedua kondisi yang dipakai,
    dan setiap resampel bootstrap memilih peserta yang sama untuk A dan B.
    """
    a, b = GroupedCounts(evaluate(records_a, iou)), GroupedCounts(evaluate(records_b, iou))
    groups = sorted(set(a.groups) & set(b.groups))
    if not groups:
        return NAN_ESTIMATE
    return paired_difference(groups, getattr(a, metric), getattr(b, metric),
                             cfg.stats.bootstrap_resamples, cfg.stats.ci_level, cfg.seed)


def verdict(estimate: Estimate) -> str:

    """Kalimat kesimpulan menurut aturan PRD §8.6 / §12.2 butir 5."""
    if math.isnan(estimate.low):
        return "tidak dapat dihitung"
    if estimate.excludes_zero:
        return "berbeda (A lebih tinggi)" if estimate.value > 0 else "berbeda (B lebih tinggi)"
    return "tidak dapat disimpulkan"
