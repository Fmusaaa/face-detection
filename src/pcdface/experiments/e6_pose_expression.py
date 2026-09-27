"""E6 — Pose dan ekspresi (PRD §9, RQ6, H5).

Data: set pose (tiap pose di setiap `pose_distances_cm`), set ekspresi (jarak
acuan), set kosong untuk FPPI. Resolusi asli, tanpa enhancement, parameter
detektor bawaan (titik operasi). Bootstrap per peserta.

Yang diukur: di pose/ekspresi mana detektor mulai kehilangan wajah, dan
seberapa jauh recall turun dibanding acuan (`depan` / `netral`) pada peserta
yang sama. Ekspresi hanyalah kondisi yang diperagakan peserta — sistem tidak
menebak ekspresi.

Tabel:
- e6_ringkasan            recall seluruh set pose / ekspresi dan FPPI kosong per detektor
- e6_pose                 recall (Wilson), FP, rerata IoU & skor per detektor × jarak × pose
- e6_per_sumbu            recall per sumbu × sudut, kiri+kanan digabung (grafik 9)
- e6_batas_sudut          sudut terbesar dengan recall ≥ target, per detektor × jarak × sumbu
- e6_pose_vs_depan        selisih recall pose − depan, berpasangan per peserta
- e6_ekspresi             recall per detektor × ekspresi (grafik 10)
- e6_ekspresi_vs_netral   selisih recall ekspresi − netral, berpasangan per peserta
- e6_perbandingan         selisih recall antar detektor pada citra yang sama, per kondisi
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pcdface.evaluation.matching import iou as box_iou
from pcdface.evaluation.stats import Estimate, wilson
from pcdface.experiments.common import (
    Record,
    evaluate,
    fppi,
    paired,
    paired_conditions,
    run_records,
    verdict,
)
from pcdface.experiments.runner import ExperimentError, RunContext
from pcdface.pose import AXIS_ORDER, REFERENCE_EXPRESSION, REFERENCE_POSE, parse_pose


def hit_score(record: Record, iou: float) -> float:
    """Skor deteksi terbaik yang cocok dengan wajah acuan; NaN bila meleset atau tanpa skor."""
    if record.scores is None or not record.gt:
        return math.nan
    matched = [s for box, s in zip(record.boxes, record.scores) if box_iou(record.gt[0], box) >= iou]
    return max(matched) if matched else math.nan


def condition_row(records: Sequence[Record], iou: float, level: float) -> dict[str, float]:
    evals = evaluate(records, iou)
    tp, fp, fn = (sum(getattr(e, k) for e in evals) for k in ("tp", "fp", "fn"))
    ious = [value for e in evals for value in e.ious]
    scores = [s for s in (hit_score(r, iou) for r in records) if not math.isnan(s)]
    row: dict[str, float] = {"citra": len(records), "peserta": len({r.group for r in records}),
                             "TP": tp, "FP": fp, "FN": fn}
    row.update(wilson(tp, tp + fn, level).as_dict("recall"))
    row.update(wilson(tp, tp + fp, level).as_dict("precision"))
    row["rerata_iou"] = float(np.mean(ious)) if ious else math.nan
    row["rerata_skor"] = float(np.mean(scores)) if scores else math.nan  # MediaPipe saja; Haar tanpa skor
    return row


def versus_reference(estimate: Estimate) -> str:
    """Kesimpulan selisih kondisi − acuan (aturan PRD §12.2 butir 5)."""
    if math.isnan(estimate.low):
        return "tidak dapat dihitung"
    if estimate.excludes_zero:
        return "lebih rendah dari acuan" if estimate.value < 0 else "lebih tinggi dari acuan"
    return "tidak dapat disimpulkan"


def angle_limit(recalls: dict[int, tuple[float, float]], target: float) -> tuple[int | None, int | None]:
    """(sudut maks titik, sudut maks konservatif): sudut terbesar yang semua sudut ≤-nya
    punya recall ≥ target — titik memakai nilai recall, konservatif memakai batas bawah Wilson."""
    out: list[int | None] = []
    for use_low in (False, True):
        limit = None
        for angle in sorted(recalls):
            value = recalls[angle][1] if use_low else recalls[angle][0]
            if math.isnan(value) or value < target:
                break
            limit = angle
        out.append(limit)
    return out[0], out[1]


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    ds, ev = cfg.dataset, cfg.evaluation
    iou, level = ev.iou_primary, cfg.stats.ci_level
    samples = ctx.samples(("pose", "ekspresi", "kosong"))
    if not any(s.meta.set in ("pose", "ekspresi") for s in samples):
        raise ExperimentError("set pose dan ekspresi belum punya citra beranotasi")
    detectors = list(cfg.experiments.e6.detectors)

    runs: dict[str, list[Record]] = {}
    for name in detectors:
        with ctx.detector(name) as detector:
            records = run_records(detector, samples, cfg, labels={"detektor": name, "mode": "operating"})
        runs[name] = records
        ctx.dump(records)
        ctx.log(f"{name} selesai")

    def select(name: str, keep: Callable[[Record], bool]) -> list[Record]:
        return [r for r in runs[name] if keep(r)]

    poses = [p for p in ds.poses if any(s.meta.set == "pose" and s.meta.pose == p for s in samples)]
    distances = [d for d in ds.pose_distances_cm
                 if any(s.meta.set == "pose" and s.meta.distance_cm == d for s in samples)]
    expressions = [e for e in ds.expressions if any(s.meta.expression == e for s in samples)]

    def pose_records(name: str, distance: int, keep: Callable[[str], bool]) -> list[Record]:
        return select(name, lambda r: r.set == "pose" and r.distance_cm == distance and keep(r.pose))

    def expression_records(name: str, expression: str | None = None) -> list[Record]:
        return select(name, lambda r: r.set == "ekspresi" and (expression is None or r.expression == expression))

    # --- ringkasan ---------------------------------------------------------
    summary = []
    for name in detectors:
        row: dict[str, float | str] = {"detektor": name}
        for label, subset in (("pose", select(name, lambda r: r.set == "pose")), ("ekspresi", expression_records(name))):
            evals = evaluate(subset, iou)
            tp, fn = sum(e.tp for e in evals), sum(e.fn for e in evals)
            row.update(wilson(tp, tp + fn, level).as_dict(f"recall_{label}"))
        row["fppi_kosong"] = fppi(select(name, lambda r: r.set == "kosong"), iou)
        summary.append(row)
    ctx.table(pd.DataFrame(summary), "e6_ringkasan", "E6 — ringkasan pose dan ekspresi",
              f"IoU {iou}. Recall: Wilson {level:.0%}. Parameter detektor bawaan, tanpa enhancement.")

    # --- pose ----------------------------------------------------------------
    if poses:
        pose_rows, versus_rows, axis_rows, limit_rows = [], [], [], []
        for name in detectors:
            for distance in distances:
                reference = pose_records(name, distance, lambda p: p == REFERENCE_POSE)
                for pose in poses:
                    subset = pose_records(name, distance, lambda p, pose=pose: p == pose)
                    if not subset:
                        continue
                    info = parse_pose(pose)
                    pose_rows.append({"detektor": name, "jarak_cm": distance, "pose": pose, "sumbu": info.axis,
                                      "sudut": info.angle, **condition_row(subset, iou, level)})
                    if pose != REFERENCE_POSE and reference:
                        diff = paired_conditions(subset, reference, iou, cfg, metric="recall")
                        versus_rows.append({"detektor": name, "jarak_cm": distance, "pose": pose,
                                            **diff.as_dict("selisih_recall"), "kesimpulan": versus_reference(diff)})

                for axis in AXIS_ORDER[1:]:
                    angles = sorted({parse_pose(p).angle for p in poses if parse_pose(p).axis == axis})
                    if not angles:
                        continue
                    recalls: dict[int, tuple[float, float]] = {}
                    for angle in [0, *angles]:
                        keep = (lambda p: p == REFERENCE_POSE) if angle == 0 else \
                            (lambda p, a=angle: parse_pose(p).axis == axis and parse_pose(p).angle == a)
                        subset = pose_records(name, distance, keep)
                        if not subset:
                            continue
                        row = condition_row(subset, iou, level)
                        axis_rows.append({"detektor": name, "jarak_cm": distance, "sumbu": axis, "sudut": angle, **row})
                        recalls[angle] = (row["recall"], row["recall_low"])
                    point, conservative = angle_limit(recalls, ev.recall_target)
                    limit_rows.append({"detektor": name, "jarak_cm": distance, "sumbu": axis,
                                       "sudut_diuji": ", ".join(str(a) for a in angles),
                                       "sudut_maks": point, "sudut_maks_konservatif": conservative})
        ctx.table(pd.DataFrame(pose_rows), "e6_pose", "E6 — recall per detektor × jarak × pose",
                  "kiri/kanan menurut peserta. Sudut nominal (±10°). rerata_skor: skor MediaPipe pada wajah yang "
                  "terdeteksi — turun sebelum wajah hilang.")
        ctx.table(pd.DataFrame(axis_rows), "e6_per_sumbu", "E6 — recall per sumbu × sudut (kiri dan kanan digabung)",
                  "Sudut 0 = pose depan. menoleh = yaw, mengangguk = pitch (menunduk + mendongak), miring = roll.")
        limits = pd.DataFrame(limit_rows).astype({"sudut_maks": "Int64", "sudut_maks_konservatif": "Int64"})
        ctx.table(limits, "e6_batas_sudut",
                  f"E6 — sudut terbesar dengan recall ≥ {ev.recall_target:g}",
                  "Semua sudut dari 0 sampai batas ini memenuhi target. Konservatif memakai batas bawah Wilson. "
                  "Kosong = sudah gagal sejak sudut uji pertama (atau pose depan).")
        ctx.table(pd.DataFrame(versus_rows), "e6_pose_vs_depan", "E6 — selisih recall pose − depan (peserta sama)",
                  "Bootstrap berpasangan per peserta; acuan = pose depan di jarak yang sama.")

    # --- ekspresi --------------------------------------------------------------
    if expressions:
        expression_rows, versus_rows = [], []
        for name in detectors:
            reference = expression_records(name, REFERENCE_EXPRESSION)
            for expression in expressions:
                subset = expression_records(name, expression)
                expression_rows.append({"detektor": name, "ekspresi": expression, **condition_row(subset, iou, level)})
                if expression != REFERENCE_EXPRESSION and reference:
                    diff = paired_conditions(subset, reference, iou, cfg, metric="recall")
                    versus_rows.append({"detektor": name, "ekspresi": expression,
                                        **diff.as_dict("selisih_recall"), "kesimpulan": versus_reference(diff)})

        ctx.table(pd.DataFrame(expression_rows), "e6_ekspresi", "E6 — recall per detektor × ekspresi",
                  f"Di {ds.reference_distance_cm} cm, wajah menghadap kamera. Ekspresi diperagakan peserta, "
                  "bukan ditebak sistem.")
        ctx.table(pd.DataFrame(versus_rows), "e6_ekspresi_vs_netral",
                  "E6 — selisih recall ekspresi − netral (peserta sama)", "Bootstrap berpasangan per peserta.")

    # --- antar detektor pada citra yang sama -----------------------------------
    conditions: list[tuple[str, str, Callable[[Record], bool]]] = []
    for distance in distances:
        for pose in poses:
            conditions.append((f"pose {distance} cm", pose,
                               lambda r, d=distance, p=pose: r.set == "pose" and r.distance_cm == d and r.pose == p))
    for expression in expressions:
        conditions.append(("ekspresi", expression, lambda r, e=expression: r.set == "ekspresi" and r.expression == e))
    compare_rows = []
    for a, b in itertools.combinations(detectors, 2):
        for group, condition, keep in conditions:
            diff = paired(select(a, keep), select(b, keep), iou, cfg, metric="recall")
            compare_rows.append({"kelompok": group, "kondisi": condition, "A": a, "B": b,
                                 **diff.as_dict("selisih_recall"), "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(compare_rows), "e6_perbandingan", "E6 — selisih recall A − B per kondisi",
              "Citra yang sama untuk kedua detektor, bootstrap berpasangan per peserta.")
