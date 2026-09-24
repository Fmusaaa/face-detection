"""E2 — Multi-wajah (PRD §9, RQ2).

Data: set multi (+ set kosong untuk FPPI), resolusi asli. Bootstrap per citra.

Tabel:
- e2_ringkasan          P/R/F1 (+IoU sensitivitas), rerata IoU, FPPI, AP, akurasi & MAE hitung
- e2_per_formasi        akurasi hitung, MAE, recall per formasi
- e2_recall_per_posisi  recall per posisi jarak (pasangan kotak kiri→kanan ke positions_cm)
- e2_perbandingan       selisih berpasangan antar detektor (F1, recall, AP, akurasi hitung)
- e2_kurva_pr           titik kurva PR
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pandas as pd

from pcdface.detection.registry import supports_scores
from pcdface.evaluation.multiface import count_accuracy, count_mae, position_hits
from pcdface.evaluation.stats import paired_difference, wilson
from pcdface.experiments.common import (
    Record,
    ap_summary,
    evaluate,
    fppi,
    operating_summary,
    paired,
    run_records,
    thin_curve,
    verdict,
)
from pcdface.experiments.runner import RunContext


def _count_row(records: list[Record], level: float) -> dict[str, float]:
    predicted = [len(r.boxes) for r in records]
    actual = [len(r.gt) for r in records]
    k, n = count_accuracy(predicted, actual)
    return {**wilson(k, n, level).as_dict("akurasi_hitung"), "mae_hitung": count_mae(predicted, actual)}


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    ev = cfg.evaluation
    level = cfg.stats.ci_level
    samples = ctx.samples(("multi", "kosong"))
    ctx.require(samples, "multi")

    op: dict[str, list[Record]] = {}
    ap_runs: dict[str, list[Record]] = {}
    for name in cfg.experiments.e2.detectors:
        with ctx.detector(name) as detector:
            op[name] = run_records(detector, samples, cfg, labels={"detektor": name, "mode": "operating"})
        ctx.dump(op[name])
        if supports_scores(name, cfg):
            with ctx.detector(name, "ap") as detector:
                ap_runs[name] = run_records(detector, samples, cfg, labels={"detektor": name, "mode": "ap"})
            ctx.dump(ap_runs[name])
        ctx.log(f"{name} selesai")

    def multi(records: list[Record]) -> list[Record]:
        return [r for r in records if r.set == "multi"]

    def empties(records: list[Record]) -> list[Record]:
        return [r for r in records if r.set == "kosong"]

    summary_rows, pr_rows, formation_rows, position_rows = [], [], [], []
    for name, records in op.items():
        for iou in ev.iou_thresholds:
            s = operating_summary(multi(records), iou, cfg)
            row = {"detektor": name, "iou": iou, **s.row(), "fppi_kosong": fppi(empties(records), iou),
                   **_count_row(multi(records), level), "ap": math.nan, "ap_low": math.nan, "ap_high": math.nan}
            if iou == ev.iou_primary and name in ap_runs:
                estimate, recall, precision = ap_summary(multi(ap_runs[name]), iou, cfg)
                row.update(estimate.as_dict("ap"))
                for r_value, p_value in zip(*thin_curve(recall, precision)):
                    pr_rows.append({"eksperimen": "e2", "detektor": name, "resolusi": "asli",
                                    "recall": float(r_value), "precision": float(p_value), "ap": estimate.value})
            summary_rows.append(row)

        by_formation: dict[str, list[Record]] = defaultdict(list)
        for record in multi(records):
            by_formation[record.formation].append(record)
        for formation in sorted(by_formation):
            subset = by_formation[formation]
            evals = evaluate(subset, ev.iou_primary)
            tp, n = sum(e.tp for e in evals), sum(e.n_gt for e in evals)
            formation_rows.append({
                "detektor": name, "formasi": formation, "posisi_cm": ";".join(map(str, subset[0].positions_cm)),
                "citra": len(subset), **_count_row(subset, level), **wilson(tp, n, level).as_dict("recall"),
            })

        hits: dict[int, list[bool]] = defaultdict(list)
        for record, e in zip(multi(records), evaluate(multi(records), ev.iou_primary)):
            for distance, hit in position_hits(record.gt, e.gt_matched, record.positions_cm):
                hits[distance].append(hit)
        for distance in sorted(hits):
            values = hits[distance]
            position_rows.append({"detektor": name, "jarak_cm": distance, "wajah": len(values),
                                  **wilson(sum(values), len(values), level).as_dict("recall")})

    ctx.table(pd.DataFrame(summary_rows), "e2_ringkasan", "E2 — ringkasan multi-wajah",
              "P, R, akurasi hitung: Wilson; F1 dan AP: bootstrap per citra. Akurasi hitung = proporsi citra "
              "dengan jumlah deteksi tepat sama dengan jumlah wajah (semua deteksi dihitung, termasuk yang palsu).")
    ctx.table(pd.DataFrame(formation_rows), "e2_per_formasi", "E2 — per formasi (IoU utama)")
    ctx.table(pd.DataFrame(position_rows), "e2_recall_per_posisi", "E2 — recall per posisi jarak",
              "Jarak tiap wajah dari urutan kotak manual kiri → kanan dipasangkan dengan positions_cm (PRD §6.6).")
    ctx.table(pd.DataFrame(pr_rows, columns=["eksperimen", "detektor", "resolusi", "recall", "precision", "ap"]),
              "e2_kurva_pr", "E2 — titik kurva precision–recall")

    compare_rows = []
    for a_name, b_name in itertools.combinations(cfg.experiments.e2.detectors, 2):
        a_multi, b_multi = multi(op[a_name]), multi(op[b_name])
        for metric in ("f1", "recall", "ap", "akurasi_hitung"):
            if metric == "ap":
                if a_name not in ap_runs or b_name not in ap_runs:
                    continue
                diff = paired(multi(ap_runs[a_name]), multi(ap_runs[b_name]), ev.iou_primary, cfg, "ap")
            elif metric == "akurasi_hitung":
                correct_a = {r.key: float(len(r.boxes) == len(r.gt)) for r in a_multi}
                correct_b = {r.key: float(len(r.boxes) == len(r.gt)) for r in b_multi}
                diff = paired_difference(
                    sorted(correct_a),
                    lambda keys: sum(correct_a[k] for k in keys) / len(keys),
                    lambda keys: sum(correct_b[k] for k in keys) / len(keys),
                    cfg.stats.bootstrap_resamples, level, cfg.seed,
                )
            else:
                diff = paired(a_multi, b_multi, ev.iou_primary, cfg, metric)
            compare_rows.append({"A": a_name, "B": b_name, "metrik": metric, **diff.as_dict("selisih"),
                                 "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(compare_rows), "e2_perbandingan", "E2 — selisih berpasangan A − B (set multi)",
              "Bootstrap berpasangan per citra.")

    # PRD §8.6: set multi-wajah di-bootstrap per citra, bukan per subjek
    assert all(r.group == r.key for r in multi(next(iter(op.values())))), "E2 harus bootstrap per citra"
