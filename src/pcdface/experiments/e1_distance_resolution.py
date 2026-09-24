"""E1 — Jarak × resolusi (PRD §9, RQ1, RQ4, H1, H2).

Data: set jarak (+ set kosong untuk FPPI). Setiap detektor dijalankan pada
1280×720 asli dan 640×360 hasil perkecilan skala 0,5 (INTER_AREA, ground truth
ikut diskalakan), masing-masing dua run: titik operasi dan ambang rendah (AP).

Tabel:
- e1_ringkasan          P/R/F1 (+IoU sensitivitas 0,3/0,4), rerata IoU, FPPI, AP
- e1_recall_per_jarak   recall per jarak (Wilson + bootstrap subjek), lebar wajah px & proporsi
- e1_jangkauan          jarak optimal, jarak maksimum efektif (PRD §8.4)
- e1_ukuran_px / e1_ukuran_proporsi   recall per bin ukuran wajah
- e1_ukuran_minimum     ukuran wajah minimum (px dan proporsi) per detektor
- e1_loglog             validasi model kamera: kemiringan log(w) vs log(Z)
- e1_loglog_per_sesi    sama, per sesi pengambilan (hanya bila > 1 sesi) — kamera bergeser?
- e1_klaim_dokumentasi  recall ≤200 cm vs >200 cm (klaim MediaPipe 2 m / 5 m)
- e1_resolusi           H2: selisih recall 1280 − 640 per detektor (berpasangan)
- e1_perbandingan       selisih berpasangan antar detektor (F1, recall, AP)
- e1_kurva_pr           titik kurva PR untuk grafik
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from pcdface.detection.registry import supports_scores
from pcdface.evaluation.distance_analysis import bin_detections, distance_range, loglog_fit, min_face_size
from pcdface.evaluation.stats import cluster_bootstrap, paired_difference, wilson
from pcdface.experiments.common import (
    GroupedCounts,
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

NEAR_LIMIT_CM = 200  # klaim short-range "within 2 meters"


def res_label(resolution: tuple[int, int]) -> str:
    return f"{resolution[0]}x{resolution[1]}"


def _join(values: tuple[int, ...]) -> str:
    return ", ".join(str(v) for v in values) if values else "–"


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    ev = cfg.evaluation
    e1 = cfg.experiments.e1
    level, n_boot = cfg.stats.ci_level, cfg.stats.bootstrap_resamples
    samples = ctx.samples(("jarak", "kosong"))
    ctx.require(samples, "jarak")

    op: dict[tuple[str, str], list[Record]] = {}
    ap_runs: dict[tuple[str, str], list[Record]] = {}
    for detector_name in e1.detectors:
        for resolution in e1.resolutions:
            key = (detector_name, res_label(resolution))
            scale = cfg.scale_for(resolution)
            labels = {"detektor": detector_name, "resolusi": key[1]}
            with ctx.detector(detector_name) as detector:
                op[key] = run_records(detector, samples, cfg, scale, labels={**labels, "mode": "operating"})
            ctx.dump(op[key])
            if supports_scores(detector_name, cfg):
                with ctx.detector(detector_name, "ap") as detector:
                    ap_runs[key] = run_records(detector, samples, cfg, scale, labels={**labels, "mode": "ap"})
                ctx.dump(ap_runs[key])
            ctx.log(f"{detector_name} @ {key[1]} selesai")

    def faces(records: list[Record]) -> list[Record]:
        return [r for r in records if r.set == "jarak"]

    def empties(records: list[Record]) -> list[Record]:
        return [r for r in records if r.set == "kosong"]

    # --- ringkasan + kurva PR -------------------------------------------------
    summary_rows, pr_rows = [], []
    for (name, res), records in op.items():
        for iou in ev.iou_thresholds:
            s = operating_summary(faces(records), iou, cfg)
            row = {"detektor": name, "resolusi": res, "iou": iou, **s.row(),
                   "fppi_kosong": fppi(empties(records), iou), "fppi_semua": fppi(records, iou)}
            row.update({"ap": math.nan, "ap_low": math.nan, "ap_high": math.nan})
            if iou == ev.iou_primary and (name, res) in ap_runs:
                estimate, recall, precision = ap_summary(faces(ap_runs[(name, res)]), iou, cfg)
                row.update(estimate.as_dict("ap"))
                for r_value, p_value in zip(*thin_curve(recall, precision)):
                    pr_rows.append({"eksperimen": "e1", "detektor": name, "resolusi": res,
                                    "recall": float(r_value), "precision": float(p_value), "ap": estimate.value})
            summary_rows.append(row)
    ctx.table(pd.DataFrame(summary_rows), "e1_ringkasan", "E1 — ringkasan per detektor dan resolusi",
              f"P dan R: interval Wilson; F1 dan AP: bootstrap per subjek ({n_boot} resampel). "
              f"AP hanya pada IoU {ev.iou_primary}. FPPI kosong = deteksi palsu per citra set kosong.")
    ctx.table(pd.DataFrame(pr_rows, columns=["eksperimen", "detektor", "resolusi", "recall", "precision", "ap"]),
              "e1_kurva_pr", "E1 — titik kurva precision–recall")

    # --- recall per jarak + jangkauan ----------------------------------------
    distance_rows, range_rows = [], []
    for (name, res), records in op.items():
        face_records = faces(records)
        evals = evaluate(face_records, ev.iou_primary)
        by_distance: dict[int, list[tuple[Record, object]]] = defaultdict(list)
        for record, e in zip(face_records, evals):
            by_distance[int(record.distance_cm)].append((record, e))
        recall_map = {}
        for distance in sorted(by_distance):
            pairs = by_distance[distance]
            n = sum(e.n_gt for _, e in pairs)
            k = sum(e.tp for _, e in pairs)
            grouped = GroupedCounts([e for _, e in pairs])
            boot = cluster_bootstrap(grouped.groups, grouped.recall, n_boot, level, cfg.seed)
            widths = np.array([r.gt[0][2] for r, _ in pairs], dtype=float)
            ratios = widths / np.array([r.width for r, _ in pairs], dtype=float)
            recall = wilson(k, n, level)
            recall_map[distance] = recall.value
            distance_rows.append({
                "detektor": name, "resolusi": res, "jarak_cm": distance, "wajah": n, "TP": k,
                **recall.as_dict("recall"),
                "recall_boot_low": boot.low, "recall_boot_high": boot.high,
                "lebar_px": float(widths.mean()), "lebar_px_sb": float(widths.std(ddof=1)) if len(widths) > 1 else 0.0,
                "lebar_proporsi": float(ratios.mean()),
            })
        result = distance_range(recall_map, ev.recall_target)
        range_rows.append({"detektor": name, "resolusi": res, "target_recall": ev.recall_target,
                           "jarak_optimal_cm": _join(result.qualifying),
                           "jarak_maks_efektif_cm": result.max_effective,
                           "jarak_maks_berturut_cm": result.max_contiguous})
    ctx.table(pd.DataFrame(distance_rows), "e1_recall_per_jarak", "E1 — recall per jarak",
              "recall: estimasi titik + Wilson; recall_boot: bootstrap per subjek (memperhitungkan frame "
              "berkorelasi). Lebar wajah dari kotak manual.")
    ctx.table(pd.DataFrame(range_rows), "e1_jangkauan", "E1 — jarak optimal dan jarak maksimum efektif",
              "Jarak optimal = recall (estimasi titik) ≥ target. 'Berturut' = semua jarak yang lebih dekat juga memenuhi.")

    # --- ukuran wajah ---------------------------------------------------------
    size_rows = {"px": [], "proporsi": []}
    minimum_rows = []
    for name in e1.detectors:
        pooled: dict[str, list[float]] = {"px": [], "proporsi": []}
        pooled_hits: list[bool] = []
        for resolution in e1.resolutions:
            res = res_label(resolution)
            face_records = faces(op[(name, res)])
            evals = evaluate(face_records, ev.iou_primary)
            px = [float(r.gt[0][2]) for r in face_records]
            ratio = [float(r.gt[0][2]) / r.width for r in face_records]
            hits = [bool(e.gt_matched[0]) for e in evals]
            pooled["px"] += px
            pooled["proporsi"] += ratio
            pooled_hits += hits
            for unit, values, edges in (("px", px, ev.size_bins_px), ("proporsi", ratio, ev.size_bins_ratio)):
                for b in bin_detections(values, hits, edges):
                    estimate = wilson(b.detected, b.n, level)
                    size_rows[unit].append({"detektor": name, "resolusi": res, "bin_bawah": b.low, "bin_atas": b.high,
                                            "wajah": b.n, "TP": b.detected, **estimate.as_dict("recall")})
            minimum_rows.append({
                "detektor": name, "resolusi": res,
                "ukuran_min_px": min_face_size(px, hits, ev.size_bins_px, ev.recall_target).threshold,
                "ukuran_min_proporsi": min_face_size(ratio, hits, ev.size_bins_ratio, ev.recall_target).threshold,
            })
        minimum_rows.append({
            "detektor": name, "resolusi": "gabungan",
            "ukuran_min_px": min_face_size(pooled["px"], pooled_hits, ev.size_bins_px, ev.recall_target).threshold,
            "ukuran_min_proporsi": min_face_size(pooled["proporsi"], pooled_hits, ev.size_bins_ratio, ev.recall_target).threshold,
        })
    ctx.table(pd.DataFrame(size_rows["px"]), "e1_ukuran_px", "E1 — recall per lebar wajah (piksel)")
    ctx.table(pd.DataFrame(size_rows["proporsi"]), "e1_ukuran_proporsi", "E1 — recall per lebar wajah (proporsi lebar frame)")
    ctx.table(pd.DataFrame(minimum_rows), "e1_ukuran_minimum", "E1 — ukuran wajah minimum untuk recall ≥ target",
              f"Batas bawah bin terkecil sehingga bin itu dan semua bin di atasnya punya recall ≥ {ev.recall_target} "
              "(PRD §12.2 butir 6). 'gabungan' = kedua resolusi digabung. Kosong = target tidak tercapai.")

    # --- validasi model kamera -------------------------------------------------
    loglog_rows = []
    for resolution in e1.resolutions:
        res = res_label(resolution)
        face_records = faces(op[(e1.detectors[0], res)])
        fit = loglog_fit([r.distance_cm for r in face_records], [r.gt[0][2] for r in face_records])
        low, high = fit.slope_ci95
        loglog_rows.append({"resolusi": res, "n": fit.n, "kemiringan": fit.slope, "kemiringan_low": low,
                            "kemiringan_high": high, "intersep": fit.intercept, "r2": fit.r2, "persamaan": fit.equation})
    ctx.table(pd.DataFrame(loglog_rows), "e1_loglog", "E1 — validasi model kamera lubang jarum",
              "Kemiringan ≈ −1 membuktikan w ∝ 1/Z. Jauh dari −1 → periksa Center Stage (PRD §8.4).")

    first = faces(op[(e1.detectors[0], res_label(e1.resolutions[0]))])
    sessions = sorted({r.session for r in first})
    if len(sessions) > 1:
        session_rows = []
        for session in sessions:
            subset = [r for r in first if r.session == session]
            distances = {r.distance_cm for r in subset}
            row = {"sesi": session, "n": len(subset), "jarak_cm": ", ".join(str(d) for d in sorted(distances))}
            if len(subset) >= 3 and len(distances) >= 2:
                fit = loglog_fit([r.distance_cm for r in subset], [r.gt[0][2] for r in subset])
                low, high = fit.slope_ci95
                row.update({"kemiringan": fit.slope, "kemiringan_low": low, "kemiringan_high": high,
                            "intersep": fit.intercept, "r2": fit.r2})
            session_rows.append(row)
        ctx.table(pd.DataFrame(session_rows), "e1_loglog_per_sesi",
                  f"E1 — validasi model kamera per sesi ({res_label(e1.resolutions[0])})",
                  "Intersep yang berbeda jauh antar sesi berarti ukuran wajah pada jarak sama tidak setara — "
                  "posisi kamera, zoom, atau Center Stage berubah. Butuh ≥ 2 jarak per sesi.")

    # --- klaim dokumentasi MediaPipe -------------------------------------------
    claim_rows = []
    for (name, res), records in op.items():
        face_records = faces(records)
        evals = evaluate(face_records, ev.iou_primary)
        near = [e for r, e in zip(face_records, evals) if r.distance_cm <= NEAR_LIMIT_CM]
        far = [e for r, e in zip(face_records, evals) if r.distance_cm > NEAR_LIMIT_CM]
        row = {"detektor": name, "resolusi": res}
        for label, subset in (("dekat", near), ("jauh", far)):
            row.update(wilson(sum(e.tp for e in subset), sum(e.n_gt for e in subset), level).as_dict(f"recall_{label}"))
        if near and far:
            gn, gf = GroupedCounts(near), GroupedCounts(far)
            diff = paired_difference(sorted(set(gn.groups) | set(gf.groups)), gn.recall, gf.recall, n_boot, level, cfg.seed)
            row.update(diff.as_dict("selisih"))
            row["kesimpulan"] = verdict(diff)
        claim_rows.append(row)
    ctx.table(pd.DataFrame(claim_rows), "e1_klaim_dokumentasi",
              f"E1 — recall ≤{NEAR_LIMIT_CM} cm (dekat) vs >{NEAR_LIMIT_CM} cm (jauh)",
              "Klaim dokumentasi: short-range bekerja terbaik ≤2 m, full-range ≤5 m. Rentang data hanya sampai "
              f"{max(cfg.dataset.distances_cm)} cm, jadi klaim 5 m hanya bisa dinyatakan 'konsisten dengan', bukan dibuktikan. "
              "Selisih = dekat − jauh, bootstrap berpasangan per subjek.")

    # --- H2: resolusi ----------------------------------------------------------
    resolution_rows = []
    if len(e1.resolutions) >= 2:
        high_res, low_res = res_label(e1.resolutions[0]), res_label(e1.resolutions[1])
        for name in e1.detectors:
            for scope, keep in (("semua jarak", lambda r: True), (f"≥{NEAR_LIMIT_CM} cm", lambda r: r.distance_cm >= NEAR_LIMIT_CM)):
                a = [r for r in faces(op[(name, high_res)]) if keep(r)]
                b = [r for r in faces(op[(name, low_res)]) if keep(r)]
                if not a:
                    continue
                diff = paired(a, b, ev.iou_primary, cfg, metric="recall")
                resolution_rows.append({"detektor": name, "cakupan": scope, "A": high_res, "B": low_res,
                                        **diff.as_dict("selisih_recall"), "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(resolution_rows), "e1_resolusi", "E1 — pengaruh resolusi (H2): recall A − B",
              "H2 memprediksi selisih positif untuk Haar di jarak jauh dan ≈ 0 untuk MediaPipe.")

    # --- perbandingan antar detektor --------------------------------------------
    compare_rows = []
    for resolution in e1.resolutions:
        res = res_label(resolution)
        for a_name, b_name in itertools.combinations(e1.detectors, 2):
            for metric in ("f1", "recall", "ap"):
                if metric == "ap":
                    if (a_name, res) not in ap_runs or (b_name, res) not in ap_runs:
                        continue
                    a, b = faces(ap_runs[(a_name, res)]), faces(ap_runs[(b_name, res)])
                else:
                    a, b = faces(op[(a_name, res)]), faces(op[(b_name, res)])
                diff = paired(a, b, ev.iou_primary, cfg, metric=metric)
                compare_rows.append({"resolusi": res, "A": a_name, "B": b_name, "metrik": metric,
                                     **diff.as_dict("selisih"), "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(compare_rows), "e1_perbandingan", "E1 — selisih berpasangan A − B (set jarak)",
              "Bootstrap berpasangan per subjek. 'Berbeda' hanya bila interval selisih tidak memuat 0 (PRD §12.2 butir 5).")
