"""E7 — Blur gerak simulasi (PRD §9, RQ7).

Data: set jarak (semua jarak) + set kosong untuk FPPI, resolusi asli. Setiap
citra diberi blur gerak linear dengan panjang kernel `blur_px` (0 = asli) dan
arah `blur_angle_deg` sebelum dideteksi — tidak perlu foto tambahan, dan
setiap level blur memakai citra yang persis sama, jadi selisihnya murni efek
blur. Parameter detektor bawaan, tanpa enhancement. Bootstrap per peserta.

Keterbatasan: blur simulasi seragam di seluruh citra; blur gerak asli hanya
mengenai bagian yang bergerak dan bisa disertai efek *rolling shutter*.

Tabel:
- e7_ringkasan          P/R/F1 dan FPPI kosong per detektor × blur
- e7_recall_per_jarak   recall per detektor × blur × jarak (grafik 12)
- e7_vs_asli            selisih recall blur − asli pada citra yang sama
- e7_perbandingan       selisih recall antar detektor per level blur
"""

from __future__ import annotations

import itertools
from functools import partial

import pandas as pd

from pcdface.evaluation.stats import wilson
from pcdface.experiments.common import (
    Record,
    evaluate,
    fppi,
    operating_summary,
    paired,
    run_records,
    verdict,
)
from pcdface.experiments.e6_pose_expression import versus_reference
from pcdface.experiments.runner import RunContext
from pcdface.preprocessing import motion_blur


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    e7, ev = cfg.experiments.e7, cfg.evaluation
    iou, level = ev.iou_primary, cfg.stats.ci_level
    samples = ctx.samples(("jarak", "kosong"))
    ctx.require(samples, "jarak")

    runs: dict[tuple[str, int], list[Record]] = {}
    for name in e7.detectors:
        with ctx.detector(name) as detector:
            for blur in e7.blur_px:
                degrade = partial(motion_blur, length_px=blur, angle_deg=e7.blur_angle_deg) if blur > 1 else None
                records = run_records(detector, samples, cfg, degrade=degrade,
                                      labels={"detektor": name, "mode": "operating", "blur_px": str(blur)})
                runs[(name, blur)] = records
                ctx.dump(records)
        ctx.log(f"{name} selesai ({len(e7.blur_px)} level blur)")

    def faces(records: list[Record]) -> list[Record]:
        return [r for r in records if r.set != "kosong"]

    distances = sorted({s.meta.distance_cm for s in samples if s.meta.set == "jarak"})

    summary_rows, distance_rows = [], []
    for (name, blur), records in runs.items():
        s = operating_summary(faces(records), iou, cfg)
        summary_rows.append({"detektor": name, "blur_px": blur, **s.row(),
                             "fppi_kosong": fppi([r for r in records if r.set == "kosong"], iou)})
        for distance in distances:
            subset = [r for r in faces(records) if r.distance_cm == distance]
            evals = evaluate(subset, iou)
            tp, fn = sum(e.tp for e in evals), sum(e.fn for e in evals)
            distance_rows.append({"detektor": name, "blur_px": blur, "jarak_cm": distance, "citra": len(subset),
                                  **wilson(tp, tp + fn, level).as_dict("recall")})
    ctx.table(pd.DataFrame(summary_rows), "e7_ringkasan", "E7 — kinerja per detektor × panjang blur gerak",
              f"Blur gerak linear {e7.blur_angle_deg:g}° dengan panjang kernel blur_px piksel pada 1280×720; "
              f"0 = citra asli. IoU {iou}. P/R: Wilson; F1: bootstrap per peserta.")
    ctx.table(pd.DataFrame(distance_rows), "e7_recall_per_jarak", "E7 — recall per jarak × blur",
              "Wajah jauh (kecil) diduga lebih cepat hilang karena blur menghapus proporsi detail yang lebih besar.")

    versus_rows, compare_rows = [], []
    for name in e7.detectors:
        for blur in e7.blur_px:
            if blur == 0:
                continue
            diff = paired(faces(runs[(name, blur)]), faces(runs[(name, 0)]), iou, cfg, metric="recall")
            versus_rows.append({"detektor": name, "blur_px": blur, **diff.as_dict("selisih_recall"),
                                "kesimpulan": versus_reference(diff)})
    for blur in e7.blur_px:
        for a, b in itertools.combinations(e7.detectors, 2):
            diff = paired(faces(runs[(a, blur)]), faces(runs[(b, blur)]), iou, cfg, metric="recall")
            compare_rows.append({"blur_px": blur, "A": a, "B": b, **diff.as_dict("selisih_recall"),
                                 "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(versus_rows), "e7_vs_asli", "E7 — selisih recall blur − asli (citra yang sama)",
              "Bootstrap berpasangan per peserta; acuan = blur 0.")
    ctx.table(pd.DataFrame(compare_rows), "e7_perbandingan", "E7 — selisih recall A − B per level blur",
              "Citra yang sama untuk kedua detektor, bootstrap berpasangan per peserta.")
