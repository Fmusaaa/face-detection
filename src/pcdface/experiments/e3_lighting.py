"""E3 — Pencahayaan × enhancement (PRD §9, RQ3, H3).

Data: set cahaya (terang, redup, backlight di jarak acuan) + set jarak di jarak
acuan untuk kondisi normal + set kosong untuk FPPI. Resolusi asli.

Varian detektor: Haar dengan `equalize` true dan false (`haar_eq`, `haar_noeq`),
lalu detektor lain dari config. Masing-masing × enhancement (none, clahe).
Bootstrap per subjek.

Tabel:
- e3_f1                 P/R/F1 per varian × enhancement × cahaya (peta panas grafik 7)
- e3_ringkasan          F1 keseluruhan dan FPPI kosong per varian × enhancement
- e3_luminansi          rerata kanal Y per kondisi cahaya dan per sesi (bukti kuantitatif kondisi)
- e3_efek_clahe         ΔF1 clahe − none per varian × cahaya (berpasangan)
- e3_perbandingan       ΔF1 antar detektor per cahaya × enhancement, termasuk equalize on − off
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from pcdface.config import HaarConfig
from pcdface.experiments.common import Record, fppi, operating_summary, paired, run_records, verdict
from pcdface.experiments.runner import RunContext


def variants(ctx: RunContext) -> list[tuple[str, str, dict]]:
    """(label varian, nama detektor, overrides)."""
    out = []
    for name in ctx.cfg.experiments.e3.detectors:
        if isinstance(ctx.cfg.detector(name), HaarConfig):
            for equalize in ctx.cfg.experiments.e3.haar_equalize:
                out.append((f"{name}_{'eq' if equalize else 'noeq'}", name, {"equalize": equalize}))
        else:
            out.append((name, name, {}))
    return out


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    ds, ev = cfg.dataset, cfg.evaluation
    iou = ev.iou_primary
    all_samples = ctx.samples(("cahaya", "jarak", "kosong"))
    samples = [s for s in all_samples
               if s.meta.set in ("cahaya", "kosong")
               or (s.meta.set == "jarak" and s.meta.distance_cm == ds.reference_distance_cm)]
    ctx.require(samples, "cahaya")
    lightings = [l for l in ds.lightings if any(s.meta.lighting == l and s.meta.set != "kosong" for s in samples)]

    runs: dict[tuple[str, str], list[Record]] = {}
    for label, name, overrides in variants(ctx):
        for enhancement in cfg.experiments.e3.enhancements:
            with ctx.detector(name, overrides=overrides) as detector:
                records = run_records(detector, samples, cfg, enhancement=enhancement,
                                      labels={"detektor": label, "enhancement": enhancement, "mode": "operating"})
            runs[(label, enhancement)] = records
            ctx.dump(records)
        ctx.log(f"{label} selesai")

    def faces(records: list[Record], lighting: str | None = None) -> list[Record]:
        return [r for r in records if r.set != "kosong" and (lighting is None or r.lighting == lighting)]

    f1_rows, summary_rows = [], []
    for (label, enhancement), records in runs.items():
        for lighting in lightings:
            s = operating_summary(faces(records, lighting), iou, cfg)
            f1_rows.append({"varian": label, "enhancement": enhancement, "cahaya": lighting, **s.row()})
        s = operating_summary(faces(records), iou, cfg)
        summary_rows.append({"varian": label, "enhancement": enhancement, **s.row(),
                             "fppi_kosong": fppi([r for r in records if r.set == "kosong"], iou)})
    ctx.table(pd.DataFrame(f1_rows), "e3_f1", "E3 — per varian × enhancement × cahaya",
              f"IoU {iou}. P/R: Wilson; F1: bootstrap per subjek. Kondisi normal = set jarak di "
              f"{ds.reference_distance_cm} cm.")
    ctx.table(pd.DataFrame(summary_rows), "e3_ringkasan", "E3 — ringkasan per varian × enhancement")

    luma: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample.meta.set != "kosong" and sample.meta.luma_mean is not None:
            luma[(sample.meta.lighting, "semua")].append(sample.meta.luma_mean)
            luma[(sample.meta.lighting, sample.meta.session or "(tanpa sesi)")].append(sample.meta.luma_mean)
    sessions = sorted({s for _, s in luma if s != "semua"})
    scopes = ["semua"] + (sessions if len(sessions) > 1 else [])
    ctx.table(pd.DataFrame([
        {"cahaya": l, "sesi": scope, "citra": len(luma[(l, scope)]), "rerata_Y": float(np.mean(luma[(l, scope)])),
         "sb_Y": float(np.std(luma[(l, scope)], ddof=1)) if len(luma[(l, scope)]) > 1 else 0.0}
        for l in lightings for scope in scopes if luma[(l, scope)]
    ]), "e3_luminansi", "E3 — rerata luminansi (kanal Y) per kondisi cahaya",
        "Bila data diambil di beberapa sesi, baris per sesi menunjukkan apakah kondisi yang sama "
        "(mis. 'redup') benar-benar setara antar sesi.")

    labels = [label for label, _, _ in variants(ctx)]
    enhancements = cfg.experiments.e3.enhancements
    clahe_rows = []
    if "none" in enhancements and "clahe" in enhancements:
        for label in labels:
            for lighting in lightings:
                diff = paired(faces(runs[(label, "clahe")], lighting), faces(runs[(label, "none")], lighting), iou, cfg)
                clahe_rows.append({"varian": label, "cahaya": lighting, **diff.as_dict("selisih_f1"),
                                   "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(clahe_rows), "e3_efek_clahe", "E3 — efek CLAHE: F1(clahe) − F1(none)",
              "Bootstrap berpasangan per subjek.")

    compare_rows = []
    for enhancement in enhancements:
        for a, b in itertools.combinations(labels, 2):
            for lighting in lightings:
                diff = paired(faces(runs[(a, enhancement)], lighting), faces(runs[(b, enhancement)], lighting), iou, cfg)
                compare_rows.append({"enhancement": enhancement, "cahaya": lighting, "A": a, "B": b,
                                     **diff.as_dict("selisih_f1"), "kesimpulan": verdict(diff)})
    ctx.table(pd.DataFrame(compare_rows), "e3_perbandingan", "E3 — selisih F1 berpasangan A − B per cahaya",
              "haar_eq vs haar_noeq menjawab efek equalizeHist (PRD §5.1); MediaPipe vs Haar menjawab H3.")
