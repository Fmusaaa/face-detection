"""E5 — Sensitivitas parameter Haar (PRD §9, P1).

Grid `scaleFactor` × `minNeighbors` pada set jarak + multi (+ kosong untuk FPPI),
resolusi asli. Menunjukkan bahwa perbandingan tidak bergantung pada satu
setelan yang kebetulan menguntungkan. Untuk MediaPipe, kurva PR di E1/E2 sudah
mencakup seluruh rentang `min_detection_confidence`.

`run all` hanya menjalankan E5 bila `experiments.e5.enabled: true`;
`run e5` selalu menjalankannya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcdface.config import HaarConfig
from pcdface.experiments.common import fppi, operating_summary, run_records
from pcdface.experiments.runner import ExperimentError, RunContext


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    e5 = cfg.experiments.e5
    iou = cfg.evaluation.iou_primary
    haar_names = [n for n, spec in cfg.detectors.items() if isinstance(spec, HaarConfig)]
    if not haar_names:
        raise ExperimentError("tidak ada detektor Haar di config")
    name = haar_names[0]
    samples = ctx.samples(("jarak", "multi", "kosong"))
    ctx.require(samples, "jarak")

    rows = []
    for scale_factor in e5.scale_factors:
        for min_neighbors in e5.min_neighbors:
            overrides = {"scale_factor": scale_factor, "min_neighbors": min_neighbors}
            with ctx.detector(name, overrides=overrides) as detector:
                records = run_records(detector, samples, cfg, labels={"detektor": name, **{k: str(v) for k, v in overrides.items()}})
            ctx.dump(records)
            faces = [r for r in records if r.set != "kosong"]
            s = operating_summary(faces, iou, cfg)
            rows.append({"scale_factor": scale_factor, "min_neighbors": min_neighbors, **s.row(),
                         "fppi_kosong": fppi([r for r in records if r.set == "kosong"], iou),
                         "median_ms": float(np.median([r.elapsed_ms for r in records]))})
            ctx.log(f"scaleFactor {scale_factor}, minNeighbors {min_neighbors}: F1 {s.f1.value:.3f}")
    spec = cfg.detector(name)
    ctx.table(pd.DataFrame(rows), "e5_sensitivitas_haar", "E5 — sensitivitas parameter Haar",
              f"Setelan utama: scaleFactor {spec.scale_factor}, minNeighbors {spec.min_neighbors}, "
              f"equalize {spec.equalize}. F1: bootstrap per subjek (jarak) / per citra (multi).")
