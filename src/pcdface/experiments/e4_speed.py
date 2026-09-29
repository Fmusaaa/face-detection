"""E4 — Kecepatan (PRD §8.5, RQ5, H4).

Sampel acak (seed config) sebanyak `sample_images` per set dimuat ke memori
dulu, sehingga waktu baca berkas tidak ikut. Per detektor × resolusi: warm-up
dibuang, lalu `repeats` panggilan bergiliran pada sampel. Waktu = `elapsed_ms`
detektor (termasuk konversi warna yang dibutuhkannya, PRD §12.2 butir 8).

Perangkat eksekusi tidak setara: Haar di CPU, MediaPipe di GPU (Metal) —
tercatat di kolom `perangkat` (PRD §8.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcdface.dataset.resize import scale_image
from pcdface.experiments.e1_distance_resolution import res_label
from pcdface.experiments.runner import ExperimentError, RunContext


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    e4 = cfg.experiments.e4
    samples = ctx.samples(tuple(cfg.dataset.sets))
    rng = np.random.default_rng(cfg.seed)
    chosen = []
    for set_name in cfg.dataset.sets:
        pool = [s for s in samples if s.meta.set == set_name]
        if pool:
            picks = rng.choice(len(pool), size=min(e4.sample_images, len(pool)), replace=False)
            chosen += [pool[i] for i in sorted(picks)]
    if not chosen:
        raise ExperimentError("tidak ada citra beranotasi untuk E4")
    originals = [s.load() for s in chosen]
    ctx.notes["e4_sampel"] = [s.key for s in chosen]

    rows = []
    for resolution in e4.resolutions:
        images = [scale_image(image, cfg.scale_for(resolution)) for image in originals]
        for name in e4.detectors:
            with ctx.detector(name) as detector:
                for index in range(e4.warmup_runs):
                    detector.detect(images[index % len(images)])
                times = np.array([detector.detect(images[i % len(images)]).elapsed_ms for i in range(e4.repeats)])
            median = float(np.median(times))
            rows.append({
                "detektor": name, "resolusi": res_label(resolution), "perangkat": ctx.device(name),
                "panggilan": int(times.size), "citra_berbeda": len(images),
                "median_ms": median, "p95_ms": float(np.percentile(times, 95)),
                "rerata_ms": float(times.mean()), "sb_ms": float(times.std(ddof=1)) if times.size > 1 else 0.0,
                "fps_median": 1000.0 / median if median > 0 else float("nan"),
            })
            ctx.log(f"{name} @ {res_label(resolution)}: median {median:.2f} ms")
    ctx.table(pd.DataFrame(rows), "e4_kecepatan", "E4 — waktu deteksi per frame",
              f"{e4.warmup_runs} warm-up dibuang, {e4.repeats} panggilan. Waktu baca berkas tidak dihitung. "
              "Kolom perangkat: Haar selalu CPU; MediaPipe GPU (Metal) di macOS. Bila berbeda, ini perbandingan "
              "implementasi pada perangkat ini, bukan algoritma pada perangkat keras yang sama (PRD §8.5).")

