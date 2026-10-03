"""E8 — Pengenalan identitas LBPH (PRD v4 §9 E8, RQ8, H7).

Data: set satu wajah (jarak, cahaya, pose, ekspresi, oklusi) **hanya** dari peserta dengan
`consent_research` dan `consent_recognition` = ya. Galeri (foto pendaftaran) = set jarak di
jarak acuan, cahaya normal — 5 foto per peserta (PRD §12.2 butir 13). Semua foto lain menjadi
foto uji, termasuk foto kondisi acuan dari set lain (`depan` 100 cm, `netral`, `tanpa`) yang
diambil di momen berbeda — inilah pembanding "acuan".

Dua sumber kotak wajah:

- ``manual`` — kotak anotasi: mengukur LBPH saja, terpisah dari galat deteksi;
- detektor `experiments.e8.detector` (bawaan ``yolo_n``) — jalur ujung-ke-ujung seperti repo
  referensi: galeri dan foto uji di-crop dari deteksi. Foto uji yang wajahnya tidak terdeteksi
  (tidak ada kotak dengan IoU ≥ `iou_primary` terhadap kotak manual) dihitung gagal.

Metrik (proporsi: Wilson 95%; akurasi keseluruhan juga bootstrap per peserta):

- akurasi rank-1 (identifikasi tertutup) — tetangga terdekat = peserta sebenarnya, tanpa ambang;
- DIR pada ambang `recognition.max_distance` — benar **dan** jarak LBPH ≤ ambang;
- FRR — foto uji asli yang ditolak (jarak > ambang, atau tidak terdeteksi);
- FAR — *leave-one-subject-out*: model dilatih tanpa peserta k, semua foto k diuji sebagai orang
  tak dikenal; FAR = proporsi yang tetap diterima sebagai orang lain.

Tabel:
- e8_ringkasan     per sumber: galeri, foto uji, akurasi rank-1, DIR, FRR, FAR, tidak terdeteksi
- e8_per_kondisi   per sumber × faktor × level (acuan, jarak, cahaya, pose per jarak, ekspresi, oklusi)
- e8_vs_acuan      selisih akurasi level − acuan pada peserta yang sama (bootstrap berpasangan)
- e8_ambang        DIR dan FAR di seluruh rentang ambang (grafik 15)
- e8_prediksi      prediksi per foto uji (audit; berisi kode peserta, bukan nama)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pcdface.boxes import Box
from pcdface.dataset.loader import Sample
from pcdface.dataset.metadata import SINGLE_FACE_SETS, read_subjects
from pcdface.evaluation.matching import iou as box_iou
from pcdface.evaluation.stats import NAN_ESTIMATE, Estimate, cluster_bootstrap, paired_difference, wilson
from pcdface.experiments.e6_pose_expression import versus_reference
from pcdface.experiments.runner import ExperimentError, RunContext
from pcdface.pose import REFERENCE_EXPRESSION, REFERENCE_OCCLUSION, REFERENCE_POSE
from pcdface.recognition.lbph import (
    LBPHRecognizer,
    consenting_subjects,
    face_patch,
    is_e8_gallery,
    is_reference_condition,
)

MANUAL = "manual"
MAX_CURVE_POINTS = 300


@dataclass
class Probe:
    """Satu foto uji untuk satu sumber kotak."""

    key: str
    subject: str
    set: str
    distance_cm: int | None
    lighting: str
    pose: str
    expression: str
    occlusion: str
    reference: bool             # kondisi acuan (depan/netral/tanpa di jarak acuan, cahaya normal)
    detected: bool
    nearest: str = ""
    distance: float = math.nan

    @property
    def correct(self) -> bool:
        return self.detected and self.nearest == self.subject

    def accepted(self, threshold: float) -> bool:
        return self.detected and self.distance <= threshold

    def identified(self, threshold: float) -> bool:
        return self.correct and self.distance <= threshold


def ordered(values: set[str], order: Sequence[str]) -> list[str]:
    """Nilai menurut urutan config; yang tidak ada di config di belakang, alfabetis."""
    rank = {value: i for i, value in enumerate(order)}
    return sorted(values, key=lambda v: (rank.get(v, len(rank)), v))


def best_match(gt: Box, boxes: Sequence[Box], iou_threshold: float) -> Box | None:
    """Deteksi dengan IoU tertinggi terhadap kotak manual, bila ≥ ambang."""
    scored = [(box_iou(gt, box), box) for box in boxes]
    scored = [item for item in scored if item[0] >= iou_threshold]
    return max(scored)[1] if scored else None


def _train(patches: dict[str, np.ndarray], samples: Sequence[Sample], ctx: RunContext,
           exclude: str | None = None) -> LBPHRecognizer | None:
    chosen = [(patches[s.key], s.meta.subject_id) for s in samples
              if s.key in patches and s.meta.subject_id != exclude]
    if len({label for _, label in chosen}) < 1:
        return None
    recognizer = LBPHRecognizer(ctx.cfg.recognition)
    recognizer.train([p for p, _ in chosen], [label for _, label in chosen])
    return recognizer


def _accuracy_by_subject(probes: Sequence[Probe], hit: Callable[[Probe], bool]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for probe in probes:
        counts[probe.subject][0] += int(hit(probe))
        counts[probe.subject][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _pooled(counts: dict[str, tuple[int, int]]) -> Callable[[Sequence[str]], float]:
    def statistic(keys: Sequence[str]) -> float:
        k = sum(counts[key][0] for key in keys if key in counts)
        n = sum(counts[key][1] for key in keys if key in counts)
        return k / n if n else math.nan

    return statistic


def condition_row(probes: Sequence[Probe], threshold: float, level: float) -> dict[str, float]:
    n = len(probes)
    distances = [p.distance for p in probes if p.detected]
    row: dict[str, float] = {"foto": n, "peserta": len({p.subject for p in probes}),
                             "tak_terdeteksi": sum(not p.detected for p in probes)}
    row.update(wilson(sum(p.correct for p in probes), n, level).as_dict("akurasi"))
    row.update(wilson(sum(p.identified(threshold) for p in probes), n, level).as_dict("dir"))
    row["rerata_jarak_lbph"] = float(np.mean(distances)) if distances else math.nan
    return row


def factor_levels(probes: Sequence[Probe], order: dict[str, Sequence[str]] | None = None
                  ) -> list[tuple[str, int | None, str, list[Probe]]]:
    """(faktor, jarak_cm, level, foto uji) untuk tabel per kondisi; level diurutkan menurut `order` (config)."""
    order = order or {}
    out: list[tuple[str, int | None, str, list[Probe]]] = []
    reference = [p for p in probes if p.reference]
    if reference:
        out.append(("acuan", reference[0].distance_cm, f"{REFERENCE_POSE}/{REFERENCE_EXPRESSION}/{REFERENCE_OCCLUSION}",
                    reference))

    def group(set_name: str, attribute: str, factor: str, by_distance: bool = False) -> None:
        chosen = [p for p in probes if p.set == set_name]
        rank = {str(value): i for i, value in enumerate(order.get(factor, ()))}
        keys = sorted({(p.distance_cm if by_distance else None, getattr(p, attribute)) for p in chosen},
                      key=lambda k: (k[0] or 0, rank.get(str(k[1]), len(rank)), str(k[1])))
        for distance, value in keys:
            subset = [p for p in chosen if getattr(p, attribute) == value
                      and (not by_distance or p.distance_cm == distance)]
            out.append((factor, distance, str(value), subset))

    group("jarak", "distance_cm", "jarak", by_distance=True)
    group("cahaya", "lighting", "cahaya")
    group("pose", "pose", "pose", by_distance=True)
    group("ekspresi", "expression", "ekspresi")
    group("oklusi", "occlusion", "oklusi")
    return out


def threshold_curve(genuine: Sequence[Probe], impostor: Sequence[float], marker: float) -> pd.DataFrame:
    """DIR(t) dan FAR(t) di seluruh rentang ambang (titik dari data + ambang config)."""
    values = [p.distance for p in genuine if p.detected] + list(impostor)
    if not values:
        return pd.DataFrame(columns=["ambang", "dir", "far"])
    grid = np.unique(np.concatenate([[0.0, marker], np.asarray(values, dtype=float)]))
    if grid.size > MAX_CURVE_POINTS:
        keep = np.unique(np.linspace(0, grid.size - 1, MAX_CURVE_POINTS).astype(int))
        grid = np.unique(np.concatenate([grid[keep], [marker]]))
    correct = np.array([p.distance if p.correct else np.inf for p in genuine], dtype=float)
    impostor_arr = np.asarray(impostor, dtype=float)
    rows = []
    for t in grid:
        rows.append({"ambang": float(t),
                     "dir": float((correct <= t).mean()) if correct.size else math.nan,
                     "far": float((impostor_arr <= t).mean()) if impostor_arr.size else math.nan})
    return pd.DataFrame(rows)


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    rc, ev, e8 = cfg.recognition, cfg.evaluation, cfg.experiments.e8
    level, n_boot, threshold = cfg.stats.ci_level, cfg.stats.bootstrap_resamples, rc.max_distance
    reference_cm = cfg.dataset.reference_distance_cm

    allowed = consenting_subjects(read_subjects(ctx.paths.subjects))
    single = [s for s in ctx.samples(SINGLE_FACE_SETS) if len(s.gt) == 1]
    samples = [s for s in single if s.meta.subject_id in allowed]
    ctx.notes["e8_foto_tanpa_izin_pengenalan"] = len(single) - len(samples)
    gallery = [s for s in samples if is_e8_gallery(s.meta, reference_cm)]
    gallery_subjects = sorted({s.meta.subject_id for s in gallery})
    if len(gallery_subjects) < 2:
        raise ExperimentError(
            "E8 butuh ≥ 2 peserta dengan consent_recognition = ya yang punya foto galeri beranotasi "
            f"(set jarak {reference_cm} cm, cahaya normal). Peserta dengan izin: {sorted(allowed) or 'tidak ada'}")
    probes = [s for s in samples if not is_e8_gallery(s.meta, reference_cm) and s.meta.subject_id in gallery_subjects]
    if not probes:
        raise ExperimentError("E8 tidak punya foto uji (semua foto peserta berizin adalah galeri)")
    ctx.notes["e8_galeri"] = {sid: sum(s.meta.subject_id == sid for s in gallery) for sid in gallery_subjects}

    # --- crop: satu kali baca per foto, kotak manual + kotak detektor -----------------------
    sources = [MANUAL, e8.detector]
    patches: dict[str, dict[str, np.ndarray]] = {source: {} for source in sources}
    with ctx.detector(e8.detector) as detector:
        for sample in gallery + probes:
            image = sample.load()
            patches[MANUAL][sample.key] = face_patch(image, sample.gt[0], rc.face_size)
            match = best_match(sample.gt[0], detector.detect(image).boxes, ev.iou_primary)
            if match is not None:
                patches[e8.detector][sample.key] = face_patch(image, match, rc.face_size)
    ctx.log(f"{len(gallery)} foto galeri, {len(probes)} foto uji, {len(gallery_subjects)} peserta")

    summary_rows, condition_rows, versus_rows, prediction_rows = [], [], [], []
    curves = []
    for source in sources:
        found = patches[source]
        model = _train(found, gallery, ctx)
        if model is None or len(model.labels) < 2:
            ctx.log(f"{source}: galeri < 2 peserta setelah deteksi — dilewati")
            continue
        results: list[Probe] = []
        for sample in probes:
            meta = sample.meta
            probe = Probe(sample.key, meta.subject_id, meta.set, meta.distance_cm, meta.lighting, meta.pose,
                          meta.expression, meta.occlusion, is_reference_condition(meta, reference_cm),
                          detected=sample.key in found)
            if probe.detected:
                prediction = model.predict(found[sample.key])
                probe.nearest, probe.distance = prediction.nearest, prediction.distance
            results.append(probe)

        # FAR: leave-one-subject-out — semua foto peserta k diuji pada model tanpa k
        impostor: list[float] = []
        for subject in model.labels:
            without = _train(found, gallery, ctx, exclude=subject)
            if without is None:
                continue
            for sample in gallery + probes:
                if sample.meta.subject_id == subject and sample.key in found:
                    impostor.append(without.predict(found[sample.key]).distance)

        n = len(results)
        accuracy = _accuracy_by_subject(results, lambda p: p.correct)
        boot = cluster_bootstrap(sorted(accuracy), _pooled(accuracy), n_boot, level, cfg.seed)
        row = {"sumber": source, "peserta": len(model.labels),
               "galeri": sum(s.key in found for s in gallery), "foto_uji": n,
               "tak_terdeteksi": sum(not p.detected for p in results)}
        row.update(wilson(sum(p.correct for p in results), n, level).as_dict("akurasi"))
        row.update({"akurasi_boot_low": boot.low, "akurasi_boot_high": boot.high})
        row.update(wilson(sum(p.identified(threshold) for p in results), n, level).as_dict("dir"))
        row.update(wilson(sum(not p.accepted(threshold) for p in results), n, level).as_dict("frr"))
        row.update(wilson(sum(d <= threshold for d in impostor), len(impostor), level).as_dict("far"))
        row["foto_penyusup"] = len(impostor)
        summary_rows.append(row)

        order = {"cahaya": cfg.dataset.lightings, "pose": cfg.dataset.poses,
                 "ekspresi": cfg.dataset.expressions, "oklusi": cfg.dataset.occlusions}
        for factor, distance, value, subset in factor_levels(results, order):
            condition_rows.append({"sumber": source, "faktor": factor, "jarak_cm": distance, "level": value,
                                   **condition_row(subset, threshold, level)})

        # selisih terhadap acuan pada peserta yang sama
        reference_pool = [p for p in results if p.reference]
        comparisons: list[tuple[str, int | None, str, list[Probe], list[Probe]]] = []
        for distance in sorted({p.distance_cm for p in results if p.set == "jarak"}):
            comparisons.append(("jarak", distance, str(distance),
                                [p for p in results if p.set == "jarak" and p.distance_cm == distance], reference_pool))
        for lighting in ordered({p.lighting for p in results if p.set == "cahaya"}, cfg.dataset.lightings):
            comparisons.append(("cahaya", None, lighting,
                                [p for p in results if p.set == "cahaya" and p.lighting == lighting], reference_pool))
        for distance in sorted({p.distance_cm for p in results if p.set == "pose"}):
            same = [p for p in results if p.set == "pose" and p.distance_cm == distance]
            base = [p for p in same if p.pose == REFERENCE_POSE]
            for pose in ordered({p.pose for p in same} - {REFERENCE_POSE}, cfg.dataset.poses):
                comparisons.append(("pose", distance, pose, [p for p in same if p.pose == pose], base))
        for set_name, attribute, reference, levels in (
                ("ekspresi", "expression", REFERENCE_EXPRESSION, cfg.dataset.expressions),
                ("oklusi", "occlusion", REFERENCE_OCCLUSION, cfg.dataset.occlusions)):
            same = [p for p in results if p.set == set_name]
            base = [p for p in same if getattr(p, attribute) == reference]
            for value in ordered({getattr(p, attribute) for p in same} - {reference}, levels):
                comparisons.append((set_name, None, value, [p for p in same if getattr(p, attribute) == value], base))
        for factor, distance, value, subset, base in comparisons:
            a, b = _accuracy_by_subject(subset, lambda p: p.correct), _accuracy_by_subject(base, lambda p: p.correct)
            groups = sorted(set(a) & set(b))
            diff: Estimate = (paired_difference(groups, _pooled(a), _pooled(b), n_boot, level, cfg.seed)
                              if groups else NAN_ESTIMATE)
            versus_rows.append({"sumber": source, "faktor": factor, "jarak_cm": distance, "level": value,
                                "peserta": len(groups), **diff.as_dict("selisih_akurasi"),
                                "kesimpulan": versus_reference(diff)})

        curve = threshold_curve(results, impostor, threshold)
        curve.insert(0, "sumber", source)
        curves.append(curve)
        for p in results:
            prediction_rows.append({"sumber": source, "foto": p.key, "peserta": p.subject, "set": p.set,
                                    "terdeteksi": p.detected, "prediksi": p.nearest if p.detected else "",
                                    "jarak_lbph": p.distance, "benar": p.correct,
                                    "diterima": p.accepted(threshold)})
        ctx.log(f"{source}: akurasi rank-1 {summary_rows[-1]['akurasi']:.3f}, FAR {summary_rows[-1]['far']:.3f}")

    if not summary_rows:
        raise ExperimentError("E8: tidak ada sumber kotak dengan galeri ≥ 2 peserta")

    ctx.table(pd.DataFrame(summary_rows), "e8_ringkasan", "E8 — pengenalan identitas LBPH",
              f"Galeri: set jarak {reference_cm} cm, cahaya normal. Akurasi rank-1 = tetangga terdekat benar "
              f"(tanpa ambang); DIR = benar dan jarak LBPH ≤ {threshold:g}; FRR = foto asli ditolak; FAR = "
              "leave-one-subject-out (foto peserta yang tidak ada di galeri tetap diterima). Proporsi: Wilson 95%; "
              f"akurasi_boot = bootstrap per peserta ({n_boot} resampel). Sumber '{e8.detector}' = crop dari "
              "deteksi (ujung-ke-ujung); wajah tak terdeteksi dihitung gagal.")
    ctx.table(pd.DataFrame(condition_rows), "e8_per_kondisi", "E8 — akurasi pengenalan per kondisi",
              "'acuan' = foto uji pose depan, ekspresi netral, dan oklusi tanpa di jarak acuan — kondisi sama "
              "dengan galeri tetapi direkam di momen lain.")
    ctx.table(pd.DataFrame(versus_rows), "e8_vs_acuan", "E8 — selisih akurasi kondisi − acuan (peserta sama)",
              "Jarak dan cahaya dibandingkan dengan kumpulan foto acuan; pose, ekspresi, dan oklusi dengan "
              "level acuan set yang sama (depan di jarak yang sama, netral, tanpa). Bootstrap berpasangan per peserta.")
    ctx.table(pd.concat(curves, ignore_index=True), "e8_ambang", "E8 — DIR dan FAR terhadap ambang jarak LBPH",
              f"Ambang config: {threshold:g} (nilai repo referensi). Ambang tidak dipilih ulang dari kurva ini "
              "(PRD §12.2 butir 13).")
    ctx.table(pd.DataFrame(prediction_rows), "e8_prediksi", "E8 — prediksi per foto uji")
