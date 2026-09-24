"""Metrik titik operasi, AP, multi-wajah, jarak, dan statistik — nilai acuan hitungan tangan."""

import math

import numpy as np
import pytest

from pcdface.evaluation.average_precision import average_precision, pool, pr_curve, score_image
from pcdface.evaluation.distance_analysis import distance_range, loglog_fit, min_face_size
from pcdface.evaluation.multiface import count_accuracy, count_mae, position_hits, positions_of
from pcdface.evaluation.operating_point import aggregate, evaluate_image
from pcdface.evaluation.stats import cluster_bootstrap, paired_difference, wilson


# ---------------------------------------------------------------------------
# Titik operasi
# ---------------------------------------------------------------------------
def test_operating_point_hand_values():
    e1 = evaluate_image("a.jpg", "S01", [(0, 0, 10, 10), (50, 0, 10, 10)], [(0, 0, 10, 10), (200, 200, 5, 5)], 0.5)
    e2 = evaluate_image("kosong.jpg", "kosong.jpg", [], [(1, 1, 5, 5)], 0.5)
    assert (e1.tp, e1.fp, e1.fn) == (1, 1, 1) and e1.gt_matched == [True, False]
    op = aggregate([e1, e2])
    assert (op.images, op.tp, op.fp, op.fn) == (2, 1, 2, 1)
    assert op.precision == pytest.approx(1 / 3)
    assert op.recall == pytest.approx(1 / 2)
    assert op.f1 == pytest.approx(0.4)                     # = 2PR/(P+R)
    assert op.f1 == pytest.approx(2 * op.precision * op.recall / (op.precision + op.recall))
    assert op.fppi == pytest.approx(1.0)
    assert op.mean_iou == pytest.approx(1.0)


def test_operating_point_undefined_values_are_nan_not_zero():
    op = aggregate([evaluate_image("a", "g", [(0, 0, 10, 10)], [], 0.5)])
    assert math.isnan(op.precision) and math.isnan(op.mean_iou)
    assert op.recall == 0.0 and op.f1 == 0.0
    assert math.isnan(aggregate([]).fppi)


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------
def test_ap_all_point_hand_value():
    # 3 GT; urutan skor TP, FP, TP, FP → recall 1/3,1/3,2/3,2/3; precision 1, 1/2, 2/3, 1/2
    curve = pr_curve(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, False, True, False]), n_gt=3)
    assert curve.recall.tolist() == pytest.approx([1 / 3, 1 / 3, 2 / 3, 2 / 3])
    assert curve.ap == pytest.approx(1 / 3 * 1 + 1 / 3 * 2 / 3)  # 5/9


def test_ap_envelope_and_missed_faces():
    # FP di urutan pertama: precision 1/2 lalu 2/3 → selubung 2/3 untuk seluruh recall
    assert pr_curve(np.array([3.0, 2.0, 1.0]), np.array([False, True, True]), 2).ap == pytest.approx(2 / 3)
    # recall tertinggi hanya 0,5 → AP maksimal 0,5
    assert pr_curve(np.array([0.9, 0.8]), np.array([True, True]), 4).ap == pytest.approx(0.5)
    assert pr_curve(np.array([]), np.array([], dtype=bool), 3).ap == 0.0
    assert math.isnan(pr_curve(np.array([0.5]), np.array([False]), 0).ap)
    assert average_precision(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)


def test_ap_depends_only_on_score_order():
    flags = np.array([True, False, True, True, False])
    raw = np.array([0.95, 0.9, 0.5, 0.3, 0.1])
    haar_like = raw * 12 - 4                       # levelWeights bisa negatif
    assert pr_curve(raw, flags, 4).ap == pytest.approx(pr_curve(haar_like, flags, 4).ap)


def test_ap_pools_images_and_marks_duplicates():
    gt = [(0, 0, 10, 10)]
    a = score_image("a", "S01", gt, [(0, 0, 10, 10), (1, 0, 10, 10)], [0.9, 0.8], 0.5)
    b = score_image("b", "S02", gt, [(50, 50, 10, 10)], [0.95], 0.5)
    assert a.is_tp.tolist() == [True, False]    # duplikat → FP
    curve = pool([a, b])
    # urutan: b FP (0,95), a TP (0,9), a FP (0,8); 2 GT → recall 0, 1/2, 1/2
    assert curve.ap == pytest.approx(0.5 * 0.5)
    assert pool([a, a]).n_gt == 2              # citra berulang (bootstrap) dihitung dua kali


# ---------------------------------------------------------------------------
# Multi-wajah
# ---------------------------------------------------------------------------
def test_count_metrics():
    assert count_accuracy([2, 3, 1], [2, 2, 1]) == (2, 3)
    assert count_mae([2, 3, 1], [2, 2, 1]) == pytest.approx(1 / 3)
    assert math.isnan(count_mae([], []))
    with pytest.raises(ValueError):
        count_accuracy([1], [1, 2])


def test_positions_follow_left_to_right_order():
    gt = [(500, 0, 50, 60), (100, 0, 80, 100), (300, 0, 60, 80)]
    assert positions_of(gt, [80, 150, 250]) == [250, 80, 150]
    assert position_hits(gt, [True, False, True], [80, 150, 250]) == [(250, True), (80, False), (150, True)]
    with pytest.raises(ValueError):
        positions_of(gt, [80, 150])


# ---------------------------------------------------------------------------
# Jarak
# ---------------------------------------------------------------------------
def test_loglog_fit_recovers_pinhole_model():
    distances = [50, 100, 150, 200, 250, 300] * 3
    fit = loglog_fit(distances, [15000 / d for d in distances])
    assert fit.slope == pytest.approx(-1.0) and fit.r2 == pytest.approx(1.0)
    assert math.exp(fit.intercept) == pytest.approx(15000)
    assert fit.equation == "w = 15000 · Z^-1.000"
    with pytest.raises(ValueError):
        loglog_fit([100, 100, 100], [1, 2, 3])


def test_distance_range_handles_non_monotone_recall():
    recall = {50: 1.0, 100: 0.95, 150: 0.9, 200: 0.6, 250: 0.92, 300: 0.2}
    result = distance_range(recall, 0.9)
    assert result.qualifying == (50, 100, 150, 250)
    assert result.max_effective == 250 and result.max_contiguous == 150
    assert distance_range({50: 0.5}, 0.9).max_effective is None


def test_min_face_size_hand_values():
    sizes = [10, 15, 25, 30, 35, 38, 45, 55, 70, 90, 300]
    hits = [False, False, True, True, True, False, True, True, True, True, True]
    edges = [0, 20, 40, 60, 80, 1000]          # bin 60–80 berisi 1, 80–1000 berisi 2
    assert min_face_size(sizes, hits, edges, 0.9).threshold == 40   # bin 20–40 recall 0,75
    assert min_face_size(sizes, hits, edges, 0.7).threshold == 20   # bin 0–20 recall 0
    result = min_face_size([10, 90], [False, True], [0, 20, 40, 60, 1000], 0.9)
    assert result.threshold == 60 and [b.n for b in result.bins] == [1, 0, 0, 1]  # bin kosong dilewati
    assert min_face_size([10], [False], [0, 20, 1000], 0.9).threshold is None


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------
def test_wilson_reference_values():
    est = wilson(8, 10)
    assert est.value == 0.8
    assert (est.low, est.high) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9433, abs=1e-4))
    assert wilson(0, 10).low == 0.0 and wilson(0, 10).high == pytest.approx(0.2775, abs=1e-4)
    assert wilson(10, 10).high == 1.0 and wilson(10, 10).low == pytest.approx(0.7225, abs=1e-4)
    assert math.isnan(wilson(0, 0).value)
    with pytest.raises(ValueError):
        wilson(5, 3)


def test_cluster_bootstrap_resamples_groups_deterministically():
    values = {"S01": 0.9, "S02": 0.7, "S03": 0.8, "S04": 1.0}
    stat = lambda keys: float(np.mean([values[k] for k in keys]))  # noqa: E731
    first = cluster_bootstrap(list(values), stat, n_resamples=500, seed=1)
    again = cluster_bootstrap(list(values) * 3, stat, n_resamples=500, seed=1)  # duplikat kode diabaikan
    assert first == again
    assert first.value == pytest.approx(0.85)
    assert first.low <= first.value <= first.high
    flat = cluster_bootstrap(["a", "b"], lambda keys: 0.5, n_resamples=100)
    assert (flat.low, flat.high) == (0.5, 0.5)


def test_paired_difference_detects_consistent_gap():
    base = {"S01": 0.6, "S02": 0.9, "S03": 0.75, "S04": 0.8}
    a = lambda keys: float(np.mean([base[k] + 0.1 for k in keys]))  # noqa: E731
    b = lambda keys: float(np.mean([base[k] for k in keys]))  # noqa: E731
    diff = paired_difference(list(base), a, b, n_resamples=300)
    assert diff.value == pytest.approx(0.1) and diff.low == pytest.approx(0.1) and diff.excludes_zero
    # tanpa pemasangan, interval masing-masing detektor tumpang tindih
    ca, cb = cluster_bootstrap(list(base), a, n_resamples=300), cluster_bootstrap(list(base), b, n_resamples=300)
    assert ca.low < cb.high
