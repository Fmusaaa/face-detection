"""IoU dan pencocokan — nilai acuan dihitung tangan."""

import pytest

from pcdface.evaluation.matching import iou, iou_matrix, match_by_score, match_greedy


def test_iou_identical_disjoint_and_partial():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0  # hanya bersentuhan
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    # kotak kecil di dalam kotak besar: 25 / 100
    assert iou((0, 0, 10, 10), (0, 0, 5, 5)) == pytest.approx(0.25)


def test_iou_is_symmetric_and_handles_empty_box():
    a, b = (3, 4, 20, 11), (10, 1, 7, 30)
    assert iou(a, b) == pytest.approx(iou(b, a))
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_square_box_iou_ceiling_against_tall_box():
    # PRD §5.1: kotak persegi selebar kotak manual berasio w/h 0,75 → IoU maks 0,75
    assert iou((0, 0, 75, 75), (0, 0, 75, 100)) == pytest.approx(0.75)


def test_iou_matrix_shape():
    assert iou_matrix([(0, 0, 1, 1)] * 2, [(0, 0, 1, 1)] * 3).shape == (2, 3)
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)


def test_greedy_takes_highest_iou_first():
    gt = [(0, 0, 10, 10), (9, 0, 10, 10)]
    preds = [(4, 0, 10, 10), (0, 0, 10, 10)]
    m = match_greedy(gt, preds, 0.3)
    assert (m.true_positive, m.false_positive, m.false_negative) == (2, 0, 0)
    assert sorted(m.pairs) == [(0, 1, 1.0), (1, 0, pytest.approx(1 / 3))]
    assert m.pred_matched == [True, True] and m.gt_matched == [True, True]


def test_greedy_threshold_and_duplicates():
    gt = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (1, 0, 10, 10), (50, 50, 5, 5)]
    m = match_greedy(gt, preds, 0.5)
    assert (m.true_positive, m.false_positive, m.false_negative) == (1, 2, 0)
    assert m.matched_ious == [1.0]
    # IoU 1/3 di bawah ambang 0,5 → tidak cocok
    m = match_greedy([(0, 0, 10, 10)], [(5, 0, 10, 10)], 0.5)
    assert (m.true_positive, m.false_positive, m.false_negative) == (0, 1, 1)


def test_greedy_empty_inputs():
    assert match_greedy([], [], 0.5).true_positive == 0
    m = match_greedy([], [(0, 0, 5, 5)], 0.5)
    assert (m.false_positive, m.false_negative) == (1, 0)
    m = match_greedy([(0, 0, 5, 5)], [], 0.5)
    assert (m.false_positive, m.false_negative) == (0, 1)


def test_match_by_score_voc_duplicate_is_fp():
    gt = [(0, 0, 10, 10), (9, 0, 10, 10)]
    preds = [(4, 0, 10, 10), (0, 0, 10, 10)]
    # p0 (skor tertinggi) memilih GT dengan IoU terbesar (A: 0,429); p1 lalu duplikat
    assert match_by_score(gt, preds, [0.9, 0.8], 0.3) == [True, False]
    # urutan skor dibalik: p1 mengambil A (IoU 1), p0 lalu ke A lagi → FP menurut VOC
    assert match_by_score(gt, preds, [0.8, 0.9], 0.3) == [False, True]


def test_match_by_score_no_gt_and_length_check():
    assert match_by_score([], [(0, 0, 5, 5)], [0.5], 0.5) == [False]
    with pytest.raises(ValueError):
        match_by_score([], [(0, 0, 5, 5)], [], 0.5)
