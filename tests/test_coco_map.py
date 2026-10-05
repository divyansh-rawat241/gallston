"""COCO mAP must behave like AP, not like the old per-image F1."""

import numpy as np

from src.evaluation.coco_map import coco_map, coco_map50
from src.evaluation.map_calculator import mean_image_f1


def _t(boxes):
    return {"boxes": np.array(boxes, dtype=float).reshape(-1, 4), "labels": np.zeros(len(boxes))}


def test_perfect_predictions_score_one():
    gt = [_t([[10, 10, 50, 50]]), _t([[20, 20, 60, 60], [100, 100, 140, 140]])]
    pred = [{"boxes": g["boxes"], "scores": np.ones(len(g["boxes"])), "labels": np.zeros(len(g["boxes"]))} for g in gt]
    assert abs(coco_map50(pred, gt) - 1.0) < 1e-6
    assert abs(coco_map(pred, gt)["map"] - 1.0) < 1e-6


def test_low_score_false_positive_does_not_hurt_ap_but_hurts_f1():
    # A false positive ranked BELOW every true positive leaves AP at 1.0,
    # whereas per-image F1 drops: the two metrics are not interchangeable.
    gt = [_t([[10, 10, 50, 50]])]
    pred = [{"boxes": np.array([[10, 10, 50, 50], [200, 200, 240, 240]], float),
             "scores": np.array([0.9, 0.3]), "labels": np.zeros(2)}]
    assert abs(coco_map50(pred, gt) - 1.0) < 1e-6
    assert mean_image_f1(pred, gt) < 0.7


def test_missed_object_lowers_ap():
    gt = [_t([[10, 10, 50, 50], [100, 100, 140, 140]])]
    pred = [{"boxes": np.array([[10, 10, 50, 50]], float), "scores": np.array([0.9]), "labels": np.zeros(1)}]
    assert coco_map50(pred, gt) < 0.6


def test_ignore_region_removes_false_positive():
    gt = np.array([[10, 10, 30, 30]])
    preds = [{"boxes": np.array([[10, 10, 30, 30], [100, 100, 120, 120]]), "scores": np.array([0.9, 0.95])}]
    assert coco_map(preds, [{"boxes": gt}])["map_50"] < 0.6
    assert coco_map(preds, [{"boxes": gt, "ignore": np.array([[100, 100, 120, 120]])}])["map_50"] == 1.0
