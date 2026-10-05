"""COCO-style mean Average Precision for detection, via torchmetrics.

This is the metric to report. `map_calculator.calculate_map` is NOT mAP: it
averages per-image F1 at a fixed threshold (see docs/LEAKAGE_AUDIT.md).

For a correct AP the predictions should keep low-confidence boxes (e.g. a
score floor of 0.001): AP integrates over the whole precision-recall curve,
and cutting predictions at 0.25 truncates it and understates AP.

Inputs use the repo's existing format: one dict per image,
  prediction: {"boxes": (n, 4) xyxy pixels, "scores": (n,), "labels": (n,)}
  target:     {"boxes": (m, 4) xyxy pixels, "labels": (m,),
               optional "ignore": (k, 4) xyxy ignore regions}

Ignore regions are passed as COCO "iscrowd" boxes: they add no false
negatives, and predictions falling on them are not counted as false
positives. Used for labels that a model-independent rule marked as
uncertain (scripts/apply_label_rule.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision


def _as_tensor(a, shape_tail=(4,), dtype=torch.float32) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a, dtype=np.float64 if dtype == torch.float32 else np.int64), dtype=dtype)
    return t.reshape(-1, *shape_tail) if shape_tail else t.reshape(-1)


def coco_map(predictions: Sequence[dict], targets: Sequence[dict]) -> dict[str, float]:
    """Return COCO metrics: map (0.5:0.95), map_50, map_75, mar_100, per size bucket."""
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    for p, t in zip(predictions, targets):
        n = len(np.asarray(p.get("scores", [])).reshape(-1))
        gt = np.asarray(t.get("boxes", np.empty((0, 4))), dtype=np.float64).reshape(-1, 4)
        ig = np.asarray(t.get("ignore", np.empty((0, 4))), dtype=np.float64).reshape(-1, 4)
        boxes = np.concatenate([gt, ig])
        crowd = torch.cat([torch.zeros(len(gt), dtype=torch.long), torch.ones(len(ig), dtype=torch.long)])
        metric.update(
            [{"boxes": _as_tensor(p.get("boxes", np.empty((0, 4)))),
              "scores": _as_tensor(p.get("scores", []), shape_tail=None),
              "labels": torch.zeros(n, dtype=torch.long)}],
            [{"boxes": _as_tensor(boxes),
              "labels": torch.zeros(len(boxes), dtype=torch.long),
              "iscrowd": crowd}],
        )
    out = metric.compute()
    keys = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100")
    return {k: float(out[k]) for k in keys}


def coco_map50(predictions: Sequence[dict], targets: Sequence[dict]) -> float:
    """COCO AP at IoU 0.5 (single class)."""
    return coco_map(predictions, targets)["map_50"]
