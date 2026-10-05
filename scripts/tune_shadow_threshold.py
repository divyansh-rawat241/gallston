"""
Shadow Analyzer Threshold Tuning Script.

Sweeps `ShadowAnalyzer.threshold` in [lo, hi] and records COCO mAP50 on the
chosen split (per-image F1 was reported before 2026-10-01; see
docs/LEAKAGE_AUDIT.md).

FIXED (2026-08-09):
  - Reads real 368/201-image split from `dataset_final_resplit/{val,test}`
    instead of stale 12-image `data/processed/val`.
  - Uses polygon-aware label loader (labels are polygons, not YOLO bboxes).
  - Loads weights from `production_models/` by default.
  - Shares `load_validation_data` + `load_ensemble` with the comprehensive
    evaluation script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.comprehensive_evaluation import load_ensemble, load_validation_data
from src.evaluation.coco_map import coco_map50
from src.evaluation.shadow_analyzer import ShadowAnalyzer


def tune_shadow_threshold(
    ensemble,
    val_images,
    val_labels,
    threshold_range=(0.5, 0.9),
    n_steps: int = 20,
    conf_threshold: float = 0.25,
):
    """Grid-search shadow threshold that maximises per-image mean F1."""
    print(f"Tuning shadow threshold in {threshold_range} over {n_steps} steps...")

    best_metric = 0.0
    best_threshold = float(threshold_range[0])
    results = []

    for threshold in tqdm(np.linspace(threshold_range[0], threshold_range[1], n_steps)):
        shadow_analyzer = ShadowAnalyzer(threshold=float(threshold))
        all_predictions, all_targets = [], []

        for img, target_boxes in zip(val_images, val_labels):
            pred_boxes, pred_scores, pred_labels = ensemble.predict(
                img, conf_threshold=conf_threshold
            )

            h, w = img.shape[:2]
            if len(pred_boxes) > 0:
                pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
                pred_scores = np.asarray(pred_scores, dtype=np.float32)
                pred_labels = np.asarray(pred_labels, dtype=np.int64)
                if pred_boxes.max() <= 1.05:
                    pred_boxes_pixel = pred_boxes * [w, h, w, h]
                else:
                    pred_boxes_pixel = pred_boxes
            else:
                pred_boxes_pixel = np.empty((0, 4), dtype=np.float32)
                pred_scores = np.empty(0, dtype=np.float32)
                pred_labels = np.empty(0, dtype=np.int64)

            if len(pred_boxes_pixel) > 0:
                adjusted = shadow_analyzer.apply_to_ensemble(
                    img, pred_boxes_pixel, pred_scores
                )
                keep = adjusted >= conf_threshold
                pred_boxes_pixel = pred_boxes_pixel[keep]
                pred_scores = adjusted[keep]
                pred_labels = pred_labels[keep]

            all_predictions.append(
                {"boxes": pred_boxes_pixel, "scores": pred_scores, "labels": pred_labels}
            )
            all_targets.append(
                {"boxes": target_boxes, "labels": np.zeros(len(target_boxes), dtype=np.int64)}
            )

        try:
            metric = coco_map50(all_predictions, all_targets)
        except Exception as exc:
            print(f"  metric calc failed at threshold={threshold:.3f}: {exc}")
            continue

        results.append({"threshold": float(threshold), "metric": float(metric)})
        if metric > best_metric:
            best_metric = float(metric)
            best_threshold = float(threshold)
            print(f"  new best {best_metric:.4f} at threshold {best_threshold:.3f}")

    print("\nShadow threshold tuning complete.")
    print(f"Best metric: {best_metric:.4f}  threshold: {best_threshold:.3f}")
    return best_threshold, best_metric, results


def main():
    parser = argparse.ArgumentParser(description="Tune ShadowAnalyzer threshold")
    parser.add_argument("--data-dir", default="dataset_final_resplit")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--yolo-weights", default="production_models/yolo_best.pt")
    parser.add_argument("--rfdetr-weights", default="production_models/rfdetr_best.pth")
    parser.add_argument("--lo", type=float, default=0.5)
    parser.add_argument("--hi", type=float, default=0.9)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--out", default="runs/final_eval/shadow_threshold_tuning.json")
    args = parser.parse_args()

    print(f"Loading ensemble  yolo={args.yolo_weights}  rfdetr={args.rfdetr_weights}")
    ensemble = load_ensemble(args.yolo_weights, args.rfdetr_weights)
    if ensemble is None:
        print("Failed to load ensemble")
        return

    print(f"Loading data  {args.data_dir}/{args.split}")
    val_images, val_labels = load_validation_data(args.data_dir, args.split)
    if len(val_images) == 0:
        print("No data found")
        return

    best_threshold, best_metric, results = tune_shadow_threshold(
        ensemble, val_images, val_labels,
        threshold_range=(args.lo, args.hi), n_steps=args.n_steps,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "split": args.split,
                "data_dir": args.data_dir,
                "metric": "coco_map50",
                "best_threshold": best_threshold,
                "best_metric": best_metric,
                "all_results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
