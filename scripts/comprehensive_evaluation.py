"""
Comprehensive Evaluation Script
Combines all optimizations: ensemble weights, shadow threshold, and TTA scales.

FIXED (2026-08-09):
  - Reads from `dataset_final_resplit/{split}` (default: test=201) instead of
    the stale 12-image `data/processed/val` split.
  - Uses `src.evaluation.label_io.load_bboxes` which handles the polygon label
    rows correctly. The old parser assumed `cls cx cy w h` and read the first
    four polygon vertices as a bbox, producing garbage boxes and zero mAP.
  - Loads weights from `production_models/` by default; overridable via CLI.
  - Reports COCO mAP50 (src/evaluation/coco_map.py). Before 2026-10-01 it
    reported per-image mean F1 under the name mAP (docs/LEAKAGE_AUDIT.md).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow `python scripts/comprehensive_evaluation.py` from repo root without
# needing to set PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.label_io import iter_split
from src.evaluation.coco_map import coco_map50
from src.evaluation.shadow_analyzer import ShadowAnalyzer
from src.models.ensemble import GallstoneEnsemble


def load_validation_data(data_dir="dataset_final_resplit", split="test"):
    """Load evaluation images + pixel bboxes for a YOLO-style split."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return [], []

    images, labels = [], []
    for _stem, img, bboxes in iter_split(split_dir):
        images.append(img)
        labels.append(bboxes)

    print(f"Loaded {len(images)} images from {split_dir}")
    return images, labels

def load_ensemble(yolo_weights: str = "production_models/yolo_best.pt",
                  rfdetr_weights: str = "production_models/rfdetr_best.pth"):
    """Load trained ensemble from production weights."""
    models = []

    try:
        from ultralytics import YOLO
        if Path(yolo_weights).exists():
            models.append(YOLO(yolo_weights))
            print(f"[ok] YOLO loaded from {yolo_weights}")
        else:
            print(f"[miss] YOLO weights not found at {yolo_weights}")
    except Exception as e:
        print(f"[err] YOLO load failed: {e}")

    try:
        from rfdetr import RFDETRLarge
        if Path(rfdetr_weights).exists():
            rf = RFDETRLarge(pretrain_weights=rfdetr_weights)
            try:
                rf.optimize_for_inference()
            except Exception:
                pass
            models.append(rf)
            print(f"[ok] RF-DETR loaded from {rfdetr_weights}")
        else:
            print(f"[miss] RF-DETR weights not found at {rfdetr_weights}")
    except Exception as e:
        print(f"[err] RF-DETR load failed: {e}")

    if len(models) == 0:
        print("✗ No models loaded!")
        return None

    # Try to load optimized weights
    weights_path = Path("configs/ensemble_weights.json")
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            config = json.load(f)
            weights = config.get('weights', [0.5, 0.5])
            print(f"✓ Using optimized weights: {[f'{w:.3f}' for w in weights]}")
    else:
        weights = [0.5, 0.5]
        print(f"Using default weights: {[f'{w:.3f}' for w in weights]}")

    return GallstoneEnsemble(models, weights=weights)

def evaluate_configuration(ensemble, val_images, val_labels, config_name, use_shadow=False, shadow_threshold=0.7, use_tta=False, tta_scales=None):
    """Evaluate a specific configuration"""
    print(f"\nEvaluating: {config_name}")

    shadow_analyzer = ShadowAnalyzer(threshold=shadow_threshold) if use_shadow else None

    all_predictions = []
    all_targets = []

    for img, target_boxes in tqdm(zip(val_images, val_labels), total=len(val_images), desc=f"  {config_name}"):
        # Get predictions
        if use_tta and tta_scales:
            pred_boxes, pred_scores, pred_labels = ensemble.predict_multi_scale(
                img,
                scales=tta_scales,
                conf_threshold=0.25,
                iou_threshold=0.5
            )
        else:
            pred_boxes, pred_scores, pred_labels = ensemble.predict(img, conf_threshold=0.25, iou_threshold=0.5)

        # Convert to pixel coordinates
        h, w = img.shape[:2]
        if len(pred_boxes) > 0:
            pred_boxes_pixel = pred_boxes * [w, h, w, h]
        else:
            pred_boxes_pixel = np.empty((0, 4))

        # Apply shadow verification if enabled
        if use_shadow and shadow_analyzer and len(pred_boxes_pixel) > 0:
            adjusted_scores = shadow_analyzer.apply_to_ensemble(img, pred_boxes_pixel, pred_scores)

            # Filter by adjusted confidence
            keep_mask = adjusted_scores >= 0.25
            pred_boxes_pixel = pred_boxes_pixel[keep_mask]
            pred_scores = adjusted_scores[keep_mask]
            pred_labels = pred_labels[keep_mask]

        all_predictions.append({
            'boxes': pred_boxes_pixel,
            'scores': pred_scores,
            'labels': pred_labels
        })
        all_targets.append({
            'boxes': target_boxes,
            'labels': np.zeros(len(target_boxes))
        })

    # Calculate mAP
    try:
        current_map = coco_map50(all_predictions, all_targets)
        print(f"  mAP50: {current_map:.4f}")
        return current_map
    except Exception as e:
        print(f"  Error calculating mAP: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Comprehensive gallstone eval")
    parser.add_argument("--data-dir", default="dataset_final_resplit")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--yolo-weights", default="production_models/yolo_best.pt")
    parser.add_argument("--rfdetr-weights", default="production_models/rfdetr_best.pth")
    parser.add_argument("--out", default="runs/final_eval/comprehensive_evaluation.json")
    args = parser.parse_args()

    print("=" * 60)
    print("COMPREHENSIVE EVALUATION - Gallstone Detection")
    print(f"split={args.split}  data={args.data_dir}")
    print("=" * 60)

    print("\nLoading ensemble...")
    ensemble = load_ensemble(args.yolo_weights, args.rfdetr_weights)

    if ensemble is None:
        print("Failed to load ensemble. Exiting.")
        return

    print("Loading evaluation data...")
    val_images, val_labels = load_validation_data(args.data_dir, args.split)

    if len(val_images) == 0:
        print("No validation data found. Exiting.")
        return

    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 60)

    results = {}

    # 1. Baseline (no optimizations)
    print("\n" + "-" * 60)
    print("BASELINE CONFIGURATION")
    print("-" * 60)
    baseline_map = evaluate_configuration(
        ensemble, val_images, val_labels,
        "Baseline (no optimizations)",
        use_shadow=False, use_tta=False
    )
    results['baseline'] = baseline_map

    # 2. With optimized ensemble weights
    print("\n" + "-" * 60)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("-" * 60)
    weights_path = Path("configs/ensemble_weights.json")
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            config = json.load(f)
            optimized_weights = config.get('weights', [0.5, 0.5])
            ensemble.weights = optimized_weights
            print(f"Using optimized weights: {[f'{w:.3f}' for w in optimized_weights]}")

        weights_map = evaluate_configuration(
            ensemble, val_images, val_labels,
            "Optimized ensemble weights",
            use_shadow=False, use_tta=False
        )
        results['optimized_weights'] = weights_map
    else:
        print("No optimized weights found. Run: python scripts/optimize_ensemble_weights.py")

    # 3. With shadow verification
    print("\n" + "-" * 60)
    print("SHADOW VERIFICATION")
    print("-" * 60)
    shadow_path = Path("configs/shadow_threshold_tuning.json")
    if shadow_path.exists():
        with open(shadow_path, 'r') as f:
            config = json.load(f)
            best_threshold = config.get('best_threshold', 0.7)
            print(f"Using optimized shadow threshold: {best_threshold:.3f}")
    else:
        best_threshold = 0.7
        print("Using default shadow threshold: 0.7")

    shadow_map = evaluate_configuration(
        ensemble, val_images, val_labels,
        f"Shadow verification (threshold={best_threshold:.2f})",
        use_shadow=True, shadow_threshold=best_threshold, use_tta=False
    )
    results['shadow_verification'] = shadow_map

    # 4. With multi-scale TTA
    print("\n" + "-" * 60)
    print("MULTI-SCALE TTA")
    print("-" * 60)
    tta_path = Path("configs/tta_scales_optimization.json")
    if tta_path.exists():
        with open(tta_path, 'r') as f:
            config = json.load(f)
            best_scales = config.get('best_scales', [640, 800, 1024])
            print(f"Using optimized TTA scales: {best_scales}")
    else:
        best_scales = [640, 800, 1024]
        print(f"Using default TTA scales: {best_scales}")

    tta_map = evaluate_configuration(
        ensemble, val_images, val_labels,
        f"Multi-scale TTA (scales={best_scales})",
        use_shadow=False, use_tta=True, tta_scales=best_scales
    )
    results['multi_scale_tta'] = tta_map

    # 5. Full optimization (all combined)
    print("\n" + "-" * 60)
    print("FULL OPTIMIZATION (ALL COMBINED)")
    print("-" * 60)
    full_map = evaluate_configuration(
        ensemble, val_images, val_labels,
        "Full optimization (weights + shadow + TTA)",
        use_shadow=True, shadow_threshold=best_threshold,
        use_tta=True, tta_scales=best_scales
    )
    results['full_optimization'] = full_map

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for config_name, map_value in results.items():
        improvement = map_value - baseline_map
        improvement_str = f" (+{improvement:.4f})" if improvement > 0 else f" ({improvement:.4f})"
        print(f"{config_name:25s}: {map_value:.4f}{improvement_str}")

    print("\n" + "=" * 60)
    print(f"FINAL mAP50: {full_map:.4f}")
    if baseline_map > 0:
        print(f"IMPROVEMENT: +{full_map - baseline_map:.4f} ({((full_map/baseline_map - 1) * 100):.2f}%)")
    else:
        print(f"IMPROVEMENT: +{full_map - baseline_map:.4f} (baseline zero)")
    print("=" * 60)

    # Save results (metric here is per-image mean F1, not COCO mAP - see docstring)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    improvement_pct = (
        (full_map / baseline_map - 1) * 100 if baseline_map > 0 else float("nan")
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "data_dir": args.data_dir,
                "yolo_weights": args.yolo_weights,
                "rfdetr_weights": args.rfdetr_weights,
                "metric": "coco_map50",
                "baseline": baseline_map,
                "results": results,
                "final": full_map,
                "improvement": full_map - baseline_map,
                "improvement_percent": improvement_pct,
            },
            f,
            indent=2,
        )

    print(f"\nSaved comprehensive results to {output_path}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if full_map >= 0.97:
        print("✓ TARGET ACHIEVED! mAP50 >= 97%")
        print("  Ready for clinical deployment.")
    elif full_map >= 0.90:
        print("⚠ CLOSE TO TARGET. mAP50 >= 90%")
        print("  Consider additional training data or model architecture improvements.")
    else:
        print("✗ BELOW TARGET. mAP50 < 90%")
        print("  Recommendations:")
        print("  1. Run RF-DETR hyperparameter optimization")
        print("  2. Collect more training data")
        print("  3. Try larger model variants (YOLO26l, RF-DETR base)")
        print("  4. Implement test-time augmentation with more transforms")

if __name__ == "__main__":
    main()