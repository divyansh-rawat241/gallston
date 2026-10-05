"""
Multi-Scale TTA Configuration Optimization
Finds optimal scale combinations for test-time augmentation
"""
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from src.models.ensemble import GallstoneEnsemble
from src.evaluation.coco_map import coco_map50

def load_validation_data(data_dir="data/processed"):
    """Load validation images and labels"""
    val_dir = Path(data_dir) / "val"
    images = []
    labels = []

    img_dir = val_dir / "images"
    label_dir = val_dir / "labels"

    if not img_dir.exists():
        print(f"Validation directory not found: {img_dir}")
        return [], []

    for img_path in img_dir.glob("*.jpg"):
        label_path = label_dir / f"{img_path.stem}.txt"

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load labels
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        x_center, y_center, width, height = map(float, parts[1:5])
                        h, w = img.shape[:2]

                        # Convert to pixel coordinates [x1, y1, x2, y2]
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        boxes.append([x1, y1, x2, y2])

        images.append(img)
        labels.append(np.array(boxes) if len(boxes) > 0 else np.empty((0, 4)))

    print(f"Loaded {len(images)} validation images")
    return images, labels

def load_ensemble():
    """Load trained ensemble"""
    models = []

    # Load YOLO model
    try:
        from ultralytics import YOLO
        yolo_path = "runs/detect/gallstone_detection/yolo26m_gallstone/weights/best.pt"
        if Path(yolo_path).exists():
            yolo_model = YOLO(yolo_path)
            models.append(yolo_model)
            print(f"Loaded YOLO model from {yolo_path}")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")

    # Load RF-DETR model
    try:
        from src.models.rf_detr_trainer import RFDETRTrainer
        rfdetr_path = "weights/rf_detr/checkpoint_best_ema.pth"
        if Path(rfdetr_path).exists():
            trainer = RFDETRTrainer({})
            rfdetr_model = trainer.load_model(rfdetr_path)
            models.append(rfdetr_model)
            print(f"Loaded RF-DETR model from {rfdetr_path}")
    except Exception as e:
        print(f"Failed to load RF-DETR model: {e}")

    if len(models) == 0:
        return None

    # Try to load optimized weights
    weights_path = Path("configs/ensemble_weights.json")
    if weights_path.exists():
        import json
        with open(weights_path, 'r') as f:
            config = json.load(f)
            weights = config.get('weights', [0.5, 0.5])
            print(f"Using optimized weights: {[f'{w:.3f}' for w in weights]}")
    else:
        weights = [0.5, 0.5]

    return GallstoneEnsemble(models, weights=weights)

def evaluate_scales(ensemble, val_images, val_labels, scales, conf_threshold=0.25, iou_threshold=0.5):
    """Evaluate ensemble with specific scale configuration"""
    all_predictions = []
    all_targets = []

    for img, target_boxes in zip(val_images, val_labels):
        # Get predictions with multi-scale TTA
        pred_boxes, pred_scores, pred_labels = ensemble.predict_multi_scale(
            img,
            scales=scales,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # Convert to pixel coordinates
        h, w = img.shape[:2]
        if len(pred_boxes) > 0:
            pred_boxes_pixel = pred_boxes * [w, h, w, h]
        else:
            pred_boxes_pixel = np.empty((0, 4))

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
        return current_map
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        return 0.0

def optimize_tta_scales(ensemble, val_images, val_labels):
    """Optimize multi-scale TTA configuration"""
    print("Optimizing multi-scale TTA configuration...")

    # Define scale combinations to test
    scale_combinations = [
        [640],                          # Single scale (baseline)
        [640, 800],                     # 2 scales
        [640, 1024],                    # 2 scales (larger gap)
        [800, 1024],                    # 2 scales (higher res)
        [640, 800, 1024],               # 3 scales (current)
        [512, 640, 800],                # 3 scales (lower res)
        [640, 800, 1024, 1280],         # 4 scales
        [512, 640, 800, 1024],          # 4 scales (wider range)
    ]

    best_map = 0.0
    best_scales = [640, 800, 1024]
    results = []

    for scales in tqdm(scale_combinations):
        print(f"\nTesting scales: {scales}")

        try:
            current_map = evaluate_scales(ensemble, val_images, val_labels, scales)

            results.append({
                'scales': scales,
                'map50': current_map,
                'num_scales': len(scales)
            })

            if current_map > best_map:
                best_map = current_map
                best_scales = scales
                print(f"✓ New best mAP50: {best_map:.4f} with scales: {best_scales}")
            else:
                print(f"  mAP50: {current_map:.4f}")

        except Exception as e:
            print(f"Error with scales {scales}: {e}")
            continue

    print(f"\nMulti-scale TTA optimization complete!")
    print(f"Best mAP50: {best_map:.4f}")
    print(f"Best scales: {best_scales}")

    # Save results
    output_path = Path("configs/tta_scales_optimization.json")
    output_path.parent.mkdir(exist_ok=True)

    import json
    with open(output_path, 'w') as f:
        json.dump({
            'best_scales': best_scales,
            'best_map50': best_map,
            'all_results': results
        }, f, indent=2)

    print(f"Saved optimization results to {output_path}")

    return best_scales, best_map

def main():
    print("Loading ensemble...")
    ensemble = load_ensemble()

    if ensemble is None:
        print("Failed to load ensemble")
        return

    print("Loading validation data...")
    val_images, val_labels = load_validation_data()

    if len(val_images) == 0:
        print("No validation data found")
        return

    print("Starting multi-scale TTA optimization...")
    best_scales, best_map = optimize_tta_scales(ensemble, val_images, val_labels)

    print(f"\nOptimal TTA scales: {best_scales}")
    print(f"Resulting mAP50: {best_map:.4f}")

if __name__ == "__main__":
    main()