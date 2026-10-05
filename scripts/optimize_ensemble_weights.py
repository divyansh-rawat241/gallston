"""
Ensemble Weight Optimization Script
Optimizes ensemble weights using grid search on validation set
"""
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
from src.models.ensemble import GallstoneEnsemble
from src.evaluation.coco_map import coco_map50
import torch

def load_models():
    """Load trained models for ensemble"""
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
        rfdetr_path = "weights/rf_detr/best_model.pth"
        if Path(rfdetr_path).exists():
            trainer = RFDETRTrainer({})
            rfdetr_model = trainer.load_model(rfdetr_path)
            models.append(rfdetr_model)
            print(f"Loaded RF-DETR model from {rfdetr_path}")
    except Exception as e:
        print(f"Failed to load RF-DETR model: {e}")

    return models

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

def optimize_weights(models, val_images, val_labels, n_steps=20):
    """Optimize ensemble weights using grid search"""
    if len(models) < 2:
        print("Need at least 2 models for weight optimization")
        return [1.0]

    print(f"Optimizing weights for {len(models)} models...")

    best_map = 0.0
    best_weights = [1.0 / len(models)] * len(models)

    # Generate weight combinations
    weight_combinations = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            if len(models) == 2:
                weights = [i / n_steps, j / n_steps]
            elif len(models) == 3:
                weights = [i / n_steps, j / n_steps, k / n_steps]
            else:
                weights = [1.0 / len(models)] * len(models)

            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                weight_combinations.append(weights)

    print(f"Testing {len(weight_combinations)} weight combinations...")

    for weights in tqdm(weight_combinations):
        ensemble = GallstoneEnsemble(models, weights=weights)

        # Evaluate on validation set
        all_predictions = []
        all_targets = []

        for img, target_boxes in zip(val_images, val_labels):
            # Get predictions
            pred_boxes, pred_scores, pred_labels = ensemble.predict(img, conf_threshold=0.25)

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
                'labels': np.zeros(len(target_boxes))  # All gallstones
            })

        # Calculate mAP
        try:
            current_map = coco_map50(all_predictions, all_targets)
            if current_map > best_map:
                best_map = current_map
                best_weights = weights
                print(f"New best mAP50: {best_map:.4f} with weights: {[f'{w:.3f}' for w in best_weights]}")
        except Exception as e:
            print(f"Error calculating mAP: {e}")
            continue

    print(f"\nOptimization complete!")
    print(f"Best mAP50: {best_map:.4f}")
    print(f"Best weights: {[f'{w:.3f}' for w in best_weights]}")

    return best_weights

def main():
    print("Loading models...")
    models = load_models()

    if len(models) < 2:
        print("Need at least 2 trained models for ensemble optimization")
        return

    print("Loading validation data...")
    val_images, val_labels = load_validation_data()

    if len(val_images) == 0:
        print("No validation data found")
        return

    print("Starting weight optimization...")
    best_weights = optimize_weights(models, val_images, val_labels, n_steps=15)

    # Save best weights
    output_path = Path("configs/ensemble_weights.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'weights': best_weights,
            'map50': best_map if 'best_map' in locals() else 0.0
        }, f, indent=2)

    print(f"Saved optimized weights to {output_path}")

if __name__ == "__main__":
    main()