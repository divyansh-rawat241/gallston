"""
Per-image mean F1 at a fixed IoU. NOT mean Average Precision.

Historically exported as `calculate_map`, which made F1 look like mAP in every
report (see docs/LEAKAGE_AUDIT.md). Use `src.evaluation.coco_map` for mAP.
"""
import warnings
import numpy as np

def mean_image_f1(predictions, targets, iou_threshold=0.5):
    """
    Calculate mean Average Precision at given IoU threshold

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        targets: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching

    Returns:
        mAP value
    """
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0

    all_precisions = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get('boxes', np.empty((0, 4)))
        pred_scores = pred.get('scores', np.empty(0))
        target_boxes = target.get('boxes', np.empty((0, 4)))

        if len(target_boxes) == 0:
            # No ground truth boxes
            if len(pred_boxes) == 0:
                all_precisions.append(1.0)  # Perfect: no predictions, no ground truth
            else:
                all_precisions.append(0.0)  # False positives
            continue

        if len(pred_boxes) == 0:
            # No predictions but there are ground truth boxes
            all_precisions.append(0.0)  # False negatives
            continue

        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Calculate IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)

        # Match predictions to targets
        matched_targets = set()
        true_positives = 0
        false_positives = 0

        for pred_idx in range(len(pred_boxes)):
            best_iou = 0.0
            best_target_idx = -1

            for target_idx in range(len(target_boxes)):
                if target_idx in matched_targets:
                    continue

                iou = iou_matrix[pred_idx, target_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_targets.add(best_target_idx)
            else:
                false_positives += 1

        # Calculate precision and recall
        false_negatives = len(target_boxes) - len(matched_targets)

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        # Use F1 score as a simple metric
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        all_precisions.append(f1)

    # Return mean of all precisions
    return np.mean(all_precisions)

def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes

    Args:
        boxes1: Array of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4) in format [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)
    """
    n = len(boxes1)
    m = len(boxes2)

    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    return iou_matrix

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def calculate_map(predictions, targets, iou_threshold=0.5):
    """Deprecated misnomer: returns mean per-image F1, not mAP."""
    warnings.warn("calculate_map returns mean per-image F1, not mAP; use "
                  "src.evaluation.coco_map.coco_map50 or mean_image_f1", DeprecationWarning, stacklevel=2)
    return mean_image_f1(predictions, targets, iou_threshold)
