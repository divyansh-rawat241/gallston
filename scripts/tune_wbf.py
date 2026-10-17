"""
tune_wbf.py
-----------
Tune Weighted Boxes Fusion for the YOLO + RF-DETR ensemble.

Problem: eval_grouped.py fuses with hardcoded equal weights and conf_type="avg".
Averaging deflates fused scores below the 0.25 operating threshold, collapsing
recall (0.64 RF-DETR alone -> 0.25 ensemble). This script runs inference once,
caches predictions, then grids over weights x conf_type x operating threshold
using true COCO metrics -- no retraining, minutes not hours.

Usage:
    python scripts/tune_wbf.py --data Dataset/dataset_grouped_neg3 --split valid \
        --yolo runs/yolo_full/yolo26m/weights/best.pt \
        --rfdetr runs/rfdetr_full/rfdetr_large/checkpoint_best_total.pth \
        --rfdetr-resolution 704
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_grouped import (  # noqa: E402
    load_split,
    yolo_fn,
    rfdetr_fn,
    wbf,
    metrics_with_ci,
    image_level,
)

OPERATING_CONFS = [0.10, 0.15, 0.20, 0.25]
WEIGHT_GRID = [(w, round(1 - w, 2)) for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
CONF_TYPES = ("avg", "max")


def wbf_weighted(per_model, w, h, weights, iou=0.5, conf_type="avg"):
    from ensemble_boxes import weighted_boxes_fusion
    scale = np.array([w, h, w, h], np.float32)
    boxes = [np.clip(b / scale, 0, 1).tolist() for b, _ in per_model]
    scores = [s.tolist() for _, s in per_model]
    labels = [[0] * len(s) for _, s in per_model]
    b, s, _ = weighted_boxes_fusion(
        boxes, scores, labels, weights=list(weights),
        iou_thr=iou, skip_box_thr=0.001, conf_type=conf_type)
    return (np.asarray(b, np.float32).reshape(-1, 4) * scale,
            np.asarray(s, np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--rfdetr", required=True)
    ap.add_argument("--rfdetr-resolution", type=int, default=704)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--out", default="runs/wbf_tuning.json")
    args = ap.parse_args()

    data = ROOT / args.data
    rows = load_split(data, args.split)
    print(f"{args.split}: {len(rows)} images")

    print("running YOLO inference...")
    yf = yolo_fn(args.yolo)
    y_out = [yf(r["img"]) for r in rows]
    print("running RF-DETR inference...")
    rf = rfdetr_fn(args.rfdetr, args.rfdetr_resolution, keep_class=None)
    r_out = [rf(r["img"]) for r in rows]
    print("inference done.")

    results = []
    for (wy, wr), ct in itertools.product(WEIGHT_GRID, CONF_TYPES):
        fused = [wbf_weighted([y_out[i], r_out[i]],
                              r["img"].shape[1], r["img"].shape[0],
                              weights=(wy, wr), conf_type=ct)
                 for i, r in enumerate(rows)]
        preds = [{"boxes": b, "scores": s, "labels": np.zeros(len(s))}
                 for b, s in fused]
        m = metrics_with_ci(rows, preds, args.bootstrap)
        # threshold metrics at several operating points (cheap, no re-inference)
        thr = {}
        for conf in OPERATING_CONFS:
            il = image_level(rows, preds, conf=conf)
            thr[str(conf)] = {"stone_recall": il["stone_recall"],
                              "fp_per_image": il["fp_per_image"],
                              "neg_images_with_fp": il.get("neg_images_with_fp")}
        results.append({"weights": [wy, wr], "conf_type": ct,
                        "map": m["map"], "map_50": m["map_50"],
                        "thresholds": thr})
        print(f"w=({wy},{wr}) {ct:3s}  mAP {m['map']:.4f}  mAP50 {m['map_50']:.4f}  "
              f"R@0.25={thr['0.25']['stone_recall']:.3f}  "
              f"R@0.15={thr['0.15']['stone_recall']:.3f}")

    results.sort(key=lambda r: (r["map"], r["map_50"]), reverse=True)
    print("\n=== top 5 by mAP ===")
    for r in results[:5]:
        print(f"  w={r['weights']} {r['conf_type']:3s}  "
              f"mAP {r['map']:.4f}  mAP50 {r['map_50']:.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")
    print(f"\nBEST: w={results[0]['weights']} {results[0]['conf_type']}  "
          f"mAP {results[0]['map']:.4f}")


if __name__ == "__main__":
    main()
