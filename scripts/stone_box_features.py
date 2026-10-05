"""Image features of every labelled stone box, for a model-independent label rule.

For labelling convention A ("label only definite stones"), each polygon label
is described by quantities that do not depend on any model:
  contrast   - mean intensity inside the polygon minus the median of a ring
               of surrounding pixels (lumen / wall around the echo)
  peak       - 95th-percentile intensity inside the polygon
  shadow     - mean intensity of lateral bands at the same depth minus the
               band directly below the box (positive = posterior shadow)
  area_frac  - box area / image area
Reads the grouped split (cleaned greyscale images + labels_polygon) and writes
a CSV, one row per box.

Usage: python scripts/stone_box_features.py --data dataset_grouped_neg2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.label_io import load_polygon_labels  # noqa: E402


def features(grey: np.ndarray, poly: np.ndarray) -> dict:
    h, w = grey.shape
    pts = (poly.reshape(-1, 2) * [w, h]).astype(np.int32)
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    inside = np.zeros_like(grey, np.uint8)
    cv2.fillPoly(inside, [pts], 1)
    if inside.sum() < 3:
        cv2.rectangle(inside, (x1, y1), (x2, y2), 1, -1)
    m = max(bw, bh)
    ring = np.zeros_like(grey, np.uint8)
    cv2.rectangle(ring, (max(0, x1 - m), max(0, y1 - m)), (min(w - 1, x2 + m), min(h - 1, y2 + m)), 1, -1)
    ring[cv2.dilate(inside, np.ones((5, 5), np.uint8)) > 0] = 0
    g = grey.astype(np.float32)
    vin = g[inside > 0]
    vring = g[ring > 0] if ring.any() else vin
    # shadow: band below the box vs same-depth bands left and right of it
    sy1, sy2 = min(h, y2 + 2), min(h, y2 + 2 + int(1.5 * bh) + 4)
    below = g[sy1:sy2, x1:x2 + 1]
    lat = np.concatenate([g[sy1:sy2, max(0, x1 - bw):x1].ravel(), g[sy1:sy2, x2 + 1:min(w, x2 + 1 + bw)].ravel()])
    shadow = float(lat.mean() - below.mean()) if below.size and lat.size else 0.0
    return {"contrast": round(float(vin.mean() - np.median(vring)), 2),
            "peak": round(float(np.percentile(vin, 95)), 1),
            "shadow": round(shadow, 2),
            "area_frac": round(bw * bh / (w * h), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_grouped_neg2")
    ap.add_argument("--out", default="runs/label_rule/box_features.csv")
    args = ap.parse_args()
    data = ROOT / args.data
    man = json.loads((data / "split_manifest.json").read_text())["images"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["name", "split", "group", "box", "contrast", "peak", "shadow", "area_frac"])
        wr.writeheader()
        for name, m in man.items():
            if name.startswith("neg_"):
                continue
            stem = Path(name).stem
            grey = cv2.imread(str(data / m["split"] / "images" / name), cv2.IMREAD_GRAYSCALE)
            for i, (_, poly) in enumerate(load_polygon_labels(data / m["split"] / "labels_polygon" / f"{stem}.txt")):
                wr.writerow({"name": name, "split": m["split"], "group": m["group"], "box": i, **features(grey, np.asarray(poly))})
                n += 1
    print(f"wrote {n} boxes to {out}")


if __name__ == "__main__":
    main()
