"""Rank images where the labels and out-of-fold model predictions disagree.

Input: per-image OOF predictions from scripts/eval_cv.py --save-preds. Each
image was predicted by a model that never saw its video, so a confident
disagreement points at either a model error or a label problem. Reviewing the
top of this list is the cheapest way to find label noise (cleanlab-style
auditing; docs/research/training_techniques.md #2).

Per image, at the operating threshold:
  missed      - labelled stone with no prediction at IoU >= 0.5
  extra       - confident prediction (>= --extra-conf) matching no label
  merge/split - one label box covering >= 2 predictions (cluster labelled as
                one box) or one prediction covering >= 2 labels
  neg_fp      - confident prediction on a stone-free image
Output: CSV ranked by a disagreement score, and contact sheets
(label boxes green, predictions red with score) of the top N.

Usage: python scripts/label_audit.py --preds runs/cv/rf704/oof_preds.json --data dataset_grouped_neg2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def inside(small: np.ndarray, big: np.ndarray, frac: float = 0.7) -> np.ndarray:
    """[i, j] = share of small box i lying inside big box j >= frac."""
    if not len(small) or not len(big):
        return np.zeros((len(small), len(big)), bool)
    x1 = np.maximum(small[:, None, 0], big[None, :, 0]); y1 = np.maximum(small[:, None, 1], big[None, :, 1])
    x2 = np.minimum(small[:, None, 2], big[None, :, 2]); y2 = np.minimum(small[:, None, 3], big[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (small[:, 2] - small[:, 0]) * (small[:, 3] - small[:, 1])
    return inter / (a[:, None] + 1e-9) >= frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="runs/cv/rf704/oof_preds.json")
    ap.add_argument("--data", default="dataset_grouped_neg2", help="dataset holding the images (any split)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--extra-conf", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default="runs/cv/rf704/label_audit")
    args = ap.parse_args()

    data, out = ROOT / args.data, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    paths = {p.name: p for p in data.glob("*/images/*")}
    recs = []
    for r in json.loads((ROOT / args.preds).read_text()):
        gt = np.asarray(r["gt"], np.float32).reshape(-1, 4)
        b = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
        s = np.asarray(r["scores"], np.float32)
        keep = s >= args.conf
        b, s = b[keep], s[keep]
        m = iou(gt, b)
        best_for_gt = m.max(1) if len(b) else np.zeros(len(gt))
        best_for_pred = m.max(0) if len(gt) else np.zeros(len(b))
        missed = int((best_for_gt < 0.5).sum())
        unmatched = best_for_pred < 0.5
        extra = int((unmatched & (s >= args.extra_conf)).sum())
        merge = int((inside(b, gt).sum(0) >= 2).sum())   # label box holding >= 2 predictions
        split = int((inside(gt, b).sum(0) >= 2).sum())   # prediction holding >= 2 labels
        neg_fp = int(len(gt) == 0 and (s >= args.extra_conf).any())
        top_unmatched = float(s[unmatched].max()) if unmatched.any() else 0.0
        score = 2 * missed + extra + 1.5 * (merge + split) + 2 * neg_fp + top_unmatched
        recs.append({"name": r["name"], "group": r["group"], "fold": r["fold"], "n_gt": len(gt),
                     "n_pred": int(len(b)), "missed": missed, "extra": extra, "merge": merge, "split": split,
                     "neg_fp": neg_fp, "top_unmatched_score": round(top_unmatched, 3), "score": round(score, 3),
                     "_gt": gt, "_b": b, "_s": s})
    recs.sort(key=lambda x: -x["score"])

    cols = [k for k in recs[0] if not k.startswith("_")]
    with open(out / "label_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in recs:
            w.writerow({k: x[k] for k in cols})

    flagged = [x for x in recs if x["score"] > 0]
    by_kind = {k: sum(1 for x in recs if x[k]) for k in ("missed", "extra", "merge", "split", "neg_fp")}
    print(f"{len(recs)} images, {len(flagged)} with any disagreement; images per kind: {by_kind}")
    by_group: dict[str, float] = {}
    for x in flagged:
        by_group[x["group"]] = by_group.get(x["group"], 0) + x["score"]
    print("groups with most disagreement:", sorted(by_group.items(), key=lambda kv: -kv[1])[:10])

    tiles = []
    for x in recs[:args.top]:
        img = cv2.imread(str(paths[x["name"]]))
        for bx in x["_gt"]:
            cv2.rectangle(img, tuple(map(int, bx[:2])), tuple(map(int, bx[2:])), (0, 220, 0), 2)
        for bx, sc in zip(x["_b"], x["_s"]):
            cv2.rectangle(img, tuple(map(int, bx[:2])), tuple(map(int, bx[2:])), (0, 0, 255), 2)
            cv2.putText(img, f"{sc:.2f}", (int(bx[0]), max(12, int(bx[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        h = 300
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        img = cv2.copyMakeBorder(img, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tag = f"{x['name'][:34]} m{x['missed']} e{x['extra']} mg{x['merge']} sp{x['split']}"
        cv2.putText(img, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(cv2.resize(img, (420, 322)))
    for i in range(0, len(tiles), 12):
        page = tiles[i:i + 12]
        page += [np.full_like(tiles[0], 255)] * (12 - len(page))
        sheet = np.vstack([np.hstack(page[r:r + 4]) for r in (0, 4, 8)])
        cv2.imwrite(str(out / f"sheet_{i // 12 + 1:02d}.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(f"wrote {out / 'label_audit.csv'} and {len(range(0, len(tiles), 12))} contact sheets")


if __name__ == "__main__":
    main()
