"""Evaluate trained detectors on the grouped TEST split with COCO mAP.

Reports, per model (and for the WBF ensemble when both are given):
  * COCO mAP50:95, mAP50, mAP75, mAR100
  * 95% confidence intervals from a video-grouped bootstrap (whole videos are
    resampled, because frames of one video are not independent)
  * image level ("does this image contain a stone?", score = max box score):
    AUROC when stone-free neg_* images are present, stone recall and false
    positives per image at the operating threshold

Predictions keep boxes down to a 0.001 score floor so AP sees the full
precision-recall curve. Run this ONCE per finished model: tuning anything
against these numbers would turn the test set into a second validation set.

Usage:
  python scripts/eval_grouped.py --yolo runs/grouped/yolo26l/weights/best.pt \
                                 --rfdetr runs/grouped/rfdetr_large/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.coco_map import coco_map  # noqa: E402
from src.evaluation.label_io import load_bboxes  # noqa: E402

SCORE_FLOOR = 0.001
OPERATING_CONF = 0.25
TTA_CONF = "max"  # "avg" penalises stones seen in only some views (tiles see each stone once)


def load_split(data: Path, split: str):
    manifest = json.loads((data / "split_manifest.json").read_text())["images"]
    out = []
    for p in sorted((data / split / "images").iterdir()):  # includes neg_* hard negatives (empty gt)
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = load_bboxes(data / split / "labels" / f"{p.stem}.txt", (w, h))
        ig_file = data / split / "labels_ignore" / f"{p.stem}.txt"  # convention-A ignore regions, if any
        ig = load_bboxes(ig_file, (w, h)) if ig_file.exists() else np.empty((0, 4), np.float32)
        out.append({"name": p.name, "img": img, "gt": gt, "ignore": ig, "group": manifest[p.name]["group"]})
    return out


def yolo_fn(path: str):
    from ultralytics import YOLO
    import torch
    m = YOLO(path)
    device = 0 if torch.cuda.is_available() else "cpu"  # checkpoints may carry device=0

    def f(img):
        r = m.predict(img, conf=SCORE_FLOOR, verbose=False, device=device)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return f


def rfdetr_fn(path: str, resolution: int | None = None, keep_class: int | None = None):
    from rfdetr import RFDETRLarge
    # the checkpoint does not store its input size: pass the training resolution
    m = RFDETRLarge(pretrain_weights=path, **({"resolution": resolution} if resolution else {}))

    def f(img):
        d = m.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=SCORE_FLOOR)  # RF-DETR wants RGB
        b, s = np.asarray(d.xyxy, np.float32).reshape(-1, 4), np.asarray(d.confidence, np.float32)
        if keep_class is not None:  # multi-class models (e.g. joint gallstone + kidney_stone)
            k = np.asarray(d.class_id) == keep_class
            b, s = b[k], s[k]
        return b, s
    return f


def with_tta(f, tiles: int = 1, hflip: bool = False, overlap: float = 0.25):
    """Wrap a predictor with flip and tiled (SAHI-style) test-time augmentation.

    Both detectors resize their input to a fixed size, so rescaling the frame
    changes nothing; a tile, by contrast, is up-sampled, which magnifies small
    stones. Tile boxes are mapped back to the full frame and everything
    (full frame + tiles, each optionally flipped) is fused by WBF.
    """
    if tiles == 1 and not hflip:
        return f

    def views(img):
        h, w = img.shape[:2]
        yield 0, 0, img
        if tiles > 1:
            th, tw = int(h / (tiles - (tiles - 1) * overlap)), int(w / (tiles - (tiles - 1) * overlap))
            for i in range(tiles):
                for j in range(tiles):
                    y0 = min(round(i * th * (1 - overlap)), h - th)
                    x0 = min(round(j * tw * (1 - overlap)), w - tw)
                    yield x0, y0, img[y0:y0 + th, x0:x0 + tw]

    def g(img):
        h, w = img.shape[:2]
        outs = []
        for x0, y0, im in views(img):
            vw = im.shape[1]
            for flip in ((False, True) if hflip else (False,)):
                b, s = f(np.ascontiguousarray(im[:, ::-1]) if flip else im)
                b = b.copy()
                if flip:
                    b[:, [0, 2]] = vw - b[:, [2, 0]]
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                outs.append((b, s))
        return wbf(outs, w, h, iou=0.55, conf_type=TTA_CONF)
    return g


def wbf(per_model, w, h, iou=0.5, conf_type="avg"):
    from ensemble_boxes import weighted_boxes_fusion
    scale = np.array([w, h, w, h], np.float32)
    boxes = [np.clip(b / scale, 0, 1).tolist() for b, _ in per_model]
    scores = [s.tolist() for _, s in per_model]
    labels = [[0] * len(s) for _, s in per_model]
    b, s, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=iou, skip_box_thr=SCORE_FLOOR, conf_type=conf_type)
    return np.asarray(b, np.float32).reshape(-1, 4) * scale, np.asarray(s, np.float32)


def metrics_with_ci(rows, preds, n_boot: int, seed: int = 0):
    targets = [{"boxes": r["gt"], "labels": np.zeros(len(r["gt"])),
                "ignore": r.get("ignore", np.empty((0, 4)))} for r in rows]
    point = coco_map(preds, targets)
    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[r["group"]].append(i)
    groups = list(by_group)
    rng = np.random.default_rng(seed)
    boots = defaultdict(list)
    for _ in range(n_boot):
        idx = [i for g in rng.choice(groups, size=len(groups), replace=True) for i in by_group[g]]
        m = coco_map([preds[i] for i in idx], [targets[i] for i in idx])
        for k in ("map", "map_50"):
            boots[k].append(m[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return {**{k: round(v, 4) for k, v in point.items()}, "ci95": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()}}


def image_level(rows, preds, conf: float = OPERATING_CONF) -> dict:
    """Image-level detection quality plus stone recall at an operating threshold."""
    from torchvision.ops import box_iou
    import torch
    has = np.array([len(r["gt"]) > 0 for r in rows])
    top = np.array([float(p["scores"].max()) if len(p["scores"]) else 0.0 for p in preds])
    out = {"operating_conf": conf}
    if has.any() and (~has).any():
        from sklearn.metrics import roc_auc_score
        out["image_auroc"] = round(float(roc_auc_score(has, top)), 4)
        out["neg_images"] = int((~has).sum())
        out["neg_fp_per_image"] = round(float(np.mean([(p["scores"] >= conf).sum() for p, h in zip(preds, has) if not h])), 3)
        out["neg_images_with_fp"] = round(float(np.mean(top[~has] >= conf)), 3)
        # per negative source (neg_polyp_*, plain neg_*): checks for an image-style shortcut
        src = np.array([r["name"].split("_")[1] if r["name"].startswith("neg_polyp") else "normal" for r in rows])
        for k in sorted(set(src[~has])):
            m = (~has) & (src == k)
            out[f"neg_{k}_with_fp"] = round(float(np.mean(top[m] >= conf)), 3)
            out[f"auroc_vs_{k}"] = round(float(roc_auc_score(np.r_[np.ones(has.sum()), np.zeros(m.sum())],
                                                              np.r_[top[has], top[m]])), 4)
    found = total = fp = 0
    for r, p in zip(rows, preds):
        keep = p["scores"] >= conf
        ig = np.asarray(r.get("ignore", np.empty((0, 4))), np.float32).reshape(-1, 4)
        if len(ig) and keep.any():  # detections on ignore regions are neither hits nor false positives
            pb = np.asarray(p["boxes"], np.float32).reshape(-1, 4)
            on_ig = box_iou(torch.as_tensor(pb), torch.as_tensor(ig)).max(1).values.numpy() >= 0.3
            keep = keep & ~on_ig
        b = torch.as_tensor(p["boxes"][keep], dtype=torch.float32).reshape(-1, 4)
        g = torch.as_tensor(r["gt"], dtype=torch.float32).reshape(-1, 4)
        total += len(g)
        if len(g) and len(b):
            iou = box_iou(g, b)
            found += int((iou.max(1).values >= 0.5).sum())
            fp += int((iou.max(0).values < 0.5).sum()) if len(r["gt"]) else len(b)
        else:
            fp += len(b)
    out["stone_recall"] = round(found / max(total, 1), 4)
    out["pos_image_hit_rate"] = round(float(np.mean(top[has] >= conf)), 4) if has.any() else None
    out["fp_per_image"] = round(fp / len(rows), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_grouped")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--yolo")
    ap.add_argument("--rfdetr")
    ap.add_argument("--rfdetr-ensemble", nargs="+", help="several RF-DETR checkpoints fused by WBF (conf avg)")
    ap.add_argument("--keep-class", type=int, help="score only this RF-DETR class id (multi-class models)")
    ap.add_argument("--rfdetr-resolution", type=int, help="input size the RF-DETR model was trained at (default 704)")
    ap.add_argument("--tta-tiles", type=int, default=1, help="N -> NxN overlapping tiles plus the full frame")
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", default="runs/grouped/eval_test.json")
    args = ap.parse_args()
    if not (args.yolo or args.rfdetr or args.rfdetr_ensemble):
        sys.exit("give --yolo, --rfdetr and/or --rfdetr-ensemble")

    data = ROOT / args.data
    rows = load_split(data, args.split)
    print(f"{args.split}: {len(rows)} images, {len({r['group'] for r in rows})} video groups")

    raw = {}
    if args.yolo:
        f = with_tta(yolo_fn(args.yolo), args.tta_tiles, args.tta_hflip)
        raw["yolo"] = [f(r["img"]) for r in rows]
    if args.rfdetr:
        f = with_tta(rfdetr_fn(args.rfdetr, args.rfdetr_resolution, args.keep_class), args.tta_tiles, args.tta_hflip)
        raw["rfdetr"] = [f(r["img"]) for r in rows]
    if args.rfdetr_ensemble:
        per = []
        for w in args.rfdetr_ensemble:
            f = rfdetr_fn(w, args.rfdetr_resolution, args.keep_class)
            per.append([f(r["img"]) for r in rows])
            print(f"  ensemble member done: {w}")
        raw["rfdetr_ensemble"] = [wbf([p[i] for p in per], r["img"].shape[1], r["img"].shape[0], iou=0.55, conf_type="avg")
                                  for i, r in enumerate(rows)]
    if args.yolo and args.rfdetr:
        raw["wbf_ensemble"] = [wbf([raw["yolo"][i], raw["rfdetr"][i]], r["img"].shape[1], r["img"].shape[0])
                               for i, r in enumerate(rows)]

    report = {"data": args.data, "split": args.split, "tta": {"tiles": args.tta_tiles, "hflip": args.tta_hflip}, "images": len(rows), "weights": {"yolo": args.yolo, "rfdetr": args.rfdetr, "rfdetr_ensemble": args.rfdetr_ensemble}}
    for name, outs in raw.items():
        preds = [{"boxes": b, "scores": s, "labels": np.zeros(len(s))} for b, s in outs]
        report[name] = metrics_with_ci(rows, preds, args.bootstrap)
        report[name]["image_level"] = image_level(rows, preds)
        m = report[name]
        print(f"{name:>13}: mAP50:95 {m['map']:.3f} {m['ci95']['map']}   mAP50 {m['map_50']:.3f} {m['ci95']['map_50']}   {m['image_level']}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
