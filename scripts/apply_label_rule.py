"""Labelling convention A: keep definite stones, turn faint companion echoes into ignore regions.

Decided 2026-10-04 after the label audit (docs/EXPERIMENTS.md E14): some videos
(notably `a32`) box every faint echogenic dot in the lumen, while most label
only clear stones. The rule below is model-independent - it uses only image
intensities measured by scripts/stone_box_features.py - and is applied the same
way to every split, so it cannot tune the test set to a model.

A labelled box is REMOVED when it is not the brightest box of its image and
  peak < PEAK_MIN  or  contrast < CONTRAST_MIN  or  contrast < REL_MIN x (max contrast in image)
The brightest box always stays, so no positive image becomes a negative.

Removed boxes are not deleted: they go to `labels_ignore/` (YOLO box format).
Training does not see them (rfdetr has no ignore support, so they are
background); evaluation passes them as COCO ignore ("iscrowd") regions, so
detections there are neither rewarded nor penalised.

Usage: python scripts/apply_label_rule.py --source dataset_grouped_neg2 --target dataset_grouped_neg3 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.label_io import load_polygon_labels, polygon_to_bbox  # noqa: E402

PEAK_MIN = 110.0      # 95th-percentile grey level inside the polygon
CONTRAST_MIN = 30.0   # mean inside minus median of the surrounding ring
REL_MIN = 0.45        # share of the image's strongest stone contrast


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset_grouped_neg2")
    ap.add_argument("--target", default="dataset_grouped_neg3")
    ap.add_argument("--features", default="runs/label_rule/box_features.csv",
                    help="from scripts/stone_box_features.py run on --source")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)

    feats: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(ROOT / args.features, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            feats[r["name"]][int(r["box"])] = {k: float(r[k]) for k in ("contrast", "peak")}

    manifest = json.loads((src / "split_manifest.json").read_text())
    stats = Counter()
    log = []
    for name, m in manifest["images"].items():
        split, stem = m["split"], Path(name).stem
        for sub in ("images", "labels", "labels_polygon", "labels_ignore"):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)
        try:
            os.link(src / split / "images" / name, dst / split / "images" / name)
        except OSError:
            shutil.copy2(src / split / "images" / name, dst / split / "images" / name)
        polys = load_polygon_labels(src / split / "labels_polygon" / f"{stem}.txt")
        f = feats.get(name, {})
        top = max(f, key=lambda i: f[i]["contrast"]) if f else None
        cmax = max((v["contrast"] for v in f.values()), default=1.0)
        keep_p, keep_b, ign_b = [], [], []
        for i, (cls, poly) in enumerate(polys):
            v = f.get(i)
            remove = v is not None and i != top and (
                v["peak"] < PEAK_MIN or v["contrast"] < CONTRAST_MIN or v["contrast"] < REL_MIN * max(cmax, 1.0))
            x1, y1, x2, y2 = polygon_to_bbox(poly)
            row = f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
            if remove:
                ign_b.append(row)
                log.append({"name": name, "split": split, "group": m["group"], "box": i, **v})
            else:
                keep_b.append(row)
                keep_p.append(f"{cls} " + " ".join(f"{c:.6f}" for c in poly.reshape(-1)))
            stats[f"{split}_{'ignored' if remove else 'kept'}"] += 1
        for sub, rows in (("labels", keep_b), ("labels_polygon", keep_p), ("labels_ignore", ign_b)):
            (dst / split / sub / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    (dst / "data.yaml").write_text((src / "data.yaml").read_text().replace(src.as_posix(), dst.as_posix()),
                                   encoding="utf-8")
    manifest["label_rule"] = {"convention": "A: definite stones only; faint companions -> ignore regions",
                              "peak_min": PEAK_MIN, "contrast_min": CONTRAST_MIN, "rel_min": REL_MIN,
                              "source": args.source, "counts": dict(stats)}
    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with open(dst / "ignored_boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "split", "group", "box", "contrast", "peak"])
        w.writeheader()
        w.writerows(log)
    print(dict(sorted(stats.items())))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
