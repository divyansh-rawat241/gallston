"""Grouped K-fold cross-validation datasets from a grouped split.

The `test` split is never touched: folds are built from train + valid only,
using the fold already assigned to each video/patient group in
`split_manifest.json` (stratified, whole groups). For fold k, groups with
fold == k form `valid`, every other train/valid group forms `train`.

Each fold is a full dataset folder (images hard-linked, labels copied) with
its own data.yaml and a copy of the manifest, so the normal tools work on it:
  python scripts/check_leakage.py --data dataset_cv/fold0 --splits train valid
  python scripts/train_grouped.py --model rfdetr --data dataset_cv/fold0 ...
  python scripts/eval_grouped.py --data dataset_cv/fold0 --split valid ...

Why: the fixed `valid` split has 17 video groups (CI about +/-0.15 mAP), wider
than most effects we test. Out-of-fold predictions over all ~95 groups give a
tighter estimate, paired comparisons between configurations, and the
predictions needed for a label audit.

Usage: python scripts/build_cv_folds.py --source dataset_grouped_neg2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_leakage import thumb  # noqa: E402  (same similarity the leak check uses)


def merge_near_duplicate_groups(src: Path, images: dict[str, dict], link: float = 0.90) -> dict[str, int]:
    """Fold per image, after joining groups whose images are near-duplicates.

    The split builder clusters negatives on its own thumbnails; a few pairs it
    kept apart still exceed the leak checker's 0.95 threshold. Joined groups
    take the fold of their largest member group.
    """
    names = sorted(images)
    groups = sorted({images[n]["group"] for n in names})
    gi = {g: i for i, g in enumerate(groups)}
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    T = np.stack([thumb(src / images[n]["split"] / "images" / n) for n in names])
    sim = T @ T.T
    for a, b in zip(*np.nonzero(np.triu(sim > link, 1))):
        ga, gb = gi[images[names[a]]["group"]], gi[images[names[b]]["group"]]
        if find(ga) != find(gb):
            print(f"  joining groups {groups[ga]} + {groups[gb]} ({names[a]} ~ {names[b]}, {sim[a, b]:.3f})")
            parent[find(ga)] = find(gb)
    size = Counter(images[n]["group"] for n in names)
    comp_fold: dict[int, tuple[int, int]] = {}
    for g in groups:
        f = next(images[n]["fold"] for n in names if images[n]["group"] == g)
        c = find(gi[g])
        if c not in comp_fold or size[g] > comp_fold[c][1]:
            comp_fold[c] = (f, size[g])
    return {n: comp_fold[find(gi[images[n]["group"]])][0] for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset_grouped_neg2")
    ap.add_argument("--target", default="dataset_cv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)

    manifest = json.loads((src / "split_manifest.json").read_text())
    images = {n: m for n, m in manifest["images"].items() if m["split"] in ("train", "valid")}
    fold_of = merge_near_duplicate_groups(src, images)
    images = {n: {**m, "fold": fold_of[n]} for n, m in images.items()}
    folds = sorted({m["fold"] for m in images.values()})
    for k in folds:
        fd = dst / f"fold{k}"
        counts = Counter()
        for name, m in images.items():
            split = "valid" if m["fold"] == k else "train"
            for sub in ("images", "labels", "labels_polygon"):
                (fd / split / sub).mkdir(parents=True, exist_ok=True)
            stem = Path(name).stem
            img = src / m["split"] / "images" / name
            try:
                os.link(img, fd / split / "images" / name)
            except OSError:
                shutil.copy2(img, fd / split / "images" / name)
            for sub in ("labels", "labels_polygon"):
                lab = src / m["split"] / sub / f"{stem}.txt"
                if lab.exists():
                    shutil.copy2(lab, fd / split / sub / f"{stem}.txt")
            kind = "neg" if name.startswith("neg_") else "pos"
            counts[f"{split}_{kind}"] += 1
        (fd / "data.yaml").write_text(
            f"path: {fd.as_posix()}\ntrain: train/images\nval: valid/images\n\nnames:\n  0: gallstone\n",
            encoding="utf-8")
        fold_manifest = {**manifest, "cv_fold": k, "cv_source": args.source,
                         "images": {n: {**m, "split": "valid" if m["fold"] == k else "train"} for n, m in images.items()}}
        (fd / "split_manifest.json").write_text(json.dumps(fold_manifest, indent=1), encoding="utf-8")
        print(f"fold{k}: {dict(sorted(counts.items()))}")
    print(f"wrote {dst} ({len(folds)} folds; test split excluded)")


if __name__ == "__main__":
    main()
