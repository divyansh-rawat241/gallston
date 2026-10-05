"""Fail loudly if a dataset split leaks between train, validation and test.

Checks:
  1. No file content (MD5) appears in more than one split.
  2. No video group (`pseudo_<video>__<frame>`) appears in more than one split.
  3. No near-duplicate images across splits: correlation of 32x32 thumbnails of
     the central 70% crop above --near-dup (default 0.95). The 2026-10-01 audit
     found no near-duplicates across different videos at this threshold, so any
     hit here means leaked content.
  4. Every image has a non-empty label file (except neg_* hard negatives).

Exits with status 1 on any leak, so it can gate a training script.

Usage:
  python scripts/check_leakage.py --data dataset_grouped
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def thumb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    crop = img[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    t = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t = (t - t.mean()) / (t.std() + 1e-6)
    return t / 32.0  # unit norm, so a dot product is the correlation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_grouped")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--near-dup", type=float, default=0.95)
    args = ap.parse_args()
    data = ROOT / args.data

    records = []  # (split, path, md5, group)
    problems: list[str] = []
    for s in args.splits:
        img_dir = data / s / "images"
        if not img_dir.exists():
            problems.append(f"missing split folder: {img_dir}")
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            label = data / s / "labels" / f"{p.stem}.txt"
            # neg_* files are deliberate stone-free hard negatives: empty label allowed
            if not label.exists() or (not label.read_text().strip() and not p.stem.startswith("neg_")):
                problems.append(f"no label / empty label: {s}/{p.name}")
            m = re.match(r"pseudo_([a-z]+\d+)__", p.stem)
            group = m.group(1) if m else f"pl:{p.stem}"
            records.append((s, p, hashlib.md5(p.read_bytes()).hexdigest(), group))

    by_md5, by_group = defaultdict(set), defaultdict(set)
    for s, _, d, g in records:
        by_md5[d].add(s)
        by_group[g].add(s)
    md5_leaks = [d for d, ss in by_md5.items() if len(ss) > 1]
    group_leaks = sorted(g for g, ss in by_group.items() if len(ss) > 1)

    X = np.stack([thumb(p) for _, p, _, _ in records])
    split_arr = np.array([s for s, *_ in records])
    near = 0
    examples = []
    for a, b in combinations(args.splits, 2):
        ia, ib = np.where(split_arr == a)[0], np.where(split_arr == b)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        C = X[ia] @ X[ib].T
        hits = np.argwhere(C > args.near_dup)
        near += len(hits)
        for i, j in hits[:3]:
            examples.append(f"{a}/{records[ia[i]][1].name} ~ {b}/{records[ib[j]][1].name} ({C[i, j]:.3f})")

    counts = {s: sum(1 for r in records if r[0] == s) for s in args.splits}
    groups = {s: len({r[3] for r in records if r[0] == s}) for s in args.splits}
    print(f"dataset: {data}")
    for s in args.splits:
        print(f"  {s:>5}: {counts[s]:>4} images, {groups[s]:>3} groups")
    print(f"exact-copy leaks across splits : {len(md5_leaks)}")
    print(f"video groups in >1 split       : {len(group_leaks)} {group_leaks[:10]}")
    print(f"near-duplicates across splits  : {near} (corr > {args.near_dup})")
    for e in examples[:6]:
        print(f"    {e}")
    print(f"label problems                 : {len(problems)}")
    for p in problems[:6]:
        print(f"    {p}")

    leaked = bool(md5_leaks or group_leaks or near or problems)
    print("\nRESULT:", "LEAK OR DEFECT FOUND - do not train on this split" if leaked else "clean")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
