"""Build a stone-only pretraining set from the Roboflow kidney-stone ultrasound data.

Source: universe.roboflow.com/kidney-ktlmt/kidney-stone-ultrasound (CC BY 4.0),
forked and exported as YOLOv8 into external/kidney_stone/. Classes there are
0 = kidney-stone, 1 = normal kidney (a whole-organ box). Only stones are kept;
images with no stone stay in as negatives. The class is named "gallstone"
so the detection head transfers unchanged. Images get the same
`remove_color_overlays` preprocessing as the gallstone data.

The kidney set has many near-duplicate video frames across its own splits, so
its `valid` score is optimistic. That does not matter here: it is only used to
pretrain, and the gallstone `valid` split decides whether pretraining helped.

Usage: python scripts/prepare_kidney_pretrain.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.overlay_removal import remove_color_overlays  # noqa: E402

STONE = 0


def to_box(parts: list[str]) -> str | None:
    vals = np.array(parts[1:], dtype=np.float32)
    if len(vals) == 4:  # already cx cy w h
        cx, cy, w, h = vals
    else:  # polygon x1 y1 x2 y2 ...
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="external/kidney_stone")
    ap.add_argument("--target", default="external/kidney_pretrain")
    args = ap.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    if dst.exists():
        shutil.rmtree(dst)

    # kidney train + test -> train; kidney valid -> valid (monitoring only)
    plan = {"train": ["train", "test"], "valid": ["valid"]}
    for out_split, in_splits in plan.items():
        (dst / out_split / "images").mkdir(parents=True)
        (dst / out_split / "labels").mkdir(parents=True)
        n_img = n_box = n_neg = 0
        for s in in_splits:
            for p in sorted((src / s / "images").iterdir()):
                lab = src / s / "labels" / f"{p.stem}.txt"
                rows = []
                if lab.exists():
                    for line in lab.read_text().splitlines():
                        parts = line.split()
                        if parts and int(parts[0]) == STONE:
                            r = to_box(parts)
                            if r:
                                rows.append(r)
                img = cv2.imread(str(p))
                if img is None:
                    continue
                cleaned, _ = remove_color_overlays(img)
                cv2.imwrite(str(dst / out_split / "images" / f"{p.stem}.jpg"), cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                (dst / out_split / "labels" / f"{p.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                n_img += 1
                n_box += len(rows)
                n_neg += not rows
        print(f"{out_split}: {n_img} images, {n_box} stone boxes, {n_neg} without stones")
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\n\nnames:\n  0: gallstone\n", encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
