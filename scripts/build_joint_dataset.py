"""Joint gallstone + kidney-stone training set; valid/test stay gallstone-only.

Two classes: 0 = gallstone, 1 = kidney_stone. Sharing one detector lets the
"bright echo + posterior shadow" features learn from ~14.7k kidney stones,
while separate classes mean neither dataset's box conventions are forced on
the other and the model can tell gallbladder from kidney context. Only
class-0 boxes are scored (scripts/eval_grouped.py --keep-class 0).

Balancing (kidney ~5.2k train images vs ~1.6k gallstone-side images):
gallstone positives are repeated `--pos-repeat` times and gallstone
negatives `--neg-repeat` times. Repeats are exact copies; RF-DETR's random
flip/resize at load time makes each pass differ. Kidney images go to train
only, so valid/test are byte-identical to the gallstone source split.

Usage: python scripts/build_joint_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gall", default="dataset_grouped_neg2")
    ap.add_argument("--kidney", default="external/kidney_pretrain")
    ap.add_argument("--target", default="dataset_joint")
    ap.add_argument("--pos-repeat", type=int, default=3)
    ap.add_argument("--neg-repeat", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    gall, kid, dst = ROOT / args.gall, ROOT / args.kidney, ROOT / args.target
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)

    counts = {}
    for split in ("train", "valid", "test"):
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)
        n = {"gall_pos": 0, "gall_neg": 0, "kidney_pos": 0, "kidney_neg": 0}
        for img in sorted((gall / split / "images").iterdir()):
            lab = gall / split / "labels" / f"{img.stem}.txt"
            is_neg = img.stem.startswith("neg_")
            reps = 1 if split != "train" else (args.neg_repeat if is_neg else args.pos_repeat)
            for r in range(reps):
                stem = img.stem if r == 0 else f"{img.stem}_rep{r}"
                link(img, dst / split / "images" / f"{stem}{img.suffix}")
                shutil.copy2(lab, dst / split / "labels" / f"{stem}.txt")
                n["gall_neg" if is_neg else "gall_pos"] += 1
        if split == "train":
            for ks in ("train", "valid"):  # all kidney data is training data here
                for img in sorted((kid / ks / "images").iterdir()):
                    rows = (kid / ks / "labels" / f"{img.stem}.txt").read_text().split("\n")
                    rows = ["1" + r[1:] for r in rows if r.strip()]  # class 0 -> kidney_stone
                    stem = f"kidney_{img.stem}" if rows else f"neg_kidney_{img.stem}"
                    link(img, dst / "train" / "images" / f"{stem}{img.suffix}")
                    (dst / "train" / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                    n["kidney_pos" if rows else "kidney_neg"] += 1
        counts[split] = n
        print(split, n)

    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        "names:\n  0: gallstone\n  1: kidney_stone\n", encoding="utf-8")
    shutil.copy2(gall / "split_manifest.json", dst / "split_manifest.json")  # valid/test groups unchanged
    (dst / "joint_info.json").write_text(json.dumps({**vars(args), "counts": counts}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
