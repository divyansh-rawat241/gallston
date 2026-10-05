"""Training set for the final model: train + valid of a grouped dataset; test stays out.

The final model (docs/FINAL_EVAL_PROTOCOL.md) uses every non-test image and a
fixed epoch count chosen from the CV folds, so it needs no validation split.
rfdetr still requires a `valid` folder; it gets a copy of the original valid
images, which are also in train - so its validation numbers are monitoring
only and must not be used for any choice. The test split is not copied.

Leakage: train/valid vs. test was checked on the source dataset
(`python scripts/check_leakage.py --data <source>` -> clean); combining train
and valid cannot create new cross-split overlap with test.

Usage: python scripts/build_final_dataset.py --source dataset_grouped_neg4 --target dataset_final_neg4 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset_grouped_neg4")
    ap.add_argument("--target", default="dataset_final_neg4")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    man = json.loads((src / "split_manifest.json").read_text())
    n = {"train": 0, "valid": 0}
    for name, m in man["images"].items():
        if m["split"] == "test":
            continue
        stem = Path(name).stem
        targets = ["train"] + (["valid"] if m["split"] == "valid" else [])
        for split in targets:
            for sub in ("images", "labels", "labels_polygon", "labels_ignore"):
                (dst / split / sub).mkdir(parents=True, exist_ok=True)
            try:
                os.link(src / m["split"] / "images" / name, dst / split / "images" / name)
            except OSError:
                shutil.copy2(src / m["split"] / "images" / name, dst / split / "images" / name)
            for sub in ("labels", "labels_polygon", "labels_ignore"):
                f = src / m["split"] / sub / f"{stem}.txt"
                if f.exists():
                    shutil.copy2(f, dst / split / sub / f.name)
            n[split] += 1
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\n\nnames:\n  0: gallstone\n", encoding="utf-8")
    (dst / "final_info.json").write_text(json.dumps({"source": args.source, "counts": n,
        "note": "train = source train + valid; valid duplicates source valid (monitoring only); test excluded"}, indent=1))
    print(n, "->", dst)


if __name__ == "__main__":
    main()
