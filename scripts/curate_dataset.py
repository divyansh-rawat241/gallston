"""Drop images listed in docs/data_exclusions.json from a grouped dataset.

Every exclusion carries a reason, so the curated dataset is reproducible and
auditable. Entries can name images (exact file name or prefix) or whole groups.
Splits are otherwise unchanged; images are hard-linked; labels, polygon labels
and ignore regions are copied.

Usage: python scripts/curate_dataset.py --source dataset_grouped_neg3 --target dataset_grouped_neg4 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset_grouped_neg3")
    ap.add_argument("--target", default="dataset_grouped_neg4")
    ap.add_argument("--exclusions", default="docs/data_exclusions.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--remove-text", action="store_true",
                    help="inpaint OCR-detected burned-in text in every image (labelled stones are protected)")
    args = ap.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    rules = json.loads((ROOT / args.exclusions).read_text(encoding="utf-8"))["exclude"]
    manifest = json.loads((src / "split_manifest.json").read_text())

    def reason_for(name: str, group: str) -> str | None:
        for r in rules:
            if r.get("group") == group or name in r.get("images", []) or \
                    any(name.startswith(p) for p in r.get("prefixes", [])):
                return r["reason"]
        return None

    kept, excluded = {}, {}
    stats = Counter()
    for name, m in manifest["images"].items():
        why = reason_for(name, m["group"])
        if why:
            excluded[name] = {**m, "reason": why}
            stats[f"excluded_{m['split']}"] += 1
            continue
        kept[name] = m
        stats[f"kept_{m['split']}"] += 1
        stem = Path(name).stem
        for sub in ("images", "labels", "labels_polygon", "labels_ignore"):
            (dst / m["split"] / sub).mkdir(parents=True, exist_ok=True)
        img_src, img_dst = src / m["split"] / "images" / name, dst / m["split"] / "images" / name
        if args.remove_text:
            from src.evaluation.label_io import load_bboxes
            from src.preprocessing.text_removal import remove_burned_in_text
            img = cv2.imread(str(img_src))
            h, w = img.shape[:2]
            prot = [load_bboxes(src / m["split"] / sub / f"{stem}.txt", (w, h))
                    for sub in ("labels", "labels_ignore") if (src / m["split"] / sub / f"{stem}.txt").exists()]
            prot = np.concatenate(prot) if prot else None
            out, n_px = remove_burned_in_text(img, protect=prot)
            stats["text_removed_images"] += n_px > 0
            if n_px:
                cv2.imwrite(str(img_dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                shutil.copy2(img_src, img_dst)
            kept[name] = {**m, "text_removed_px": n_px}
        else:
            try:
                os.link(img_src, img_dst)
            except OSError:
                shutil.copy2(img_src, img_dst)
        for sub in ("labels", "labels_polygon", "labels_ignore"):
            f = src / m["split"] / sub / f"{stem}.txt"
            if f.exists():
                shutil.copy2(f, dst / m["split"] / sub / f.name)

    (dst / "data.yaml").write_text((src / "data.yaml").read_text().replace(src.as_posix(), dst.as_posix()),
                                   encoding="utf-8")
    manifest["images"] = kept
    manifest["excluded"] = excluded
    manifest["curation"] = {"source": args.source, "exclusions_file": args.exclusions,
                            "remove_text": args.remove_text, "counts": dict(stats)}
    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(dict(sorted(stats.items())))
    for r in rules:
        n = sum(1 for e in excluded.values() if e["reason"] == r["reason"])
        print(f"  {n:4d}  {r['reason'][:90]}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
