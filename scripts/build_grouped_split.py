"""Build a leakage-free, video-grouped train / valid / test split.

Replaces `resplit_dataset_corrected.py`, which split image-by-image and copied
into a target folder it never emptied (see docs/LEAKAGE_AUDIT.md).

Rules enforced here:
  1. Deduplicate by file content (MD5), not by path.
  2. Drop images flagged "bad" during manual review (review_state_*.json).
  3. A whole video (`pseudo_<video>__<frame>`) goes to exactly one split.
     `pl_*` images have no recorded source and match no video, so each is
     its own group.
  4. Split separately within each stratum (`a*` videos, `c*` videos, `pl_*`)
     so every split sees both video sources.
  5. The target folder is deleted and rebuilt from scratch.
  6. Coloured overlays (burned-in arrows) are inpainted out of every image
     (src/preprocessing/overlay_removal.py); disable with --keep-overlays.

Labels are written as YOLO boxes (`cls cx cy w h`) derived from the polygons;
the original polygons are kept in `labels_polygon/` for segmentation work.
A 5-fold group assignment over all groups is saved for cross-validation.

Usage:
  python scripts/build_grouped_split.py --overwrite
  python scripts/build_grouped_split.py --test-frac 0.15 --valid-frac 0.15 --seed 42 --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.label_io import load_polygon_labels, polygon_to_bbox  # noqa: E402
from src.preprocessing.overlay_removal import remove_color_overlays  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")


def group_of(stem: str) -> str:
    m = re.match(r"pseudo_([a-z]+\d+)__", stem)
    return m.group(1) if m else f"pl:{stem}"


def stratum_of(group: str) -> str:
    if group.startswith("neg"):  # "neg:<cluster>", "negpolyp:<patient>", ...
        return group.split(":", 1)[0]
    return "pl" if group.startswith("pl:") else re.match(r"[a-z]+", group).group(0)


def _thumb(path: Path) -> np.ndarray:
    # on the cleaned image, as check_leakage.py sees it (tint and text removed)
    g = cv2.cvtColor(remove_color_overlays(cv2.imread(str(path)))[0], cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    t = cv2.resize(g[int(.15 * h):int(.85 * h), int(.15 * w):int(.85 * w)], (32, 32)).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def collect_negatives(neg_dir: Path, known_md5: set[str], link: float = 0.90, tag: str = "") -> dict[str, dict]:
    """Stone-free gallbladder images as hard negatives (empty label files).

    Near-duplicate frames (thumbnail correlation > `link`, single linkage) form
    one group and stay in one split. Files named like UIdataGB's "f12 (3).jpg"
    carry a patient ID; those are also linked, so a patient never spans splits
    (UIdataGB ships many augmented copies per patient).
    """
    paths = [p for p in sorted(neg_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    recs, seen = [], set(known_md5)
    for p in paths:
        d = hashlib.md5(p.read_bytes()).hexdigest()
        if d not in seen:
            seen.add(d)
            recs.append((d, p))
    T = np.stack([_thumb(p) for _, p in recs])
    sim = T @ T.T
    parent = list(range(len(recs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(sim > link, 1))):
        parent[find(i)] = find(j)
    first_of_patient: dict[str, int] = {}
    for i, (_, p) in enumerate(recs):
        m = re.match(r"([a-zA-Z]+\d+)\s*\(", p.name)
        if m:
            pid = m.group(1).lower()
            if pid in first_of_patient:
                parent[find(i)] = find(first_of_patient[pid])
            else:
                first_of_patient[pid] = i
    items = {}
    prefix = f"neg{tag}"
    for i, (d, p) in enumerate(recs):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem).strip("_")
        items[d] = {"image": p, "label": None, "stem": f"neg_{tag + '_' if tag else ''}{stem}", "group": f"{prefix}:{find(i)}"}
    print(f"negatives{' (' + tag + ')' if tag else ''}: {len(items)} unique images in {len(set(r['group'] for r in items.values()))} near-dup groups")
    return items


def collect(source: Path, review_dir: Path) -> dict[str, dict]:
    bad: set[str] = set()
    for f in review_dir.glob("review_state_*.json"):
        bad |= set(json.loads(f.read_text()).get("bad", []))

    items: dict[str, dict] = {}  # md5 -> record
    dropped_bad = dropped_dup = missing_label = 0
    for split in ("train", "val", "valid", "test"):
        img_dir = source / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if p.name in bad:
                dropped_bad += 1
                continue
            label = source / split / "labels" / f"{p.stem}.txt"
            if not label.exists():
                missing_label += 1
                continue
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in items:
                dropped_dup += 1
                continue
            items[digest] = {"image": p, "label": label, "stem": p.stem, "group": group_of(p.stem)}
    print(f"collected {len(items)} unique images | dropped: {dropped_bad} review-bad, "
          f"{dropped_dup} duplicate contents, {missing_label} without labels")
    return items


def assign_groups(groups: dict[str, list[str]], fracs: dict[str, float], seed: int,
                  boxes: dict[str, int] | None = None) -> dict[str, str]:
    """Greedy, stratified assignment of whole groups to splits.

    Balances image count and, when `boxes` (group -> box count) is given, box
    count too: a few multi-stone videos would otherwise skew one split.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        by_stratum[stratum_of(g)].append(g)

    assignment: dict[str, str] = {}
    # negatives last, so adding them leaves the RNG stream - and thus the
    # positive split - identical to a build without negatives
    for stratum, glist in sorted(by_stratum.items(), key=lambda kv: (kv[0].startswith("neg"), kv[0])):
        rng.shuffle(glist)
        glist.sort(key=lambda g: len(groups[g]), reverse=True)  # big groups first, ties shuffled
        nb = boxes or {g: len(groups[g]) for g in glist}
        tot_i = sum(len(groups[g]) for g in glist)
        tot_b = sum(nb[g] for g in glist) or 1
        have_i = {s: 0 for s in SPLITS}
        have_b = {s: 0 for s in SPLITS}

        def deficit(s: str) -> float:
            # remaining shortfall of this split, as a share of its target, on both counts
            di = (fracs[s] * tot_i - have_i[s]) / max(fracs[s] * tot_i, 1e-9)
            db = (fracs[s] * tot_b - have_b[s]) / max(fracs[s] * tot_b, 1e-9)
            return 0.5 * (di + db)

        for g in glist:
            split = max(SPLITS, key=deficit)
            assignment[g] = split
            have_i[split] += len(groups[g])
            have_b[split] += nb[g]
        print(f"  stratum {stratum:>2}: {len(glist)} groups, images {have_i}, boxes {have_b}")
    return assignment


def kfold_groups(groups: dict[str, list[str]], k: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        by_stratum[stratum_of(g)].append(g)
    fold_of: dict[str, int] = {}
    # negatives last, so adding them leaves the RNG stream - and thus the
    # positive split - identical to a build without negatives
    for stratum, glist in sorted(by_stratum.items(), key=lambda kv: (kv[0].startswith("neg"), kv[0])):
        rng.shuffle(glist)
        glist.sort(key=lambda g: len(groups[g]), reverse=True)
        load = [0] * k
        for g in glist:
            f = min(range(k), key=lambda i: load[i])
            fold_of[g] = f
            load[f] += len(groups[g])
    return fold_of


def write_box_label(src: Path, dst: Path) -> int:
    rows = []
    for cls, poly in load_polygon_labels(src):
        x1, y1, x2, y2 = polygon_to_bbox(poly)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        rows.append(f"{cls} {x1 + w / 2:.6f} {y1 + h / 2:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset_final")
    ap.add_argument("--review-dir", default="dataset_final_resplit")
    ap.add_argument("--target", default="dataset_grouped")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="delete the target folder if it exists")
    ap.add_argument("--negatives", help="folder of stone-free gallbladder images to add as hard negatives")
    ap.add_argument("--extra-negatives", nargs="*", default=[], metavar="TAG=DIR",
                    help="more hard-negative folders, each its own stratum, e.g. polyp=external/uidatagb/polyps")
    ap.add_argument("--keep-overlays", action="store_true", help="do not inpaint burned-in coloured arrows")
    args = ap.parse_args()

    source, target = ROOT / args.source, ROOT / args.target
    if target.exists():
        if not args.overwrite:
            sys.exit(f"{target} exists; pass --overwrite to rebuild it from scratch")
        shutil.rmtree(target)

    items = collect(source, ROOT / args.review_dir)
    if args.negatives:
        items.update(collect_negatives(ROOT / args.negatives, set(items)))
    for spec in args.extra_negatives:
        tag, folder = spec.split("=", 1)
        items.update(collect_negatives(ROOT / folder, set(items), tag=tag))
    groups: dict[str, list[str]] = defaultdict(list)
    for digest, rec in items.items():
        groups[rec["group"]].append(digest)

    fracs = {"test": args.test_frac, "valid": args.valid_frac}
    fracs["train"] = 1.0 - fracs["test"] - fracs["valid"]
    print(f"{len(groups)} groups -> assigning whole groups with fractions {fracs}")
    box_count = {g: sum(len(load_polygon_labels(items[d]["label"])) if items[d]["label"] else 0 for d in ds)
                 for g, ds in groups.items()}
    split_of_group = assign_groups(groups, fracs, args.seed, boxes=box_count)
    fold_of_group = kfold_groups(groups, args.folds, args.seed)

    for s in SPLITS:
        for sub in ("images", "labels", "labels_polygon"):
            (target / s / sub).mkdir(parents=True, exist_ok=True)

    manifest = {"seed": args.seed, "fractions": fracs, "source": args.source,
                "groups": {}, "images": {}, "counts": {s: {"images": 0, "boxes": 0, "groups": 0} for s in SPLITS}}
    overlay_actions = defaultdict(int)
    for g, digests in sorted(groups.items()):
        split = split_of_group[g]
        manifest["groups"][g] = {"split": split, "fold": fold_of_group[g], "images": len(digests)}
        manifest["counts"][split]["groups"] += 1
        for d in digests:
            rec = items[d]
            name = f"{rec['stem']}.jpg" if rec["label"] is None else rec["image"].name
            dst = target / split / "images" / name
            action = "none"
            if args.keep_overlays:
                shutil.copy2(rec["image"], dst)
            else:
                cleaned, action = remove_color_overlays(cv2.imread(str(rec["image"])))
                cv2.imwrite(str(dst), cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            overlay_actions[action] += 1
            if rec["label"] is None:  # hard negative
                for sub in ("labels", "labels_polygon"):
                    (target / split / sub / f"{rec['stem']}.txt").write_text("", encoding="utf-8")
                n_boxes = 0
            else:
                shutil.copy2(rec["label"], target / split / "labels_polygon" / rec["label"].name)
                n_boxes = write_box_label(rec["label"], target / split / "labels" / rec["label"].name)
            manifest["images"][name] = {"md5": d, "group": g, "split": split, "fold": fold_of_group[g], "overlay": action}
            manifest["counts"][split]["images"] += 1
            manifest["counts"][split]["boxes"] += n_boxes

    (target / "data.yaml").write_text(
        f"path: {target.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        "names:\n  0: gallstone\n", encoding="utf-8")
    manifest["overlay_actions"] = dict(overlay_actions)
    print(f"overlay removal: {dict(overlay_actions)}")
    (target / "split_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("\nfinal split (whole videos per split):")
    for s in SPLITS:
        c = manifest["counts"][s]
        print(f"  {s:>5}: {c['images']:>4} images  {c['boxes']:>4} boxes  {c['groups']:>3} groups")
    print(f"\nwrote {target}  ->  now run: python scripts/check_leakage.py --data {args.target}")


if __name__ == "__main__":
    main()
