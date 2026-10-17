"""
detect_text_artifacts.py
-----------------------
Screen every image for burned-in text, watermarks and figure-page artifacts.

Motivation
----------
Visual review found that a subset of images were screenshots of journal
figures, stock-agency photos and DICOM headers rather than raw clinical scans.
Two things make that dangerous:

  1. the words on those images ("GALLBLADDER", "Cholelithiasis", "Stone")
     are a shortcut label the network can read instead of looking;
  2. some carry patient identifiers and third-party copyright marks.

Manual review does not scale to thousands of frames, so this scores every image
with cheap, interpretable statistics and ranks the worst offenders for review.

Signals
-------
  white_frac   fraction of pixels at/near 255. Raw B-mode ultrasound is dark;
               a figure page, a white sidebar or a text banner is bright.
  black_frac   fraction of pixels at/near 0. Letterboxing and page margins.
  white_run    longest horizontal run of near-white pixels, as a fraction of
               width. Text lines and page furniture make very wide runs.
  flat_frac    fraction of pixels whose 4-neighbourhood is identical, i.e.
               regions with no speckle texture. Scanned paper and solid
               annotation panels are flat; ultrasound is not.

Usage
-----
    python scripts/detect_text_artifacts.py --data "Gallstone dataset/dataset_grouped_neg2"
    python scripts/detect_text_artifacts.py --data <root> --top 60 --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

WHITE_PX = 245
BLACK_PX = 10
LONGEST_RUN_TARGET = 0.55   # fraction of width for the "text line" signal
FLAT_TOL = 2                # neighbour delta treated as identical


def longest_true_run(row: np.ndarray) -> int:
    """Longest run of True in a 1-D boolean array."""
    if not row.any():
        return 0
    # positions where value changes
    idx = np.flatnonzero(np.diff(row))
    if idx.size == 0:
        return row.size
    bounds = np.concatenate(([-1], idx, [row.size - 1]))
    return int(np.max(np.diff(bounds)))


def score_image(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("L")
        a = np.asarray(im, dtype=np.uint8)
        h, w = a.shape

    white_mask = a >= WHITE_PX
    black_mask = a <= BLACK_PX

    white_frac = float(white_mask.mean())
    black_frac = float(black_mask.mean())

    # longest near-white horizontal run, subsampled rows for speed
    rows = white_mask[:: max(1, h // 120)]
    run = max((longest_true_run(r) for r in rows), default=0)
    white_run = run / w if w else 0.0

    # flat, texture-free fraction: a pixel counts as flat only if every one of
    # its 4-neighbours has (nearly) the same value. Ultrasound speckle makes
    # this rare; scanned paper and solid annotation panels make it common.
    eq_v = np.abs(np.diff(a, axis=0).astype(np.int16)) <= FLAT_TOL   # (h-1, w)
    eq_h = np.abs(np.diff(a, axis=1).astype(np.int16)) <= FLAT_TOL   # (h, w-1)
    m = min(eq_v.shape[0], eq_h.shape[0])
    n = min(eq_v.shape[1], eq_h.shape[1])
    if m > 0 and n > 0:
        flat_frac = float((eq_v[:m, :n] & eq_h[:m, :n]).mean())
    else:
        flat_frac = 1.0

    return {
        "w": int(w), "h": int(h),
        "white_frac": round(white_frac, 5),
        "black_frac": round(black_frac, 5),
        "white_run": round(white_run, 4),
        "flat_frac": round(flat_frac, 4),
    }


def risk_score(s: dict) -> float:
    """Rank images by likelihood of carrying burned-in text/figure furniture.

    Calibrated on this dataset: the 1,286 pseudo_*/neg_* images have
    white_frac <= 0.0083 (median 0.0000), while figure-page images reach 0.67.
    white_frac therefore separates cleanly and dominates the score.

    black_frac and white_run are deliberately NOT scored: B-mode ultrasound
    is a bright wedge on a black background, and a skin line or reverberation
    band can span the full width. Both are high for ordinary clinical images
    and produce heavy false positives. They are still reported for manual
    inspection, just not used for the risk number.
    """
    return round(5.0 * s["white_frac"], 5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Screen dataset for burned-in text/figures")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    ap.add_argument("--top", type=int, default=40, help="how many worst to list")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--out", help="write full JSON report here")
    args = ap.parse_args()

    root = Path(args.data)
    results: dict[str, dict] = {}
    by_prefix: dict[str, list[float]] = defaultdict(list)

    for split in args.splits:
        img_dir = root / split / "images"
        if not img_dir.exists():
            print(f"[{split}] missing, skipped")
            continue
        files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        print(f"[{split}] scoring {len(files)} images...", flush=True)
        for p in files:
            try:
                s = score_image(p)
            except Exception as e:
                print(f"  WARN unreadable {p.name}: {e}")
                continue
            s["split"] = split
            s["risk"] = risk_score(s)
            results[p.stem] = s
            prefix = re.match(r"([a-z_]+)", p.stem)
            key = prefix.group(1) if prefix else "other"
            by_prefix[key].append(s["risk"])

    print("\n" + "=" * 72)
    print("BURNED-IN TEXT / FIGURE SCREEN")
    print("=" * 72)

    print("\nrisk distribution by filename family:")
    for k in sorted(by_prefix):
        v = sorted(by_prefix[k])
        n_hi = sum(1 for r in v if r > 0.05)
        print(f"  {k:<12} n={len(v):>5}  median={v[len(v) // 2]:.5f}  "
              f"max={max(v):.5f}  above-threshold={n_hi}")

    worst = sorted(results.items(), key=lambda kv: -kv[1]["risk"])[: args.top]
    print(f"\ntop {min(args.top, len(worst))} highest-risk images:")
    print(f"  {'stem':<34} {'split':<6} {'risk':>8} {'white':>8} {'run':>7} {'flat':>7} {'blk':>7}")
    for stem, s in worst:
        print(f"  {stem:<34} {s['split']:<6} {s['risk']:>8.4f} "
              f"{s['white_frac']:>8.4f} {s['white_run']:>7.3f} "
              f"{s['flat_frac']:>7.3f} {s['black_frac']:>7.3f}")

    n_flag = sum(1 for s in results.values() if s["risk"] > args.threshold)
    flagged = {k for k, s in results.items() if s["risk"] > args.threshold}
    outside_pl = {k for k in flagged if not k.startswith("pl_")}

    print(f"\nimages scoring above {args.threshold}: {n_flag} / {len(results)}")
    if outside_pl:
        print(f"  !! {len(outside_pl)} of them are NOT pl_* -- inspect these:")
        for k in sorted(outside_pl)[:15]:
            print(f"     {k}")
    else:
        print("  all flagged images belong to the pl_* family "
              "(no contamination outside pl_*)")
    print("\nNOTE: this detector keys on page-background brightness. It reliably")
    print("      finds figure pages but MISSES dark/colored watermarks and DICOM")
    print("      header text, which can sit on an otherwise dark scan. The pl_*")
    print("      prefix remains the authoritative filter; use this as a cross-check.")
    print("=" * 72)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
