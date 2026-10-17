"""
audit_labels.py
---------------
Integrity audit for a YOLO-style detection/segmentation dataset.

Validates, per split:
  1. file pairing      - every image has a label + polygon file and vice versa
  2. row format        - labels/ rows are 5 tokens, labels_polygon/ rows are odd
  3. class ids         - all rows use the declared class index
  4. coordinate sanity - normalized values in [0,1], non-degenerate w/h
  5. bbox <-> polygon  - labels/ box matches the min/max extent of labels_polygon/
  6. duplicates        - identical boxes repeated in one file
  7. negatives         - files named neg_* must have no rows
  8. distribution      - stones per image, box area percentiles (small-object risk)

Exit code is non-zero when a hard error is found, so this can be wired into CI
as a test.

Usage:
    python scripts/audit_labels.py --data "Gallstone dataset/dataset_grouped_neg2"
    python scripts/audit_labels.py --data <root> --json runs/final_eval/label_audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# A bbox derived from a polygon may differ slightly from the stored bbox
# (rounding, independent annotation). ~2e-3 normalized is ~1.3 px at 640.
DEFAULT_TOL = 2e-3

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".bmp", ".tif", ".tiff"}
AREA_HARD_MIN = 1e-8          # normalized area; below this the box is degenerate
# Heuristic: a gallstone filling >5% of the frame is unusual. Flagged, not
# rejected -- a gallbladder packed with stones may legitimately be large, so a
# human still has to look. Boxes above AREA_SUSPICIOUS_HIGH (~0.5) are almost
# always an organ or full-frame mislabelling.
AREA_SUSPICIOUS_HIGH = 0.05
AREA_ALMOST_CERTAIN_BAD = 0.50


def _rows(path: Path) -> list[list[float]]:
    """Parse a YOLO label file into a list of float rows."""
    out: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                out.append([float(p) for p in parts])
            except ValueError:
                # keep the raw token count so the format check can report it
                out.append([float("nan")] * len(parts))
    return out


def _shoelace(coords: list[float]) -> float:
    """Absolute area of a normalised polygon given a flat [x1,y1,x2,y2,...] list."""
    pts = list(zip(coords[0::2], coords[1::2]))
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def audit_split(root: Path, split: str, n_classes: int, tol: float) -> dict:
    base = root / split
    img_dir = base / "images"
    lbl_dir = base / "labels"
    pol_dir = base / "labels_polygon"

    rep: dict = {
        "split": split,
        "exists": img_dir.exists(),
        "errors": {},
        "warnings": {},
        "counts": {},
    }
    err: Counter = Counter()
    warn: Counter = Counter()

    if not img_dir.exists():
        rep["errors"]["missing_images_dir"] = 1
        return rep

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
    polys = {p.stem: p for p in pol_dir.glob("*.txt")} if pol_dir.exists() else {}

    rep["counts"] = {
        "images": len(images),
        "label_files": len(labels),
        "polygon_files": len(polys),
    }

    if not labels:
        rep["errors"]["no_label_files"] = 1
        return rep

    # ---- 1. pairing -------------------------------------------------
    for stem in images:
        if stem not in labels:
            err["image_without_label"] += 1
        if stem not in polys:
            warn["image_without_polygon"] += 1
    for stem in labels:
        if stem not in images:
            err["label_without_image"] += 1
    for stem in polys:
        if stem not in images:
            err["polygon_without_image"] += 1

    areas: list[float] = []
    per_image_rows: list[int] = []
    n_empty_label = 0
    n_nonempty = 0
    n_neg_named = 0
    n_neg_violation = 0
    tok_counts: Counter = Counter()
    poly_tok_counts: Counter = Counter()
    worst_delta = 0.0
    n_compared = 0
    n_mismatch = 0
    oversized: dict = {}

    for stem, img_path in images.items():
        lbl = labels.get(stem)
        if lbl is None:
            continue

        rows = _rows(lbl)
        is_neg = stem.startswith("neg")
        if is_neg:
            n_neg_named += 1
            if rows:
                n_neg_violation += 1
                err["neg_named_but_has_rows"] += 1

        # ---- 2/3/4. row format, class, coordinates -------------------
        parsed: list[tuple[float, float, float, float]] = []
        for r in rows:
            tok_counts[len(r)] += 1
            if len(r) != 5:
                err["label_row_not_5_tokens"] += 1
                continue
            cls, cx, cy, w, h = r
            if any(v != v for v in r):  # NaN from a parse failure
                err["label_row_unparseable"] += 1
                continue
            if int(cls) != cls or not (0 <= int(cls) < n_classes):
                err["bad_class_id"] += 1
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                err["centre_out_of_range"] += 1
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                err["wh_out_of_range_or_zero"] += 1
                continue
            if w * h < AREA_HARD_MIN:
                err["degenerate_zero_area"] += 1
                continue
            if cx - w / 2 < -tol or cx + w / 2 > 1 + tol:
                err["box_crosses_image_edge"] += 1
            if cy - h / 2 < -tol or cy + h / 2 > 1 + tol:
                err["box_crosses_image_edge"] += 1
            parsed.append((cls, cx, cy, w, h))
            areas.append(w * h)
            if w * h > AREA_SUSPICIOUS_HIGH:
                oversized.setdefault(split, []).append(
                    {"file": stem, "area": round(w * h, 5),
                     "w": round(w, 4), "h": round(h, 4)}
                )
                if w * h > AREA_ALMOST_CERTAIN_BAD:
                    warn["box_almost_certainly_bad_annotation"] += 1
                else:
                    warn["box_oversized_review_me"] += 1

        # ---- 6. duplicates -------------------------------------------
        seen = set()
        for _cls, cx, cy, w, h in parsed:
            key = (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
            if key in seen:
                warn["duplicate_box_in_file"] += 1
            seen.add(key)

        if rows:
            n_nonempty += 1
        else:
            n_empty_label += 1
        per_image_rows.append(len(parsed))

        # ---- 5. bbox <-> polygon agreement ---------------------------
        pol = polys.get(stem)
        if pol is None or not parsed:
            continue
        prows = _rows(pol)
        for r in prows:
            poly_tok_counts[len(r)] += 1
        good = [r for r in prows if len(r) >= 7 and len(r) % 2 == 1]
        if not good:
            if prows:
                warn["polygon_rows_malformed"] += 1
            continue

        # polygon centroids vs stored centroids, greedy nearest match
        poly_boxes = []
        for r in good:
            co = r[1:]
            xs, ys = co[0::2], co[1::2]
            poly_boxes.append(
                (min(xs), min(ys), max(xs), max(ys), _shoelace(co))
            )

        for _cls, cx, cy, w, h in parsed:
            bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            best, best_d = None, 1e9
            for pb in poly_boxes:
                d = max(
                    abs(pb[0] - bx1), abs(pb[1] - by1),
                    abs(pb[2] - bx2), abs(pb[3] - by2),
                )
                if d < best_d:
                    best_d, best = d, pb
            if best is None:
                continue
            n_compared += 1
            worst_delta = max(worst_delta, best_d)
            if best_d > tol:
                n_mismatch += 1

    if n_mismatch:
        err["bbox_polygon_mismatch"] = n_mismatch

    def pct(v: list[float], p: float) -> float:
        if not v:
            return 0.0
        s = sorted(v)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]

    rep["counts"].update({
        "label_rows": sum(tok_counts.values()),
        "polygon_rows": sum(poly_tok_counts.values()),
        "empty_label_files": n_empty_label,
        "nonempty_label_files": n_nonempty,
        "neg_prefixed_files": n_neg_named,
        "neg_with_rows": n_neg_violation,
        "boxes": len(areas),
        "stones_per_image_mean": round(statistics.fmean(per_image_rows), 3)
        if per_image_rows else 0.0,
        "stones_per_image_max": max(per_image_rows) if per_image_rows else 0,
    })
    rep["box_area_normalised"] = {
        "min": round(min(areas), 6) if areas else 0.0,
        "p05": round(pct(areas, 5), 6),
        "p25": round(pct(areas, 25), 6),
        "median": round(pct(areas, 50), 6),
        "p75": round(pct(areas, 75), 6),
        "p95": round(pct(areas, 95), 6),
        "max": round(max(areas), 6) if areas else 0.0,
    }
    rep["label_row_token_counts"] = dict(sorted(tok_counts.items()))
    rep["polygon_row_token_counts"] = dict(sorted(poly_tok_counts.items()))
    rep["bbox_vs_polygon"] = {
        "compared": n_compared,
        "mismatched": n_mismatch,
        "worst_abs_delta": round(worst_delta, 6),
        "tolerance": tol,
    }
    rep["oversized_boxes"] = oversized
    rep["errors"] = dict(err)
    rep["warnings"] = dict(warn)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a YOLO detection/segmentation dataset")
    ap.add_argument("--data", required=True, help="dataset root (contains split dirs)")
    ap.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    ap.add_argument("--n-classes", type=int, default=1)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--json", help="also write the report to this path")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}")
        return 2

    print("=" * 72)
    print("LABEL AUDIT")
    print(f"root: {root}")
    print("=" * 72)

    reports = []
    hard_errors = 0
    soft_warnings = 0

    for split in args.splits:
        rep = audit_split(root, split, args.n_classes, args.tol)
        reports.append(rep)
        if not rep["exists"]:
            print(f"\n[{split}] MISSING images dir - skipped")
            continue

        c = rep["counts"]
        print(f"\n[{split}]")
        print(f"  images {c.get('images',0):>6}   label files {c.get('label_files',0):>6}"
              f"   polygon files {c.get('polygon_files',0):>6}")
        print(f"  label rows {c.get('label_rows',0):>6}   polygon rows {c.get('polygon_rows',0):>6}")
        print(f"  empty-label (negative) {c.get('empty_label_files',0):>6}"
              f"   non-empty {c.get('nonempty_label_files',0):>6}")
        print(f"  stones/image mean {c.get('stones_per_image_mean',0):.2f}"
              f"   max {c.get('stones_per_image_max',0)}")

        a = rep.get("box_area_normalised", {})
        if a:
            print(f"  box area (norm)  p05 {a['p05']:.5f}  p25 {a['p25']:.5f}"
                  f"  median {a['median']:.5f}  p75 {a['p75']:.5f}  max {a['max']:.5f}")

        bp = rep.get("bbox_vs_polygon", {})
        if bp.get("compared"):
            status = "OK" if bp["mismatched"] == 0 else "MISMATCH"
            print(f"  bbox vs polygon: {bp['compared']} compared, "
                  f"{bp['mismatched']} mismatched ({status}), "
                  f"worst delta {bp['worst_abs_delta']:.5f} (tol {bp['tolerance']})")

        if rep["errors"]:
            hard_errors += sum(rep["errors"].values())
            print("  ERRORS:")
            for k, v in sorted(rep["errors"].items(), key=lambda kv: -kv[1]):
                print(f"      {k:<38} {v}")
        if rep["warnings"]:
            soft_warnings += sum(rep["warnings"].values())
            print("  warnings:")
            for k, v in sorted(rep["warnings"].items(), key=lambda kv: -kv[1]):
                print(f"      {k:<38} {v}")
        if not rep["errors"] and not rep["warnings"]:
            print("  clean")

    print("\n" + "=" * 72)
    print(f"hard errors: {hard_errors}    warnings: {soft_warnings}")
    print("=" * 72)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"root": str(root), "splits": reports}, indent=2))
        print(f"report written to {out}")

    if hard_errors:
        print("\nRESULT: FAILED (see errors above)")
        return 1
    print("\nRESULT: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())