"""
review_sheet.py
--------------
Render an annotated contact sheet for visual label QC.

Given a list of image stems (or read them straight out of a label-audit JSON),
draws each image's ground-truth boxes on top of the image, tiles the results,
and writes a single PNG that a human (or an agent with vision) can review at a
glance.

This exists because numeric audits flag problems but cannot adjudicate them:
"box area 0.99" says something is wrong, only the picture says what.

Usage:
    # review every oversized box flagged by audit_labels.py
    python scripts/review_sheet.py \
        --data "Gallstone dataset/dataset_grouped_neg2" \
        --audit runs/final_eval/label_audit.json \
        --split test --out runs/final_eval/review_test.png

    # review specific stems, with a wider box outline
    python scripts/review_sheet.py --data <root> --split train \
        --stems pseudo_c26__57_dd39a4aedb9b pseudo_c29__28_e184d9171901
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 300, 300
CAPTION_H = 46
COLS = 4
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

try:
    FONT = ImageFont.load_default(14)
    FONT_SM = ImageFont.load_default(11)
except TypeError:  # very old Pillow
    FONT = ImageFont.load_default()
    FONT_SM = ImageFont.load_default()


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Return [(x1, y1, x2, y2), ...] in pixels for a YOLO label file."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(p) for p in parts)
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def collect_from_audit(audit_path: Path, split: str) -> list[str]:
    """Pull the oversized-box filenames this split flagged."""
    data = json.loads(audit_path.read_text())
    for s in data["splits"]:
        if s["split"] == split:
            return [e["file"] for e in s.get("oversized_boxes", {}).get(split, [])]
    return []


def sample_stems(root: Path, split: str, n: int, stratify: bool,
                 seed: int) -> list[str]:
    """Randomly sample stems that actually carry boxes.

    Stratified mode sorts positives by box area and takes an even spread, so
    the sheet shows small, medium and large stones rather than being dominated
    by whichever size happens to be most common.
    """
    import random

    rng = random.Random(seed)
    # NOTE: `root` here is already the split directory
    lbl_dir = root / "labels"

    entries: list[tuple[str, float]] = []
    for p in sorted(lbl_dir.glob("*.txt")):
        boxes = read_boxes(p)
        if not boxes:
            continue
        biggest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        entries.append((p.stem, biggest))

    if not entries:
        return []

    if not stratify:
        picked = rng.sample(entries, min(n, len(entries)))
        return [s for s, _ in picked]

    entries.sort(key=lambda e: e[1])
    out: list[str] = []
    per = max(1, n // 4)
    size = len(entries)
    # four equal bands of the area distribution
    for band in range(4):
        lo = (size * band) // 4
        hi = (size * (band + 1)) // 4
        chunk = entries[lo:hi]
        out.extend(s for s, _ in rng.sample(chunk, min(per, len(chunk))))
    return out[:n]


def load_tile(img_path: Path, boxes: list[tuple[float, float, float, float]],
              caption: str, scale_note: str = "") -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        px = (x1 * W, y1 * H, x2 * W, y2 * H)
        d.rectangle(px, outline=(0, 230, 0), width=3)
        area = (x2 - x1) * (y2 - y1)
        tag = f"#{i + 1} area={area * 100:.1f}%"
        d.rectangle([px[0], max(0, px[1] - 18), px[0] + 150, px[1]], fill=(0, 0, 0))
        d.text((px[0] + 3, max(0, px[1] - 17)), tag, fill=(255, 255, 0), font=FONT_SM)

    # frame size tells you what the box means in real pixels
    d.rectangle([0, 0, W - 1, H - 1], outline=(90, 90, 90), width=1)
    d.text((4, 4), f"{W}x{H}", fill=(180, 220, 255), font=FONT_SM)

    tile = Image.new("RGB", (TILE_W, TILE_H + CAPTION_H), (24, 24, 28))
    scale = min(TILE_W / W, TILE_H / H)
    tw, th = max(1, int(W * scale)), max(1, int(H * scale))
    tile.paste(img.resize((tw, th), Image.LANCZOS), (0, 0))

    td = ImageDraw.Draw(tile)
    td.text((4, TILE_H + 3), caption[:42], fill=(235, 235, 235), font=FONT)
    if scale_note:
        td.text((4, TILE_H + 20), scale_note, fill=(150, 200, 255), font=FONT_SM)
    td.text((4, TILE_H + 32), f"{len(boxes)} box(es)", fill=(200, 200, 160), font=FONT_SM)
    return tile


def main() -> int:
    ap = argparse.ArgumentParser(description="Annotated contact sheet for label QC")
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--audit", help="label_audit.json to pull flagged stems from")
    ap.add_argument("--stems", nargs="*", help="explicit stems (overrides --audit)")
    ap.add_argument("--sample", type=int, help="render a random sample instead")
    ap.add_argument("--stratify", action="store_true",
                    help="stratify the sample by box area (small/med/large stones)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data) / args.split
    img_dir, lbl_dir = root / "images", root / "labels"

    if args.sample:
        stems = sample_stems(root, args.split, args.sample, args.stratify, args.seed)
        src = f"random sample n={args.sample}{' stratified by size' if args.stratify else ''}"
    elif args.stems:
        stems = list(args.stems)
        src = "explicit"
    elif args.audit:
        stems = collect_from_audit(Path(args.audit), args.split)
        src = "audit oversized_boxes"
    else:
        print("need --audit, --stems or --sample")
        return 2

    if not stems:
        print(f"no stems to review for split '{args.split}'")
        return 0

    tiles = []
    missing = []
    for stem in stems:
        img_path = next(
            (img_dir / f"{stem}{e}" for e in IMG_EXTS if (img_dir / f"{stem}{e}").exists()),
            None,
        )
        if img_path is None:
            missing.append(stem)
            continue
        boxes = read_boxes(lbl_dir / f"{stem}.txt")
        tiles.append(load_tile(img_path, boxes, stem))

    if missing:
        print(f"WARNING: {len(missing)} stem(s) not found: {missing[:5]}")

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new(
        "RGB", (COLS * TILE_W, rows * (TILE_H + CAPTION_H)), (14, 14, 16)
    )
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % COLS) * TILE_W, (i // COLS) * (TILE_H + CAPTION_H)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"split={args.split}  source={src}  tiles={len(tiles)}  "
          f"size={sheet.size}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())