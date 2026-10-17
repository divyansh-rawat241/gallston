"""
visualize_conformal.py
---------------------
Render what a split-conformal bounding box actually looks like.

Split conformal takes per-coordinate residuals from a calibration split and
expands every predicted box outward by a fixed quantile. The guarantee is
distributional -- P(ground truth inside the expanded box) >= 1-alpha -- not
per-image. That makes the expansion easy to misjudge: the extra border is only
a few pixels, so at full-frame scale it looks like nothing happened and the
feature appears broken.

Each case is therefore drawn as a pair:

    [ full frame, both boxes ]   [ zoom on the lesion, same boxes ]

so the expansion is legible. Numbers give the quantile in pixels at native
resolution and the relative growth of the box.

IMPORTANT
---------
The default quantiles are the ones this project actually fitted, recorded in
docs/BENCHMARKS.md (x1 0.0044, y1 0.0051, x2 0.0094, y2 0.0088 at alpha=0.05).
Those came from the legacy leaked split under the RF-DETR BGR fusion bug, so
they are ILLUSTRATIVE ONLY. Refit on a clean split before using any such figure
in a manuscript.

Coordinate spaces
-----------------
YOLO label rows are normalized (cx, cy, w, h) in [0, 1]. Everything the
drawing code touches is in PIXELS. read_boxes_norm and read_boxes_px keep those
two spaces apart deliberately -- conflating them silently produces 0x0 boxes.

Usage
-----
    python scripts/visualize_conformal.py \
        --data "Gallstone dataset/dataset_grouped_neg2" --split test \
        --target-areas 0.0006 0.0024 0.0130 \
        --out runs/final_eval/conformal_demo_test.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_Q = (0.0044, 0.0051, 0.0094, 0.0088)
DEFAULT_ALPHA = 0.05

PT = (60, 220, 90)        # point estimate, solid green
CF = (255, 176, 60)       # conformal expansion, dashed orange
BG = (14, 14, 18)
CAPTION_H = 78
LEGEND_H = 26
ZOOM = 520

try:
    FONT = ImageFont.load_default(13)
    FONT_S = ImageFont.load_default(11)
    FONT_B = ImageFont.load_default(15)
except TypeError:
    FONT = FONT_S = FONT_B = ImageFont.load_default()


# ---------------------------------------------------------------- label IO
def read_boxes_norm(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Ground-truth boxes as NORMALIZED xyxy in [0,1]. Use for size stats."""
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


def norm_to_px(boxes, W: int, H: int):
    """Normalized xyxy -> pixel xyxy."""
    return [(b[0] * W, b[1] * H, b[2] * W, b[3] * H) for b in boxes]


# ------------------------------------------------------------ conformal
def expand(box, q, W: int, H: int):
    """Push each edge outward by its per-coordinate quantile, then clip.

    Expansion is one-sided by construction: the residual for the left edge is
    max(0, pred.x1 - gt.x1), so the conformal box can only ever grow outward.
    """
    x1, y1, x2, y2 = box
    return (max(0.0, x1 - q[0] * W),
            max(0.0, y1 - q[1] * H),
            min(float(W), x2 + q[2] * W),
            min(float(H), y2 + q[3] * H))


def dashed_rect(d, box, colour, width=2, dash=8, gap=5):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if x2 <= x1 or y2 <= y1:
        return
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2),
                             (x2, y2, x1, y2), (x1, y2, x1, y1)):
        dx, dy = bx - ax, by - ay
        n = int(max(abs(dx), abs(dy), 1))
        pos = 0
        while pos < n:
            end = min(pos + dash, n)
            d.line([(ax + dx * pos / n, ay + dy * pos / n),
                    (ax + dx * end / n, ay + dy * end / n)],
                   fill=colour, width=width)
            pos = end + gap


# ---------------------------------------------------------------- drawing
def render_case(img: Image.Image, boxes_px, q, title, subtitle):
    """One row: full frame (left) + zoom on largest lesion (right) + caption."""
    W, H = img.size
    main_w = ZOOM
    scale = main_w / W
    main_h = max(1, int(H * scale))

    row = Image.new("RGB", (main_w * 2 + 24, max(main_h, ZOOM) + CAPTION_H), BG)

    # ---- left: full frame ------------------------------------------------
    view = img.resize((main_w, main_h), Image.LANCZOS)
    dv = ImageDraw.Draw(view)
    for b in boxes_px:
        c = expand(b, q, W, H)
        dashed_rect(dv, [c[0] * scale, c[1] * scale,
                         c[2] * scale, c[3] * scale], CF, width=3)
        dv.rectangle([b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale],
                     outline=PT, width=3)
    row.paste(view, (0, 0))
    ImageDraw.Draw(row).text((4, main_h + 4), "FULL FRAME", fill=(190, 190, 200),
                             font=FONT_S)

    # ---- right: zoom on the largest box ----------------------------------
    zpanel = Image.new("RGB", (ZOOM, ZOOM), BG)
    dz = ImageDraw.Draw(zpanel)
    if boxes_px:
        b = max(boxes_px, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        half = max(b[2] - b[0], b[3] - b[1]) * 0.60 + 8
        zx1, zy1 = max(0.0, cx - half), max(0.0, cy - half)
        zx2, zy2 = min(float(W), cx + half), min(float(H), cy + half)
        cw, ch = max(1, int(zx2 - zx1)), max(1, int(zy2 - zy1))
        crop = img.crop((int(zx1), int(zy1), int(zx2), int(zy2)))
        zs = min(ZOOM / cw, ZOOM / ch)
        crop = crop.resize((max(1, int(cw * zs)), max(1, int(ch * zs))),
                           Image.LANCZOS)
        ox, oy = (ZOOM - crop.width) // 2, (ZOOM - crop.height) // 2
        zpanel.paste(crop, (ox, oy))

        def z(bb):
            return [ox + (bb[0] - zx1) * zs, oy + (bb[1] - zy1) * zs,
                    ox + (bb[2] - zx1) * zs, oy + (bb[3] - zy1) * zs]

        for bb in boxes_px:
            c = expand(bb, q, W, H)
            dashed_rect(dz, z(c), CF, width=3)
            dz.rectangle(z(bb), outline=PT, width=3)

    row.paste(zpanel, (main_w + 24, 0))
    ImageDraw.Draw(row).text((main_w + 28, max(main_h, ZOOM) + 4),
                             "ZOOM on lesion", fill=(190, 190, 200), font=FONT_S)

    # ---- caption ---------------------------------------------------------
    dc = ImageDraw.Draw(row)
    y = max(main_h, ZOOM) + 20
    dc.text((4, y), title, fill=(238, 238, 244), font=FONT_B)
    dc.text((4, y + 20), subtitle, fill=(150, 205, 255), font=FONT_S)
    return row


def build_sheet(rows, q, alpha, out_note):
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 10 * (len(rows) - 1) + LEGEND_H
    sheet = Image.new("RGB", (W, H), BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10

    d = ImageDraw.Draw(sheet)
    ly = H - LEGEND_H + 5
    d.rectangle([4, ly, 24, ly + 12], outline=PT, width=2)
    d.text((30, ly), "point estimate", fill=(205, 205, 205), font=FONT_S)
    d.rectangle([160, ly, 180, ly + 12], outline=CF, width=2)
    d.text((186, ly), "conformal expansion (>= 1-alpha marginal coverage)",
           fill=(205, 205, 205), font=FONT_S)
    d.text((W - 470, ly), f"alpha={alpha:.2f}   " + out_note, fill=(150, 150, 160),
           font=FONT_S)
    return sheet


# ---------------------------------------------------------------- picking
def pick_by_area(lbl_dir: Path, targets):
    """Choose images whose largest normalized box area is nearest each target."""
    entries = []
    for p in sorted(lbl_dir.glob("*.txt")):
        boxes = read_boxes_norm(p)
        if boxes:
            a = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
            entries.append((p.stem, a))
    picked = []
    for t in targets:
        if entries:
            stem, _ = min(entries, key=lambda e: abs(e[1] - t))
            picked.append(stem)
            entries = [e for e in entries if e[0] != stem]
    return picked


def find_image(img_dir: Path, stem: str):
    for e in IMG_EXTS:
        p = img_dir / f"{stem}{e}"
        if p.exists():
            return p
    return None


def predict_boxes(model, img_path: Path, conf: float, keep_cls: set[int] | None):
    """Run a detector and return pixel xyxy boxes, optionally filtered by class."""
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    out = []
    for b, c in zip(xyxy, cls):
        if keep_cls is None or c in keep_cls:
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualise conformal box expansion")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stems", nargs="*")
    ap.add_argument("--target-areas", nargs="*", type=float,
                    default=[0.0006, 0.0024, 0.0130])
    ap.add_argument("--q", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                    default=list(DEFAULT_Q))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--labels", default="small,median,large")
    ap.add_argument("--model", help="ultralytics weights; when set, draw predictions "
                                    "instead of ground truth")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--class-id", type=int, default=0,
                    help="keep only this predicted class (0 = gallstone)")
    ap.add_argument("--source", default="gt", choices=["gt", "pred"],
                    help="label printed on the figure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data) / args.split
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.exists():
        print(f"missing images dir: {img_dir}")
        return 2

    model = None
    if args.model:
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"model: {args.model}  task={model.task}  names={getattr(model,'names',None)}")

    q = tuple(args.q)
    stems = list(args.stems) if args.stems else pick_by_area(lbl_dir, args.target_areas)
    names = args.labels.split(",")

    rows = []
    for i, stem in enumerate(stems):
        ip = find_image(img_dir, stem)
        if ip is None:
            print(f"  skip: no image for {stem}")
            continue

        img = Image.open(ip).convert("RGB")
        W, H = img.size

        if model is not None:
            boxes_px = predict_boxes(model, ip, args.conf, {args.class_id})
            gt_n = len(read_boxes_norm(lbl_dir / f"{stem}.txt"))
            if not boxes_px:
                print(f"  {stem}: no detections, skipped")
                continue
        else:
            boxes_n = read_boxes_norm(lbl_dir / f"{stem}.txt")
            if not boxes_n:
                continue
            boxes_px = norm_to_px(boxes_n, W, H)
            gt_n = len(boxes_px)

        big = max(boxes_px, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
        pw, ph = big[2] - big[0], big[3] - big[1]
        cf = expand(big, q, W, H)
        dw, dh = cf[2] - cf[0], cf[3] - cf[1]
        growth = (dw / pw - 1) * 100 if pw > 0 else 0.0

        tag = names[i].strip() if i < len(names) else f"case{i+1}"
        title = f"[{tag}] {stem}   native {W}x{H}   {len(boxes_px)} box(es)"
        if model is not None:
            title += f"   gt_stones={gt_n}   MODEL PREDICTION"
        else:
            title += "   ground-truth box"
        subtitle = (
            f"point {pw:.0f}x{ph:.0f}px  ->  conformal {dw:.0f}x{dh:.0f}px   "
            f"border L+{q[0]*W:.1f} T+{q[1]*H:.1f} R+{q[2]*W:.1f} B+{q[3]*H:.1f}px"
            f"   growth {growth:+.0f}%")

        rows.append(render_case(img, boxes_px, q, title, subtitle))

    if not rows:
        print("nothing rendered")
        return 1

    sheet = build_sheet(rows, q, args.alpha,
                        "quantiles from legacy leaked split - illustrative only")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"alpha={args.alpha}  q(norm) x1={q[0]} y1={q[1]} x2={q[2]} y2={q[3]}")
    print(f"rendered {len(rows)} case(s) -> {out}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())