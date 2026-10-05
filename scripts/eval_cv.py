"""Pool out-of-fold predictions from grouped K-fold CV and score them.

For every fold k, the fold-k model predicts the fold-k `valid` images (groups
it never trained on). The pooled out-of-fold (OOF) predictions cover every
train+valid image exactly once, so the metrics rest on ~95 video groups
instead of 17. `test` is not used.

Also writes the per-image OOF predictions (`--save-preds`), the input for the
label audit (images where model and label disagree) and for paired
comparisons between configurations.

Usage:
  python scripts/eval_cv.py --cv dataset_cv --runs runs/cv/rf704 --out runs/cv/rf704/oof_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from eval_grouped import image_level, load_split, metrics_with_ci, rfdetr_fn, yolo_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", default="dataset_cv")
    ap.add_argument("--runs", required=True, help="folder with fold0..foldK run outputs")
    ap.add_argument("--model", choices=["rfdetr", "yolo"], default="rfdetr")
    ap.add_argument("--rfdetr-resolution", type=int)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-preds", help="write per-image OOF predictions here (json)")
    args = ap.parse_args()

    cv, runs = ROOT / args.cv, ROOT / args.runs
    rows, preds = [], []
    for fd in sorted(cv.glob("fold*")):
        run = runs / fd.name
        if args.model == "rfdetr":
            # best_total is written at the end of training; a run cut short only has best_ema
            cands = [run / "rfdetr_large" / n for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth")]
            w = next((c for c in cands if c.exists()), cands[0])
            print(f"{fd.name}: using {w.name}")
        else:
            w = next(run.glob("*/weights/best.pt"))
        if not w.exists():
            sys.exit(f"missing weights for {fd.name}: {w}")
        f = rfdetr_fn(str(w), args.rfdetr_resolution) if args.model == "rfdetr" else yolo_fn(str(w))
        fold_rows = load_split(fd, "valid")
        for r in fold_rows:
            b, s = f(r["img"])
            r["fold"] = fd.name
            rows.append(r)
            preds.append({"boxes": b, "scores": s, "labels": np.zeros(len(s))})
        print(f"{fd.name}: {len(fold_rows)} OOF images")

    if args.save_preds:  # before the slow bootstrap, so a long run never loses them
        (ROOT / args.save_preds).write_text(json.dumps([
            {"name": r["name"], "group": r["group"], "fold": r["fold"], "gt": r["gt"].tolist(),
             "boxes": p["boxes"].tolist(), "scores": p["scores"].tolist()} for r, p in zip(rows, preds)]))
    report = {"cv": args.cv, "runs": args.runs, "images": len(rows),
              "groups": len({r["group"] for r in rows}),
              "oof": metrics_with_ci(rows, preds, args.bootstrap)}
    report["oof"]["image_level"] = image_level(rows, preds)
    m = report["oof"]
    print(f"OOF ({len(rows)} images, {report['groups']} groups): mAP50:95 {m['map']:.3f} {m['ci95']['map']}  "
          f"mAP50 {m['map_50']:.3f} {m['ci95']['map_50']}  {m['image_level']}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
