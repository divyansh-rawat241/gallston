"""OCR every image and flag burned-in text that could leak the answer or identify a patient.

Coloured overlays are removed in preprocessing, but white/grey text survives.
A detector can learn words like "stone" or "cholelithiasis" instead of the
anatomy (shortcut learning), and names / IDs are patient data. This script
runs offline OCR (rapidocr-onnxruntime) on each image as the model sees it and
writes a CSV with the recognised text and three flags:

  answer_text - diagnostic words that reveal the label (stone, calculus,
                cholelithiasis, gallstone, sludge, polyp, normal, ...)
  watermark   - publisher / stock-photo / teaching-site marks
  phi         - patterns that look like patient identifiers (PT:/name:,
                MRN/ID numbers, dates of birth)

Plain scanner annotation (depth, gain, "GB", probe names) is recorded but not
flagged: it appears on every clinical screen and carries no label information.

Usage: python scripts/text_audit.py --data dataset_grouped_neg3 --out runs/text_audit/neg3.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

ANSWER = re.compile(r"stone|calcul|lithias|gall\s*stone|sludge|polyp|choleste|crystal|normal|shadow|"
                    r"cholecyst|adenomyo|carcinoma|mass|tumou?r|lesion|hyperecho|echogenic", re.I)
WATERMARK = re.compile(r"elsevier|alamy|getty|shutterstock|dreamstime|istock|science\s*photo|radiopaedia|"
                       r"imaging\s*study|netter|wikimedia|copyright|©|www\.|\.com|credit", re.I)
PHI = re.compile(r"\bpt\s*[:.]|patient\s*[:.]|name\s*[:.]|\bmrn\b|\bid\s*[:#]|\bdob\b|"
                 r"\b\d{6,}\b|\b(mr|mrs|ms)\.?\s+[a-z]", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_grouped_neg3")
    ap.add_argument("--out", default="runs/text_audit/neg3.csv")
    ap.add_argument("--min-score", type=float, default=0.6)
    args = ap.parse_args()
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    data, out = ROOT / args.data, ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    man = json.loads((data / "split_manifest.json").read_text())["images"]
    n_flag = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "split", "group", "answer_text", "watermark", "phi", "text"])
        w.writeheader()
        for i, (name, m) in enumerate(sorted(man.items())):
            img = cv2.imread(str(data / m["split"] / "images" / name))
            res, _ = ocr(img)
            words = [t for _, t, s in (res or []) if float(s) >= args.min_score]
            text = " | ".join(words)
            row = {"name": name, "split": m["split"], "group": m["group"],
                   "answer_text": int(bool(ANSWER.search(text))), "watermark": int(bool(WATERMARK.search(text))),
                   "phi": int(bool(PHI.search(text))), "text": text[:300]}
            n_flag += row["answer_text"] or row["watermark"] or row["phi"]
            w.writerow(row)
            if i % 200 == 0:
                print(f"{i}/{len(man)} images, {n_flag} flagged", flush=True)
    print(f"done: {len(man)} images, {n_flag} flagged -> {out}")


if __name__ == "__main__":
    main()
