"""Erase burned-in text (scanner labels, captions) from ultrasound frames.

Why: the OCR audit (scripts/text_audit.py, 2026-10-04) found scanner text on
106/136 normal-gallbladder negatives ("Voluson E6", frequency/depth readouts)
but on almost no stone images. A detector can learn "this scanner's text =>
no stone" instead of anatomy. Removing text from every image, at training and
inference alike, takes that cue away.

Text boxes come from offline OCR (rapidocr-onnxruntime); each box is padded,
then filled with the speckle-matched inpainting used for coloured overlays
(`overlay_removal._textured_fill`). Boxes that overlap a protected region
(e.g. a labelled stone) are left alone.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .overlay_removal import _textured_fill

MIN_SCORE = 0.5   # OCR confidence for a text box
PAD = 3           # pixels added around each text box


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def text_mask(image_bgr: np.ndarray, protect: np.ndarray | None = None) -> np.ndarray:
    """uint8 mask (255 = text) of OCR-detected text; `protect` = xyxy boxes never masked."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    res, _ = _ocr()(image_bgr)
    for quad, _txt, score in res or []:
        if float(score) < MIN_SCORE:
            continue
        q = np.asarray(quad, np.float32)
        x1, y1 = np.floor(q.min(0)).astype(int) - PAD
        x2, y2 = np.ceil(q.max(0)).astype(int) + PAD
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if protect is not None and len(protect):
            p = np.asarray(protect, np.float32).reshape(-1, 4)
            if ((p[:, 0] < x2) & (p[:, 2] > x1) & (p[:, 1] < y2) & (p[:, 3] > y1)).any():
                continue
        mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def remove_burned_in_text(image_bgr: np.ndarray, protect: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Return (image with text inpainted, number of masked pixels). Output is greyscale-in-BGR."""
    mask = text_mask(image_bgr, protect)
    n = int((mask > 0).sum())
    if n == 0:
        return image_bgr, 0
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(_textured_fill(grey, mask), cv2.COLOR_GRAY2BGR), n
