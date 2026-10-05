"""Remove coloured annotation overlays (arrows, markers) burned into B-mode images.

Why: 52% of training images carry yellow arrows pointing at the stone, against
21% of test images (docs/LEAKAGE_AUDIT.md). A detector can learn "find the
arrow" instead of "find the stone", and real scans have no arrows.

B-mode ultrasound is greyscale: images without overlays have exactly zero
chroma (measured 2026-10-02). So any pixel with Lab chroma above CHROMA_MIN is
overlay - including faint, JPEG-blurred dotted stone outlines that a
saturation threshold misses. Those pixels are masked, dilated and inpainted,
and the result is always returned as greyscale so no colour cue survives.
Images tinted as a whole (e.g. GE Voluson sepia) carry chroma proportional to
brightness at a near-constant hue; pixels that break that tint model (yellow
labels such as "GB", green/blue annotation text) are inpainted before the
image is converted to greyscale.

Plain inpainting leaves smooth, waxy patches (local texture 0.55x the
surrounding tissue) right next to the stones the arrows pointed at, which is
itself a cue. The fill therefore adds speckle-like noise scaled to the local
texture of the neighbouring tissue; NOISE_SCALE = 0.55 brings filled regions to
~1.00x their surroundings (calibrated on 80 annotated frames, 2026-10-02).
The noise seed is derived from the image content, so it is deterministic but
never the same pattern twice.

Apply the SAME function at training and inference time.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

CHROMA_MIN = 4.0      # Lab chroma above which a pixel is overlay (clean frames: 0)
NOISE_SCALE = 0.55    # speckle strength in filled regions (texture ratio ~1.0)
TINT_FRACTION = 0.30  # raw (undilated) coloured share above which the whole image is tinted;
                      # heavily annotated frames reach ~30%, Voluson sepia frames ~100%


def raw_overlay_mask(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return (chroma > CHROMA_MIN).astype(np.uint8) * 255


def overlay_mask(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.dilate(raw_overlay_mask(image_bgr), np.ones((3, 3), np.uint8), iterations=2)


def _local_std(grey: np.ndarray, k: int = 7) -> np.ndarray:
    g = grey.astype(np.float32)
    m = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))


def _textured_fill(grey: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.inpaint(grey, mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
    # spread the neighbouring tissue's texture level into the hole
    sigma = cv2.inpaint(np.clip(_local_std(grey), 0, 255).astype(np.uint8), mask, 9,
                        cv2.INPAINT_TELEA).astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(grey.tobytes()))
    noise = cv2.GaussianBlur(rng.standard_normal(grey.shape).astype(np.float32), (0, 0), 1.0)
    noise /= noise.std() + 1e-6
    out = base.copy()
    m = mask > 0
    out[m] += NOISE_SCALE * noise[m] * sigma[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def _off_tint_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pixels whose colour departs from a whole-image tint (hue or chroma/L ratio)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    tissue = (L > 20) & (chroma > CHROMA_MIN)
    if tissue.sum() < 1000:
        return np.zeros(L.shape, np.uint8)
    h0 = np.median(hue[tissue])
    r0 = np.median((chroma / (L + 1.0))[tissue])
    dh = np.abs((hue - h0 + 180.0) % 360.0 - 180.0)
    off = (chroma > 8.0) & ((dh > 20.0) | (chroma / (L + 1.0) > 1.8 * r0))
    return off.astype(np.uint8) * 255


def remove_color_overlays(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (cleaned BGR image, action) where action is 'none', 'inpainted' or 'greyscaled'."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return image_bgr, "none"
    raw = raw_overlay_mask(image_bgr)
    frac = float((raw > 0).mean())
    mask = cv2.dilate(raw, np.ones((3, 3), np.uint8), iterations=2)
    to_grey = lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if frac == 0.0:
        return to_grey(image_bgr), "none"
    if frac > TINT_FRACTION:
        tint_off = _off_tint_mask(image_bgr)
        if tint_off.any():
            grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            mask = cv2.dilate(tint_off, np.ones((3, 3), np.uint8), iterations=2)
            return cv2.cvtColor(_textured_fill(grey, mask), cv2.COLOR_GRAY2BGR), "greyscaled"
        return to_grey(image_bgr), "greyscaled"
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(_textured_fill(grey, mask), cv2.COLOR_GRAY2BGR), "inpainted"
