"""
preprocessing.py
----------------
Full preprocessing pipeline for Sri Lankan currency note images.

Pipeline:
  Input Image
    → Grayscale → Gaussian Blur → Adaptive Threshold → Morphological Ops
    → Find Contours → Filter → Detect Rectangle
    → Perspective Warp (1024×512)
    → HSV conversion + CLAHE on V channel
    → Output: normalised numpy array
"""

import base64
import cv2
import numpy as np
from typing import Optional, Tuple

WARP_W = 1024
WARP_H = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_grayscale(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _blur(gray, ksize=5):
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)

def _adaptive_threshold(blurred):
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=4)

def _morphology(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(closed, kernel, iterations=1)

def _find_note_contour(processed):
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = processed.shape[0] * processed.shape[1]
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < 0.15 * img_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def _order_points(pts):
    rect    = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _perspective_warp(img, pts):
    rect = _order_points(pts)
    dst  = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]],
                    dtype=np.float32)
    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    h, w   = warped.shape[:2]
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        warped = cv2.resize(warped, (WARP_W, WARP_H))
    return warped

def _apply_clahe_hsv(bgr):
    hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_eq  = cv2.merge([h, s, clahe.apply(v)])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Step image encoder  (base64 is imported at top of file)
# ---------------------------------------------------------------------------

def _encode_step(img: np.ndarray) -> str:
    """Encode a single step image as a base64 PNG string."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(image_input, debug=False):
    """
    Returns
    -------
    (bgr, hsv)              when debug=False
    (bgr, hsv, steps)       when debug=True

    steps is a dict with keys:
        step1_original      — raw photo with green contour drawn
        step2_grayscale     — grayscale image
        step3_threshold     — adaptive threshold binary image
        step4_morphology    — after morphological operations
        step5_warp          — perspective warped 1024x512
        step6_clahe         — final CLAHE normalised output
        corners_detected    — bool
        fallback_used       — bool

    bgr / hsv are None if note boundary not found.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {image_input}")
    else:
        img = image_input.copy()

    original = img.copy()
    gray     = _to_grayscale(img)
    blurred  = _blur(gray)
    thresh   = _adaptive_threshold(blurred)
    morph    = _morphology(thresh)
    pts      = _find_note_contour(morph)

    if pts is None:
        h, w          = img.shape[:2]
        pts           = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],
                                  dtype=np.float32)
        fallback_used = True
    else:
        fallback_used = False

    warped     = _perspective_warp(original, pts)
    normalised = _apply_clahe_hsv(warped)
    hsv        = cv2.cvtColor(normalised, cv2.COLOR_BGR2HSV)

    if debug:
        # Draw detected contour on original for step 1 visualisation
        orig_vis = original.copy()
        if not fallback_used:
            cv2.polylines(orig_vis, [pts.astype(np.int32)], True, (0, 255, 0), 3)

        steps = {
            "step1_original":   _encode_step(orig_vis),
            "step2_grayscale":  _encode_step(gray),
            "step3_threshold":  _encode_step(thresh),
            "step4_morphology": _encode_step(morph),
            "step5_warp":       _encode_step(warped),
            "step6_clahe":      _encode_step(normalised),
            "corners_detected": not fallback_used,
            "fallback_used":    fallback_used,
        }
        return normalised, hsv, steps

    return normalised, hsv


def preprocess_for_ae(img_bgr, h, w):
    """Resize and normalise to [0,1] float32 for autoencoder input."""
    resized = cv2.resize(img_bgr, (w, h))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image> [--debug]")
        sys.exit(1)
    path  = sys.argv[1]
    debug = "--debug" in sys.argv
    result = preprocess(path, debug=debug)
    bgr = result[0]
    if bgr is not None:
        cv2.imwrite("preprocessed_output.jpg", bgr)
        print(f"Saved → preprocessed_output.jpg  {bgr.shape}")
    else:
        print("ERROR: Could not locate note boundary.")