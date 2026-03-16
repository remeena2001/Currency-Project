"""
denomination_detector.py
------------------------
Denomination detection using 2D Hue+Saturation histogram comparison.
NO machine learning training required — just builds reference histograms
from genuine images and compares at runtime using Bhattacharyya distance.

Usage (CLI)
-----------
    python denomination_detector.py calibrate dataset/    # build references
    python denomination_detector.py detect    my_note.jpg # identify a note
"""

import json
import os
import cv2
import numpy as np

from config import (
    DENOMINATIONS, SIDES, DATASET_DIR, MODELS_DIR,
    HIST_H_BINS, HIST_S_BINS, DENOM_UNKNOWN_THRESHOLD,
)

HIST_PATH = os.path.join(MODELS_DIR, "denomination_histograms.json")


# ── Histogram computation ─────────────────────────────────────────────────────

def _compute_hist(bgr: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 2D Hue+Saturation histogram from a BGR image.
    Returns a flat float32 array of length HIST_H_BINS × HIST_S_BINS.
    """
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [HIST_H_BINS, HIST_S_BINS],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Bhattacharyya distance between two normalised histograms.
    0.0 = identical,  higher = more different.
    """
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(dataset_dir: str = DATASET_DIR):
    """
    Build mean reference histograms for each denomination by averaging
    histograms from all genuine front images.

    Saves to models/denomination_histograms.json.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    references = {}

    for denom in DENOMINATIONS:
        # Use front images only for denomination detection
        folder = os.path.join(dataset_dir, f"{denom}_F")
        if not os.path.isdir(folder):
            print(f"  [WARN] {folder} not found — skipping Rs.{denom}")
            continue

        hists = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            hists.append(_compute_hist(img))

        if not hists:
            print(f"  [WARN] No images for Rs.{denom} — skipping")
            continue

        mean_hist = np.mean(np.stack(hists, axis=0), axis=0)
        cv2.normalize(mean_hist, mean_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        references[str(denom)] = mean_hist.flatten().tolist()
        print(f"  Rs.{denom:<5} → {len(hists)} images averaged")

    with open(HIST_PATH, "w") as f:
        json.dump(references, f)
    print(f"\n  Saved → {HIST_PATH}")
    return references


# ── Inference ─────────────────────────────────────────────────────────────────

_refs_cache = None

def _load_references() -> dict:
    global _refs_cache
    if _refs_cache is not None:
        return _refs_cache
    if not os.path.exists(HIST_PATH):
        raise FileNotFoundError(
            f"Denomination histograms not found at '{HIST_PATH}'.\n"
            "Run:  python denomination_detector.py calibrate dataset/"
        )
    with open(HIST_PATH) as f:
        raw = json.load(f)
    shape         = (HIST_H_BINS, HIST_S_BINS)
    _refs_cache   = {
        int(k): np.array(v, dtype=np.float32).reshape(shape)
        for k, v in raw.items()
    }
    return _refs_cache


def detect_denomination(img_bgr: np.ndarray) -> tuple:
    """
    Identify denomination using Bhattacharyya distance to reference histograms.

    Returns
    -------
    (denomination: int, confidence: float, distances: dict)
        denomination — best match (e.g. 1000), or 0 if unknown
        confidence   — 1 − best_distance  (higher = more confident)
        distances    — {denom: distance} for all denominations
    """
    refs   = _load_references()
    query  = _compute_hist(img_bgr)
    dists  = {d: _bhattacharyya(query, h) for d, h in refs.items()}
    best   = min(dists, key=dists.get)
    dist   = dists[best]

    if dist > DENOM_UNKNOWN_THRESHOLD:
        return 0, 0.0, dists                     # unknown

    confidence = round(float(1.0 - dist), 4)
    return best, confidence, dists


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python denomination_detector.py calibrate <dataset_dir>")
        print("  python denomination_detector.py detect    <image_path>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "calibrate":
        print(f"Calibrating from: {sys.argv[2]}")
        calibrate(sys.argv[2])

    elif cmd == "detect":
        img = cv2.imread(sys.argv[2])
        if img is None:
            print("Cannot load image.")
            sys.exit(1)
        denom, conf, dists = detect_denomination(img)
        if denom == 0:
            print("Result: UNKNOWN denomination")
        else:
            print(f"Result: Rs.{denom}  (confidence: {conf:.3f})")
        print("\nAll distances:")
        for d, dist in sorted(dists.items()):
            marker = " ← best" if d == denom else ""
            print(f"  Rs.{d:<5}  {dist:.4f}{marker}")
