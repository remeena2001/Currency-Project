"""
calibrate.py
------------
Compute per-model decision thresholds AFTER training.
Run this once after train.py completes.

threshold = mean_genuine_error + k × std_genuine_error

k=3.0  (default) → covers ~99.7% of genuine notes
k=2.5  → more aggressive (catches more fakes, slightly more false positives)
k=3.5  → more lenient  (fewer false positives, might miss some fakes)

Usage
-----
    python calibrate.py                    # k=3.0, all models
    python calibrate.py --k 2.5           # tighter thresholds
    python calibrate.py --k 3.5           # looser thresholds
    python calibrate.py --denom 1000      # one denomination only
    python calibrate.py --dataset /path   # custom dataset path
"""

import argparse
import json
import os

import cv2
import numpy as np

from config import (
    CALIBRATION_K, DATASET_DIR, DENOMINATIONS,
    MODELS_DIR, SIDES,
)
from autoencoder import (
    build_autoencoder, model_path, threshold_path,
    reconstruction_error, prepare_input,
)
from preprocessing import preprocess

THRESHOLDS_SUMMARY = os.path.join(MODELS_DIR, "thresholds.json")


def calibrate_one(denomination: int, side: str,
                  k: float, dataset_dir: str) -> dict:
    """
    Compute threshold for one (denomination, side) model.
    Returns a dict with threshold, mean, std, p95, p99.
    """
    mp = model_path(denomination, side)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Model not found: '{mp}'. Train it first.")

    # Load model
    model = build_autoencoder()
    model.load_weights(mp)

    # Load genuine images
    folder = os.path.join(dataset_dir, f"{denomination}_{side}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: '{folder}'")

    errors = []
    exts   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(exts):
            continue
        raw = cv2.imread(os.path.join(folder, fname))
        if raw is None:
            continue
        bgr, _ = preprocess(raw)
        if bgr is None:
            continue
        err = reconstruction_error(model, bgr)
        errors.append(err)

    if not errors:
        raise ValueError(f"No images processed in '{folder}'")

    errors    = np.array(errors, dtype=np.float64)
    mean_e    = float(np.mean(errors))
    std_e     = float(np.std(errors))
    threshold = float(mean_e + k * std_e)

    result = {
        "denomination": denomination,
        "side":         side,
        "k":            k,
        "threshold":    threshold,
        "mean_error":   mean_e,
        "std_error":    std_e,
        "p95":          float(np.percentile(errors, 95)),
        "p99":          float(np.percentile(errors, 99)),
        "n_images":     len(errors),
    }

    # Save per-model threshold
    tp = threshold_path(denomination, side)
    with open(tp, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=float, default=CALIBRATION_K,
                        help="Std multiplier (default 3.0)")
    parser.add_argument("--denom",   type=int,   default=None)
    parser.add_argument("--side",    type=str,   default=None, choices=["F","B"])
    parser.add_argument("--dataset", type=str,   default=DATASET_DIR)
    args = parser.parse_args()

    denoms = [args.denom] if args.denom else DENOMINATIONS
    sides  = [args.side]  if args.side  else SIDES

    print(f"\nCalibrating thresholds  (k = {args.k})\n")
    summary = {}

    for denom in denoms:
        for side in sides:
            key = f"{denom}_{side}"
            try:
                r = calibrate_one(denom, side, args.k, args.dataset)
                summary[key] = r
                print(f"  Rs.{denom} {side}  →  threshold={r['threshold']:.6f}  "
                      f"(mean={r['mean_error']:.6f}, std={r['std_error']:.6f}, "
                      f"p99={r['p99']:.6f},  n={r['n_images']})")
            except Exception as e:
                print(f"  [SKIP] {key} — {e}")

    # Save combined summary
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(THRESHOLDS_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Thresholds saved → {THRESHOLDS_SUMMARY}")
    print("\n  Tuning guide:")
    print("    Too many false positives?  →  python calibrate.py --k 3.5")
    print("    Want to catch more fakes?  →  python calibrate.py --k 2.5\n")


if __name__ == "__main__":
    main()
