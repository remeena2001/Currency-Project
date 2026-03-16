"""
config.py
---------
All tunable parameters in one place.
Edit this file — nothing else needs changing for basic tuning.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DB_PATH     = os.path.join(BASE_DIR, "history.db")

# ── Denominations ─────────────────────────────────────────────────────────────
DENOMINATIONS = [20, 50, 100, 500, 1000, 5000]
SIDES         = ["F", "B"]   # F = front, B = back

# ── Preprocessing ─────────────────────────────────────────────────────────────
WARP_W = 1024   # output width  (px)
WARP_H = 512    # output height (px)

# ── Denomination detection ────────────────────────────────────────────────────
# 2D Hue+Saturation histogram comparison (Bhattacharyya distance)
HIST_H_BINS = 36    # hue bins   (0–179 → 36 bins of 5°)
HIST_S_BINS = 32    # sat bins   (0–255 → 32 bins)
DENOM_UNKNOWN_THRESHOLD = 0.45   # Bhattacharyya dist above this → "unknown"

# ── Security feature ROIs ─────────────────────────────────────────────────────
# Relative fractions of (WARP_W, WARP_H) — (x1, y1, x2, y2)
DEFAULT_ROIS = {
    "security_thread":     (0.18, 0.00, 0.26, 1.00),
    "serial_number_tl":    (0.00, 0.00, 0.36, 0.28),
    "serial_number_br":    (0.64, 0.72, 1.00, 1.00),
    "watermark":           (0.04, 0.20, 0.28, 0.80),
    "latent_image":        (0.04, 0.48, 0.30, 0.92),
    "identification_mark": (0.72, 0.28, 0.96, 0.72),
}
DENOMINATION_ROIS = {}   # per-denomination overrides if needed

# ── 6-Stage intensity pipeline ────────────────────────────────────────────────
INTENSITY_THRESHOLD = 20.0   # % active edge pixels — >= this → PASS
CANNY_LOW           = 50
CANNY_HIGH          = 150

# ── Autoencoder ───────────────────────────────────────────────────────────────
AE_INPUT_H    = 64
AE_INPUT_W    = 128
AE_LATENT_DIM = 128
AE_EPOCHS     = 80
AE_BATCH_SIZE = 8
AE_LR         = 0.001
AE_PATIENCE   = 15          # early stopping patience

# ── Region weights for reconstruction error ───────────────────────────────────
# Zones with security features are weighted more heavily.
# Values are multipliers (1 = normal, 3 = 3× penalty for errors in that zone).
WEIGHT_SECURITY_STRIP = 3.0   # left strip  (watermark / security thread)
WEIGHT_SERIAL_STRIP   = 3.0   # right strip (serial number / colour-shift ink)
WEIGHT_CENTRAL_BAND   = 3.0   # centre band (latent image / microprinting)
WEIGHT_PLAIN          = 1.0   # plain background areas

# ── Calibration ───────────────────────────────────────────────────────────────
# threshold = mean_genuine_error + K × std_genuine_error
# k=3.0  covers ~99.7% of genuine notes (few false positives)
# k=2.5  more aggressive (catches more fakes, slightly more false positives)
CALIBRATION_K = 3.0

# ── Final verdict ─────────────────────────────────────────────────────────────
UNCERTAIN_LOWER = 0.35
UNCERTAIN_UPPER = 0.65
