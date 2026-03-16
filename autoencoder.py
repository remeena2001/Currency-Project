"""
autoencoder.py
--------------
Lightweight Convolutional Autoencoder with region-weighted reconstruction error.

Region-weighted error
---------------------
Security feature zones (watermark strip, serial strip, central band)
are weighted 3× higher than plain background areas.
This means the model penalises errors in security feature regions much more
heavily, making it more sensitive to forgeries in those areas.

Weight map layout (64×128):
  ┌──────────────────────────────────────────────────────┐
  │plain │ watermark/    │  central  │  serial/      │plain│
  │      │ sec. thread   │  latent   │  colour-shift │     │
  │  1×  │     3×        │    3×     │      3×       │ 1×  │
  └──────────────────────────────────────────────────────┘
"""

import json
import os

import cv2
import numpy as np

from config import (
    AE_INPUT_H, AE_INPUT_W, AE_LATENT_DIM, MODELS_DIR,
    WEIGHT_SECURITY_STRIP, WEIGHT_SERIAL_STRIP,
    WEIGHT_CENTRAL_BAND, WEIGHT_PLAIN,
)


# ── Weight map ────────────────────────────────────────────────────────────────

def build_weight_map(h: int = AE_INPUT_H, w: int = AE_INPUT_W) -> np.ndarray:
    """
    Build a (H, W, 1) float32 weight map.
    High-weight zones correspond to security feature regions.
    """
    wmap = np.full((h, w), WEIGHT_PLAIN, dtype=np.float32)

    # Left strip — watermark / windowed security thread  (x: 0–20%)
    x1 = int(w * 0.00); x2 = int(w * 0.20)
    wmap[:, x1:x2] = WEIGHT_SECURITY_STRIP

    # Right strip — serial number / colour-shift ink  (x: 80–100%)
    x1 = int(w * 0.80); x2 = w
    wmap[:, x1:x2] = WEIGHT_SERIAL_STRIP

    # Central band — latent image / microprinting  (x: 40–60%)
    x1 = int(w * 0.40); x2 = int(w * 0.60)
    wmap[:, x1:x2] = WEIGHT_CENTRAL_BAND

    return wmap[..., np.newaxis]   # (H, W, 1) for broadcasting with (H, W, 3)


# ── Architecture ──────────────────────────────────────────────────────────────

def build_autoencoder():
    """
    Build and compile a lightweight convolutional autoencoder.
    Input/output: (AE_INPUT_H, AE_INPUT_W, 3)  float32 [0,1].
    Uses region-weighted MSE loss.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    import tensorflow.keras.backend as K

    weight_map_np = build_weight_map()   # (H, W, 1) numpy

    # ── Custom weighted loss ─────────────────────────────────────────────────
    weight_tensor = tf.constant(weight_map_np, dtype=tf.float32)

    def weighted_mse(y_true, y_pred):
        sq_err     = K.square(y_pred - y_true)            # (B, H, W, 3)
        weighted   = sq_err * weight_tensor               # broadcast over batch+channels
        return K.mean(weighted)

    # ── Encoder ──────────────────────────────────────────────────────────────
    inp = layers.Input(shape=(AE_INPUT_H, AE_INPUT_W, 3), name="input")
    x   = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x   = layers.MaxPooling2D(2)(x)

    h_bot = AE_INPUT_H // 8
    w_bot = AE_INPUT_W // 8

    x   = layers.Flatten()(x)
    x   = layers.Dense(AE_LATENT_DIM, activation="relu", name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x   = layers.Dense(h_bot * w_bot * 64, activation="relu")(x)
    x   = layers.Reshape((h_bot, w_bot, 64))(x)
    x   = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x   = layers.UpSampling2D(2)(x)
    x   = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x   = layers.UpSampling2D(2)(x)
    x   = layers.Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
    x   = layers.UpSampling2D(2)(x)
    out = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same", name="output")(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=weighted_mse)
    return model


# ── Path helpers ──────────────────────────────────────────────────────────────

def model_path(denomination: int, side: str) -> str:
    return os.path.join(MODELS_DIR, f"ae_{denomination}_{side}.keras")

def threshold_path(denomination: int, side: str) -> str:
    return os.path.join(MODELS_DIR, f"ae_{denomination}_{side}_thresh.json")


# ── Image prep ────────────────────────────────────────────────────────────────

def prepare_input(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to AE input, convert BGR→RGB, normalise to [0,1]."""
    resized = cv2.resize(img_bgr, (AE_INPUT_W, AE_INPUT_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ── Reconstruction error ──────────────────────────────────────────────────────

def reconstruction_error(model, img_bgr: np.ndarray) -> float:
    """
    Compute region-weighted mean-squared reconstruction error for one image.
    Lower = more like a genuine note. Higher = suspicious.
    """
    x      = prepare_input(img_bgr)[np.newaxis, ...]   # (1, H, W, 3)
    x_hat  = model.predict(x, verbose=0)                # (1, H, W, 3)
    wmap   = build_weight_map()                         # (H, W, 1)
    sq_err = (x[0] - x_hat[0]) ** 2                    # (H, W, 3)
    return float(np.mean(sq_err * wmap))


# ── Model cache + loader ──────────────────────────────────────────────────────

_cache: dict = {}

def get_model(denomination: int, side: str):
    """Return (model, threshold) — loaded once and cached in memory."""
    key = (denomination, side)
    if key not in _cache:
        import tensorflow as tf
        mp = model_path(denomination, side)
        tp = threshold_path(denomination, side)
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"Model not found: '{mp}'. Run python train.py first."
            )
        # Rebuild with custom loss so Keras can load it
        model = build_autoencoder()
        model.load_weights(mp.replace(".keras", ".weights.h5")
                           if mp.endswith(".keras")
                           and not os.path.exists(mp)
                           else mp)
        threshold = 0.02
        if os.path.exists(tp):
            with open(tp) as f:
                threshold = json.load(f).get("threshold", threshold)
        _cache[key] = (model, threshold)
    return _cache[key]


def ae_confidence(denomination: int, side: str, img_bgr: np.ndarray) -> float:
    """
    Returns confidence score in [0, 1].
    1.0 = very low reconstruction error = almost certainly genuine.
    0.0 = error far above threshold = suspicious.
    """
    model, threshold = get_model(denomination, side)
    err   = reconstruction_error(model, img_bgr)
    score = max(0.0, 1.0 - err / (threshold * 2.0))
    return round(float(score), 4)
