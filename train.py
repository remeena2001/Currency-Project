"""
train.py
--------
Train all 12 autoencoders (6 denominations × 2 sides).

Usage
-----
    python train.py                        # train all 12 models
    python train.py --denom 1000           # both sides of Rs.1000
    python train.py --denom 1000 --side F  # front only
    python train.py --epochs 100           # override epoch count
    python train.py --dataset /path/to/dataset
"""

import argparse
import os

import cv2
import numpy as np

from config import (
    AE_EPOCHS, AE_BATCH_SIZE, AE_PATIENCE,
    DATASET_DIR, DENOMINATIONS, MODELS_DIR, SIDES,
)
from autoencoder import build_autoencoder, model_path, prepare_input


# ── Data loading + augmentation ───────────────────────────────────────────────

def load_images(denomination: int, side: str,
                dataset_dir: str = DATASET_DIR) -> np.ndarray:
    """
    Load genuine images, run full preprocessing, resize to AE input.
    Returns float32 array (N, H, W, 3) in [0, 1].
    """
    from preprocessing import preprocess

    folder = os.path.join(dataset_dir, f"{denomination}_{side}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: '{folder}'")

    imgs, skipped = [], 0
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(exts):
            continue
        raw = cv2.imread(os.path.join(folder, fname))
        if raw is None:
            skipped += 1
            continue
        bgr, _ = preprocess(raw)
        if bgr is None:
            skipped += 1
            continue
        imgs.append(prepare_input(bgr))

    if not imgs:
        raise ValueError(f"No images found in '{folder}'")
    if skipped:
        print(f"  [WARN] Skipped {skipped} unreadable images")
    return np.array(imgs, dtype=np.float32)


def augment(X: np.ndarray) -> np.ndarray:
    """4× augmentation: original + h-flip + v-flip + brightness jitter."""
    aug = []
    for img in X:
        aug.append(img)
        aug.append(img[:, ::-1, :])
        aug.append(img[::-1, :, :])
        aug.append(np.clip(img * np.random.uniform(0.90, 1.10), 0.0, 1.0))
    arr = np.array(aug, dtype=np.float32)
    np.random.shuffle(arr)
    return arr


# ── Single model training ─────────────────────────────────────────────────────

def train_one(denomination: int, side: str, epochs: int,
              dataset_dir: str = DATASET_DIR):
    import tensorflow as tf

    print(f"\n{'='*52}")
    print(f"  Rs.{denomination}  {side}  —  training")
    print(f"{'='*52}")

    X       = augment(load_images(denomination, side, dataset_dir))
    n       = int(len(X) * 0.8)
    X_tr, X_val = X[:n], X[n:]
    print(f"  Train: {len(X_tr)}   Val: {len(X_val)}")

    model = build_autoencoder()

    os.makedirs(MODELS_DIR, exist_ok=True)
    mp = model_path(denomination, side)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=AE_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            mp, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    model.fit(
        X_tr, X_tr,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=AE_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload best weights
    model.load_weights(mp) if mp.endswith(".keras") else None
    print(f"  Saved → {mp}")
    return model, X_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--denom",   type=int, default=None)
    parser.add_argument("--side",    type=str, default=None, choices=["F","B"])
    parser.add_argument("--epochs",  type=int, default=AE_EPOCHS)
    parser.add_argument("--dataset", type=str, default=DATASET_DIR)
    args = parser.parse_args()

    denoms = [args.denom] if args.denom else DENOMINATIONS
    sides  = [args.side]  if args.side  else SIDES

    success, failed = [], []
    for denom in denoms:
        for side in sides:
            try:
                train_one(denom, side, args.epochs, args.dataset)
                success.append(f"Rs.{denom} {side}")
            except Exception as e:
                print(f"  [SKIP] Rs.{denom} {side} — {e}")
                failed.append(f"Rs.{denom} {side}")

    print(f"\n{'='*52}")
    print(f"  ✅ Trained : {success}")
    if failed:
        print(f"  ❌ Skipped : {failed}")
    print(f"\n  ➡  Now run: python calibrate.py")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
