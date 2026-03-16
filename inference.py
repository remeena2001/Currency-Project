"""
inference.py
------------
Currency verification using autoencoder reconstruction error only.
Preprocessing steps are returned as base64 images for the UI.

CLI usage
---------
    python inference.py front.jpg back.jpg
    python inference.py front.jpg back.jpg --denom 1000
    python inference.py front.jpg back.jpg --json
"""

import base64
import json

import cv2
import numpy as np

from config                import UNCERTAIN_LOWER, UNCERTAIN_UPPER
from preprocessing         import preprocess
from denomination_detector import detect_denomination
from autoencoder           import ae_confidence, get_model, reconstruction_error


def _encode(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def _pct(v: float) -> float:
    return round(float(np.clip(v, 0, 100)), 1)


def analyse_bytes(front_bytes: bytes, back_bytes: bytes,
                  denom_override: int = None) -> dict:
    """
    Full verification pipeline.
    1. Decode raw bytes
    2. Preprocess both sides  (warp + CLAHE)
    3. Detect denomination
    4. Run autoencoder on front + back
    5. Compute confidence and verdict
    Returns a dict matching the index.html UI contract.
    """

    # ── 1. Decode ─────────────────────────────────────────────────────────
    def decode(b: bytes) -> np.ndarray:
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Use JPEG, PNG, or BMP.")
        return img

    front_raw = decode(front_bytes)
    back_raw  = decode(back_bytes)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    front_bgr, front_hsv, front_dbg = preprocess(front_raw, debug=True)
    back_bgr,  back_hsv,  back_dbg  = preprocess(back_raw,  debug=True)

    if front_bgr is None or back_bgr is None:
        raise ValueError(
            "Could not locate the note boundary in one or both images. "
            "Ensure the note is clearly visible against a plain background."
        )

    # ── 3. Denomination detection ─────────────────────────────────────────
    if denom_override:
        denom, denom_conf = denom_override, 1.0
    else:
        denom, denom_conf, _ = detect_denomination(front_bgr)
        if denom == 0:
            denom = 1000

    # ── 4. Autoencoder reconstruction error ───────────────────────────────
    try:
        conf_f     = ae_confidence(denom, "F", front_bgr)
        conf_b     = ae_confidence(denom, "B", back_bgr)
        model_conf = (conf_f + conf_b) / 2.0

        model_f, thresh_f = get_model(denom, "F")
        model_b, thresh_b = get_model(denom, "B")
        err_f = reconstruction_error(model_f, front_bgr)
        err_b = reconstruction_error(model_b, back_bgr)

    except FileNotFoundError:
        conf_f = conf_b = model_conf = 0.5
        err_f  = err_b  = 0.0
        thresh_f = thresh_b = 0.02

    # ── 5. Verdict — autoencoder only ─────────────────────────────────────
    if model_conf >= UNCERTAIN_UPPER:
        verdict = "GENUINE"
    elif model_conf <= UNCERTAIN_LOWER:
        verdict = "SUSPICIOUS"
    else:
        verdict = "UNCERTAIN"

    # ── 6. Build preprocessing image dicts for UI ─────────────────────────
    # steps_dict contains individual step images keyed step1–step6
    def prep_steps(raw_bgr, final_bgr, steps_dict):
        return {
            "original":         _encode(raw_bgr),
            "final":            _encode(final_bgr),
            "step1_original":   steps_dict.get("step1_original",   ""),
            "step2_grayscale":  steps_dict.get("step2_grayscale",  ""),
            "step3_threshold":  steps_dict.get("step3_threshold",  ""),
            "step4_morphology": steps_dict.get("step4_morphology", ""),
            "step5_warp":       steps_dict.get("step5_warp",       ""),
            "step6_clahe":      steps_dict.get("step6_clahe",      ""),
        }

    def prep_info(raw_bgr, final_bgr, steps_dict):
        h, w = raw_bgr.shape[:2]
        return {
            "was_rotated":      h > w,
            "corners_detected": not steps_dict.get("fallback_used", False),
            "final_size":       f"{final_bgr.shape[1]}x{final_bgr.shape[0]}",
            "method":           "perspective_warp" if not steps_dict.get("fallback_used") else "resize",
        }

    # ── 7. Assemble response ──────────────────────────────────────────────
    return {
        "verdict":               verdict,
        "denomination":          f"Rs. {denom}",
        "denom_conf":            round(denom_conf * 100, 1),
        "model_conf":            _pct(model_conf * 100),
        "front_conf":            _pct(conf_f * 100),
        "back_conf":             _pct(conf_b * 100),
        "front_error":           round(float(err_f), 6),
        "back_error":            round(float(err_b), 6),
        "front_thresh":          round(float(thresh_f), 6),
        "back_thresh":           round(float(thresh_b), 6),
        "preprocess_front":      prep_steps(front_raw, front_bgr, front_dbg),
        "preprocess_back":       prep_steps(back_raw,  back_bgr,  back_dbg),
        "preprocess_info_front": prep_info(front_raw, front_bgr, front_dbg),
        "preprocess_info_back":  prep_info(back_raw,  back_bgr,  back_dbg),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("front")
    parser.add_argument("back")
    parser.add_argument("--denom", type=int, default=None)
    parser.add_argument("--json",  action="store_true")
    args = parser.parse_args()

    with open(args.front, "rb") as f: fb = f.read()
    with open(args.back,  "rb") as f: bb = f.read()

    result = analyse_bytes(fb, bb, denom_override=args.denom)

    if args.json:
        out = {k: v for k, v in result.items() if "preprocess" not in k}
        print(json.dumps(out, indent=2))
    else:
        icon = {"GENUINE":"✅","UNCERTAIN":"⚠️","SUSPICIOUS":"❌"}.get(result["verdict"],"❓")
        print(f"\n  {icon}  {result['verdict']}")
        print(f"  Denomination : {result['denomination']}  ({result['denom_conf']}% confident)")
        print(f"  Model conf.  : {result['model_conf']}%")
        print(f"  Front score  : {result['front_conf']}%  (err={result['front_error']:.6f}  thresh={result['front_thresh']:.6f})")
        print(f"  Back  score  : {result['back_conf']}%  (err={result['back_error']:.6f}  thresh={result['back_thresh']:.6f})")
