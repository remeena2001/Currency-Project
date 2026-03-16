# Sri Lankan Currency Authenticity Detector

An AI-powered web application that detects counterfeit Sri Lankan currency notes using **Convolutional Autoencoder anomaly detection**. Upload the front and back photos of a note — the system automatically preprocesses the image, identifies the denomination, and returns a **GENUINE / UNCERTAIN / SUSPICIOUS** verdict.

> Trained on genuine notes only. No fake note images required for training.

---

## How It Works

The core idea is **reconstruction error anomaly detection**:

1. Each autoencoder is trained exclusively on genuine notes — it learns exactly what a real note looks like
2. When a genuine note is passed, the model reconstructs it accurately → **low error → GENUINE**
3. When a fake note is passed, the model cannot reconstruct unfamiliar patterns → **high error → SUSPICIOUS**

Security feature zones (watermark, serial number, security thread) are given **3× higher weight** in the loss function, making the model more sensitive to forgeries in those critical areas.

---

## Features

- ✅ Supports all 6 Sri Lankan denominations — Rs. 20, 50, 100, 500, 1000, 5000
- ✅ Analyses both front and back of each note — 12 models total
- ✅ Auto-detects denomination using 2D Hue+Saturation histogram comparison
- ✅ 6-step preprocessing pipeline handles any angle, lighting, or camera
- ✅ Step-by-step preprocessing visualisation in the web UI
- ✅ Confidence scores with raw reconstruction error vs threshold breakdown
- ✅ Scan history stored in SQLite
- ✅ Runs fully offline — no internet connection required

---

## Project Structure

```
currency_detector/
│
├── config.py                  ← All settings (paths, model params, thresholds)
├── preprocessing.py           ← 6-step image pipeline (warp + CLAHE)
├── denomination_detector.py   ← 2D histogram + Bhattacharyya denomination ID
├── autoencoder.py             ← CNN autoencoder + region-weighted loss
├── train.py                   ← Train all 12 models
├── calibrate.py               ← Compute decision thresholds after training
├── inference.py               ← End-to-end verification pipeline
├── database.py                ← SQLite scan history
├── api.py                     ← FastAPI web server
├── index.html                 ← Web UI
├── currency_detector.ipynb    ← Jupyter notebook (train + run everything)
├── requirements.txt
│
├── dataset/                   ← Your images go here
│   ├── 20_F/    20_B/
│   ├── 50_F/    50_B/
│   ├── 100_F/   100_B/
│   ├── 500_F/   500_B/
│   ├── 1000_F/  1000_B/
│   └── 5000_F/  5000_B/
│
└── models/                    ← Auto-created after training
    ├── denomination_histograms.json
    ├── thresholds.json
    ├── ae_1000_F.keras
    ├── ae_1000_F_thresh.json
    └── ...
```

---

## Prerequisites

- Python 3.9 or higher
- At least **150+ genuine note images per denomination per side** (150 front + 150 back for each denomination)

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/currency-detector.git
cd currency-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place your genuine note images into the `dataset/` folder following this naming convention:

```
dataset/
├── 1000_F/   ← front images of Rs.1000 notes
├── 1000_B/   ← back  images of Rs.1000 notes
└── ...
```

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.tiff`

---

## Running with Jupyter Notebook (Recommended)

Open `currency_detector.ipynb` in VS Code and run cells in order:

| Cell | Action | Time |
|------|--------|------|
| 0 | Install packages | ~2 min (once only) |
| 1 | Imports & setup | instant |
| 2 | Check dataset folders | instant |
| 3 | Calibrate denomination detector | ~30 sec |
| 4 | Train 12 autoencoders | ~10–20 min |
| 5 | Calibrate thresholds | ~2 min |
| 6 | Visual test — preprocessing | instant |
| 7 | Full inference test | instant |
| 8 | Start server → localhost:8000 | instant |

> After the first run, only re-run **Cell 1** and **Cell 8** to start the server.

---

## Running from Terminal

```bash
# Step 1 — Build denomination reference histograms
python denomination_detector.py calibrate dataset/

# Step 2 — Train all 12 autoencoders
python train.py

# Step 3 — Calibrate decision thresholds
python calibrate.py

# Step 4 — Start the web server
python api.py
```

Open your browser at **http://localhost:8000**

---

## Threshold Tuning

After training you can retune sensitivity **without retraining** by adjusting `k`:

```bash
python calibrate.py --k 3.0   # default — covers 99.7% of genuine notes
python calibrate.py --k 2.5   # tighter — catches more fakes
python calibrate.py --k 3.5   # looser  — fewer false positives
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/analyse` | Verify a note (multipart: front + back image) |
| `GET` | `/history` | Last 20 scan records |
| `GET` | `/health` | Server status |

### Example — POST /analyse

**Request** — `multipart/form-data`
- `front` — front side image file
- `back` — back side image file
- `denom` *(optional)* — override denomination (e.g. `1000`)

**Response**
```json
{
  "verdict": "GENUINE",
  "denomination": "Rs. 1000",
  "denom_conf": 94.2,
  "model_conf": 87.5,
  "front_conf": 91.0,
  "back_conf": 84.0,
  "front_error": 0.003421,
  "back_error": 0.004812,
  "front_thresh": 0.012500,
  "back_thresh": 0.013100,
  "preprocess_front": { "original": "...", "final": "...", "step1": "...", ... },
  "preprocess_back":  { ... }
}
```

---

## Preprocessing Pipeline

Every image — regardless of how it was photographed — goes through 6 automatic steps:

```
Input photo (any angle, any size, any lighting)
    │
    ▼  Step 1 — Grayscale conversion
    ▼  Step 2 — Gaussian blur (noise removal)
    ▼  Step 3 — Adaptive threshold (uneven lighting handled)
    ▼  Step 4 — Morphological operations (boundary strengthened)
    ▼  Step 5 — Perspective warp → flat 1024×512 rectangle
    ▼  Step 6 — CLAHE brightness normalisation (HSV channel)
    │
    ▼  Clean 1024×512 image → Autoencoder
```

The web UI shows each step as a separate image so you can inspect exactly what the model receives.

---

## Model Architecture

```
Input (64×128×3)
    Conv2D(16) → MaxPool
    Conv2D(32) → MaxPool        Encoder
    Conv2D(64) → MaxPool
    Flatten → Dense(128)  ←── Bottleneck
    Dense → Reshape
    ConvTranspose(64) → UpSample
    ConvTranspose(32) → UpSample    Decoder
    ConvTranspose(16) → UpSample
    ConvTranspose(3, sigmoid)
Output (64×128×3)
```

**Region-weighted loss map:**

```
┌─────────────────────────────────────────────────────┐
│  1×  │  Watermark / Security thread  │  1×  │  Serial  │  1×  │
│      │           3×                  │      │   3×     │      │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow & Keras |
| Image Processing | OpenCV |
| Backend | FastAPI |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Notebook | Jupyter |
| Language | Python 3.9+ |

---

## License

MIT License — free to use and modify.
