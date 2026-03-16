# Sri Lankan Currency Detector

A web app that checks whether a Sri Lankan currency note is **real or fake** using AI.

Just upload a photo of the front and back of a note — the app tells you if it's **GENUINE**, **SUSPICIOUS**, or **UNCERTAIN**.

---

## What It Does

- Upload any photo of a note (any angle, any lighting)
- The app automatically cleans and straightens the image
- AI model checks if the note matches a genuine note
- Shows result with a confidence score
- Works for all 6 denominations — Rs. 20, 50, 100, 500, 1000 and 5000

---

## How the AI Works

The model is trained on **genuine notes only**.

It learns what a real note looks like. When you give it a fake note, the model cannot reconstruct it properly — that high error tells the system the note is suspicious.

No fake note images are needed for training.

---

## Requirements

- Python 3.9 or higher
- 150+ photos of genuine notes per denomination (front and back)

---

## Installation

```bash
# 1. Clone the project
git clone https://github.com/your-username/currency-detector.git
cd currency-detector

# 2. Install packages
pip install -r requirements.txt
```

---

## Dataset Folder Setup

Put your note images inside the `dataset/` folder like this:

```
dataset/
├── 20_F/       ← front photos of Rs.20
├── 20_B/       ← back  photos of Rs.20
├── 50_F/
├── 50_B/
├── 100_F/
├── 100_B/
├── 500_F/
├── 500_B/
├── 1000_F/
├── 1000_B/
├── 5000_F/
└── 5000_B/
```

---

## How to Run

Open `currency_detector.ipynb` in VS Code and run the cells one by one:

| Cell | What it does |
|------|-------------|
| Cell 0 | Install packages |
| Cell 1 | Setup |
| Cell 2 | Check your dataset |
| Cell 3 | Build denomination detector |
| Cell 4 | Train the AI models (~10–20 min) |
| Cell 5 | Set detection thresholds |
| Cell 6 | Test preprocessing |
| Cell 7 | Test the full pipeline |
| Cell 8 | Start the website |

After Cell 8 runs, open your browser and go to:

```
http://localhost:8000
```

> Next time you just need to run **Cell 1** then **Cell 8** to start the app.

---

## Using the Website

1. Upload the **front** photo of the note
2. Upload the **back** photo of the note
3. Click **Verify Note**
4. See the result — GENUINE ✅ / UNCERTAIN ⚠️ / SUSPICIOUS ❌

The website also shows you every preprocessing step so you can see exactly what the AI is looking at.

---

## Tech Stack

- **Python** — main language
- **TensorFlow & Keras** — AI model
- **OpenCV** — image processing
- **FastAPI** — web server
- **HTML & CSS** — website frontend
- **SQLite** — scan history

---

## License

MIT — free to use.
