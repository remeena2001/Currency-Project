"""
api.py — FastAPI server.

    python api.py           # dev server with auto-reload
    uvicorn api:app --port 8000
"""
import os, traceback
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from database  import get_history, save_scan
from inference import analyse_bytes

app = FastAPI(title="Sri Lankan Currency Detector", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    if not os.path.exists(_HTML):
        return HTMLResponse("<h2>index.html not found — place it next to api.py</h2>",
                            status_code=404)
    return HTMLResponse(open(_HTML, encoding="utf-8").read())


@app.post("/analyse")
async def analyse(front: UploadFile = File(...), back: UploadFile = File(...),
                  denom: int = Query(default=None)):
    fb = await front.read()
    bb = await back.read()
    if not fb: return JSONResponse({"error":"Front image missing"}, 400)
    if not bb: return JSONResponse({"error":"Back image missing"},  400)
    try:
        result = analyse_bytes(fb, bb, denom_override=denom)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 422)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e),
                             "hint": "Run: python denomination_detector.py calibrate dataset/  then  python train.py  then  python calibrate.py"}, 503)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, 500)
    try: save_scan(result)
    except: pass
    return JSONResponse(result)


@app.get("/history")
async def history(limit: int = Query(20, ge=1, le=200)):
    return JSONResponse(get_history(limit))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
