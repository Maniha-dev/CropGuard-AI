"""CropGuard HTTP API + live web page.

Run:   uvicorn src.api.server:app --reload --port 8000
Open:  http://localhost:8000/        (live camera page)
Docs:  http://localhost:8000/docs
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.decision.spray_decision import REQUIRED_FRAMES, SprayDecision
from src.detection.detector import load_model
from src.detection.inference import detect_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "web" / "index.html"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_CONFIDENCE = 0.60
SESSION_TTL_SECONDS = 600
MAX_SESSIONS = 500
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]

_model_lock = threading.Lock()  # one inference at a time (the model is shared)
_sessions = {}  # session id -> [SprayDecision, last_seen]
_sessions_lock = threading.Lock()

def _decision_for(session_id):
    """Each browser tab gets its own 'N pests in a row' counter."""
    now = time.time()
    with _sessions_lock:
        for key in [k for k, (_, seen) in _sessions.items() if now - seen > SESSION_TTL_SECONDS]:
            del _sessions[key]
        if session_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(min(_sessions, key=lambda k: _sessions[k][1]))
        entry = _sessions.setdefault(session_id, [SprayDecision(), now])
        entry[1] = now
        return entry[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(require_trained=True)
    # Warm-up: the first inference is slow, so do it before the first user arrives
    app.state.model(np.zeros((64, 64, 3), np.uint8), verbose=False)
    yield


app = FastAPI(title="CropGuard-AI API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(404, "web/index.html is missing.")
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    model = app.state.model
    return {"status": "ok", "task": model.task, "classes": list(model.names.values())}


def _detection_to_dict(d):
    x1, y1, x2, y2 = d.box
    return {
        "class_name": d.class_name,
        "confidence": round(float(d.confidence), 4),
        "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
    }


def _run_detection(frame, confidence, session_id):
    with _model_lock:
        start = time.perf_counter()
        detections, highest = detect_frame(app.state.model, frame, confidence)
        elapsed_ms = (time.perf_counter() - start) * 1000

    engine = _decision_for(session_id)
    decision = engine.update(bool(detections), highest)
    height, width = frame.shape[:2]
    return {
        "image": {"width": width, "height": height},
        "confidence_threshold": confidence,
        "pest_detected": bool(detections),
        "highest_confidence": round(highest, 4),
        "detections": [_detection_to_dict(d) for d in detections],
        "decision": decision,  # "SPRAY" or "DON'T SPRAY"
        "confirmed_frames": engine.consecutive_detections,
        "required_frames": REQUIRED_FRAMES,
        "inference_ms": round(elapsed_ms, 1),
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
    session: str = Query("default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
):
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 10 MB.")

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode file as an image.")

    return await run_in_threadpool(_run_detection, frame, confidence, session)
