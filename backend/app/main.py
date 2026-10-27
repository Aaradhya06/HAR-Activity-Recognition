"""
FastAPI Application — HAR Deep Vision Backend
=============================================
Provides REST endpoints for:
  - Health check
  - Model information
  - Video prediction
  - Frame-sequence prediction
  - Webcam frame streaming
  - Metrics retrieval
  - Class listing
  - Model reload
  - Training trigger
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import HARInferenceEngine, ModelState

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Global state ──────────────────────────────────────────────────────────────

_engine: HARInferenceEngine | None = None
_config: dict = {}
_training_in_progress: bool = False


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        logger.error("config.yaml not found at %s", config_path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup / shutdown."""
    global _engine, _config
    logger.info("HAR backend starting up...")
    _config = load_config()
    if _config:
        try:
            _engine = HARInferenceEngine(_config)
            logger.info("Inference engine initialized. State: %s", _engine.state.value)
        except Exception as e:
            logger.error("Failed to initialize inference engine: %s", e)
    yield
    logger.info("HAR backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HAR Deep Vision API",
    description="Human Activity Recognition using Deep Learning & Computer Vision",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_engine() -> HARInferenceEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    return _engine


def _validate_video_format(filename: str) -> None:
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format: {ext}. Allowed: {allowed}",
        )


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    engine = _engine
    return {
        "status": "ok",
        "model_state": engine.state.value if engine else "not_initialized",
        "device": str(engine.device) if engine else "unknown",
    }


@app.get("/api/model-info")
async def model_info():
    """Return model architecture and status information."""
    engine = _require_engine()
    info = engine.get_model_info()
    return info


@app.get("/api/classes")
async def get_classes():
    """Return list of supported activity classes."""
    engine = _require_engine()
    return {
        "class_names": engine.class_names,
        "num_classes": engine.num_classes,
        "model_state": engine.state.value,
    }


def _get_results_dir() -> Path:
    raw = _config.get("evaluation", {}).get("results_dir", "../results")
    p = Path(raw)
    if p.exists():
        return p
    clean = raw.replace("../", "")
    p_proj = Path(__file__).resolve().parent.parent.parent / clean
    if p_proj.exists():
        return p_proj
    p_backend = Path(__file__).resolve().parent.parent / clean
    if p_backend.exists():
        return p_backend
    return p_proj


@app.get("/api/metrics")
async def get_metrics():
    """Return latest evaluation metrics if available."""
    results_dir = _get_results_dir()
    metrics_path = results_dir / "metrics.json"

    if not metrics_path.exists():
        return {
            "available": False,
            "message": "No evaluation results found. Run evaluate.py first.",
        }

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["available"] = True
    return metrics


@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    use_pose: bool = True,
):
    """
    Predict activity from an uploaded video file.

    Returns structured prediction with confidence, top-k predictions,
    inference latency, and FPS.
    """
    engine = _require_engine()
    _validate_video_format(file.filename or "video.mp4")

    # Check file size
    max_mb = _config.get("api", {}).get("max_video_size_mb", 100)
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Maximum: {max_mb} MB.",
        )

    # Save to temp file
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = engine.predict_video(tmp_path, use_pose=use_pose)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return result


@app.post("/api/predict/frame-sequence")
async def predict_frame_sequence(
    files: list[UploadFile] = File(...),
    use_pose: bool = True,
):
    """
    Predict from an ordered sequence of uploaded image frames.
    Minimum: sequence_length frames required.
    """
    engine = _require_engine()
    import cv2
    import numpy as np

    seq_len = engine.inf_cfg.get("sequence_length", 16)
    if len(files) < 1:
        raise HTTPException(status_code=422, detail="At least 1 frame required.")

    frames_bgr = []
    for f in files:
        content = await f.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=422, detail=f"Cannot decode frame: {f.filename}")
        frames_bgr.append(frame)

    try:
        result = engine.predict_frame_sequence(frames_bgr, use_pose=use_pose)
    except Exception as e:
        logger.error("Frame sequence prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.post("/api/model/load")
async def reload_model():
    """Reload model checkpoint from disk (e.g. after training)."""
    engine = _require_engine()
    result = engine.reload_model()
    return result


@app.get("/api/inference-stats")
async def inference_stats():
    """Return inference timing statistics."""
    engine = _require_engine()
    return engine.get_inference_stats()


@app.get("/api/results/confusion-matrix")
async def get_confusion_matrix():
    """Serve confusion matrix image."""
    results_dir = _get_results_dir()
    cm_path = results_dir / "confusion_matrix.png"
    if not cm_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix not yet generated.")
    return FileResponse(str(cm_path), media_type="image/png")


@app.get("/api/results/training-curves")
async def get_training_curves():
    """Serve training curves image."""
    results_dir = _get_results_dir()
    curve_path = results_dir / "training_curves.png"
    if not curve_path.exists():
        raise HTTPException(status_code=404, detail="Training curves not yet generated.")
    return FileResponse(str(curve_path), media_type="image/png")


@app.get("/api/results/validation-curves")
async def get_validation_curves():
    """Serve validation curves image."""
    results_dir = _get_results_dir()
    curve_path = results_dir / "validation_curves.png"
    if not curve_path.exists():
        raise HTTPException(status_code=404, detail="Validation curves not yet generated.")
    return FileResponse(str(curve_path), media_type="image/png")


@app.get("/api/results/training-history")
async def get_training_history():
    """Return training history JSON."""
    results_dir = _get_results_dir()
    history_path = results_dir / "training_history.json"
    if not history_path.exists():
        return {"available": False, "message": "Training history not found."}
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    history["available"] = True
    return history


@app.get("/api/results/sample-video/{action}/{filename}")
async def get_sample_video(action: str, filename: str):
    """Serve a sample action video file for demonstration."""
    vid_path = Path(__file__).resolve().parent.parent.parent / "data" / "ucf101" / action / filename
    if not vid_path.exists():
        raise HTTPException(status_code=404, detail="Sample video not found.")
    return FileResponse(str(vid_path), media_type="video/mp4")


# ── Static SPA Serving (if frontend is built) ─────────────────────────────────
dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
public_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "public"

if (dist_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

# Serve public/ assets (icons, images) at root path
if public_dir.exists():
    app.mount("/public", StaticFiles(directory=str(public_dir)), name="public")


@app.get("/")
async def root():
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "HAR Deep Vision API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/icon.svg")
async def get_icon():
    """Serve the app icon (referenced by frontend dist/index.html)."""
    # Try dist first, then public
    for search_dir in [dist_dir, public_dir]:
        icon_path = search_dir / "icon.svg"
        if icon_path.exists():
            return FileResponse(str(icon_path), media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Icon not found")


@app.get("/{full_path:path}")
async def spa_catchall(full_path: str):
    """
    Catch-all route for SPA (React Router).
    Serves index.html for any non-API path so client-side routing works.
    This must be defined LAST so it doesn't shadow API routes.
    """
    # Don't catch API paths
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi"):
        raise HTTPException(status_code=404, detail=f"Not found: /{full_path}")
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Frontend not built. Run npm run build in frontend/.")
