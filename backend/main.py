"""DeepGuard AI - FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.config import (
    ACCURACY_GRAPH_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PATH,
    LOSS_GRAPH_PATH,
    METRICS_PATH,
    PR_CURVE_PATH,
    ROC_CURVE_PATH,
)
from src.video_utils import VideoReadError, is_readable_video

from .config import settings
from .inference import model_service, read_json
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepguard")

CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    await run_in_threadpool(model_service.load)  # load ONCE, reuse for every request
    log.info("Backend ready in %.2fs (model loaded: %s)", time.perf_counter() - t0, model_service.loaded)
    yield


app = FastAPI(title="DeepGuard AI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check the backend console."})


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.loaded,
        checkpoint_exists=model_service.checkpoint_exists,
        device=model_service.device,
        model_load_seconds=model_service.load_seconds,
        message="Model ready." if model_service.loaded else (model_service.error or "Model not loaded."),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    meta = model_service.checkpoint_meta()
    cfg = meta.get("config")
    history = read_json(HISTORY_PATH)
    metrics = read_json(METRICS_PATH)
    if cfg is None and history:
        cfg = history.get("config")
    architecture = {
        "name": "CNN-LSTM",
        "backbone": "ResNet18 (ImageNet pretrained)",
        "backbone_frozen": cfg.get("freeze_backbone", True) if cfg else True,
        "feature_dim": 512,
        "lstm_hidden_size": cfg.get("hidden_size", 64) if cfg else 64,
        "lstm_layers": cfg.get("num_layers", 1) if cfg else 1,
        "bidirectional": cfg.get("bidirectional", True) if cfg else True,
        "classifier": "LayerNorm -> TemporalDiff -> BiLSTM -> Avg+MaxPool -> Classifier -> 2 Logits",
        "classes": {"0": "REAL", "1": "FAKE"},
        "input": "N sampled frames, center-cropped to 224x224, ImageNet-normalised",
    }
    return ModelInfoResponse(
        checkpoint_available=model_service.checkpoint_exists,
        model_loaded=model_service.loaded,
        device=model_service.device,
        architecture=architecture,
        num_frames=cfg.get("num_frames") if cfg else None,
        training_config=cfg,
        best_epoch=meta.get("best_epoch") or (history or {}).get("best_epoch"),
        best_val_f1=meta.get("best_val_f1"),
        dataset_split=meta.get("split_summary") or (history or {}).get("split_summary"),
        split_warnings=meta.get("split_warnings") or (history or {}).get("split_warnings") or [],
        training=history,
        evaluation=metrics,
        message="ok" if model_service.loaded else (model_service.error or "Model not loaded."),
    )


@app.get("/api/artifacts/confusion-matrix")
def confusion_matrix_png():
    if not CONFUSION_MATRIX_PATH.exists():
        raise HTTPException(404, "Confusion matrix not generated yet. Run: python -m src.evaluate")
    return FileResponse(CONFUSION_MATRIX_PATH, media_type="image/png")


@app.get("/api/artifacts/roc-curve")
def roc_curve_png():
    if not ROC_CURVE_PATH.exists():
        raise HTTPException(404, "ROC curve not generated yet. Run: python -m src.evaluate")
    return FileResponse(ROC_CURVE_PATH, media_type="image/png")


@app.get("/api/artifacts/pr-curve")
def pr_curve_png():
    if not PR_CURVE_PATH.exists():
        raise HTTPException(404, "Precision-Recall curve not generated yet. Run: python -m src.evaluate")
    return FileResponse(PR_CURVE_PATH, media_type="image/png")


@app.get("/api/artifacts/loss-graph")
def loss_graph_png():
    if not LOSS_GRAPH_PATH.exists():
        raise HTTPException(404, "Loss graph not generated yet. Run: python -m src.evaluate")
    return FileResponse(LOSS_GRAPH_PATH, media_type="image/png")


@app.get("/api/artifacts/accuracy-graph")
def accuracy_graph_png():
    if not ACCURACY_GRAPH_PATH.exists():
        raise HTTPException(404, "Accuracy graph not generated yet. Run: python -m src.evaluate")
    return FileResponse(ACCURACY_GRAPH_PATH, media_type="image/png")


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not model_service.loaded:
        raise HTTPException(503, model_service.error or "Model is not loaded. Run: python initialize.py")

    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in settings.allowed_extensions))
        raise HTTPException(400, f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + CHUNK:
        raise HTTPException(413, f"File is larger than the {settings.max_upload_mb} MB limit.")

    t0 = time.perf_counter()
    fd, tmp_name = tempfile.mkstemp(prefix="deepguard_", suffix=ext)
    tmp = Path(tmp_name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:  # stream to disk: never holds the whole video in RAM
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"File is larger than the {settings.max_upload_mb} MB limit.")
                out.write(chunk)
        await file.close()
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        upload_s = time.perf_counter() - t0

        readable, reason = await run_in_threadpool(is_readable_video, tmp)
        if not readable:
            raise HTTPException(422, f"Not a readable video: {reason}")

        try:
            result = await run_in_threadpool(model_service.predict, tmp)
        except VideoReadError as e:
            raise HTTPException(422, f"Could not decode the video: {e}")

        total = time.perf_counter() - t0
        log.info(
            "predict %s (%.1f MB): %s (real=%.3f, fake=%.3f) in %.2fs",
            filename,
            size / 1e6,
            result["prediction"],
            result["real_probability"],
            result["fake_probability"],
            total,
        )
        t = result["timing"]
        return PredictionResponse(
            filename=filename,
            file_size_bytes=size,
            prediction=result["prediction"],
            real_probability=result["real_probability"],
            fake_probability=result["fake_probability"],
            status="success",
            probabilities=result["probabilities"],
            processing_time_seconds=round(total, 3),
            frames_processed=result["frames_processed"],
            model_loaded=True,
            timing={
                "upload_seconds": round(upload_s, 4),
                "decode_seconds": t["decode_seconds"],
                "preprocess_seconds": t["preprocess_seconds"],
                "inference_seconds": t["inference_seconds"],
                "model_seconds": t["total_seconds"],
            },
            video=result["video"],
            diagnostic=result.get("diagnostic"),
            warning=result["warning"],
        )
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not delete temp file %s", tmp)
