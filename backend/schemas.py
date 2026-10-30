"""Pydantic response models (also documented automatically at /docs)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_exists: bool
    device: Optional[str]
    model_load_seconds: Optional[float]
    message: str


class Probabilities(BaseModel):
    real: float
    fake: float


class Timing(BaseModel):
    upload_seconds: float
    decode_seconds: float
    preprocess_seconds: float
    inference_seconds: float
    model_seconds: float


class VideoInfo(BaseModel):
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    decode_method: Optional[str] = None
    sampled_indices: list[int] = []


class PredictionResponse(BaseModel):
    filename: str
    file_size_bytes: int
    prediction: str
    real_probability: float
    fake_probability: float
    status: str = "success"
    probabilities: Probabilities
    processing_time_seconds: float
    frames_processed: int
    model_loaded: bool
    timing: Timing
    video: VideoInfo
    diagnostic: Optional[dict[str, Any]] = None
    warning: str



class ModelInfoResponse(BaseModel):
    checkpoint_available: bool
    model_loaded: bool
    device: Optional[str]
    architecture: dict[str, Any]
    num_frames: Optional[int]
    training_config: Optional[dict[str, Any]]
    best_epoch: Optional[int]
    best_val_f1: Optional[float]
    dataset_split: Optional[dict[str, Any]]
    split_warnings: list[str]
    training: Optional[dict[str, Any]]
    evaluation: Optional[dict[str, Any]]
    message: str


class ErrorResponse(BaseModel):
    detail: str
