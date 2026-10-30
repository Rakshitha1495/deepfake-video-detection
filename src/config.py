"""Shared paths and default settings for the ML pipeline.

Every path is resolved from the project root so the scripts work no matter
which folder you launch them from.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / ".cache"
FEATURE_CACHE_DIR = CACHE_DIR / "features"

CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"

# Pretrained weights are downloaded ONCE into this folder and reused afterwards.
TORCH_HOME = CACHE_DIR / "torch"
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
RESNET18_WEIGHTS_FILE = TORCH_HOME / "hub" / "checkpoints" / "resnet18-f37072fd.pth"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

CLASS_NAMES = ["REAL", "FAKE"]  # index 0 = REAL, index 1 = FAKE

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CONFIDENCE_THRESHOLD = 0.60  # probability threshold for decisive REAL/FAKE vs UNCERTAIN

ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.png"
PR_CURVE_PATH = ARTIFACTS_DIR / "precision_recall_curve.png"
LOSS_GRAPH_PATH = ARTIFACTS_DIR / "training_validation_loss.png"
ACCURACY_GRAPH_PATH = ARTIFACTS_DIR / "training_validation_accuracy.png"

# Defaults tuned for CPU execution on Windows 11
DEFAULT_TRAIN_CONFIG = {
    "num_frames": 8,            # frames sampled uniformly per video
    "hidden_size": 64,         # LSTM hidden size
    "num_layers": 1,            # LSTM layers
    "dropout": 0.35,            # dropout rate
    "bidirectional": True,      # capture forward & backward temporal cues
    "batch_size": 8,            # batch size for stable gradients on cached features
    "epochs": 25,               # epochs
    "learning_rate": 1e-3,      # AdamW learning rate
    "weight_decay": 1e-2,       # AdamW weight decay
    "seed": 42,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "freeze_backbone": True,
    "early_stopping_patience": 7,
    "confidence_threshold": 0.60,
    "max_videos_per_class": 50,
    "time_budget_minutes": 20,
}


def ensure_dirs() -> None:
    for d in (REAL_DIR, FAKE_DIR, MODELS_DIR, ARTIFACTS_DIR, FEATURE_CACHE_DIR, TORCH_HOME):
        d.mkdir(parents=True, exist_ok=True)

