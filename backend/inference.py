"""Holds the ONE model instance used by every request.

The checkpoint is loaded once at backend startup. Requests only decode their
own video and run a forward pass; nothing is retrained, re-downloaded or
re-created per request.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .config import settings

log = logging.getLogger("deepguard")


class ModelService:
    def __init__(self) -> None:
        self.predictor = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()  # one inference at a time: avoids CPU thread oversubscription

    @property
    def loaded(self) -> bool:
        return self.predictor is not None

    @property
    def checkpoint_exists(self) -> bool:
        return settings.checkpoint_path.exists()

    def load(self) -> None:
        if not self.checkpoint_exists:
            self.error = (f"Model checkpoint not found at {settings.checkpoint_path}. "
                          "Run the one-time training first:  python initialize.py")
            log.warning(self.error)
            return
        try:
            import torch
            from src.predict import Predictor
            if settings.torch_threads > 0:
                torch.set_num_threads(settings.torch_threads)
            self.predictor = Predictor(settings.checkpoint_path, settings.device)
            self.error = None
            log.info("Model loaded in %.2fs on %s", self.predictor.load_seconds, self.predictor.device)
        except Exception as e:  # keep the API up so the UI can show the problem
            self.predictor = None
            self.error = f"Failed to load model checkpoint: {e}"
            log.exception("Model load failed")

    def predict(self, path: Path) -> dict:
        if self.predictor is None:
            raise RuntimeError(self.error or "Model is not loaded.")
        with self._lock:
            return self.predictor.predict(path)

    @property
    def device(self) -> Optional[str]:
        return str(self.predictor.device) if self.predictor else None

    @property
    def load_seconds(self) -> Optional[float]:
        return round(self.predictor.load_seconds, 3) if self.predictor else None

    def checkpoint_meta(self) -> dict:
        if self.predictor:
            return self.predictor.checkpoint_meta
        return {}


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


model_service = ModelService()
