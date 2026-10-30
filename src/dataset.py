"""PyTorch datasets plus a one-time feature cache.

Speed trick: while the ResNet18 backbone is frozen, its output for a given
video never changes. We therefore run every training video through ResNet18
exactly ONCE, store the (T, 512) features in .cache/features, and train the
LSTM on those cached features. Every epoch after that costs milliseconds per
video instead of seconds, and re-running training skips decoding entirely.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CACHE_DIR, FEATURE_CACHE_DIR
from .video_utils import VideoReadError, preprocess_frames, read_video_frames

FRAME_CACHE_DIR = CACHE_DIR / "frames"
_FEATURE_VERSION = "resnet18-imagenet1k-v1-centercrop224"


def _cache_key(path: str, num_frames: int, tag: str) -> str:
    p = Path(path)
    st = p.stat()
    raw = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{num_frames}|{tag}"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_frames_cached(path: str, num_frames: int) -> np.ndarray:
    """uint8 frames for one video, decoded once and then read from disk."""
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = FRAME_CACHE_DIR / f"{_cache_key(path, num_frames, 'frames')}.npy"
    if f.exists():
        return np.load(f)
    frames, _ = read_video_frames(path, num_frames)
    np.save(f, frames)
    return frames


@torch.inference_mode()
def build_feature_cache(items: list[dict], model, num_frames: int, device, log=print) -> dict:
    """Compute (or load) ResNet18 features for every item.

    Returns {"features": {path: np.ndarray}, "failed": [...], "timing": {...}}.
    Corrupted / unreadable videos are skipped and reported, never silently dropped.
    """
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.backbone.eval()
    features: dict[str, np.ndarray] = {}
    failed: list[dict] = []
    t_decode = t_cnn = 0.0
    hits = 0
    start = time.perf_counter()
    for i, it in enumerate(items, 1):
        cache_file = FEATURE_CACHE_DIR / f"{_cache_key(it['path'], num_frames, _FEATURE_VERSION)}.npy"
        if cache_file.exists():
            features[it["path"]] = np.load(cache_file)
            hits += 1
            continue
        try:
            t0 = time.perf_counter()
            frames, _ = read_video_frames(it["path"], num_frames)
            x = torch.from_numpy(preprocess_frames(frames)).unsqueeze(0).to(device)
            t1 = time.perf_counter()
            feats = model.extract_features(x).squeeze(0).cpu().numpy().astype(np.float32)
            t2 = time.perf_counter()
            t_decode += t1 - t0
            t_cnn += t2 - t1
            np.save(cache_file, feats)
            features[it["path"]] = feats
        except (VideoReadError, OSError, ValueError) as e:
            failed.append({"path": it["rel_path"], "reason": str(e)})
            log(f"  [skip] {it['rel_path']}: {e}")
        if i % 10 == 0 or i == len(items):
            log(f"  features: {i}/{len(items)} videos  ({time.perf_counter() - start:.1f}s elapsed)")
    return {
        "features": features,
        "failed": failed,
        "timing": {
            "total_seconds": round(time.perf_counter() - start, 3),
            "decode_seconds": round(t_decode, 3),
            "cnn_seconds": round(t_cnn, 3),
            "cache_hits": hits,
            "computed": len(items) - hits - len(failed),
        },
    }


class FeatureSequenceDataset(Dataset):
    """Cached ResNet18 features: (T, 512) float tensor + label."""

    def __init__(self, items: list[dict], features: dict[str, np.ndarray]):
        self.items = [it for it in items if it["path"] in features]
        self.features = features

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return torch.from_numpy(self.features[it["path"]]), torch.tensor(it["label"], dtype=torch.long)


class VideoFrameDataset(Dataset):
    """Raw normalised frames (T, 3, 224, 224). Only needed when the backbone is fine-tuned."""

    def __init__(self, items: list[dict], num_frames: int, log=print):
        self.num_frames = num_frames
        self.items = []
        for it in items:
            try:
                load_frames_cached(it["path"], num_frames)
                self.items.append(it)
            except (VideoReadError, OSError, ValueError) as e:
                log(f"  [skip] {it['rel_path']}: {e}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        frames = load_frames_cached(it["path"], self.num_frames)
        return torch.from_numpy(preprocess_frames(frames)), torch.tensor(it["label"], dtype=torch.long)
