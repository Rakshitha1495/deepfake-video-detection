"""Fast video decoding and frame preprocessing (OpenCV + NumPy only).

This module deliberately has no PyTorch dependency so it can be tested and
reused on its own. It is shared by training, evaluation and the backend, so
training-time and inference-time preprocessing are guaranteed to match.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# For videos longer than this, seeking to each sampled frame is cheaper than
# decoding every frame sequentially.
SEEK_THRESHOLD_FRAMES = 450
# Safety cap when the container does not report a frame count.
MAX_SEQUENTIAL_FRAMES = 5000

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


class VideoReadError(Exception):
    """Raised when a file cannot be opened or decoded as a video."""


def sample_indices(total_frames: int, num_frames: int) -> list[int]:
    """Evenly spaced frame indices across the whole video.

    If a clip has fewer frames than requested, some real frames are reused so
    the sequence length stays fixed (no synthetic frames are created).
    """
    if total_frames <= 0:
        return []
    if num_frames <= 1:
        return [total_frames // 2]
    return [int(round(x)) for x in np.linspace(0, total_frames - 1, num_frames)]


def _square_resize(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square (keeps aspect ratio) and resize to size x size, RGB."""
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame_bgr[top:top + side, left:left + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (size, size), interpolation=interp)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _read_by_seeking(cap: cv2.VideoCapture, indices: list[int], size: int) -> list[np.ndarray] | None:
    frames = []
    cache: dict[int, np.ndarray] = {}
    for idx in indices:
        if idx in cache:
            frames.append(cache[idx])
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None  # caller falls back to sequential decoding
        cache[idx] = _square_resize(frame, size)
        frames.append(cache[idx])
    return frames


def _read_sequentially(cap: cv2.VideoCapture, indices: list[int], size: int) -> list[np.ndarray]:
    """grab() every frame but only retrieve()/convert the ones we need."""
    wanted = set(indices)
    last = max(indices)
    picked: dict[int, np.ndarray] = {}
    pos = 0
    while pos <= last:
        if not cap.grab():
            break
        if pos in wanted:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                picked[pos] = _square_resize(frame, size)
        pos += 1
    if not picked:
        return []
    # If the reported frame count was too high, map missing indices to the
    # nearest decoded frame instead of failing.
    available = sorted(picked)
    out = []
    for idx in indices:
        if idx in picked:
            out.append(picked[idx])
        else:
            nearest = min(available, key=lambda a: abs(a - idx))
            out.append(picked[nearest])
    return out


def _count_frames_sequentially(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    try:
        while n < MAX_SEQUENTIAL_FRAMES and cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def read_video_frames(path: str | Path, num_frames: int, size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    """Decode `num_frames` evenly spaced RGB frames, center-cropped to size x size.

    Returns (frames uint8 array of shape (T, size, size, 3), info dict).
    Raises VideoReadError for unreadable or empty videos.
    """
    path = str(path)
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise VideoReadError("File could not be opened as a video (unsupported codec or corrupted file).")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if total <= 0:
            cap.release()
            total = _count_frames_sequentially(path)
            cap = cv2.VideoCapture(path)
        if total <= 0:
            raise VideoReadError("Video contains no decodable frames.")

        indices = sample_indices(total, num_frames)
        frames = None
        method = "sequential"
        if total > SEEK_THRESHOLD_FRAMES:
            frames = _read_by_seeking(cap, indices, size)
            method = "seek"
            if frames is None:  # seeking unsupported for this container/codec
                cap.release()
                cap = cv2.VideoCapture(path)
                method = "sequential-fallback"
        if frames is None:
            frames = _read_sequentially(cap, indices, size)
        if not frames:
            raise VideoReadError("Video frames could not be decoded.")
    finally:
        cap.release()

    info = {
        "total_frames": total,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_seconds": round(total / fps, 3) if fps > 0 else None,
        "sampled_indices": indices,
        "decode_method": method,
        "decode_seconds": round(time.perf_counter() - t0, 4),
    }
    return np.stack(frames, axis=0), info


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
    """uint8 (T, H, W, 3) RGB -> float32 (T, 3, H, W) normalised with ImageNet stats."""
    x = frames.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x, dtype=np.float32)


def is_readable_video(path: str | Path) -> tuple[bool, str]:
    """Cheap check: can OpenCV open the file and decode its first frame?"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "File could not be opened as a video."
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "No frames could be decoded from the file."
        return True, "ok"
    finally:
        cap.release()


def validate_video_file(path: str | Path, min_frames: int = 1) -> tuple[bool, str, dict]:
    """Comprehensive validation of video readability, frame count, and dimensions."""
    p = Path(path)
    if not p.exists():
        return False, f"File does not exist: {p.name}", {}
    if p.stat().st_size == 0:
        return False, f"File is 0 bytes: {p.name}", {}

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            return False, f"OpenCV could not open video: {p.name}", {}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            return False, f"Could not decode first frame from: {p.name}", {}

        if total <= 0:
            total = _count_frames_sequentially(str(p))

        if total < min_frames:
            return False, f"Insufficient frames ({total} < {min_frames}): {p.name}", {}

        meta = {
            "total_frames": total,
            "fps": fps,
            "width": width,
            "height": height,
            "file_size": p.stat().st_size,
        }
        return True, "ok", meta
    finally:
        cap.release()

