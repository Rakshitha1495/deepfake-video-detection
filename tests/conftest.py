import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_video(path: Path, frames: int = 30, size=(160, 120), fps: float = 15.0, seed: int = 0) -> Path:
    """Write a small synthetic test clip (moving square on noise). Used ONLY by tests."""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not create a test video"
    w, h = size
    for i in range(frames):
        img = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        x = (i * 4) % (w - 20)
        img[40:60, x:x + 20] = (0, 200, 255)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def video_file(tmp_path):
    return make_video(tmp_path / "clip.mp4")


@pytest.fixture
def corrupt_file(tmp_path):
    p = tmp_path / "broken.mp4"
    p.write_bytes(b"this is not a video" * 100)
    return p
