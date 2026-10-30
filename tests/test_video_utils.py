import numpy as np
import pytest

from src.video_utils import (
    VideoReadError,
    is_readable_video,
    preprocess_frames,
    read_video_frames,
    sample_indices,
)


def test_sample_indices_evenly_spaced():
    assert sample_indices(100, 5) == [0, 25, 50, 74, 99]
    assert sample_indices(0, 5) == []
    # fewer frames than requested: real frames are reused, length stays fixed
    assert len(sample_indices(3, 8)) == 8


def test_read_frames_shape(video_file):
    frames, info = read_video_frames(video_file, 8)
    assert frames.shape == (8, 224, 224, 3)
    assert frames.dtype == np.uint8
    assert info["total_frames"] == 30
    assert info["decode_seconds"] >= 0


def test_short_video_still_returns_requested_frames(tmp_path):
    from tests.conftest import make_video
    frames, _ = read_video_frames(make_video(tmp_path / "short.mp4", frames=3), 8)
    assert frames.shape[0] == 8


def test_corrupt_video_raises(corrupt_file):
    with pytest.raises(VideoReadError):
        read_video_frames(corrupt_file, 8)
    assert is_readable_video(corrupt_file)[0] is False


def test_preprocess_normalises(video_file):
    frames, _ = read_video_frames(video_file, 4)
    x = preprocess_frames(frames)
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == np.float32
    assert -3.0 < x.min() and x.max() < 3.0


def test_long_video_uses_seeking(tmp_path):
    from tests.conftest import make_video
    frames, info = read_video_frames(make_video(tmp_path / "long.mp4", frames=600, size=(64, 48)), 6)
    assert frames.shape == (6, 224, 224, 3)
    assert info["decode_method"] in ("seek", "sequential-fallback")
