import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend import inference as inf  # noqa: E402
from backend.config import settings  # noqa: E402
from src.model import CNNLSTM, save_checkpoint  # noqa: E402


def _client_with_checkpoint(path):
    object.__setattr__(settings, "checkpoint_path", path)
    inf.model_service.predictor = None
    from backend.main import app
    return TestClient(app)


def test_missing_checkpoint_gives_helpful_503(tmp_path, video_file):
    with _client_with_checkpoint(tmp_path / "missing.pt") as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "initialize.py" in h["message"]
        r = c.post("/api/predict", files={"file": ("a.mp4", video_file.read_bytes(), "video/mp4")})
        assert r.status_code == 503


def test_predict_flow(tmp_path, video_file, corrupt_file):
    m = CNNLSTM(hidden_size=16, pretrained_backbone=False)
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, m, {"hidden_size": 16, "num_layers": 1, "dropout": 0.3,
                              "freeze_backbone": True, "num_frames": 5}, {"best_epoch": 1})
    with _client_with_checkpoint(ckpt) as c:
        assert c.get("/api/health").json()["model_loaded"] is True
        r = c.post("/api/predict", files={"file": ("clip.mp4", video_file.read_bytes(), "video/mp4")})
        body = r.json()
        assert body["prediction"] in ("REAL", "FAKE", "UNCERTAIN", "Real", "Fake", "Uncertain")
        assert abs(body["probabilities"]["real"] + body["probabilities"]["fake"] - 1) < 1e-4


        assert body["frames_processed"] == 5
        predictor = inf.model_service.predictor
        c.post("/api/predict", files={"file": ("clip.mp4", video_file.read_bytes(), "video/mp4")})
        assert inf.model_service.predictor is predictor  # model reused, not reloaded

        assert c.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
        bad = c.post("/api/predict", files={"file": ("b.mp4", corrupt_file.read_bytes(), "video/mp4")})
        assert bad.status_code == 422
        assert c.get("/api/model-info").json()["num_frames"] == 5
