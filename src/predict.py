"""Inference: load the checkpoint ONCE, then predict videos.

REAL = class index 0
FAKE = class index 1

Prediction outcomes:
- REAL if real_probability > fake_probability
- FAKE if fake_probability >= real_probability (deterministic tie-breaker)

CLI usage:  python -m src.predict path\\to\\video.mp4 [more videos...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from .config import CHECKPOINT_PATH, CLASS_NAMES
from .model import load_checkpoint
from .video_utils import preprocess_frames, read_video_frames

WARNING_TEXT = (
    "Experimental CNN-LSTM model trained on Celeb-DF v2. "
    "A probability is a model score, not definitive proof of manipulation."
)


def decide_prediction(prob_real: float, prob_fake: float) -> str:
    """Classify video into strictly REAL or FAKE.

    if real_probability > fake_probability:
        prediction = 'REAL'
    else:
        prediction = 'FAKE'
    """
    if prob_real > prob_fake:
        return "REAL"
    return "FAKE"


class Predictor:
    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        device: str | None = None,
    ):
        t0 = time.perf_counter()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)
        self.model.eval()
        self.config = ckpt.get("config", {})
        self.checkpoint_meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "split_files")}
        self.num_frames = int(self.config.get("num_frames", 8))
        self.load_seconds = time.perf_counter() - t0
        self._warmup()

    @torch.inference_mode()
    def _warmup(self) -> None:
        """One tiny forward pass so the first real request is not delayed."""
        x = torch.zeros(1, self.num_frames, 3, 224, 224, device=self.device)
        self.model(x)

    @torch.inference_mode()
    def predict(self, video_path: str | Path) -> dict:
        t0 = time.perf_counter()
        frames, info = read_video_frames(video_path, self.num_frames)
        t1 = time.perf_counter()
        x = torch.from_numpy(preprocess_frames(frames)).unsqueeze(0).to(self.device)
        t2 = time.perf_counter()
        self.model.eval()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        t3 = time.perf_counter()

        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        # Strict binary decision rule with FAKE as tie-breaker
        prediction = decide_prediction(prob_real, prob_fake)

        diagnostic = {
            "real_probability": round(prob_real, 6),
            "fake_probability": round(prob_fake, 6),
            "predicted_class": prediction,
            "frames_processed": int(frames.shape[0]),
            "class_index_mapping": {"0": "REAL", "1": "FAKE"},
        }

        return {
            "prediction": prediction,
            "real_probability": round(prob_real, 6),
            "fake_probability": round(prob_fake, 6),
            "status": "success",
            "probabilities": {
                "real": round(prob_real, 6),
                "fake": round(prob_fake, 6),
            },
            "frames_processed": int(frames.shape[0]),
            "diagnostic": diagnostic,
            "video": {
                k: info[k]
                for k in (
                    "total_frames",
                    "fps",
                    "width",
                    "height",
                    "duration_seconds",
                    "decode_method",
                    "sampled_indices",
                )
            },
            "timing": {
                "decode_seconds": round(t1 - t0, 4),
                "preprocess_seconds": round(t2 - t1, 4),
                "inference_seconds": round(t3 - t2, 4),
                "total_seconds": round(t3 - t0, 4),
            },
            "warning": WARNING_TEXT,
        }


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python -m src.predict <video> [video ...]")
        return 1
    if not CHECKPOINT_PATH.exists():
        print(f"No trained model at {CHECKPOINT_PATH}. Run: python initialize.py")
        return 1

    predictor = Predictor()
    print(f"Model loaded in {predictor.load_seconds:.2f}s on {predictor.device}\n")

    for v in argv:
        try:
            res = predictor.predict(v)
            d = res["diagnostic"]
            print(f"File: {v}")
            print(f"  Prediction:   {res['prediction']}")
            print(f"  Real Prob:    {d['real_probability']:.4f} ({d['real_probability']*100:.1f}%)")
            print(f"  Fake Prob:    {d['fake_probability']:.4f} ({d['fake_probability']*100:.1f}%)")
            print(f"  Frames:       {d['frames_processed']}")
            print(f"  Inference:    {res['timing']['total_seconds']:.3f}s")
            print()
        except Exception as e:
            print(json.dumps({"file": v, "error": str(e)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

