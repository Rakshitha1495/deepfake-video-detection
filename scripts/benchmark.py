"""Measure real timings on YOUR machine and save them to artifacts/benchmark.json.

    python scripts/benchmark.py path\\to\\video.mp4 --repeats 5

Measures: import time, model load, first prediction, repeated predictions
(decode / preprocess / inference / total) and peak process memory.
Nothing here is estimated - every number is measured.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T_IMPORT0 = time.perf_counter()
import psutil  # noqa: E402
import torch  # noqa: E402
from src.config import ARTIFACTS_DIR, CHECKPOINT_PATH, HISTORY_PATH  # noqa: E402
from src.predict import Predictor  # noqa: E402
IMPORT_SECONDS = time.perf_counter() - T_IMPORT0


class PeakMemory:
    def __init__(self, interval=0.02):
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self.interval = interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def summarise(runs: list[dict], key: str) -> dict:
    vals = [r["timing"][key] for r in runs]
    return {"mean": round(statistics.mean(vals), 4), "min": round(min(vals), 4), "max": round(max(vals), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint. Run: python initialize.py")
        return 1

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        predictor = Predictor()
        load_s = time.perf_counter() - t0

        results = []
        for v in a.videos:
            first = predictor.predict(v)
            repeated = [predictor.predict(v) for _ in range(a.repeats)]
            results.append({
                "video": v,
                "video_info": first["video"],
                "frames_processed": first["frames_processed"],
                "first_prediction": first["timing"],
                "repeated_predictions": {k: summarise(repeated, k) for k in
                                         ("decode_seconds", "preprocess_seconds", "inference_seconds", "total_seconds")},
                "prediction": first["prediction"],
            })

    training = None
    if HISTORY_PATH.exists():
        h = json.loads(HISTORY_PATH.read_text())
        training = {"status": h.get("status"), "total_training_seconds": h.get("timing", {}).get("total_training_seconds")}

    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": str(predictor.device)},
        "import_seconds": round(IMPORT_SECONDS, 3),
        "model_load_seconds": round(load_s, 3),
        "startup_total_seconds": round(IMPORT_SECONDS + load_s, 3),
        "peak_memory_mb": round(mem.peak / 1e6, 1),
        "initial_training": training,
        "videos": results,
        "target_check": {
            "repeated_inference_under_60s": all(r["repeated_predictions"]["total_seconds"]["max"] < 60 for r in results),
        },
    }
    out = ARTIFACTS_DIR / "benchmark.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
