"""One-time project initialisation.

    python initialize.py              # check everything, train ONLY if no checkpoint exists
    python initialize.py --force      # retrain even though a checkpoint exists
    python initialize.py --check      # only run the checks, never train

Command specified by requirement:
    python initialize.py --force --epochs 25 --frames 8 --max-videos-per-class 50
"""
from __future__ import annotations

import importlib.util
import platform
import sys
import time
from pathlib import Path

REQUIRED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",
    "pydantic": "pydantic",
}


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    check_only = "--check" in args
    train_args = [a for a in args if a not in ("--force", "--check")]

    print("=" * 60)
    print("DeepGuard AI – System Initialisation & Setup")
    print("=" * 60)

    print("\n1. Python environment")
    v = sys.version_info
    if v[:2] == (3, 11):
        ok(f"Python {platform.python_version()}")
    elif (3, 10) <= v[:2] <= (3, 12):
        warn(f"Python {platform.python_version()} (3.11 recommended, this should still work)")
    else:
        fail(f"Python {platform.python_version()} - please use Python 3.11")
        return 1

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    (ok if in_venv else warn)(
        "Virtual environment active"
        if in_venv
        else "No virtual environment detected (.venv recommended)"
    )

    print("\n2. Dependencies")
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        fail("Missing packages: " + ", ".join(missing))
        print("  Install with:  pip install -r requirements.txt")
        return 1
    import torch
    ok(f"All installed (PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()})")

    from src.config import (
        CHECKPOINT_PATH,
        DEFAULT_TRAIN_CONFIG,
        FAKE_DIR,
        METRICS_PATH,
        REAL_DIR,
        ensure_dirs,
    )
    from src.data_split import (
        assign_groups,
        discover_videos,
        split_by_group,
        validate_dataset,
    )
    from src.model import pretrained_weights_cached

    ensure_dirs()

    print("\n3. Pretrained ResNet18 weights")
    if pretrained_weights_cached():
        ok("Found in local cache (.cache/torch) - no download needed")
    else:
        warn("Not cached yet - will be downloaded ONCE (~45 MB) when training starts")

    print("\n4. Dataset Validation")
    raw_items = discover_videos(max_per_class=None)
    val_report = validate_dataset(raw_items, min_frames=1)

    corrupted = val_report["corrupted"]
    unreadable = val_report["unreadable"]
    duplicates = val_report["duplicates"]
    invalid = val_report["invalid"]
    valid_items = val_report["valid"]

    n_real = val_report["counts"]["real"]
    n_fake = val_report["counts"]["fake"]
    n_total = val_report["counts"]["total"]

    print("\nDataset:")
    print(f"REAL videos: {n_real}")
    print(f"FAKE videos: {n_fake}")
    print(f"Total videos: {n_total}\n")

    if corrupted:
        fail(f"Corrupted videos ({len(corrupted)}):")
        for c in corrupted:
            print(f"    {c['path']}: {c['reason']}")
    if unreadable:
        fail(f"Unreadable videos ({len(unreadable)}):")
        for u in unreadable:
            print(f"    {u['path']}: {u['reason']}")
    if duplicates:
        fail(f"Duplicate videos ({len(duplicates)}):")
        for d in duplicates:
            print(f"    {d['path']}: {d['reason']}")
    if invalid:
        fail(f"Invalid label videos ({len(invalid)}):")
        for iv in invalid:
            print(f"    {iv['path']}: {iv['reason']}")

    # Strict check: Fail clearly if dataset is incomplete or labels are inconsistent
    if n_real != 50 or n_fake != 50:
        fail(
            f"Dataset requirement failed! Expected exactly 50 REAL and 50 FAKE videos (100 total), "
            f"but found {n_real} REAL and {n_fake} FAKE videos."
        )
        return 1

    ok("All 100 videos verified: 50 REAL and 50 FAKE, decodable and valid.")

    # Compute actual splits using source-group partitioning
    grouped_items, group_method = assign_groups(valid_items)
    val_fraction = DEFAULT_TRAIN_CONFIG["val_fraction"]
    test_fraction = DEFAULT_TRAIN_CONFIG["test_fraction"]
    seed = DEFAULT_TRAIN_CONFIG["seed"]
    split_info = split_by_group(grouped_items, val_fraction, test_fraction, seed)

    s_train = split_info["summary"]["train"]
    s_val = split_info["summary"]["val"]
    s_test = split_info["summary"]["test"]

    print("\nDataset Splits (Stratified Group Partitioning):")
    print(f"  Training videos:   {s_train['videos']:2d} ({s_train['real']} REAL / {s_train['fake']} FAKE in {s_train['groups']} groups)")
    print(f"  Validation videos: {s_val['videos']:2d} ({s_val['real']} REAL / {s_val['fake']} FAKE in {s_val['groups']} groups)")
    print(f"  Testing videos:    {s_test['videos']:2d} ({s_test['real']} REAL / {s_test['fake']} FAKE in {s_test['groups']} groups)")
    print(f"  Total splits sum:  {s_train['videos'] + s_val['videos'] + s_test['videos']} videos")

    print("\n5. Trained model checkpoint")
    have_ckpt = CHECKPOINT_PATH.exists()
    if have_ckpt and not force:
        ok(f"{CHECKPOINT_PATH.name} exists - training skipped (use --force to retrain)")
        if not METRICS_PATH.exists():
            warn("Evaluation metrics missing - running evaluation on test split")
            from src.evaluate import evaluate
            evaluate()
        print("\nReady. Start the backend with:  uvicorn backend.main:app --port 8000")
        return 0

    if not have_ckpt:
        warn("No checkpoint yet - training is required")
    if check_only:
        print("\n--check given: not training.")
        return 0

    print("\n6. Training & Evaluation Pipeline\n")
    from src.train import main as train_main

    t0 = time.perf_counter()
    train_main(train_args)
    duration_min = (time.perf_counter() - t0) / 60
    print(f"\nInitialisation finished in {duration_min:.2f} minutes.")
    print("Start the backend with:  uvicorn backend.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
