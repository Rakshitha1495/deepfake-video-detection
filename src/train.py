"""Train the CNN-LSTM deepfake detector on dataset/raw/{real,fake}.

Improvements:
- AdamW optimizer with weight decay.
- Cosine annealing learning-rate scheduler.
- Gradient clipping (max_norm=1.0).
- Early stopping with configurable patience.
- Automatic overfitting detection.
- Best checkpoint selection strictly by validation performance.
- ResNet18 feature caching for lightning-fast CPU training.
- Complete metric monitoring (train/val loss, accuracy, precision, recall, F1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEFAULT_TRAIN_CONFIG,
    HISTORY_PATH,
    ensure_dirs,
)
from .data_split import (
    assign_groups,
    class_distribution,
    discover_videos,
    split_by_group,
    validate_dataset,
)
from .dataset import FeatureSequenceDataset, VideoFrameDataset, build_feature_cache
from .metrics import compute_metrics
from .model import CNNLSTM, count_parameters, pretrained_weights_cached, save_checkpoint


def log(msg: str = "") -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv=None) -> dict:
    d = DEFAULT_TRAIN_CONFIG
    p = argparse.ArgumentParser(description="Train the CNN-LSTM deepfake detector")
    p.add_argument("--epochs", type=int, default=d["epochs"], help="Number of training epochs")
    p.add_argument("--frames", type=int, default=d["num_frames"], help="Frames sampled per video (default 8)")
    p.add_argument("--batch-size", type=int, default=d["batch_size"], help="Training batch size")
    p.add_argument("--lr", type=float, default=d["learning_rate"], help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"], help="AdamW weight decay")
    p.add_argument("--hidden-size", type=int, default=d["hidden_size"], help="LSTM hidden size")
    p.add_argument("--seed", type=int, default=d["seed"], help="Random seed for reproducibility")
    p.add_argument("--max-videos-per-class", type=int, default=d["max_videos_per_class"], help="Max videos per class (50 for full dataset)")
    p.add_argument("--patience", type=int, default=d.get("early_stopping_patience", 7), help="Early stopping patience")
    p.add_argument("--time-budget-minutes", type=float, default=d["time_budget_minutes"])
    p.add_argument("--stop-at-budget", action="store_true", help="Stop after time budget is exceeded")
    p.add_argument("--finetune-backbone", action="store_true", help="Fine-tune ResNet18 (slower on CPU)")
    p.add_argument("--skip-eval", action="store_true", help="Skip test evaluation after training")

    a = p.parse_args(argv)
    cfg = dict(d)
    cfg.update({
        "epochs": a.epochs,
        "num_frames": a.frames,
        "batch_size": a.batch_size,
        "learning_rate": a.lr,
        "weight_decay": a.weight_decay,
        "hidden_size": a.hidden_size,
        "seed": a.seed,
        "early_stopping_patience": a.patience,
        "max_videos_per_class": a.max_videos_per_class,
        "time_budget_minutes": a.time_budget_minutes,
        "freeze_backbone": not a.finetune_backbone,
    })
    cfg["_stop_at_budget"] = a.stop_at_budget
    cfg["_skip_eval"] = a.skip_eval
    return cfg


def run_epoch(model, loader, criterion, device, optimizer=None, use_features=True) -> dict:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss, y_true, y_pred, y_prob = 0.0, [], [], []
    ctx = torch.enable_grad() if training else torch.inference_mode()

    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.forward_features(x) if use_features else model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            probs = torch.softmax(logits.detach(), dim=1)
            y_true += y.tolist()
            y_pred += probs.argmax(1).tolist()
            y_prob += probs[:, 1].tolist()

    n = max(len(y_true), 1)
    m = compute_metrics(y_true, y_pred, y_prob)
    return {
        "loss": total_loss / n,
        "accuracy": m["accuracy"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
    }


def train(cfg: dict) -> dict:
    ensure_dirs()
    set_seed(cfg["seed"])
    device = pick_device()
    n_threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(n_threads)

    t_start = time.perf_counter()
    budget_s = cfg["time_budget_minutes"] * 60
    timing: dict = {}

    log(f"Device: {device}  |  CPU threads allocated: {torch.get_num_threads()}")

    # 1. Discover all videos
    items = discover_videos(max_per_class=cfg["max_videos_per_class"], seed=cfg["seed"])
    dist = class_distribution(items)
    log(f"\nDataset Discovery:")
    log(f"  REAL videos: {dist['real']}")
    log(f"  FAKE videos: {dist['fake']}")
    log(f"  Total:       {dist['total']}")

    if dist["real"] == 0 or dist["fake"] == 0:
        raise SystemExit("Need at least one video in BOTH dataset/raw/real and dataset/raw/fake. Add videos and retry.")

    # 2. Comprehensive validation of all discovered videos
    log("\nValidating videos...")
    val_rep = validate_dataset(items, min_frames=cfg["num_frames"])
    if val_rep["corrupted"]:
        log(f"  [WARN] Corrupted videos ({len(val_rep['corrupted'])}):")
        for c in val_rep["corrupted"]:
            log(f"    - {c['path']}: {c['reason']}")
    if val_rep["unreadable"]:
        log(f"  [WARN] Unreadable videos ({len(val_rep['unreadable'])}):")
        for u in val_rep["unreadable"]:
            log(f"    - {u['path']}: {u['reason']}")
    if val_rep["duplicates"]:
        log(f"  [WARN] Duplicate videos ({len(val_rep['duplicates'])}):")
        for d in val_rep["duplicates"]:
            log(f"    - {d['path']}: {d['reason']}")

    items = val_rep["valid"]
    log(f"  Validation passed: {len(items)} valid videos ready for pipeline.")

    # 3. Stratified group splitting
    items, group_method = assign_groups(items)
    split = split_by_group(items, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"])

    log("\nDataset Splits (Source-Group Partitioning):")
    for name, s in split["summary"].items():
        log(f"  {name.capitalize():5s}: {s['videos']:2d} videos ({s['real']} REAL / {s['fake']} FAKE) in {s['groups']} source groups")
    log(f"  Grouping method: {group_method}  |  {split['num_groups']} independent source groups")
    for w in split["warnings"]:
        log(f"  WARNING: {w}")

    if len(split["splits"]["train"]) == 0:
        raise SystemExit("Training split is empty; add more videos.")

    # 4. Build CNN-LSTM model
    cached_before = pretrained_weights_cached()
    t0 = time.perf_counter()
    model = CNNLSTM(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pretrained_backbone=True,
        freeze_backbone=cfg["freeze_backbone"],
        bidirectional=cfg.get("bidirectional", True),
    ).to(device)
    timing["model_build_seconds"] = round(time.perf_counter() - t0, 3)
    timing["pretrained_weights_downloaded"] = not cached_before
    log(f"\nResNet18 weights: {'loaded from local cache' if cached_before else 'downloaded once to .cache/torch'}")
    param_counts = count_parameters(model)
    log(f"Model parameters: Total={param_counts['total']:,}, Trainable={param_counts['trainable']:,}")

    # 5. Extract or load cached features
    failed: list[dict] = []
    use_features = cfg["freeze_backbone"]
    if use_features:
        log(f"\nExtracting ResNet18 spatial features ({cfg['num_frames']} frames/video, cached in .cache/features)...")
        fc = build_feature_cache(items, model, cfg["num_frames"], device, log=log)
        timing["feature_extraction"] = fc["timing"]
        failed = fc["failed"]
        ds = {k: FeatureSequenceDataset(v, fc["features"]) for k, v in split["splits"].items()}
    else:
        log(f"\nFine-tuning mode: decoding raw frames...")
        ds = {k: VideoFrameDataset(v, cfg["num_frames"], log=log) for k, v in split["splits"].items()}

    # Data loaders (use batch_size=8 on cached features for stable gradients)
    train_loader = DataLoader(
        ds["train"],
        batch_size=min(cfg["batch_size"], len(ds["train"])),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=min(max(cfg["batch_size"], 8), max(len(ds["val"]), 1)),
        shuffle=False,
    )

    # Class-weighted CrossEntropyLoss to compensate for imbalance
    train_labels = [it["label"] for it in ds["train"].items]
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer: AdamW with weight decay
    lstm_params = [
        {"params": list(model.feature_norm.parameters()) + list(model.temporal_proj.parameters()) + list(model.lstm.parameters()) + list(model.classifier.parameters()),
         "lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]}
    ]
    if not cfg["freeze_backbone"]:
        trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        if trainable_backbone:
            lstm_params.append({"params": trainable_backbone, "lr": cfg["learning_rate"] * 0.1, "weight_decay": cfg["weight_decay"]})

    optimizer = torch.optim.AdamW(lstm_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)

    has_val = len(ds["val"]) > 0
    patience = cfg.get("early_stopping_patience", 7)
    log(f"\nTraining Settings: {cfg['epochs']} epochs, batch size {cfg['batch_size']}, AdamW lr={cfg['learning_rate']}, patience={patience}.")
    log(f"Model selection: validation performance (balanced F1 + accuracy, tiebreaker lower val loss).")

    history = []
    best = {"score": -1e9, "epoch": None, "val_f1": None, "val_loss": None}
    epochs_without_improvement = 0
    epoch_times: list[float] = []
    status = "completed"
    budget_messages: list[str] = []
    public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    for epoch in range(1, cfg["epochs"] + 1):
        te = time.perf_counter()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, use_features)
        va = run_epoch(model, val_loader, criterion, device, None, use_features) if has_val else None
        scheduler.step()
        dt = time.perf_counter() - te
        epoch_times.append(dt)

        row = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 5),
            "train_accuracy": tr["accuracy"],
            "val_loss": round(va["loss"], 5) if va else None,
            "val_accuracy": va["accuracy"] if va else None,
            "val_f1": va["f1"] if va else None,
            "val_precision": va["precision"] if va else None,
            "val_recall": va["recall"] if va else None,
            "epoch_seconds": round(dt, 3),
        }
        history.append(row)

        # Automatic overfitting detection
        overfitting_flag = ""
        if va and tr["loss"] < 0.45 and (va["loss"] - tr["loss"]) > 0.40:
            overfitting_flag = " [OVERFITTING DETECTED]"

        # Best validation model selection:
        # Score balances validation accuracy, validation F1, and lower validation loss
        if va:
            val_f1 = va["f1"] if va["f1"] is not None else 0.0
            val_acc = va["accuracy"] if va["accuracy"] is not None else 0.0
            val_score = 0.5 * val_f1 + 0.5 * val_acc - 0.1 * va["loss"]
        else:
            val_score = -tr["loss"]
            val_f1 = 0.0

        improved = val_score > best["score"]
        if improved:
            best = {
                "score": val_score,
                "epoch": epoch,
                "val_f1": va["f1"] if va else None,
                "val_loss": va["loss"] if va else None,
            }
            epochs_without_improvement = 0
            save_checkpoint(CHECKPOINT_PATH, model, public_cfg, {
                "best_epoch": epoch,
                "best_val_f1": va["f1"] if va else None,
                "best_val_score": round(val_score, 4),
                "class_names": CLASS_NAMES,
                "split_files": {k: [it["rel_path"] for it in v.items] for k, v in ds.items()},
                "split_summary": split["summary"],
                "split_warnings": split["warnings"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            epochs_without_improvement += 1

        val_txt = f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} F1 {va['f1']:.3f}" if va else "no val data"
        save_tag = "  <- BEST CHECKPOINT" if improved else ""
        log(f"Epoch {epoch:2d}/{cfg['epochs']}: train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} | {val_txt} | {dt:.2f}s{save_tag}{overfitting_flag}")

        # Early stopping check
        if epochs_without_improvement >= patience and epoch >= 10:
            log(f"\n[EARLY STOPPING] Validation performance has not improved for {patience} consecutive epochs.")
            log(f"Stopping early to prevent overfitting. Best model restored from Epoch {best['epoch']}.")
            status = f"early_stopped_epoch_{epoch}"
            break

        # Time budget check
        elapsed = time.perf_counter() - t_start
        if elapsed > budget_s and not budget_messages:
            budget_messages.append(f"Training reached time budget of {cfg['time_budget_minutes']} min.")
            if cfg["_stop_at_budget"]:
                log(f"Stopping at time budget request after epoch {epoch}.")
                status = f"stopped_at_time_budget_epoch_{epoch}"
                break

    timing["epochs_seconds"] = round(sum(epoch_times), 3)
    timing["total_training_seconds"] = round(time.perf_counter() - t_start, 3)

    record = {
        "status": status,
        "config": public_cfg,
        "device": str(device),
        "best_epoch": best["epoch"],
        "best_val_f1": best["val_f1"],
        "selection_criterion": "Validation performance: 0.5*F1 + 0.5*Accuracy - 0.1*Loss",
        "history": history,
        "class_distribution": dist,
        "grouping_method": group_method,
        "num_source_groups": split["num_groups"],
        "split_summary": split["summary"],
        "split_warnings": split["warnings"],
        "failed_videos": failed,
        "timing": timing,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    HISTORY_PATH.write_text(json.dumps(record, indent=2))

    log(f"\nTraining completed in {timing['total_training_seconds']:.1f}s ({timing['total_training_seconds'] / 60:.2f} min).")
    log(f"Best checkpoint saved from Epoch {best['epoch']} -> {CHECKPOINT_PATH}")
    log(f"Training history saved -> {HISTORY_PATH}")
    return record


def main(argv=None) -> None:
    cfg = parse_args(argv)
    train(cfg)
    if not cfg["_skip_eval"]:
        from .evaluate import evaluate
        evaluate()


if __name__ == "__main__":
    sys.exit(main())
