"""Evaluation on held-out TEST split stored in checkpoint.

Usage:  python -m src.evaluate
Generates:
- artifacts/evaluation_metrics.json
- artifacts/confusion_matrix.png
- artifacts/roc_curve.png
- artifacts/precision_recall_curve.png
- artifacts/training_validation_loss.png
- artifacts/training_validation_accuracy.png
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import (
    ACCURACY_GRAPH_PATH,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    DATASET_DIR,
    HISTORY_PATH,
    LOSS_GRAPH_PATH,
    METRICS_PATH,
    PR_CURVE_PATH,
    ROC_CURVE_PATH,
    ensure_dirs,
)
from .dataset import build_feature_cache, load_frames_cached
from .metrics import (
    compute_metrics,
    save_accuracy_graph_png,
    save_confusion_matrix_png,
    save_loss_graph_png,
    save_pr_curve_png,
    save_roc_curve_png,
)
from .model import load_checkpoint
from .video_utils import VideoReadError, preprocess_frames


def _items_from_rel_paths(rel_paths: list[str]) -> list[dict]:
    items = []
    for rel in rel_paths:
        label = 0 if rel.split("/", 1)[0] == "real" else 1
        items.append({"path": str(DATASET_DIR / rel), "rel_path": rel, "label": label})
    return items


@torch.inference_mode()
def evaluate() -> dict:
    ensure_dirs()
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT_PATH}. Run training first: python initialize.py")

    t0 = time.perf_counter()
    model, ckpt = load_checkpoint(CHECKPOINT_PATH, "cpu")
    model.eval()
    cfg = ckpt.get("config", {})
    num_frames = int(cfg.get("num_frames", 8))

    test_paths = ckpt.get("split_files", {}).get("test", [])
    items = _items_from_rel_paths(test_paths)
    missing = [it["rel_path"] for it in items if not (DATASET_DIR / it["rel_path"]).exists()]
    items = [it for it in items if it["rel_path"] not in missing]

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    failed: list[dict] = [{"path": m, "reason": "file missing"} for m in missing]

    if items:
        fc = build_feature_cache(items, model, num_frames, "cpu", log=lambda *_: None)
        failed += fc["failed"]
        for it in items:
            if it["path"] not in fc["features"]:
                continue
            feats = torch.from_numpy(fc["features"][it["path"]]).unsqueeze(0)
            logits = model.forward_features(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            prob_fake = float(probs[1])

            y_true.append(it["label"])
            y_pred.append(int(prob_fake >= 0.5))
            y_prob.append(prob_fake)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    warnings = list(ckpt.get("split_warnings", []))
    n_real = sum(1 for y in y_true if y == 0)
    n_fake = sum(1 for y in y_true if y == 1)

    if len(y_true) == 0:
        warnings.append("The test split is empty, so no test metrics can be reported. Add more videos.")
    elif len(set(y_true)) < 2:
        warnings.append("The test split contains only one class; accuracy/precision/recall are not meaningful.")
    if 0 < len(y_true) < 10:
        warnings.append(
            f"Only {len(y_true)} test videos: each video changes accuracy by "
            f"{100 / len(y_true):.0f} percentage points. Treat these numbers as a rough sanity check."
        )

    # Generate and save plots
    save_confusion_matrix_png(metrics["confusion_matrix"], CONFUSION_MATRIX_PATH)
    if len(y_true) > 0 and len(set(y_true)) == 2:
        save_roc_curve_png(y_true, y_prob, ROC_CURVE_PATH)
        save_pr_curve_png(y_true, y_prob, PR_CURVE_PATH)

    # Plot training/validation curves if history exists
    if HISTORY_PATH.exists():
        try:
            hist_data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            history = hist_data.get("history", [])
            if history:
                save_loss_graph_png(history, LOSS_GRAPH_PATH)
                save_accuracy_graph_png(history, ACCURACY_GRAPH_PATH)
        except Exception as e:
            warnings.append(f"Could not generate training loss/accuracy graphs: {e}")

    result = {
        "split": "test",
        "num_test_videos": len(y_true),
        "class_distribution": {"real": n_real, "fake": n_fake, "total": len(y_true)},
        "metrics": metrics,
        "meaningful": len(y_true) > 0 and len(set(y_true)) == 2,
        "warnings": warnings,
        "failed_videos": failed,
        "num_frames": num_frames,
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_f1": ckpt.get("best_val_f1"),
        "best_val_score": ckpt.get("best_val_score"),
        "evaluation_seconds": round(time.perf_counter() - t0, 3),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("DEEPGUARD AI – TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Test videos evaluated: {len(y_true)} (REAL: {n_real}, FAKE: {n_fake})")
    if metrics["accuracy"] is not None:
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
        ap_str = f"{metrics['average_precision']:.4f}" if metrics["average_precision"] is not None else "N/A"
        print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision (Fake):  {metrics['precision']:.4f}")
        print(f"Recall (Fake):     {metrics['recall']:.4f}")
        print(f"F1 Score (Fake):   {metrics['f1']:.4f}")
        print(f"ROC-AUC:           {auc_str}")
        print(f"Average Precision: {ap_str}")
        print(f"\nConfusion Matrix [[TN, FP], [FN, TP]]:\n{metrics['confusion_matrix']}")
        if metrics["classification_report_str"]:
            print("\nClassification Report:")
            print(metrics["classification_report_str"])

    for w in warnings:
        print(f"WARNING: {w}")

    print("\nGenerated Artifacts:")
    print(f"  Metrics JSON:      {METRICS_PATH}")
    print(f"  Confusion Matrix:  {CONFUSION_MATRIX_PATH}")
    print(f"  ROC Curve:         {ROC_CURVE_PATH}")
    print(f"  PR Curve:          {PR_CURVE_PATH}")
    print(f"  Loss Graph:        {LOSS_GRAPH_PATH}")
    print(f"  Accuracy Graph:    {ACCURACY_GRAPH_PATH}")
    print("=" * 60 + "\n")

    return result


if __name__ == "__main__":
    evaluate()
    sys.exit(0)
