"""Classification metrics, classification reports, and evaluation plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CLASS_NAMES


def compute_metrics(y_true, y_pred, y_prob_fake) -> dict[str, Any]:
    """Metrics with FAKE (=1) as positive class and REAL (=0) as negative class.

    Includes accuracy, precision, recall, f1, roc_auc, confusion matrix,
    and scikit-learn classification report.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob_fake = np.asarray(y_prob_fake, dtype=float)

    out: dict[str, Any] = {
        "num_samples": int(len(y_true)),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "roc_auc_note": None,
        "average_precision": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "classification_report": None,
        "classification_report_str": "",
    }

    if len(y_true) == 0:
        out["roc_auc_note"] = "No samples."
        return out

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    out.update({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_str": report_str,
    })

    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob_fake))
        out["average_precision"] = float(average_precision_score(y_true, y_prob_fake))
    else:
        out["roc_auc_note"] = "Not defined: only one class is present in this split."

    return out


def _setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    return plt


def save_confusion_matrix_png(cm: list[list[int]], path: Path, title: str = "Test Confusion Matrix") -> None:
    plt = _setup_plot()
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=120)
    cax = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=CLASS_NAMES, fontsize=10)
    ax.set_yticks([0, 1], labels=CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")

    vmax = cm_arr.max() if cm_arr.size else 0
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            color = "white" if vmax and val > vmax / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=15, fontweight="bold")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_loss_graph_png(history: list[dict], path: Path) -> None:
    plt = _setup_plot()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss") for h in history]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    ax.plot(epochs, train_loss, label="Train Loss", color="#4F46E5", linewidth=2, marker="o", markersize=4)

    valid_val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
    valid_val_loss = [v for v in val_loss if v is not None]
    if valid_val_loss:
        ax.plot(valid_val_epochs, valid_val_loss, label="Val Loss", color="#06B6D4", linewidth=2, marker="s", markersize=4)

    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_graph_png(history: list[dict], path: Path) -> None:
    plt = _setup_plot()
    epochs = [h["epoch"] for h in history]
    train_acc = [h.get("train_accuracy", 0.0) for h in history]
    val_acc = [h.get("val_accuracy") for h in history]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    ax.plot(epochs, train_acc, label="Train Accuracy", color="#8B5CF6", linewidth=2, marker="o", markersize=4)

    valid_val_epochs = [e for e, v in zip(epochs, val_acc) if v is not None]
    valid_val_acc = [v for v in val_acc if v is not None]
    if valid_val_acc:
        ax.plot(valid_val_epochs, valid_val_acc, label="Val Accuracy", color="#10B981", linewidth=2, marker="s", markersize=4)

    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_png(y_true, y_prob_fake, path: Path) -> None:
    if len(set(y_true)) < 2:
        return
    plt = _setup_plot()
    fpr, tpr, _ = roc_curve(y_true, y_prob_fake, pos_label=1)
    auc = roc_auc_score(y_true, y_prob_fake)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Random Classifier (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve_png(y_true, y_prob_fake, path: Path) -> None:
    if len(set(y_true)) < 2:
        return
    plt = _setup_plot()
    precision, recall, _ = precision_recall_curve(y_true, y_prob_fake, pos_label=1)
    ap = average_precision_score(y_true, y_prob_fake)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
    ax.plot(recall, precision, color="#059669", linewidth=2.5, label=f"PR Curve (AP = {ap:.3f})")

    baseline = sum(y_true) / len(y_true)
    ax.axhline(baseline, color="#9CA3AF", linestyle="--", label=f"Baseline (Prevalence = {baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
