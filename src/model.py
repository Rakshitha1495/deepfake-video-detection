"""CNN-LSTM deepfake classifier.

Video frames -> ResNet18 (pretrained, frozen) -> 512-d feature per frame
-> LSTM over the frame sequence -> fully connected layer -> 2 logits (Real, Fake).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from . import config  # noqa: F401  (sets TORCH_HOME before torchvision downloads)
from .config import RESNET18_WEIGHTS_FILE

FEATURE_DIM = 512  # ResNet18 output size after global average pooling


def build_backbone(pretrained: bool) -> nn.Module:
    """ResNet18 without its ImageNet classification layer.

    pretrained=True downloads the ImageNet weights once into .cache/torch and
    reuses that local file afterwards. For inference we pass pretrained=False,
    because every weight (backbone included) is restored from our checkpoint.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Identity()
    return net


class CNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.35,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(pretrained_backbone)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # When fine-tuning, freeze low-level conv1/bn1/layer1 to conserve CPU memory & preserve edge features
            for name, p in self.backbone.named_parameters():
                if any(name.startswith(prefix) for prefix in ("conv1", "bn1", "layer1", "layer2")):
                    p.requires_grad = False

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.feature_norm = nn.LayerNorm(FEATURE_DIM)
        # Combines spatial features (512) and inter-frame temporal delta (512) -> (hidden_size * 2)
        self.temporal_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # Temporal pooling: concatenated mean and max pooling across all time steps
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, 512). All B*T frames go through ResNet18 as ONE batch."""
        b, t = frames.shape[:2]
        feats = self.backbone(frames.reshape(b * t, *frames.shape[2:]))
        return feats.reshape(b, t, -1)

    def forward_features(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, 512) -> logits (B, 2). Used directly in training with cached features."""
        # Normalize spatial features
        normed = self.feature_norm(feats)
        # Compute inter-frame temporal difference features
        diff = torch.zeros_like(normed)
        if normed.size(1) > 1:
            diff[:, 1:] = normed[:, 1:] - normed[:, :-1]
        combo = torch.cat([normed, diff], dim=-1)
        h = self.temporal_proj(combo)
        out, _ = self.lstm(h)  # (B, T, out_dim)
        avg_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(self.dropout(pooled))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.extract_features(frames)
        else:
            feats = self.extract_features(frames)
        return self.forward_features(feats)

    def train(self, mode: bool = True):  # keep frozen BatchNorm statistics fixed
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def model_kwargs_from_config(cfg: dict) -> dict:
    return {
        "hidden_size": int(cfg.get("hidden_size", 64)),
        "num_layers": int(cfg.get("num_layers", 1)),
        "dropout": float(cfg.get("dropout", 0.35)),
        "freeze_backbone": bool(cfg.get("freeze_backbone", True)),
        "bidirectional": bool(cfg.get("bidirectional", True)),
    }


def pretrained_weights_cached() -> bool:
    return RESNET18_WEIGHTS_FILE.exists()


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(path: Path, model: CNNLSTM, cfg: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "config": cfg, **extra}, tmp)
    tmp.replace(path)  # atomic: a half-written checkpoint never replaces a good one


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[CNNLSTM, dict]:
    """Rebuild the model from a checkpoint WITHOUT downloading any weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CNNLSTM(pretrained_backbone=False, **model_kwargs_from_config(cfg))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
