"""
Loss Functions for Deepfake Detection
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for hard example mining.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothedBCEWithLogits(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    Helps prevent overconfidence on synthetic face artifacts.
    """
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets)


def get_loss_function(loss_name: str = "bce", **kwargs) -> nn.Module:
    """Loss factory."""
    name = loss_name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "focal":
        return BinaryFocalLoss(
            alpha=kwargs.get("alpha", 0.5),
            gamma=kwargs.get("gamma", 2.0)
        )
    elif name in ["smooth", "label_smoothing"]:
        return LabelSmoothedBCEWithLogits(smoothing=kwargs.get("smoothing", 0.05))
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
