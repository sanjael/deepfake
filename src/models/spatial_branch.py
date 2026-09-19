"""
Spatial Branch Feature Extractor: EfficientNet-B4
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B4_Weights


class EfficientNetB4SpatialBranch(nn.Module):
    """
    Spatial Forensic Feature Extractor based on EfficientNet-B4.
    Extracts deep visual/texture forensic representations (1792-dim embedding).
    
    Can operate in two modes:
    1. Standalone Binary Classifier (Baseline 1): Features -> Dropout -> Linear -> Logits
    2. Spatial Feature Backbone (Proposed FGSA-Net): Returns raw 1792-dim spatial features.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 1,
        feature_dim: int = 1792
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Load EfficientNet-B4 backbone
        if pretrained:
            weights = EfficientNet_B4_Weights.DEFAULT
            base_model = models.efficientnet_b4(weights=weights)
        else:
            base_model = models.efficientnet_b4(weights=None)

        # Feature extractor up to Global Average Pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Classification head for standalone baseline
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled spatial feature vectors.
        Input: (B, 3, H, W)
        Output: (B, 1792)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            logits: (B, 1) raw unbounded logits for BCEWithLogitsLoss
            features (optional): (B, 1792) if return_features=True
        """
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        
        if return_features:
            return logits, feat
        return logits
