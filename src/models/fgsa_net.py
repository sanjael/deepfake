"""
FGSA-Net: Frequency-Guided Spatial Attention Network
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network with Discrete Wavelet Decomposition
"""

from typing import Tuple, Dict, Optional, Union
import torch
import torch.nn as nn

from .spatial_branch import EfficientNetB4SpatialBranch
from .frequency_branch import FrequencyBranch
from ..fusion.attention_fusion import SpatialFrequencyAttentionFusion


class FGSANet(nn.Module):
    """
    Frequency-Guided Spatial Attention Network (FGSA-Net).
    
    Architecture:
    1. Spatial Branch: EfficientNet-B4 (extracts deep visual texture anomalies)
    2. Frequency Branch: 2D-DWT Decomposition + ConvNet (extracts wavelet forensic artifacts)
    3. Attention Fusion: Learnable cross-modal gating to dynamically weight spatial vs frequency evidence
    4. Classifier: Binary Real/Fake classification head
    """

    def __init__(
        self,
        feature_dim: int = 1792,
        pretrained_spatial: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Dual Branches
        self.spatial_branch = EfficientNetB4SpatialBranch(
            pretrained=pretrained_spatial,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        
        self.frequency_branch = FrequencyBranch(
            in_channels=12, # 4 DWT subbands * 3 RGB channels
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )

        # Cross-Modal Attention Fusion
        self.attention_fusion = SpatialFrequencyAttentionFusion(
            feature_dim=feature_dim,
            reduction_ratio=4
        )

        # Final Classifier Head on Fused Representation
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for FGSA-Net.
        Args:
            x: Input RGB tensor (B, 3, H, W)
            return_attention_weights: If True, returns attention maps (theta_s, theta_f)
        Returns:
            logits: (B, 1) raw logit output
            theta_s (optional): Spatial attention weights
            theta_f (optional): Frequency attention weights
        """
        # 1. Extract spatial features
        f_s = self.spatial_branch.extract_features(x)

        # 2. Extract frequency features
        f_f = self.frequency_branch.extract_features(x)

        # 3. Dynamic attention fusion
        f_fused, theta_s, theta_f = self.attention_fusion(f_s, f_f)

        # 4. Classification
        logits = self.classifier(f_fused)

        if return_attention_weights:
            return logits, theta_s, theta_f
        return logits
