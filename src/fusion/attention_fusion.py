"""
Spatial-Frequency Attention Fusion Mechanism
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyAttentionFusion(nn.Module):
    """
    Learnable Attention-Guided Fusion Module for Spatial and Frequency Features.
    
    Given:
    - f_s: Spatial feature vector from EfficientNet-B4 (B, D)
    - f_f: Frequency feature vector from DWT/DCT branch (B, D)
    
    Computes dynamic attention weights:
    - theta_s: Attention weight for spatial evidence
    - theta_f: Attention weight for frequency evidence
    
    Output:
    F_fusion = theta_s * f_s + theta_f * f_f
    """

    def __init__(self, feature_dim: int = 1792, reduction_ratio: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = max(64, feature_dim // reduction_ratio)

        # Cross-modal gating network
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2)
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            f_s: Spatial features (B, D)
            f_f: Frequency features (B, D)
        Returns:
            f_fused: Fused feature representation (B, D)
            theta_s: Spatial attention map (B, D)
            theta_f: Frequency attention map (B, D)
        """
        # Concatenate multi-modal features for joint context
        concat_feat = torch.cat([f_s, f_f], dim=1) # (B, 2*D)
        gates = self.gate_net(concat_feat)          # (B, 2*D)

        # Split into individual spatial and frequency attention weights
        gate_s, gate_f = torch.chunk(gates, 2, dim=1) # (B, D), (B, D)
        
        # Softmax / Sigmoid gating normalization across the two modalities
        # Stack along dim=1 -> (B, 2, D), apply softmax across modalities
        stacked_gates = torch.stack([gate_s, gate_f], dim=1) # (B, 2, D)
        attn_weights = F.softmax(stacked_gates, dim=1)       # (B, 2, D)
        
        theta_s = attn_weights[:, 0, :] # (B, D)
        theta_f = attn_weights[:, 1, :] # (B, D)

        # Attentive fusion
        f_fused = (theta_s * f_s) + (theta_f * f_f) # (B, D)

        return f_fused, theta_s, theta_f
