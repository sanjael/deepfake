"""
Frequency Branch Feature Extractor: DWT + DCT
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..frequency.dwt_extractor import DWTExtractor2D


class FrequencyConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x) + self.shortcut(x))


class FrequencyBranch(nn.Module):
    """
    Deep Frequency-Domain Forensic Feature Extractor.
    Processes 2D-DWT sub-bands (LL, LH, HL, HH across 3 RGB channels = 12 channels).
    Produces a frequency forensic embedding (e.g., 512-dim or 1792-dim).
    """

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 1792,
        num_classes: int = 1,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.dwt_extractor = DWTExtractor2D(wavelet="haar")

        # Frequency deep convolutional backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.layer1 = FrequencyConvBlock(64, 128, stride=2)    # 128 -> 64
        self.layer2 = FrequencyConvBlock(128, 256, stride=2)   # 64 -> 32
        self.layer3 = FrequencyConvBlock(256, 512, stride=2)   # 32 -> 16
        self.layer4 = FrequencyConvBlock(512, feature_dim, stride=2) # 16 -> 8

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head for standalone baseline
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )

    def compute_dwt_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, 3, H, W) spatial RGB image into (B, 12, H/2, W/2) DWT subband tensor.
        """
        return self.dwt_extractor.extract_tensor_4ch(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, H, W) or (B, 12, H/2, W/2)
        Output: (B, feature_dim)
        """
        if x.shape[1] == 3:
            x_dwt = self.compute_dwt_input(x)
        else:
            x_dwt = x

        out = self.stem(x_dwt)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)
        feat = torch.flatten(out, 1)
        return feat

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits
