"""Neural network architectures for deepfake detection."""
from .spatial_branch import EfficientNetB4SpatialBranch
from .frequency_branch import FrequencyBranch
from .fgsa_net import FGSANet

__all__ = [
    "EfficientNetB4SpatialBranch",
    "FrequencyBranch",
    "FGSANet"
]
