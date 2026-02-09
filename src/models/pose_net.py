"""
Complete pose estimation network combining backbone and head.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .backbone import VoxelCNN, EventResNet
from .pose_head import PoseHead, PoseHeadUncertainty


class PoseNet(nn.Module):
    """
    Complete spacecraft pose estimation network.
    
    Combines a CNN backbone with a pose regression head.
    """
    
    def __init__(
        self,
        backbone: str = "voxel_cnn",  # "voxel_cnn" or "resnet"
        in_channels: int = 10,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.1,
        with_uncertainty: bool = False
    ):
        super().__init__()
        
        # Build backbone
        if backbone == "voxel_cnn":
            self.backbone = VoxelCNN(
                in_channels=in_channels,
                feature_dim=feature_dim,
                base_channels=base_channels
            )
        elif backbone == "resnet":
            self.backbone = EventResNet(
                in_channels=in_channels,
                feature_dim=feature_dim,
                base_channels=base_channels * 2  # ResNet uses more channels
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Build pose head
        if with_uncertainty:
            self.pose_head = PoseHeadUncertainty(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.pose_head = PoseHead(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        self.with_uncertainty = with_uncertainty
        self.backbone_name = backbone
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Voxel grid of shape (B, C, H, W)
            
        Returns:
            Dictionary with:
                - translation: (B, 3)
                - quaternion: (B, 4)
                - trans_log_var: (B, 3) [if with_uncertainty]
                - quat_log_var: (B, 4) [if with_uncertainty]
        """
        features = self.backbone(x)
        
        if self.with_uncertainty:
            translation, trans_log_var, quaternion, quat_log_var = self.pose_head(features)
            return {
                "translation": translation,
                "quaternion": quaternion,
                "trans_log_var": trans_log_var,
                "quat_log_var": quat_log_var
            }
        else:
            translation, quaternion = self.pose_head(features)
            return {
                "translation": translation,
                "quaternion": quaternion
            }
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> PoseNet:
    """
    Build model from config dictionary.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        PoseNet model
    """
    return PoseNet(
        backbone=config.get("backbone", "voxel_cnn"),
        in_channels=config.get("in_channels", 10),
        feature_dim=config.get("feature_dim", 512),
        hidden_dim=config.get("hidden_dim", 256),
        base_channels=config.get("base_channels", 32),
        dropout=config.get("dropout", 0.1),
        with_uncertainty=config.get("with_uncertainty", False)
    )
