"""
Complete pose estimation network combining backbone and head.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .backbone import VoxelCNN, EventResNet, PretrainedEfficientNet
from .pose_head import PoseHead, PoseHead6D, PoseHeadUncertainty


class PoseNet(nn.Module):
    """
    Complete spacecraft pose estimation network.
    
    Combines a CNN backbone with a pose regression head.
    Supports multiple backbone types and rotation representations.
    """
    
    def __init__(
        self,
        backbone: str = "voxel_cnn",  # "voxel_cnn", "resnet", or "efficientnet"
        in_channels: int = 10,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.1,
        with_uncertainty: bool = False,
        rotation_repr: str = "quaternion",  # "quaternion" or "6d"
        pretrained: bool = False,
        freeze_backbone: bool = False
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
        elif backbone == "efficientnet":
            self.backbone = PretrainedEfficientNet(
                in_channels=in_channels,
                feature_dim=feature_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Build pose head
        self.rotation_repr = rotation_repr
        
        if with_uncertainty:
            self.pose_head = PoseHeadUncertainty(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif rotation_repr == "6d":
            self.pose_head = PoseHead6D(
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
                - rotation_6d: (B, 6) [if rotation_repr == "6d"]
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
        elif self.rotation_repr == "6d":
            translation, rotation_6d, quaternion = self.pose_head(features)
            return {
                "translation": translation,
                "rotation_6d": rotation_6d,
                "quaternion": quaternion
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
        with_uncertainty=config.get("with_uncertainty", False),
        rotation_repr=config.get("rotation_repr", "quaternion"),
        pretrained=config.get("pretrained", False),
        freeze_backbone=config.get("freeze_backbone", False)
    )

