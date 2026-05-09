"""
Keypoint-based Pose Estimation Network.

Combines FPN backbone + Keypoint Heatmap Head for 2D keypoint detection,
with PnP solving at inference for 6DoF pose recovery.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Optional

from .fpn import FPN
from .keypoint_head import KeypointHead


class KeypointPoseNet(nn.Module):
    """
    Full keypoint-based pose estimation pipeline.
    
    Training: Predict heatmaps → soft-argmax → 2D keypoints
    Inference: 2D keypoints → PnP → 6DoF pose
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        n_keypoints: int = 8,
        heatmap_size: int = 64,
        fpn_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.n_keypoints = n_keypoints
        self.heatmap_size = heatmap_size
        
        # FPN backbone
        self.fpn = FPN(
            in_channels=in_channels,
            fpn_channels=fpn_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Keypoint detection head
        self.keypoint_head = KeypointHead(
            in_channels=fpn_channels,
            n_keypoints=n_keypoints,
            heatmap_size=heatmap_size
        )
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input voxel grid
        
        Returns:
            dict with:
                - heatmaps: (B, N, H_map, W_map) raw heatmaps for loss
                - keypoints_norm: (B, N, 2) normalized keypoint coords [0, 1]
        """
        fpn_features = self.fpn(x)
        heatmaps, keypoints_norm = self.keypoint_head(fpn_features)
        
        return {
            "heatmaps": heatmaps,
            "keypoints_norm": keypoints_norm,
        }
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def solve_pnp(keypoints_2d_px, keypoints_3d, camera_matrix):
    """
    Recover 6DoF pose from 2D-3D correspondences using EPnP.
    
    Args:
        keypoints_2d_px: (N, 2) predicted 2D keypoints in pixel coordinates
        keypoints_3d: (N, 3) known 3D keypoints in body frame
        camera_matrix: (3, 3) camera intrinsic matrix
    
    Returns:
        success: bool
        translation: (3,) translation vector
        quaternion: (4,) quaternion [Qx, Qy, Qz, Qw]
    """
    # EPnP requires float64
    pts_3d = keypoints_3d.astype(np.float64)
    pts_2d = keypoints_2d_px.astype(np.float64)
    K = camera_matrix.astype(np.float64)
    
    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, K, None,
        flags=cv2.SOLVEPNP_EPNP
    )
    
    if not success:
        return False, np.zeros(3), np.array([0, 0, 0, 1.0])
    
    # Convert rotation vector to quaternion
    from scipy.spatial.transform import Rotation
    R_mat, _ = cv2.Rodrigues(rvec)
    quat = Rotation.from_matrix(R_mat).as_quat()  # [Qx, Qy, Qz, Qw]
    
    return True, tvec.flatten(), quat
