"""
Pose regression heads for spacecraft pose estimation.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PoseHead(nn.Module):
    """
    Dual-branch pose regression head.
    
    Outputs:
        - Translation: 3D position (Tx, Ty, Tz)
        - Quaternion: 4D orientation (Qx, Qy, Qz, Qw), normalized
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Translation branch
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Quaternion branch
        self.quaternion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Feature vector of shape (B, feature_dim)
            
        Returns:
            translation: (B, 3) - position in 3D
            quaternion: (B, 4) - normalized quaternion
        """
        shared = self.shared(features)
        
        translation = self.translation_head(shared)
        quaternion = self.quaternion_head(shared)
        
        # Normalize quaternion to unit norm
        quaternion = quaternion / (quaternion.norm(dim=1, keepdim=True) + 1e-8)
        
        return translation, quaternion


class PoseHeadUncertainty(nn.Module):
    """
    Pose head with uncertainty estimation (aleatoric uncertainty).
    
    Outputs translation, quaternion, and their associated uncertainties.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Translation branch (mean + log_variance)
        self.trans_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3)
        )
        self.trans_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Quaternion branch (mean + log_variance)
        self.quat_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)
        )
        self.quat_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Feature vector of shape (B, feature_dim)
            
        Returns:
            translation: (B, 3)
            trans_log_var: (B, 3) - log variance for translation
            quaternion: (B, 4) - normalized
            quat_log_var: (B, 4) - log variance for quaternion
        """
        shared = self.shared(features)
        
        translation = self.trans_mean(shared)
        trans_log_var = self.trans_var(shared)
        
        quaternion = self.quat_mean(shared)
        quaternion = quaternion / (quaternion.norm(dim=1, keepdim=True) + 1e-8)
        quat_log_var = self.quat_var(shared)
        
        return translation, trans_log_var, quaternion, quat_log_var
