"""
Keypoint heatmap prediction head + differentiable coordinate extraction.

Takes FPN features and predicts N heatmaps, then uses soft-argmax
to extract sub-pixel 2D keypoint coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """
    Predicts keypoint heatmaps from FPN features.
    
    Architecture:
        FPN features (256ch, 64x64)
        → 3x DeconvBlock (256ch each, upsample to 64x64 — identity if already 64x64)
        → 1x1 Conv → N heatmaps
        → Soft-argmax → (N, 2) coordinates
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        n_keypoints: int = 8,
        heatmap_size: int = 64
    ):
        super().__init__()
        
        self.n_keypoints = n_keypoints
        self.heatmap_size = heatmap_size
        
        # Refinement convolutions (no upsampling since FPN already gives 64x64)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Final 1x1 conv: predict one heatmap per keypoint
        self.heatmap_conv = nn.Conv2d(256, n_keypoints, 1)
        
        # Pre-compute coordinate grids for soft-argmax
        # These will be set on first forward pass based on actual heatmap size
        self.register_buffer('_coord_x', None)
        self.register_buffer('_coord_y', None)
    
    def _init_coord_grids(self, h, w, device):
        """Initialize coordinate grids for soft-argmax."""
        # Normalized coordinates [0, 1]
        x = torch.linspace(0, 1, w, device=device)
        y = torch.linspace(0, 1, h, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self._coord_x = xx  # (H, W)
        self._coord_y = yy  # (H, W)
    
    def soft_argmax(self, heatmaps):
        """
        Differentiable soft-argmax to extract (x, y) from heatmaps.
        Forces float32 to prevent AMP float16 overflow.
        
        Args:
            heatmaps: (B, N, H, W)
        
        Returns:
            coords: (B, N, 2) — normalized coordinates in [0, 1]
        """
        B, N, H, W = heatmaps.shape
        
        # Initialize grids if needed
        if self._coord_x is None or self._coord_x.shape != (H, W):
            self._init_coord_grids(H, W, heatmaps.device)
        
        # Apply sigmoid to bound outputs between 0 and 1 for MSE loss targeting
        heatmaps = torch.sigmoid(heatmaps)
        
        # CRITICAL: force float32 for numerical stability with AMP
        heatmaps_f32 = heatmaps.float()
        
        # Softmax over spatial dimensions
        heatmaps_flat = heatmaps_f32.view(B, N, -1)
        
        # Multiply by a temperature (e.g. 50.0) so the peak at ~1.0 dominates the background at ~0.0
        # Otherwise, exp(1.0)/sum(exp(0.0)) over 4096 pixels is extremely flat.
        weights = F.softmax(heatmaps_flat * 50.0, dim=2)
        weights = weights.view(B, N, H, W)
        
        # Weighted sum of coordinates
        x_coord = (weights * self._coord_x.float()).sum(dim=(2, 3))  # (B, N)
        y_coord = (weights * self._coord_y.float()).sum(dim=(2, 3))  # (B, N)
        
        coords = torch.stack([x_coord, y_coord], dim=2)  # (B, N, 2)
        return coords
    
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: (B, 256, H/4, W/4) from FPN
        
        Returns:
            heatmaps: (B, N, H_map, W_map) sigmoid heatmaps in [0, 1]
            keypoints_norm: (B, N, 2) predicted keypoint coords in [0, 1]
        """
        x = self.refine(fpn_features)
        raw_heatmaps = self.heatmap_conv(x)  # (B, N, H, W)
        
        # Extract keypoint coordinates via soft-argmax (which internally applies Sigmoid)
        keypoints_norm = self.soft_argmax(raw_heatmaps)
        
        # Return Sigmoided heatmaps for MSE loss comparison against [0, 1] Gaussians
        heatmaps = torch.sigmoid(raw_heatmaps)
        
        return heatmaps, keypoints_norm
