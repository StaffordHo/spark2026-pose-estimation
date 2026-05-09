"""
Feature Pyramid Network (FPN) for multi-scale feature extraction.

Extracts intermediate features from EfficientNet-B0 and builds a top-down
feature pyramid. Outputs a fused feature map at 1/4 resolution suitable for
accurate keypoint heatmap prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FPN(nn.Module):
    """
    Feature Pyramid Network on top of EfficientNet-B0.
    
    Extracts features at 4 scales (1/4, 1/8, 1/16, 1/32) and fuses them
    into a single high-resolution feature map via top-down pathway.
    
    EfficientNet-B0 feature stages (at 256x256 input):
        Stage 2: stride 4,  24 channels,  64x64
        Stage 4: stride 8,  40 channels,  32x32
        Stage 6: stride 16, 112 channels, 16x16
        Stage 8: stride 32, 320 channels, 8x8
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        fpn_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base_model = models.efficientnet_b0(weights=weights)
            print("Loaded ImageNet-pretrained EfficientNet-B0 for FPN", flush=True)
        else:
            base_model = models.efficientnet_b0(weights=None)
        
        # Adapt first conv for N-channel input
        original_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                pretrained_weight = original_conv.weight.data
                avg_weight = pretrained_weight.mean(dim=1, keepdim=True)
                new_conv.weight.data = avg_weight.repeat(1, in_channels, 1, 1)
        
        base_model.features[0][0] = new_conv
        
        # Split backbone into stages for multi-scale feature extraction
        # EfficientNet-B0 features[0..8], verified dimensions at 256x256 input:
        #   After stage 0: 32ch, 128x128      After stage 5: 112ch, 16x16
        #   After stage 1: 16ch, 128x128      After stage 6: 192ch, 8x8
        #   After stage 2: 24ch, 64x64        After stage 7: 320ch, 8x8
        #   After stage 3: 40ch, 32x32        After stage 8: 1280ch, 8x8 (expansion)
        #   After stage 4: 80ch, 16x16
        # We pick 4 scale levels for FPN:
        self.stage1 = nn.Sequential(*base_model.features[0:3])   # → 24ch, 64x64
        self.stage2 = nn.Sequential(*base_model.features[3:4])   # → 40ch, 32x32
        self.stage3 = nn.Sequential(*base_model.features[4:6])   # → 112ch, 16x16
        self.stage4 = nn.Sequential(*base_model.features[6:8])   # → 320ch, 8x8
        
        # Channel sizes from EfficientNet-B0
        c2, c3, c4, c5 = 24, 40, 112, 320
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral4 = nn.Conv2d(c5, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral2 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral1 = nn.Conv2d(c2, fpn_channels, 1)
        
        # Smooth convolutions (reduce aliasing after upsampling)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        
        self.fpn_channels = fpn_channels
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone stages except the adapted first conv."""
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in stage.parameters():
                param.requires_grad = False
        # Unfreeze the adapted first conv layer
        for param in self.stage1[0][0].parameters():
            param.requires_grad = True
        print("FPN backbone frozen (except conv1)", flush=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input voxel grid
        
        Returns:
            p2: (B, fpn_channels, H/4, W/4) highest resolution FPN output
        """
        # Bottom-up pathway
        c2 = self.stage1(x)   # (B, 24, H/4, W/4)
        c3 = self.stage2(c2)  # (B, 40, H/8, W/8)
        c4 = self.stage3(c3)  # (B, 112, H/16, W/16)
        c5 = self.stage4(c4)  # (B, 320, H/32, W/32)
        
        # Top-down pathway with lateral connections
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        
        # Smooth
        p2 = self.smooth1(p2)
        
        return p2  # (B, fpn_channels, H/4, W/4)
