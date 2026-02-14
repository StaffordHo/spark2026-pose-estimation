"""
CNN Backbones for processing event voxel grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import models


class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class VoxelCNN(nn.Module):
    """
    3D CNN backbone for processing spatio-temporal voxel grids.
    
    Input shape: (B, C, D, H, W) where D is temporal depth
    Output: Feature vector of size `feature_dim`
    """
    
    def __init__(
        self,
        in_channels: int = 10,  # 2 * num_bins (ON/OFF per bin)
        feature_dim: int = 512,
        base_channels: int = 32
    ):
        super().__init__()
        
        # Reshape input from (B, C, H, W) to (B, 1, C, H, W) for 3D conv
        self.in_channels = in_channels
        
        # 3D CNN layers - treat channel dim as temporal dim
        self.conv1 = ConvBlock3D(1, base_channels, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = ConvBlock3D(base_channels, base_channels * 2, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = ConvBlock3D(base_channels * 2, base_channels * 4, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4 = ConvBlock3D(base_channels * 4, base_channels * 8, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv5 = ConvBlock3D(base_channels * 8, base_channels * 8, kernel_size=3, stride=1)
        
        # Global average pooling and FC
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, feature_dim)
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Voxel grid of shape (B, C, H, W)
            
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        # Add depth dimension: (B, C, H, W) -> (B, 1, C, H, W)
        x = x.unsqueeze(1)
        
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.conv5(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ConvBlock2D(nn.Module):
    """2D Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style backbone."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class EventResNet(nn.Module):
    """
    ResNet-style 2D CNN backbone for event data.
    Treats voxel grid channels as input channels directly.
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        feature_dim: int = 512,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.conv1 = ConvBlock2D(in_channels, base_channels, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, feature_dim)
        
        self.feature_dim = feature_dim
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Voxel grid of shape (B, C, H, W)
            
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class PretrainedEfficientNet(nn.Module):
    """
    Pretrained EfficientNet-B0 backbone adapted for event voxel grid input.
    
    Loads ImageNet-pretrained weights and adapts the first conv layer
    from 3 channels to `in_channels` (default 10) for event voxel grid input.
    
    Features:
    - Smart weight initialization: averages pretrained 3ch weights across N channels
    - Backbone freezing for staged fine-tuning
    - Outputs feature vector of size `feature_dim`
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base_model = models.efficientnet_b0(weights=weights)
            print("Loaded ImageNet-pretrained EfficientNet-B0 weights", flush=True)
        else:
            base_model = models.efficientnet_b0(weights=None)
        
        # Adapt first conv layer for N-channel input
        original_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        if pretrained:
            # Smart initialization: average pretrained 3ch weights across N channels
            with torch.no_grad():
                # Original weights shape: (32, 3, 3, 3)
                # Repeat and average to get (32, in_channels, 3, 3)
                pretrained_weight = original_conv.weight.data
                avg_weight = pretrained_weight.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)
                new_conv.weight.data = avg_weight.repeat(1, in_channels, 1, 1)
        
        base_model.features[0][0] = new_conv
        
        # Extract features (remove classifier)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # EfficientNet-B0 outputs 1280 features
        self.fc = nn.Linear(1280, feature_dim)
        
        self.feature_dim = feature_dim
        self._frozen = freeze_backbone
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all backbone layers except the first conv and final FC."""
        for name, param in self.features.named_parameters():
            if not name.startswith("0.0."):  # Don't freeze our new first conv
                param.requires_grad = False
        self._frozen = True
        print("Backbone frozen (except conv1 and FC)", flush=True)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        for param in self.features.parameters():
            param.requires_grad = True
        self._frozen = False
        print("Backbone unfrozen", flush=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Voxel grid of shape (B, C, H, W)
            
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
