"""Lightweight U-Net for binary vegetable segmentation.

The network produces raw logits. Apply sigmoid only during inference.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Down(nn.Module):
    """Halve the spatial dimensions and increase feature channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Up(nn.Module):
    """Upsample and concatenate the encoder skip features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class LightweightUNet(nn.Module):
    """Three-level U-Net with a configurable starting channel count."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = Down(c, 2 * c)
        self.enc3 = Down(2 * c, 4 * c)
        self.bottleneck = Down(4 * c, 8 * c)
        self.dec3 = Up(8 * c, 4 * c, 4 * c)
        self.dec2 = Up(4 * c, 2 * c, 2 * c)
        self.dec1 = Up(2 * c, c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bottleneck(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)
