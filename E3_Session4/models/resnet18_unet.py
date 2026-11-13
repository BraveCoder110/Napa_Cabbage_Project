"""ResNet18-encoder U-Net for binary semantic segmentation."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    """Two convolution-normalization-ReLU operations."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResNet18UNet(nn.Module):
    """Output raw foreground logits; apply sigmoid only for predictions."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64x64
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1  # 32x32
        self.enc2 = backbone.layer2  # 16x16
        self.enc3 = backbone.layer3  # 8x8
        self.enc4 = backbone.layer4  # 4x4
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 32)
        self.final_up = nn.Sequential(ConvBlock(32, 16), nn.Conv2d(16, 1, 1))
        self.pretrained_encoder = pretrained

    def forward(self, x):
        input_size = x.shape[-2:]
        stem = self.stem(x)
        e1 = self.enc1(self.pool(stem))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        y = self.up3(e4, e3)
        y = self.up2(y, e2)
        y = self.up1(y, e1)
        y = self.up0(y, stem)
        y = nn.functional.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        return self.final_up(y)
