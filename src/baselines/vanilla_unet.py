"""Vanilla UNet — Ronneberger et al., MICCAI 2015.

Faithful re-implementation of the original architecture: stacked
double-conv (3x3 conv -> BN -> ReLU) encoder, MaxPool down, transposed
conv up, channel-concat skips. Channel schedule 64 -> 128 -> 256 -> 512
-> 1024 at the bottleneck. No pretraining anywhere.

Exposes `self.final` (1x1 Conv2d producing `num_classes` channels) so the
`_BaselineWrapper` in `__init__.py` can hook it for multi-class logits.
Forward returns the raw logits (no sigmoid) — the wrapper handles that
via `sigmoid_wrapped=False`.
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaUNet(nn.Module):
    """Original UNet (Ronneberger 2015) with `num_classes` output channels."""

    def __init__(self, num_classes: int = 1, input_channels: int = 1,
                 base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.enc1 = _DoubleConv(input_channels, c)        # 64
        self.enc2 = _DoubleConv(c, c * 2)                 # 128
        self.enc3 = _DoubleConv(c * 2, c * 4)             # 256
        self.enc4 = _DoubleConv(c * 4, c * 8)             # 512
        self.bottleneck = _DoubleConv(c * 8, c * 16)      # 1024

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c * 2, c)

        self.final = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
