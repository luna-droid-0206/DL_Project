"""
models/decoder.py
─────────────────
UNet encoder and decoder blocks.

Encoder  : 4 down-sampling stages → bottleneck  (with skip connections)
Decoder  : 4 up-sampling stages   (uses skip connections from encoder)
Seg head : 1×1 Conv → sigmoid     (binary mask)

Channel progression:
  in(1) → 64 → 128 → 256 → 512 → bottleneck(1024)
                                       ↓ (fusion)
           64 ← 128 ← 256 ← 512 ←────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ── Building Blocks ────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    """
    Conv → BN → ReLU → Conv → BN → ReLU
    Optionally applies a residual projection if in_ch != out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class _Down(nn.Module):
    """MaxPool2d(2) → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """
    Bilinear upsample × 2 → concatenate skip → DoubleConv.
    Skip connection doubles the input channels before the conv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self,
                x:    torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ (edge case for non-power-of-2 inputs)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2,
                       dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ── Encoder ────────────────────────────────────────────────────────────────────

class UNetEncoder(nn.Module):
    """
    4-stage UNet encoder.

    Returns:
        bottleneck : (B, 1024, H/16, W/16)
        skips      : list of 4 tensors at [64, 128, 256, 512] channels
    """

    def __init__(self, in_ch: int = Config.IN_CHANNELS):
        super().__init__()
        f = Config.UNET_FEATURES    # [64, 128, 256, 512]

        self.stem  = _DoubleConv(in_ch, f[0])        # (B,  64, H,    W)
        self.down1 = _Down(f[0], f[1])               # (B, 128, H/2,  W/2)
        self.down2 = _Down(f[1], f[2])               # (B, 256, H/4,  W/4)
        self.down3 = _Down(f[2], f[3])               # (B, 512, H/8,  W/8)
        self.down4 = _Down(f[3], Config.BOTTLENECK_CH)  # (B,1024, H/16, W/16)

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)    # skip1
        s1 = self.down1(s0)  # skip2
        s2 = self.down2(s1)  # skip3
        s3 = self.down3(s2)  # skip4
        bn = self.down4(s3)  # bottleneck

        return bn, [s0, s1, s2, s3]


# ── Decoder ────────────────────────────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """
    4-stage UNet decoder with skip connections from the encoder.

    Input:
        bottleneck : (B, 1024, H/16, W/16)  (possibly text-fused)
        skips      : [s0(64), s1(128), s2(256), s3(512)]

    Output:
        logits     : (B, 1, H, W)  (raw, before sigmoid)
    """

    def __init__(self, out_ch: int = Config.OUT_CHANNELS):
        super().__init__()
        f  = Config.UNET_FEATURES       # [64, 128, 256, 512]
        bn = Config.BOTTLENECK_CH       # 1024

        self.up1 = _Up(bn,    f[3], f[3])   # 1024+512 → 512
        self.up2 = _Up(f[3],  f[2], f[2])   # 512+256  → 256
        self.up3 = _Up(f[2],  f[1], f[1])   # 256+128  → 128
        self.up4 = _Up(f[1],  f[0], f[0])   # 128+64   → 64

        self.head = nn.Conv2d(f[0], out_ch, kernel_size=1)

    def forward(self,
                bottleneck: torch.Tensor,
                skips:      list) -> torch.Tensor:
        s0, s1, s2, s3 = skips

        x = self.up1(bottleneck, s3)   # → (B, 512, H/8,  W/8)
        x = self.up2(x, s2)            # → (B, 256, H/4,  W/4)
        x = self.up3(x, s1)            # → (B, 128, H/2,  W/2)
        x = self.up4(x, s0)            # → (B,  64, H,    W)

        return self.head(x)             # → (B,   1, H,    W)
