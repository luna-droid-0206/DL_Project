"""
models/fusion.py
────────────────
Cross-Attention Fusion module.

Fuses spatial UNet bottleneck features with a text embedding vector
using multi-head cross-attention:

  Query   : flattened spatial image features  (B, HW, C)
  Key/Val : projected text embedding          (B,  1, C)

The attended features are reshaped back to (B, C, H, W) and added
to the original bottleneck via a residual connection.
"""

import torch
import torch.nn as nn
from config import Config


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention between image spatial features and text.

    Args:
        img_channels  (int): Channels in the UNet bottleneck feature map.
        text_dim      (int): Dimension of the incoming text embedding.
        num_heads     (int): Number of attention heads.
        dropout       (float): Attention dropout probability.
    """

    def __init__(self,
                 img_channels: int = Config.BOTTLENECK_CH,
                 text_dim:     int = Config.TEXT_EMBED_DIM,
                 num_heads:    int = Config.NUM_ATTENTION_HEADS,
                 dropout:      float = 0.1):
        super().__init__()

        self.img_channels = img_channels

        # Project text embedding to match image feature channels
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, img_channels),
            nn.LayerNorm(img_channels),
        )

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = img_channels,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.LayerNorm(img_channels),
            nn.Linear(img_channels, img_channels * 2),
            nn.GELU(),
            nn.Linear(img_channels * 2, img_channels),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(img_channels)

    def forward(self,
                img_feat:   torch.Tensor,
                text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat   : (B, C, H, W)  — UNet bottleneck features
            text_embed : (B, D)        — pre-computed text embedding

        Returns:
            fused      : (B, C, H, W)  — text-guided spatial features
        """
        B, C, H, W = img_feat.shape

        # Flatten spatial dims → (B, HW, C)
        q = img_feat.view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)

        # Project text → (B, 1, C)  (single "token" for text)
        kv = self.text_proj(text_embed).unsqueeze(1)       # (B, 1, C)

        # Cross-attention: image queries attend to text key/value
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)

        # Residual + LayerNorm
        q = self.norm(q + attn_out)

        # Feed-forward
        q = q + self.ffn(q)

        # Reshape back to (B, C, H, W)
        fused = q.permute(0, 2, 1).view(B, C, H, W)
        return fused
