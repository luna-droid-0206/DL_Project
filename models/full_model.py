"""
models/full_model.py
────────────────────
End-to-end Multimodal Brain Tumour Segmentation network.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  MRI slice (B,1,H,W)                                │
  │       │                                             │
  │  UNetEncoder ──► bottleneck (B,1024,H/16,W/16)      │
  │       │                    │                        │
  │  skip connections    CrossAttentionFusion ◄─ text   │
  │       │                    │                        │
  │  UNetDecoder ◄─────────────┘                        │
  │       │                                             │
  │  Sigmoid ──► segmentation mask (B,1,H,W)            │
  └─────────────────────────────────────────────────────┘

Setting use_text=False bypasses CrossAttentionFusion for ablation.
"""

import torch
import torch.nn as nn

from models.decoder import UNetEncoder, UNetDecoder
from models.fusion  import CrossAttentionFusion
from config import Config


class MultimodalSegNet(nn.Module):
    """
    Multimodal segmentation network combining UNet with cross-attention
    text guidance via pre-computed CLIP embeddings.

    Args:
        use_text (bool): Enable cross-attention fusion with text embeddings.
                         Set False for image-only ablation.
    """

    def __init__(self, use_text: bool = True):
        super().__init__()

        self.use_text = use_text

        # ── Backbone ──────────────────────────────────────────────────────────
        self.encoder = UNetEncoder(in_ch=Config.IN_CHANNELS)
        self.decoder = UNetDecoder(out_ch=Config.OUT_CHANNELS)

        # ── Text Fusion ───────────────────────────────────────────────────────
        if use_text:
            self.fusion = CrossAttentionFusion(
                img_channels = Config.BOTTLENECK_CH,
                text_dim     = Config.TEXT_EMBED_DIM,
                num_heads    = Config.NUM_ATTENTION_HEADS,
            )

        # ── Projection head for embedding extraction (t-SNE / CLIP align) ────
        # Projects bottleneck global avg-pool → TEXT_EMBED_DIM for alignment
        self.img_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(Config.BOTTLENECK_CH, Config.TEXT_EMBED_DIM),
            nn.LayerNorm(Config.TEXT_EMBED_DIM),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                image:      torch.Tensor,
                text_embed: torch.Tensor | None = None) -> dict:
        """
        Args:
            image      : (B, 1, H, W)   — normalised MRI slice
            text_embed : (B, TEXT_DIM)  — pre-computed CLIP text embedding

        Returns dict with keys:
            "mask_logits"  : (B, 1, H, W)   raw logits (apply sigmoid)
            "mask_pred"    : (B, 1, H, W)   sigmoid probabilities
            "img_embed"    : (B, TEXT_DIM)  image embedding for t-SNE / loss
            "bottleneck"   : (B, C, h, w)   for Grad-CAM
        """
        # Encode
        bottleneck, skips = self.encoder(image)

        # Text fusion (optional)
        if self.use_text and text_embed is not None:
            fused = self.fusion(bottleneck, text_embed)
        else:
            fused = bottleneck

        # Decode
        logits = self.decoder(fused, skips)

        # Image embedding (for t-SNE and contrastive alignment)
        img_embed = self.img_proj(bottleneck)
        img_embed = nn.functional.normalize(img_embed, dim=-1)

        return {
            "mask_logits": logits,
            "mask_pred":   torch.sigmoid(logits),
            "img_embed":   img_embed,
            "bottleneck":  fused,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def predict(self,
                image:      torch.Tensor,
                text_embed: torch.Tensor | None = None,
                threshold:  float = 0.5) -> torch.Tensor:
        """Inference helper — returns binary mask (B, 1, H, W)."""
        with torch.no_grad():
            out = self.forward(image, text_embed)
        return (out["mask_pred"] >= threshold).float()

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {train:,}")
        return train
