"""
models/clip_encoder.py
──────────────────────
CLIP-based image encoder (ViT-B/32).

During training, text embeddings are loaded from pre-computed .npy files,
so only the image encoder is used online.  The text encoder is included
for re-encoding raw text and for ROUGE/BLEU evaluation utilities.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from config import Config


class CLIPImageEncoder(nn.Module):
    """
    Wraps the CLIP vision encoder (ViT-B/32).

    Returns the patch-level token embeddings reshaped to a spatial grid
    so they can participate in the cross-attention fusion at the UNet
    bottleneck.

    Input:  (B, 1, H, W) grayscale — replicated to (B, 3, 224, 224)
    Output: (B, 512)  global CLS embedding (used for t-SNE / ablation)
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, freeze: bool = True):
        super().__init__()
        clip        = CLIPModel.from_pretrained(self.MODEL_ID)
        self.vision = clip.vision_model
        self.proj   = clip.visual_projection   # (768 → 512)

        if freeze:
            for p in self.vision.parameters():
                p.requires_grad_(False)
            for p in self.proj.parameters():
                p.requires_grad_(False)

        self.resize = _GrayscaleToRGB(target_size=Config.CLIP_IMAGE_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            embed: (B, 512)  — L2-normalised global image embedding
        """
        x_rgb  = self.resize(x)                       # (B, 3, 224, 224)
        out    = self.vision(pixel_values=x_rgb)
        cls    = out.pooler_output                     # (B, 768)
        embed  = self.proj(cls)                        # (B, 512)
        embed  = nn.functional.normalize(embed, dim=-1)
        return embed


class CLIPTextEncoder(nn.Module):
    """
    Wraps the CLIP text encoder.
    Only needed when re-encoding raw text at evaluation time.

    Input:  tokenised text (from CLIPProcessor)
    Output: (B, 512)  L2-normalised text embedding
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, freeze: bool = True):
        super().__init__()
        clip      = CLIPModel.from_pretrained(self.MODEL_ID)
        self.text = clip.text_model
        self.proj = clip.text_projection   # (512 → 512)

        if freeze:
            for p in self.text.parameters():
                p.requires_grad_(False)
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out   = self.text(input_ids=input_ids,
                          attention_mask=attention_mask)
        embed = self.proj(out.pooler_output)
        embed = nn.functional.normalize(embed, dim=-1)
        return embed


# ── Helper ─────────────────────────────────────────────────────────────────────

class _GrayscaleToRGB(nn.Module):
    """Upsample grayscale (B, 1, H, W) → (B, 3, target, target)."""

    def __init__(self, target_size: int = 224):
        super().__init__()
        self.target = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)                 # (B, 3, H, W)
        x = nn.functional.interpolate(
            x,
            size=(self.target, self.target),
            mode="bilinear",
            align_corners=False
        )
        return x
