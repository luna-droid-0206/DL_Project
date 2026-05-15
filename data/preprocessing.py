"""
data/preprocessing.py
─────────────────────
Image and text preprocessing utilities for BraTS 2020.
"""

import random
import numpy as np
import cv2
import torch


class Preprocessor:
    """
    Handles all preprocessing for images, masks, and text embeddings.

    Args:
        image_size (int): Target spatial resolution for MRI slices.
        augment    (bool): Whether to apply random augmentations.
    """

    def __init__(self, image_size: int = 256, augment: bool = False):
        self.image_size = image_size
        self.augment    = augment

    # ── Image ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        """
        Robustly collapse any ndarray to (H, W).

        Handles shapes:
          (H, W)       -> unchanged
          (1, H, W)    -> squeeze axis 0
          (H, W, 1)    -> squeeze axis 2
          (C, H, W)    -> take arr[0]        (C is the smallest axis)
          (H, W, C)    -> take arr[..., 0]   (C is the smallest axis)
          (N, H, W, C) -> take first sample, first channel

        The channel axis is identified as the axis with the MINIMUM size,
        since spatial dimensions (H, W) are always much larger than the
        channel count (1 or 4).  This avoids the fragile shape[0] < shape[-1]
        heuristic which can misfire on small or non-square slices and cause
        image-mask spatial misalignment.
        """
        arr = np.squeeze(arr)          # remove ALL size-1 dims first
        if arr.ndim == 2:
            return arr                 # already (H, W)
        if arr.ndim == 3:
            # Identify channel axis as the one with the smallest size
            ch_axis = int(np.argmin(arr.shape))
            if ch_axis == 0:
                return arr[0]          # (C, H, W) -> first channel
            else:
                return arr[..., 0]     # (H, W, C) -> first channel
        # Fallback for 4-D+: take [0] along axis 0 repeatedly
        while arr.ndim > 2:
            arr = arr[0]
        return arr

    def load_image(self, path: str) -> np.ndarray:
        """
        Load a .npy MRI slice.
        - Converts to float32
        - Min-max normalises to [0, 1]
        - Resizes to (image_size, image_size)

        Returns:
            np.ndarray of shape (H, W), dtype float32
        """
        img = np.load(path).astype(np.float32)
        img = self._to_2d(img)

        # Min-max normalisation per slice
        lo, hi = img.min(), img.max()
        if hi - lo > 1e-6:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)

        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)
        return img  # (H, W)

    def load_mask(self, path: str) -> np.ndarray:
        """
        Load a .npy segmentation mask.
        - Binarises (any label > 0 → 1, background → 0)
        - Resizes with nearest-neighbour to preserve label values

        Returns:
            np.ndarray of shape (H, W), dtype float32, values in {0, 1}
        """
        mask = np.load(path).astype(np.float32)
        mask = self._to_2d(mask)

        mask = (mask > 0).astype(np.float32)
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        return mask  # (H, W)

    def apply_augmentation(self,
                           img: np.ndarray,
                           mask: np.ndarray):
        """
        Applies identical random spatial augmentations to image and mask.
        Photometric jitter is applied to the image only.

        Returns:
            img, mask — both np.ndarray (H, W)
        """
        # Horizontal flip
        if random.random() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Vertical flip
        if random.random() > 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # 90° rotation (0 / 90 / 180 / 270)
        k = random.choice([0, 1, 2, 3])
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()

        # Brightness / contrast jitter (image only)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast scale
            beta  = random.uniform(-0.1, 0.1)  # brightness shift
            img   = np.clip(alpha * img + beta, 0.0, 1.0)

        return img, mask

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """(H, W) → (1, H, W) float tensor."""
        return torch.from_numpy(img.copy()).unsqueeze(0).float()

    # ── Text ───────────────────────────────────────────────────────────────────

    def load_text_embedding(self, path: str) -> np.ndarray:
        """
        Load a pre-computed CLIP text embedding from a .npy file.

        Handles all common shapes from TextBraTS:
          (D,)         → single vector, use as-is
          (1, D)       → single token, squeeze
          (N, D)       → token-level embeddings, mean-pool → (D,)

        Returns:
            np.ndarray of shape (TEXT_EMBED_DIM,), dtype float32, L2-normalised
        """
        embed = np.load(path).astype(np.float32)

        if embed.ndim == 1:
            # Already (D,) — use as-is
            pass
        elif embed.ndim == 2:
            # (N_tokens, D) or (1, D) — mean-pool across token dimension
            embed = embed.mean(axis=0)   # → (D,)
        else:
            # Higher-dim fallback: squeeze then mean-pool
            embed = embed.squeeze()
            if embed.ndim == 2:
                embed = embed.mean(axis=0)
            elif embed.ndim > 2:
                embed = embed.reshape(-1, embed.shape[-1]).mean(axis=0)

        # L2-normalise
        norm = np.linalg.norm(embed)
        if norm > 1e-6:
            embed = embed / norm

        return embed.astype(np.float32)  # (TEXT_EMBED_DIM,)

    def load_text_report(self, path: str) -> str:
        """Load raw radiology text from a .txt file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def text_embedding_to_tensor(self, embed: np.ndarray) -> torch.Tensor:
        """(D,) numpy → (D,) float tensor."""
        return torch.from_numpy(embed.copy()).float()
