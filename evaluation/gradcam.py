"""
evaluation/gradcam.py
─────────────────────
Grad-CAM — visualises model focus regions (6.3).

Hooks into the UNet bottleneck and computes:
  cam = ReLU( Σ_c  α_c · A_c )
  α_c = global_avg_pool( ∂L/∂A_c )
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import Config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for MultimodalSegNet.
    Target layer: encoder.down4.conv (UNet bottleneck).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model  = model
        self.device = next(model.parameters()).device
        self._grads = None
        self._acts  = None

        if target_layer is None:
            target_layer = model.encoder.down4.conv.conv[-3]
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self._acts = out.detach()
        def bwd(module, gin, gout):
            self._grads = gout[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def compute(self, image: torch.Tensor,
                text_embed: torch.Tensor = None) -> np.ndarray:
        """Returns (H, W) Grad-CAM heatmap normalised to [0, 1]."""
        self.model.eval()
        image = image.to(self.device).requires_grad_(True)
        if text_embed is not None:
            text_embed = text_embed.to(self.device)

        out   = self.model(image, text_embed)
        score = out["mask_logits"].mean()
        self.model.zero_grad()
        score.backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        H, W = image.shape[-2], image.shape[-1]
        cam  = F.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        return cam

    def visualize(self, image: np.ndarray, cam: np.ndarray,
                  gt_mask: np.ndarray = None,
                  save_path: str = None, title: str = "Grad-CAM") -> str:
        """Side-by-side: Input | Heatmap | Overlay | GT mask."""
        ncols = 4 if gt_mask is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5),
                                 facecolor="#0f1117")
        fig.suptitle(title, fontsize=13, color="#e0e0e0")

        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input MRI", color="#e0e0e0")
        axes[1].imshow(cam, cmap="inferno")
        axes[1].set_title("Grad-CAM", color="#e0e0e0")

        rgb     = np.stack([image, image, image], axis=-1)
        hm_rgb  = cm.inferno(cam)[..., :3]
        overlay = np.clip(0.55 * rgb + 0.45 * hm_rgb, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", color="#e0e0e0")

        if gt_mask is not None:
            axes[3].imshow(gt_mask, cmap="Reds", alpha=0.85)
            axes[3].set_title("Ground Truth", color="#e0e0e0")

        for ax in axes:
            ax.axis("off")
            ax.set_facecolor("#0f1117")

        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(Config.PLOTS_DIR,
                                     "attention_maps", "gradcam.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Grad-CAM saved: {save_path}")
        return save_path

    def batch_visualize(self, images: np.ndarray, cams: np.ndarray,
                        gt_masks: np.ndarray = None,
                        n: int = 6, run_name: str = "run") -> str:
        """Grid of n Grad-CAM attention maps."""
        n  = min(n, len(images))
        nc = 4 if gt_masks is not None else 3
        fig, axes = plt.subplots(n, nc, figsize=(5 * nc, 5 * n),
                                 facecolor="#0f1117")
        fig.suptitle("Grad-CAM Attention Maps", fontsize=14, color="#e0e0e0")

        for row in range(n):
            img = images[row].squeeze()
            cam = cams[row]
            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 1].imshow(cam, cmap="inferno")
            hm_rgb  = cm.inferno(cam)[..., :3]
            overlay = np.clip(0.55 * np.stack([img]*3, -1) + 0.45 * hm_rgb, 0, 1)
            axes[row, 2].imshow(overlay)
            if gt_masks is not None:
                axes[row, 3].imshow(gt_masks[row].squeeze(),
                                    cmap="Reds", alpha=0.85)
            for col in range(nc):
                axes[row, col].axis("off")

        fig.tight_layout()
        path = os.path.join(Config.PLOTS_DIR, "attention_maps",
                            f"{run_name}_gradcam_grid.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Grad-CAM grid: {path}")
        return path
