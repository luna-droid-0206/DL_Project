"""
evaluation/tsne.py
──────────────────
t-SNE embedding analysis (6.5).

Collects image and text embeddings from the model, then plots them in
a 2-D t-SNE space to demonstrate multimodal alignment.

  - Image embeddings: extracted from model.img_proj (bottleneck projection)
  - Text  embeddings: pre-computed CLIP vectors loaded from dataset
  - Points are coloured by modality (image=blue, text=orange)
    and patient ID to visualise inter-patient structure.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from config import Config


class TSNEVisualizer:
    """
    Extracts embeddings from a DataLoader and produces t-SNE plots.
    """

    def __init__(self, model: nn.Module, device: str = None):
        self.model  = model
        self.device = device or Config.DEVICE

    # ── Extract ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_embeddings(self,
                           loader:      DataLoader,
                           n_samples:   int = Config.TSNE_NUM_SAMPLES
                           ) -> dict:
        """
        Returns:
            {
              "img_embeds"  : np.ndarray (N, D),
              "text_embeds" : np.ndarray (N, D),
              "patient_ids" : list[str],
            }
        """
        self.model.eval()
        img_embeds   = []
        text_embeds  = []
        patient_ids  = []

        collected = 0
        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
            if collected >= n_samples:
                break

            image      = batch["image"].to(self.device)
            text_embed = batch["text_embed"].to(self.device)
            patients   = batch["patient_name"]

            out = self.model(image, text_embed)
            ie  = out["img_embed"].cpu().numpy()         # (B, D)
            te  = text_embed.cpu().numpy()               # (B, D)

            # L2-normalise text
            norms = np.linalg.norm(te, axis=1, keepdims=True) + 1e-8
            te    = te / norms

            for i in range(len(ie)):
                if collected >= n_samples:
                    break
                img_embeds.append(ie[i])
                text_embeds.append(te[i])
                patient_ids.append(patients[i])
                collected += 1

        return {
            "img_embeds":  np.array(img_embeds),
            "text_embeds": np.array(text_embeds),
            "patient_ids": patient_ids,
        }

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self,
             embeddings:  dict,
             run_name:    str  = "run",
             perplexity:  int  = 30,
             n_iter:      int  = 1000) -> str:
        """
        Runs t-SNE on concatenated [image_embeds; text_embeds] and plots:
          - Modality view (image=circle, text=star)
          - Patient view  (coloured by patient ID)

        Args:
            embeddings: output of extract_embeddings()
        Returns:
            save_path (str)
        """
        img_e  = embeddings["img_embeds"]    # (N, D)
        txt_e  = embeddings["text_embeds"]   # (N, D)
        pids   = embeddings["patient_ids"]

        N = len(img_e)
        all_embeds = np.concatenate([img_e, txt_e], axis=0)  # (2N, D)
        labels     = (["Image"] * N) + (["Text"] * N)

        # ── Run t-SNE ─────────────────────────────────────────────────────────
        tsne   = TSNE(n_components=2, perplexity=perplexity,
                      n_iter=n_iter, random_state=Config.SEED,
                      learning_rate="auto", init="pca")
        coords = tsne.fit_transform(all_embeds)              # (2N, 2)

        img_coords  = coords[:N]
        txt_coords  = coords[N:]

        # Unique patients for colourmap
        unique_pids = sorted(set(pids))
        pid2idx     = {p: i for i, p in enumerate(unique_pids)}
        cmap        = plt.cm.get_cmap("tab20", len(unique_pids))

        # ── Figure: two subplots ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                                 facecolor="#0f1117")
        fig.suptitle("t-SNE Embedding Space: Image ↔ Text Alignment",
                     fontsize=14, color="#e0e0e0")

        # ── Left: modality view ───────────────────────────────────────────────
        ax = axes[0]
        ax.scatter(img_coords[:, 0], img_coords[:, 1],
                   s=40, alpha=0.75, marker="o",
                   c="#7c83fd", label="Image Embeddings", zorder=2)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
                   s=90, alpha=0.85, marker="*",
                   c="#ffb86c", label="Text Embeddings",  zorder=3)

        # Draw lines connecting paired image-text points
        for i in range(N):
            ax.plot(
                [img_coords[i, 0], txt_coords[i, 0]],
                [img_coords[i, 1], txt_coords[i, 1]],
                color="#444466", lw=0.5, alpha=0.3, zorder=1
            )

        ax.set_title("Modality View", color="#e0e0e0")
        ax.legend(framealpha=0.2, fontsize=9)
        ax.set_facecolor("#1a1d2e")
        ax.set_xlabel("t-SNE Dim 1", color="#a0a0b0")
        ax.set_ylabel("t-SNE Dim 2", color="#a0a0b0")
        ax.grid(True, alpha=0.3)

        # ── Right: patient view ───────────────────────────────────────────────
        ax = axes[1]
        for i, pid in enumerate(pids):
            colour = cmap(pid2idx[pid])
            ax.scatter(img_coords[i, 0], img_coords[i, 1],
                       s=40, color=colour, alpha=0.75, marker="o", zorder=2)
            ax.scatter(txt_coords[i, 0], txt_coords[i, 1],
                       s=90, color=colour, alpha=0.85, marker="*", zorder=3)

        # Legend for first 10 patients only (avoid clutter)
        for pid in unique_pids[:10]:
            ax.scatter([], [], color=cmap(pid2idx[pid]),
                       marker="o", s=40, label=pid)
        ax.legend(framealpha=0.15, fontsize=7, ncol=2,
                  title="Patient", title_fontsize=8)
        ax.set_title("Patient View (○=Image  ★=Text)", color="#e0e0e0")
        ax.set_facecolor("#1a1d2e")
        ax.set_xlabel("t-SNE Dim 1", color="#a0a0b0")
        ax.set_ylabel("t-SNE Dim 2", color="#a0a0b0")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        path = os.path.join(Config.PLOTS_DIR, "tsne",
                            f"{run_name}_tsne.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ t-SNE plot saved: {path}")
        return path
