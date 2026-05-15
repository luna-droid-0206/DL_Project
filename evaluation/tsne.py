"""
evaluation/tsne.py
------------------
t-SNE embedding analysis (6.5).

Collects image and text embeddings from the model, then plots them in
a 2-D t-SNE space to demonstrate multimodal alignment.

  - Image embeddings: extracted from model.img_proj (bottleneck projection)
  - Text  embeddings: pre-computed CLIP vectors loaded from dataset
  - Points are coloured by modality (image=blue, text=orange)
    and patient ID to visualise inter-patient structure.

Key fix over original:
  Because every slice from the same patient shares the same pre-computed
  text embedding, the old code collapsed all text points to one t-SNE
  location causing all lines to fan from a single star.  The fixed version
  deduplicates text embeddings (one anchor per unique patient) so the
  t-SNE is more informative.
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

    # -------------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------------

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

            # L2-normalise text embeddings
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

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    def plot(self,
             embeddings:  dict,
             run_name:    str  = "run",
             perplexity:  int  = 30,
             n_iter:      int  = 1000) -> str:
        """
        Runs t-SNE on image + deduplicated-text embeddings and plots:
          - Modality view  : images as circles, one text star per patient
          - Patient view   : coloured by patient ID

        Key fix: because every slice from the same patient shares the same
        pre-computed text embedding, the original code fed N identical text
        vectors into t-SNE causing all text points to collapse to one spot
        and all lines to fan out from it.  We now deduplicate: one text
        anchor per unique patient, then connect each image slice to its
        patient's anchor.
        """
        img_e  = embeddings["img_embeds"]    # (N, D)
        txt_e  = embeddings["text_embeds"]   # (N, D)  -- repeated per patient
        pids   = embeddings["patient_ids"]

        N = len(img_e)

        # -- Deduplicate text embeddings (one per unique patient) -------------
        unique_pids = sorted(set(pids))
        pid2idx     = {p: i for i, p in enumerate(unique_pids)}
        cmap        = plt.cm.get_cmap("tab20", len(unique_pids))

        txt_dedup = {}           # patient_name -> first text embed seen
        for pid, te in zip(pids, txt_e):
            if pid not in txt_dedup:
                txt_dedup[pid] = te
        txt_pids = list(txt_dedup.keys())
        txt_vecs = np.array([txt_dedup[p] for p in txt_pids])  # (P, D)

        # Mean cosine similarity between each image and its paired text
        paired_cos = np.sum(
            img_e * np.array([txt_dedup[p] for p in pids]), axis=1
        )
        mean_cos = float(np.mean(paired_cos))

        # -- Concatenate for joint t-SNE --------------------------------------
        P          = len(txt_vecs)
        all_embeds = np.concatenate([img_e, txt_vecs], axis=0)  # (N+P, D)

        # Auto-scale perplexity: must be strictly < number of samples
        perplexity = min(perplexity, max(2, (N + P) // 3))

        # -- Run t-SNE --------------------------------------------------------
        tsne   = TSNE(n_components=2, perplexity=perplexity,
                      n_iter=n_iter, random_state=Config.SEED,
                      learning_rate="auto", init="pca")
        coords = tsne.fit_transform(all_embeds)   # (N+P, 2)

        img_coords = coords[:N]    # (N, 2)
        txt_coords = coords[N:]    # (P, 2)  one per patient

        # -- Figure -----------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0f1117")
        fig.suptitle(
            f"t-SNE Embedding Space: Image <-> Text Alignment"
            f"  |  Mean cosine sim = {mean_cos:.3f}",
            fontsize=13, color="#e0e0e0"
        )

        # Left: modality view
        ax = axes[0]
        ax.scatter(img_coords[:, 0], img_coords[:, 1],
                   s=40, alpha=0.75, marker="o",
                   c="#7c83fd", label="Image Embeddings", zorder=2)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
                   s=160, alpha=0.90, marker="*",
                   c="#ffb86c", label="Text Anchors (per patient)", zorder=3)

        # Draw lines from each image slice to its patient's text anchor
        for i, pid in enumerate(pids):
            p_idx = txt_pids.index(pid)
            ax.plot(
                [img_coords[i, 0], txt_coords[p_idx, 0]],
                [img_coords[i, 1], txt_coords[p_idx, 1]],
                color="#444466", lw=0.5, alpha=0.25, zorder=1
            )

        # Annotate each text anchor with a short patient label
        for j, pid in enumerate(txt_pids):
            label = pid.replace("BraTS20_Training_", "P")
            ax.annotate(label,
                        (txt_coords[j, 0], txt_coords[j, 1]),
                        fontsize=6, color="#ffb86c",
                        xytext=(4, 4), textcoords="offset points")

        ax.set_title("Modality View", color="#e0e0e0")
        ax.legend(framealpha=0.2, fontsize=9)
        ax.set_facecolor("#1a1d2e")
        ax.set_xlabel("t-SNE Dim 1", color="#a0a0b0")
        ax.set_ylabel("t-SNE Dim 2", color="#a0a0b0")
        ax.grid(True, alpha=0.3)

        # Right: patient view
        ax = axes[1]
        for i, pid in enumerate(pids):
            colour = cmap(pid2idx[pid])
            ax.scatter(img_coords[i, 0], img_coords[i, 1],
                       s=40, color=colour, alpha=0.75, marker="o", zorder=2)
        for j, pid in enumerate(txt_pids):
            colour = cmap(pid2idx[pid])
            ax.scatter(txt_coords[j, 0], txt_coords[j, 1],
                       s=160, color=colour, alpha=0.90, marker="*", zorder=3)

        # Legend: first 10 patients only
        for pid in unique_pids[:10]:
            label = pid.replace("BraTS20_Training_", "Patient ")
            ax.scatter([], [], color=cmap(pid2idx[pid]),
                       marker="o", s=40, label=label)
        ax.legend(framealpha=0.15, fontsize=7, ncol=2,
                  title="Patient", title_fontsize=8)
        ax.set_title("Patient View (circle=Image Slices  star=Text Anchor)",
                     color="#e0e0e0")
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
        print(f"  Saved t-SNE plot: {path}")
        return path
