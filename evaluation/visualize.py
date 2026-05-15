"""
evaluation/visualize.py
───────────────────────
All required project visualizations:
  6.1  Training curves  (Loss, Dice, IoU vs Epoch)
  6.2  Segmentation results  (input / GT / pred / overlay)
  6.4  Ablation study  (with-text vs without-text)
  6.6  Error analysis  (Dice distribution + failure cases)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from config import Config

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3d52",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

PALETTE = {
    "train":    "#7c83fd",
    "val":      "#fd7c83",
    "with":     "#50fa7b",
    "without":  "#ffb86c",
    "tumor":    "#ff5555",
    "overlay":  "#f1fa8c",
}


class Visualizer:

    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or Config.PLOTS_DIR

    def _savefig(self, fig, subdir: str, fname: str):
        path = os.path.join(self.save_dir, subdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Saved: {path}")
        return path

    # ── 6.1 Training Curves ───────────────────────────────────────────────────

    def plot_training_curves(self, history: dict, run_name: str = "run"):
        """
        Three side-by-side plots: Loss | Dice | IoU vs Epoch.
        """
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Performance", fontsize=15, y=1.02,
                     color="#e0e0e0")

        pairs = [
            ("loss", "Loss",        "train_loss", "val_loss"),
            ("dice", "Dice Score",  "train_dice", "val_dice"),
            ("iou",  "IoU Score",   "train_iou",  "val_iou"),
        ]

        for ax, (_, ylabel, tr_key, vl_key) in zip(axes, pairs):
            ax.plot(epochs, history[tr_key], color=PALETTE["train"],
                    lw=2, label="Train")
            ax.plot(epochs, history[vl_key], color=PALETTE["val"],
                    lw=2, linestyle="--", label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(framealpha=0.2)
            ax.grid(True)
            ax.set_facecolor("#1a1d2e")

        fig.tight_layout()
        return self._savefig(fig, "training_curves",
                             f"{run_name}_training_curves.png")

    # ── 6.2 Segmentation Results ──────────────────────────────────────────────

    def plot_segmentation_results(self,
                                  images:    np.ndarray,
                                  gt_masks:  np.ndarray,
                                  pred_masks: np.ndarray,
                                  n_samples: int = 4,
                                  fname:     str = "seg_results.png"):
        """
        Grid: [Input MRI | GT mask | Predicted mask | Overlay] × n_samples
        """
        n = min(n_samples, len(images))
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        fig.suptitle("Segmentation Results", fontsize=14, y=1.01,
                     color="#e0e0e0")

        col_titles = ["Input MRI", "Ground Truth", "Prediction", "Overlay"]
        for col, t in enumerate(col_titles):
            axes[0, col].set_title(t, fontsize=11, color="#e0e0e0")

        for row in range(n):
            img  = images[row].squeeze()
            gt   = gt_masks[row].squeeze()
            pred = pred_masks[row].squeeze()

            # Input
            axes[row, 0].imshow(img,  cmap="gray", vmin=0, vmax=1)

            # GT mask — overlay on image (red for tumor)
            axes[row, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].imshow(np.ma.masked_where(gt < 0.5, gt),
                                cmap="Reds", alpha=0.6)

            # Pred mask — overlay on image (yellow for tumor)
            axes[row, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(np.ma.masked_where(pred < 0.5, pred),
                                cmap="YlOrRd", alpha=0.6)

            # Overlay (GT=red, Pred=yellow combined)
            axes[row, 3].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].imshow(np.ma.masked_where(gt   < 0.5, gt),
                                cmap="Reds",   alpha=0.5)
            axes[row, 3].imshow(np.ma.masked_where(pred < 0.5, pred),
                                cmap="YlOrRd", alpha=0.45)

            for col in range(4):
                axes[row, col].axis("off")
                axes[row, col].set_facecolor("#0f1117")

        fig.tight_layout()
        return self._savefig(fig, "segmentation", fname)

    # ── 6.4 Ablation Study ────────────────────────────────────────────────────

    def plot_ablation_study(self,
                            history_with:    dict,
                            history_without: dict):
        """
        Two bar-charts comparing Dice and IoU between:
          - Model with    text
          - Model without text
        Plus val-Dice curves on one figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Ablation Study: With Text vs Without Text",
                     fontsize=14, color="#e0e0e0")

        epochs_with = range(1, len(history_with["val_dice"]) + 1)
        epochs_without = range(1, len(history_without["val_dice"]) + 1)

        # Val Dice curve
        axes[0].plot(epochs_with, history_with["val_dice"],
                     color=PALETTE["with"], lw=2, label="With Text")
        axes[0].plot(epochs_without, history_without["val_dice"],
                     color=PALETTE["without"], lw=2,
                     linestyle="--", label="Without Text")
        axes[0].set_title("Val Dice Score vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Dice Score")
        axes[0].legend(framealpha=0.2)
        axes[0].grid(True)

        # Val IoU curve
        axes[1].plot(epochs_with, history_with["val_iou"],
                     color=PALETTE["with"], lw=2, label="With Text")
        axes[1].plot(epochs_without, history_without["val_iou"],
                     color=PALETTE["without"], lw=2,
                     linestyle="--", label="Without Text")
        axes[1].set_title("Val IoU Score vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU Score")
        axes[1].legend(framealpha=0.2)
        axes[1].grid(True)

        # Bar chart: final best values
        metrics  = ["Best Dice", "Best IoU"]
        w_vals   = [max(history_with["val_dice"]),
                    max(history_with["val_iou"])]
        wo_vals  = [max(history_without["val_dice"]),
                    max(history_without["val_iou"])]

        x   = np.arange(len(metrics))
        bw  = 0.35
        b1  = axes[2].bar(x - bw/2, w_vals,  bw,
                          color=PALETTE["with"],    label="With Text",
                          edgecolor="#e0e0e0", linewidth=0.5)
        b2  = axes[2].bar(x + bw/2, wo_vals, bw,
                          color=PALETTE["without"], label="Without Text",
                          edgecolor="#e0e0e0", linewidth=0.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(metrics)
        axes[2].set_ylim(0, 1.0)
        axes[2].set_title("Peak Performance Comparison")
        axes[2].legend(framealpha=0.2)
        axes[2].bar_label(b1, fmt="%.3f", padding=3, color="#e0e0e0",
                          fontsize=9)
        axes[2].bar_label(b2, fmt="%.3f", padding=3, color="#e0e0e0",
                          fontsize=9)

        for ax in axes:
            ax.set_facecolor("#1a1d2e")
            ax.grid(True)

        fig.tight_layout()
        return self._savefig(fig, "ablation", "ablation_study.png")

    # ── 6.6 Error Analysis ────────────────────────────────────────────────────

    def plot_dice_distribution(self,
                               dice_scores: list,
                               run_name:    str = "run"):
        """
        Histogram + KDE of per-sample Dice scores.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        scores  = np.array(dice_scores)

        sns.histplot(scores, bins=30, kde=True, ax=ax,
                     color=PALETTE["train"],
                     edgecolor="#0f1117", linewidth=0.4,
                     line_kws={"color": PALETTE["val"], "lw": 2})

        ax.axvline(scores.mean(), color=PALETTE["overlay"],
                   lw=2, linestyle="--",
                   label=f"Mean = {scores.mean():.3f}")
        ax.axvline(np.percentile(scores, 25), color=PALETTE["without"],
                   lw=1.5, linestyle=":",
                   label=f"Q25 = {np.percentile(scores, 25):.3f}")

        ax.set_title("Dice Score Distribution", fontsize=13)
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Count")
        ax.legend(framealpha=0.2)
        ax.grid(True)
        ax.set_facecolor("#1a1d2e")
        fig.tight_layout()
        return self._savefig(fig, "error_analysis",
                             f"{run_name}_dice_distribution.png")

    def plot_failure_cases(self,
                           images:     np.ndarray,
                           gt_masks:   np.ndarray,
                           pred_masks: np.ndarray,
                           dice_scores: list,
                           n_worst:    int = 6,
                           run_name:   str = "run"):
        """
        Show the n_worst-performing slices (lowest Dice).
        """
        scores  = np.array(dice_scores)
        indices = np.argsort(scores)[:n_worst]

        fig, axes = plt.subplots(n_worst, 3,
                                 figsize=(12, 4 * n_worst))
        fig.suptitle("Failure Cases (Lowest Dice Scores)",
                     fontsize=14, color="#e0e0e0")

        for row, idx in enumerate(indices):
            img  = images[idx].squeeze()
            gt   = gt_masks[idx].squeeze()
            pred = pred_masks[idx].squeeze()

            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 0].set_title(f"Input  (Dice={scores[idx]:.3f})",
                                   fontsize=9)
            axes[row, 1].imshow(gt,   cmap="Reds",   alpha=0.85)
            axes[row, 1].set_title("Ground Truth", fontsize=9)
            axes[row, 2].imshow(pred, cmap="YlOrRd", alpha=0.85)
            axes[row, 2].set_title("Prediction", fontsize=9)

            for col in range(3):
                axes[row, col].axis("off")

        fig.tight_layout()
        return self._savefig(fig, "error_analysis",
                             f"{run_name}_failure_cases.png")
