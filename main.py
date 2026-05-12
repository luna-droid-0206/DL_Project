"""
main.py
───────
Entry point for the Multimodal Brain Tumour Segmentation project.

Runs:
  1. Data loading
  2. Model initialisation
  3. Training  (with-text model)
  4. Ablation  (without-text model)
  5. Full evaluation with all visualisations
  6. t-SNE embedding analysis
  7. Grad-CAM attention maps
  8. Error analysis
"""

import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import BraTSDataset
from models.full_model import MultimodalSegNet
from training.train import Trainer
from training.metrics import SegmentationMetrics, compute_rouge, compute_bleu
from evaluation.visualize import Visualizer
from evaluation.gradcam import GradCAM
from evaluation.tsne import TSNEVisualizer


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_loaders(use_text: bool = True):
    train_ds = BraTSDataset(mode="train", use_text=use_text, augment=True)
    val_ds   = BraTSDataset(mode="val",   use_text=use_text, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = Config.BATCH_SIZE,
        shuffle     = True,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = Config.PIN_MEMORY,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = Config.BATCH_SIZE,
        shuffle     = False,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = Config.PIN_MEMORY,
    )
    return train_loader, val_loader, val_ds


# ── Inference helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, n: int = 64):
    """Collect n predictions for visualisation."""
    model.eval()
    images, gt_masks, pred_masks, reports = [], [], [], []
    device = Config.DEVICE

    for batch in loader:
        img  = batch["image"].to(device)
        mask = batch["mask"].to(device)
        te   = batch["text_embed"].to(device)
        rpts = batch["text_report"]

        out  = model(img, te)
        pred = out["mask_pred"].cpu().numpy()
        img  = img.cpu().numpy()
        mask = mask.cpu().numpy()

        for i in range(len(img)):
            if len(images) >= n:
                break
            images.append(img[i])
            gt_masks.append(mask[i])
            pred_masks.append(pred[i])
            reports.append(rpts[i])

        if len(images) >= n:
            break

    return (np.array(images), np.array(gt_masks),
            np.array(pred_masks), reports)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(args):
    set_seed()
    Config.create_dirs()
    Config.summary()

    viz = Visualizer()

    # ── 1. Build loaders ──────────────────────────────────────────────────────
    print("\n[1/7] Building data loaders …")
    train_loader, val_loader, val_ds = build_loaders(use_text=True)

    # ── 2. With-text model ────────────────────────────────────────────────────
    print("\n[2/7] Training multimodal model (with text) …")
    model_with = MultimodalSegNet(use_text=True).to(Config.DEVICE)
    model_with.count_parameters()

    trainer_with = Trainer(model_with, train_loader, val_loader,
                           use_text=True, run_name="with_text")

    if args.skip_train:
        ckpt = os.path.join(Config.CHECKPOINT_DIR, "with_text_best.pth")
        if os.path.exists(ckpt):
            trainer_with.load_checkpoint(ckpt)
            print("  Loaded with-text checkpoint.")
        history_with = trainer_with.history
    else:
        history_with = trainer_with.train()

    # ── 3. Without-text model (ablation) ─────────────────────────────────────
    print("\n[3/7] Training image-only model (ablation) …")
    train_loader_no, val_loader_no, _ = build_loaders(use_text=False)
    model_without = MultimodalSegNet(use_text=False).to(Config.DEVICE)

    trainer_without = Trainer(model_without, train_loader_no, val_loader_no,
                              use_text=False, run_name="without_text")

    if args.skip_train:
        ckpt = os.path.join(Config.CHECKPOINT_DIR, "without_text_best.pth")
        if os.path.exists(ckpt):
            trainer_without.load_checkpoint(ckpt)
        history_without = trainer_without.history
    else:
        history_without = trainer_without.train()

    # ── 4. Training curves ────────────────────────────────────────────────────
    print("\n[4/7] Plotting training curves …")
    viz.plot_training_curves(history_with,    run_name="with_text")
    viz.plot_training_curves(history_without, run_name="without_text")

    # ── 5. Full evaluation ────────────────────────────────────────────────────
    print("\n[5/7] Full evaluation …")
    eval_with    = trainer_with.evaluate()
    eval_without = trainer_without.evaluate(val_loader_no)

    print("\n  ── With-text model ──")
    for k, v in eval_with.items():
        print(f"    {k:12s}: {v:.4f}")

    print("\n  ── Without-text model ──")
    for k, v in eval_without.items():
        print(f"    {k:12s}: {v:.4f}")

    # ── 6. Segmentation visualisations ───────────────────────────────────────
    print("\n[6/7] Generating visualisations …")
    imgs, gt, preds, reports = collect_predictions(model_with, val_loader)

    viz.plot_segmentation_results(imgs, gt, preds, n_samples=8,
                                  fname="with_text_seg_results.png")

    # Ablation visualisation
    viz.plot_ablation_study(history_with, history_without)

    # Dice distribution
    metrics_acc = SegmentationMetrics()
    model_with.eval()
    with torch.no_grad():
        for batch in val_loader:
            img  = batch["image"].to(Config.DEVICE)
            msk  = batch["mask"].to(Config.DEVICE)
            te   = batch["text_embed"].to(Config.DEVICE)
            out  = model_with(img, te)
            metrics_acc.update(out["mask_pred"], msk)

    viz.plot_dice_distribution(metrics_acc.per_sample_dice(),
                               run_name="with_text")
    viz.plot_failure_cases(imgs, gt, preds,
                           metrics_acc.per_sample_dice()[:len(imgs)],
                           n_worst=6, run_name="with_text")

    # ── 7. Grad-CAM ───────────────────────────────────────────────────────────
    print("\n  Grad-CAM …")
    gcam   = GradCAM(model_with)
    n_gcam = min(6, len(imgs))
    cams   = []
    for i in range(n_gcam):
        img_t = torch.from_numpy(imgs[i]).unsqueeze(0).to(Config.DEVICE)
        te_t  = val_ds[i]["text_embed"].unsqueeze(0).to(Config.DEVICE)
        cams.append(gcam.compute(img_t, te_t))

    gcam.batch_visualize(imgs[:n_gcam], np.array(cams),
                         gt_masks=gt[:n_gcam], run_name="with_text")

    # ── 7b. t-SNE ─────────────────────────────────────────────────────────────
    print("\n  t-SNE …")
    tsne_viz    = TSNEVisualizer(model_with, device=Config.DEVICE)
    embeddings  = tsne_viz.extract_embeddings(val_loader)
    tsne_viz.plot(embeddings, run_name="with_text")

    # ── ROUGE / BLEU ──────────────────────────────────────────────────────────
    if reports:
        sample_report = reports[0]
        rouge = compute_rouge(sample_report, sample_report)
        bleu  = compute_bleu(sample_report,  sample_report)
        print(f"\n  ROUGE-L (self-reference): {rouge['rougeL']:.4f}")
        print(f"  BLEU-1  (self-reference): {bleu:.4f}")

    print("\n✓ All done.  Outputs saved to:", Config.OUTPUT_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Brain Tumour Segmentation"
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training and load existing checkpoints"
    )
    args = parser.parse_args()
    run(args)
