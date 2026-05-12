"""
training/train.py
─────────────────
Training loop, validation, checkpointing, and history tracking.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from config import Config
from training.losses import CombinedLoss
from training.metrics import SegmentationMetrics


class Trainer:
    """
    Manages the full training pipeline.

    Args:
        model     : MultimodalSegNet instance
        train_loader / val_loader : DataLoader instances
        use_text  : whether to pass text embeddings to the model
        run_name  : label for checkpoint files (e.g. "with_text")
    """

    def __init__(self,
                 model:        nn.Module,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 use_text:     bool = True,
                 run_name:     str  = "run"):

        self.model        = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.use_text     = use_text
        self.run_name     = run_name
        self.device       = Config.DEVICE

        # Loss
        self.criterion = CombinedLoss(
            dice_weight = Config.DICE_WEIGHT,
            bce_weight  = Config.BCE_WEIGHT,
        )

        # Optimiser
        self.optimiser = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = Config.LR,
            weight_decay = Config.WEIGHT_DECAY,
        )

        # Scheduler
        if Config.SCHEDULER == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimiser, T_max=Config.NUM_EPOCHS, eta_min=1e-6
            )
        else:
            self.scheduler = StepLR(self.optimiser, step_size=10, gamma=0.5)

        # History
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou":  [], "val_iou":  [],
        }

        self.best_val_dice = 0.0

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        metrics  = SegmentationMetrics()
        loss_sum = 0.0

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            pbar = tqdm(loader,
                        desc="Train" if train else "Val  ",
                        leave=False,
                        dynamic_ncols=True)

            for batch in pbar:
                image      = batch["image"].to(self.device)
                mask       = batch["mask"].to(self.device)
                text_embed = batch["text_embed"].to(self.device) \
                             if self.use_text else None

                # Forward
                out    = self.model(image, text_embed)
                losses = self.criterion(out["mask_logits"], mask)
                loss   = losses["loss"]

                # Backward (train only)
                if train:
                    self.optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimiser.step()

                loss_sum += loss.item()
                metrics.update(out["mask_pred"].detach(), mask)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{metrics._dice[-1]:.4f}",
                })

        m = metrics.compute()
        m["loss"] = loss_sum / len(loader)
        return m

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run full training for Config.NUM_EPOCHS.
        Returns the complete history dict.
        """
        print(f"\n{'='*55}")
        print(f"  Training  [{self.run_name}]  |  device: {self.device}")
        print(f"{'='*55}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            t0 = time.time()

            train_m = self._run_epoch(self.train_loader, train=True)
            val_m   = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_dice"].append(train_m["dice"])
            self.history["val_dice"].append(val_m["dice"])
            self.history["train_iou"].append(train_m["iou"])
            self.history["val_iou"].append(val_m["iou"])

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{Config.NUM_EPOCHS} | "
                f"Loss {train_m['loss']:.4f}/{val_m['loss']:.4f} | "
                f"Dice {train_m['dice']:.4f}/{val_m['dice']:.4f} | "
                f"IoU {train_m['iou']:.4f}/{val_m['iou']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_m["dice"] > self.best_val_dice:
                self.best_val_dice = val_m["dice"]
                self._save_checkpoint(epoch, tag="best")
                print(f"  ✓ New best val Dice: {self.best_val_dice:.4f}")

            # Periodic checkpoint
            if epoch % Config.SAVE_EVERY == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch}")

        # Save final history
        self._save_history()
        print(f"\n  Best Val Dice: {self.best_val_dice:.4f}")
        return self.history

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str):
        fname = os.path.join(
            Config.CHECKPOINT_DIR,
            f"{self.run_name}_{tag}.pth"
        )
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "best_val_dice":   self.best_val_dice,
            "history":         self.history,
            "use_text":        self.use_text,
        }, fname)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.history        = ckpt.get("history", self.history)
        self.best_val_dice  = ckpt.get("best_val_dice", 0.0)
        print(f"  Loaded checkpoint: {path}")
        return ckpt["epoch"]

    def _save_history(self):
        path = os.path.join(
            Config.CHECKPOINT_DIR, f"{self.run_name}_history.json"
        )
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # ── Full validation with HD95 ─────────────────────────────────────────────

    def evaluate(self, loader: DataLoader = None) -> dict:
        """
        Full evaluation pass with HD95 computation.
        Returns all metric values.
        """
        loader   = loader or self.val_loader
        metrics  = SegmentationMetrics()
        loss_sum = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                image      = batch["image"].to(self.device)
                mask       = batch["mask"].to(self.device)
                text_embed = batch["text_embed"].to(self.device) \
                             if self.use_text else None

                out    = self.model(image, text_embed)
                losses = self.criterion(out["mask_logits"], mask)
                loss_sum += losses["loss"].item()
                metrics.update(out["mask_pred"], mask, compute_hd=True)

        result = metrics.compute()
        result["loss"] = loss_sum / len(loader)
        return result
