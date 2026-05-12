"""
training/losses.py
──────────────────
Loss functions for binary tumour segmentation.

  CombinedLoss = α·DiceLoss + β·BCEWithLogitsLoss
  (α=0.6, β=0.4  by default from Config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    L_dice = 1 - (2|P∩G| + ε) / (|P| + |G| + ε)

    Args:
        smooth (float): Laplace smoothing to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W)  — raw model output (before sigmoid)
            target : (B, 1, H, W)  — binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits)

        probs  = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        union        = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss to down-weight easy background pixels.
    Useful for severe foreground/background imbalance.

    L_focal = -α_t · (1 - p_t)^γ · log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, target,
                                                   reduction="none")
        probs  = torch.sigmoid(logits)
        p_t    = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal  = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice + BCE (+ optional Focal) losses.

        L = dice_w · L_dice + bce_w · L_bce

    Args:
        dice_weight (float): Weight for Dice loss term.
        bce_weight  (float): Weight for BCE  loss term.
        use_focal   (bool) : Replace BCE with Focal loss.
        pos_weight  (float): Class imbalance weight for BCE
                             (higher → penalise false negatives more).
    """

    def __init__(self,
                 dice_weight: float = Config.DICE_WEIGHT,
                 bce_weight:  float = Config.BCE_WEIGHT,
                 use_focal:   bool  = False,
                 pos_weight:  float = 2.0):
        super().__init__()

        self.dice_w = dice_weight
        self.bce_w  = bce_weight

        self.dice = DiceLoss(smooth=1.0)

        if use_focal:
            self.bce = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            pw = torch.tensor([pos_weight])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> dict:
        """
        Args:
            logits : (B, 1, H, W)
            target : (B, 1, H, W)

        Returns:
            dict with keys: "loss", "dice_loss", "bce_loss"
        """
        # Move pos_weight to same device as logits (if using BCEWithLogitsLoss)
        if hasattr(self.bce, "pos_weight") and \
                self.bce.pos_weight is not None and \
                self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        dice_loss = self.dice(logits, target)
        bce_loss  = self.bce(logits, target)
        total     = self.dice_w * dice_loss + self.bce_w * bce_loss

        return {
            "loss":      total,
            "dice_loss": dice_loss.detach(),
            "bce_loss":  bce_loss.detach(),
        }
