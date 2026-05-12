"""
training/metrics.py
───────────────────
Evaluation metrics for binary tumour segmentation.

  - Dice Coefficient
  - Intersection over Union (IoU / Jaccard)
  - Hausdorff Distance (95th percentile)
  - Precision & Recall
  - ROUGE-L & BLEU-1  (text quality)
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import Config


# ── Segmentation ───────────────────────────────────────────────────────────────

def dice_coefficient(pred: torch.Tensor,
                     target: torch.Tensor,
                     threshold: float = 0.5,
                     smooth: float = 1.0) -> float:
    """
    Dice = (2|P∩G| + ε) / (|P| + |G| + ε)

    Args:
        pred    : (B, 1, H, W) — sigmoid probabilities or binary predictions
        target  : (B, 1, H, W) — binary ground truth
        threshold: binarisation threshold for pred
    Returns:
        Mean Dice score over batch (float)
    """
    pred   = (pred >= threshold).float()
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union        = pred.sum(dim=1) + target.sum(dim=1)
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred: torch.Tensor,
              target: torch.Tensor,
              threshold: float = 0.5,
              smooth: float = 1.0) -> float:
    """
    IoU = |P∩G| / (|P∪G|) = |P∩G| / (|P| + |G| - |P∩G|)
    """
    pred   = (pred >= threshold).float()
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union        = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou          = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision_recall(pred: torch.Tensor,
                     target: torch.Tensor,
                     threshold: float = 0.5):
    """
    Returns (precision, recall) averaged over the batch.
    """
    pred   = (pred >= threshold).float()
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    prec = (tp / (tp + fp + 1e-6)).mean().item()
    rec  = (tp / (tp + fn + 1e-6)).mean().item()
    return prec, rec


def hausdorff_distance_95(pred: np.ndarray,
                           target: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance between two binary masks.

    Args:
        pred   : (H, W) binary numpy array
        target : (H, W) binary numpy array
    Returns:
        HD95 in pixels (float); 0.0 if either mask is empty.
    """
    pred_pts   = np.argwhere(pred   > 0.5).astype(float)
    target_pts = np.argwhere(target > 0.5).astype(float)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 0.0

    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return max(d1, d2)


# ── Text (ROUGE / BLEU) ────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Args:
        hypothesis : model-generated text
        reference  : ground-truth text
    Returns:
        dict with keys "rouge1", "rouge2", "rougeL"
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}


def compute_bleu(hypothesis: str, reference: str) -> float:
    """
    Compute sentence-level BLEU-1 with smoothing.
    """
    smooth  = SmoothingFunction().method1
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0
    return sentence_bleu(
        [ref_tok], hyp_tok,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth
    )


# ── Aggregator ─────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Running accumulator for all segmentation metrics over an epoch.

    Usage:
        m = SegmentationMetrics()
        for batch in loader:
            m.update(pred, mask)
        results = m.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._dice  = []
        self._iou   = []
        self._prec  = []
        self._rec   = []
        self._hd95  = []

    def update(self,
               pred:   torch.Tensor,
               target: torch.Tensor,
               compute_hd: bool = False):
        """
        Args:
            pred   : (B, 1, H, W) sigmoid probabilities
            target : (B, 1, H, W) binary ground truth
        """
        self._dice.append(dice_coefficient(pred, target))
        self._iou.append(iou_score(pred, target))
        p, r = precision_recall(pred, target)
        self._prec.append(p)
        self._rec.append(r)

        if compute_hd:
            pred_np   = (pred.detach().cpu().numpy() >= 0.5)
            target_np = target.detach().cpu().numpy()
            for b in range(pred_np.shape[0]):
                hd = hausdorff_distance_95(
                    pred_np[b, 0], target_np[b, 0]
                )
                self._hd95.append(hd)

    def compute(self) -> dict:
        out = {
            "dice":      float(np.mean(self._dice)),
            "iou":       float(np.mean(self._iou)),
            "precision": float(np.mean(self._prec)),
            "recall":    float(np.mean(self._rec)),
        }
        if self._hd95:
            out["hd95"] = float(np.mean(self._hd95))
        return out

    def per_sample_dice(self) -> list[float]:
        """Return per-batch Dice list (for error analysis / distribution plot)."""
        return self._dice.copy()
