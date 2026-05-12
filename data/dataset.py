"""
data/dataset.py
───────────────
PyTorch Dataset for BraTS 2020 multimodal segmentation.

Patient–slice mapping
─────────────────────
Images are stored as a flat global index (image_0.npy, image_1.npy, …).
Each patient has exactly SLICES_PER_PATIENT (155) slices.

  patient 0-based index = image_index // SLICES_PER_PATIENT
  patient name          = "BraTS20_Training_{idx+1:03d}"

Train split  → patients 001 … N_train
Val   split  → patients N_train+1 … N_train+N_val
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from data.preprocessing import Preprocessor


def _sorted_npy(directory: str, prefix: str) -> list[str]:
    """Return .npy files sorted numerically by the integer suffix."""
    pattern = os.path.join(directory, f"{prefix}_*.npy")
    files   = glob.glob(pattern)
    files.sort(key=lambda p: int(
        os.path.splitext(os.path.basename(p))[0].split("_")[-1]
    ))
    return files


def _count_train_patients(flair_split_path: str) -> int:
    """Count how many patients are in the training split."""
    train_imgs = _sorted_npy(
        os.path.join(flair_split_path, "train", "images"), "image"
    )
    return len(train_imgs) // Config.SLICES_PER_PATIENT


class BraTSDataset(Dataset):
    """
    PyTorch Dataset that loads:
        - MRI slices       : (1, H, W) float tensor
        - Segmentation mask: (1, H, W) float tensor
        - Text embedding   : (TEXT_EMBED_DIM,) float tensor
        - Raw text         : str  (for ROUGE / BLEU evaluation)
        - Patient name     : str

    Args:
        mode        : "train" or "val"
        use_text    : if False, text embedding is a zero vector (ablation)
        augment     : apply random augmentations (train only)
    """

    def __init__(self,
                 mode:     str  = "train",
                 use_text: bool = True,
                 augment:  bool = True):

        assert mode in ("train", "val"), "mode must be 'train' or 'val'"

        self.mode     = mode
        self.use_text = use_text
        self.prep     = Preprocessor(
            image_size = Config.IMAGE_SIZE,
            augment    = augment and (mode == "train")
        )
        self._augment = augment and (mode == "train")

        flair_root = Config.FLAIR_BRATS_PATH
        text_root  = Config.TEXT_BRATS_PATH

        # ── Collect image / mask paths ─────────────────────────────────────────
        img_dir  = os.path.join(flair_root, mode, "images")
        mask_dir = os.path.join(flair_root, mode, "masks")

        self.image_paths = _sorted_npy(img_dir,  "image")
        self.mask_paths  = _sorted_npy(mask_dir, "mask")

        assert len(self.image_paths) == len(self.mask_paths), (
            f"[Dataset] Mismatch: {len(self.image_paths)} images "
            f"vs {len(self.mask_paths)} masks in {mode}/"
        )

        # ── Build patient offset ───────────────────────────────────────────────
        # Val images are 0-indexed locally; patients continue after train set
        if mode == "train":
            self._patient_offset = 0
        else:
            self._patient_offset = _count_train_patients(flair_root)

        # ── Index all patient text data ────────────────────────────────────────
        # {patient_name: {"embed": path, "txt": path}}
        self._text_index = self._build_text_index(text_root)

        print(
            f"[BraTSDataset] mode={mode} | "
            f"slices={len(self.image_paths)} | "
            f"patients~{len(self.image_paths) // Config.SLICES_PER_PATIENT} | "
            f"use_text={use_text}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_text_index(text_root: str) -> dict:
        """Scan TextBraTSData and build a dict keyed by patient folder name."""
        index = {}
        if not os.path.isdir(text_root):
            print(f"[BraTSDataset] WARNING: text root not found: {text_root}")
            return index

        for patient_dir in sorted(os.listdir(text_root)):
            full_path = os.path.join(text_root, patient_dir)
            if not os.path.isdir(full_path):
                continue

            embed_files = glob.glob(os.path.join(full_path, "*_flair_text.npy"))
            txt_files   = glob.glob(os.path.join(full_path, "*_flair_text.txt"))

            if embed_files and txt_files:
                index[patient_dir] = {
                    "embed": embed_files[0],
                    "txt":   txt_files[0],
                }
        return index

    def _patient_name_for(self, local_image_idx: int) -> str:
        """Map a local (split-relative) image index to its BraTS patient name."""
        global_patient_idx = (
            local_image_idx // Config.SLICES_PER_PATIENT
            + self._patient_offset
        )
        return f"BraTS20_Training_{global_patient_idx + 1:03d}"

    def _get_text_data(self, patient_name: str):
        """Return (text_embedding_np, raw_text_str) for a patient."""
        if patient_name in self._text_index:
            info = self._text_index[patient_name]
            embed = self.prep.load_text_embedding(info["embed"])
            text  = self.prep.load_text_report(info["txt"])
        else:
            # Fallback: zero embedding + empty text
            embed = np.zeros(Config.TEXT_EMBED_DIM, dtype=np.float32)
            text  = ""
        return embed, text

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        img  = self.prep.load_image(self.image_paths[idx])
        mask = self.prep.load_mask(self.mask_paths[idx])

        # Augment (train only)
        if self._augment:
            img, mask = self.prep.apply_augmentation(img, mask)

        # Convert to tensors: (1, H, W)
        img_t  = self.prep.to_tensor(img)
        mask_t = self.prep.to_tensor(mask)

        # Text
        patient_name          = self._patient_name_for(idx)
        embed_np, text_report = self._get_text_data(patient_name)

        # Ablation: zero-out text embedding
        if not self.use_text:
            embed_np = np.zeros_like(embed_np)

        embed_t = self.prep.text_embedding_to_tensor(embed_np)

        return {
            "image":        img_t,        # (1, H, W)
            "mask":         mask_t,       # (1, H, W)
            "text_embed":   embed_t,      # (TEXT_EMBED_DIM,)
            "text_report":  text_report,  # str
            "patient_name": patient_name, # str
            "image_path":   self.image_paths[idx],
        }
