import os
import torch

class Config:
    # ── Dataset Paths (Kaggle) ──────────────────────────────────────────────────
    FLAIR_BRATS_PATH = "/kaggle/input/datasets/hussainnasirkhan/flair-brats2020/FLAIR_BRATS2020_split"
    TEXT_BRATS_PATH  = "/kaggle/input/datasets/mrlucario06/textbrats/TextBraTSData"

    # ── Output Paths ────────────────────────────────────────────────────────────
    OUTPUT_DIR     = "./outputs"
    CHECKPOINT_DIR = "./outputs/checkpoints"
    PLOTS_DIR      = "./outputs/plots"

    # ── Dataset ─────────────────────────────────────────────────────────────────
    SLICES_PER_PATIENT = 155       # 155 MRI slices per patient
    IMAGE_SIZE         = 256       # Resize slices to 256x256
    CLIP_IMAGE_SIZE    = 224       # CLIP input size

    # ── Text Embedding ──────────────────────────────────────────────────────────
    TEXT_EMBED_DIM = 768           # TextBraTS uses 768-dim embeddings (ViT-L/BERT)

    # ── Model Architecture ──────────────────────────────────────────────────────
    IN_CHANNELS         = 1        # Grayscale MRI
    OUT_CHANNELS        = 1        # Binary segmentation mask
    UNET_FEATURES       = [64, 128, 256, 512]
    BOTTLENECK_CH       = 1024
    NUM_ATTENTION_HEADS = 8

    # ── Training ────────────────────────────────────────────────────────────────
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 50
    LR           = 1e-4
    WEIGHT_DECAY = 1e-5
    SCHEDULER    = "cosine"        # "cosine" or "step"

    # ── Loss Weights ────────────────────────────────────────────────────────────
    DICE_WEIGHT = 0.6
    BCE_WEIGHT  = 0.4

    # ── Ablation ────────────────────────────────────────────────────────────────
    USE_TEXT = True                # False → image-only ablation model

    # ── Device ──────────────────────────────────────────────────────────────────
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    # ── Logging ─────────────────────────────────────────────────────────────────
    SAVE_EVERY = 5                 # Save checkpoint every N epochs
    LOG_EVERY  = 10                # Log every N batches

    # ── Evaluation ──────────────────────────────────────────────────────────────
    HAUSDORFF_PERCENTILE = 95      # Use 95th-percentile Hausdorff Distance
    TSNE_NUM_SAMPLES     = 200     # Samples for t-SNE plot

    # ── Random Seed ─────────────────────────────────────────────────────────────
    SEED = 42

    @classmethod
    def create_dirs(cls):
        subdirs = [
            "training_curves", "segmentation",
            "attention_maps", "ablation", "tsne", "error_analysis"
        ]
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        for s in subdirs:
            os.makedirs(os.path.join(cls.PLOTS_DIR, s), exist_ok=True)

    @classmethod
    def summary(cls):
        print("=" * 55)
        print("  Brain Tumor Segmentation — Configuration")
        print("=" * 55)
        print(f"  Device        : {cls.DEVICE}")
        print(f"  Image size    : {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  Batch size    : {cls.BATCH_SIZE}")
        print(f"  Epochs        : {cls.NUM_EPOCHS}")
        print(f"  Learning rate : {cls.LR}")
        print(f"  Use text      : {cls.USE_TEXT}")
        print("=" * 55)
