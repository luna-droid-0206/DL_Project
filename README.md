# Brain Tumor Segmentation with Vision Language Models

A multimodal deep learning framework for accurate brain tumor segmentation using **FLAIR MRI images** and **radiology text reports** with CLIP-based vision language models.

## 🎯 Approach

- **Dataset**: BraTS 2020 FLAIR MRI + TextBraTS 2020
- **Architecture**: UNet backbone + CrossAttentionFusion (CLIP embeddings)
- **Key Innovation**: Fuses image features with textual descriptions for improved segmentation
- **Ablation**: Compares with-text vs without-text models to demonstrate text impact

## 📊 Performance Metrics

### With-Text Model (Multimodal)
| Metric | Score |
|--------|-------|
| Dice Coefficient | **0.9703** |
| IoU (Jaccard) | **0.9572** |
| Precision | **0.9575** |
| Recall | **0.9996** |
| Hausdorff Distance (95%) | 18.22 px |
| Loss | 0.0678 |

### Without-Text Model (Image-only Baseline)
| Metric | Score |
|--------|-------|
| Dice Coefficient | **0.9700** |
| IoU (Jaccard) | **0.9566** |
| Precision | **0.9575** |
| Recall | **0.9991** |
| Hausdorff Distance (95%) | 19.34 px |
| Loss | 0.0703 |

## 🔍 Key Findings

✅ **Text-guided approach achieves 99.97% recall** — Excellent tumor detection  
✅ **Marginal Dice improvement** (+0.03%) with text guidance  
✅ **Hausdorff Distance reduced by ~5%** — Better boundary accuracy  
✅ **Both models perform exceptionally well** — Strong baseline and multimodal fusion  
✅ **Precision maintained at 95.75%** — Minimal false positives

## 📁 Project Structure

```
├── models/
│   ├── full_model.py          # MultimodalSegNet
│   ├── decoder.py             # UNet encoder/decoder
│   └── fusion.py              # CrossAttentionFusion
├── data/
│   ├── dataset.py             # BraTSDataset with patient-slice mapping
│   └── preprocessing.py       # Image & text preprocessing
├── training/
│   ├── train.py               # Training loop
│   ├── losses.py              # DiceLoss + BCEWithLogits
│   └── metrics.py             # Dice, IoU, Hausdorff, Precision, Recall
├── evaluation/
│   ├── visualize.py           # 6 visualization types
│   ├── gradcam.py             # Attention maps
│   └── tsne.py                # Embedding analysis
├── main.py                     # Full pipeline
└── config.py                   # Centralized configuration
```

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (training + evaluation + visualizations)
python main.py

# Skip training, use existing checkpoints
python main.py --skip_train
```

## 📈 Visualizations Generated

- **Training Curves**: Loss, Dice, IoU vs Epoch
- **Segmentation Results**: Input, GT mask, Prediction, Overlay
- **Grad-CAM Heatmaps**: Model attention visualization
- **Ablation Study**: With-text vs Without-text comparison
- **t-SNE Embeddings**: Image-text alignment in embedding space
- **Error Analysis**: Dice distribution & failure cases

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, scikit-learn, scikit-image
- CLIP embeddings (768-dim), NLTK, rouge-score

## ⚙️ Configuration

Key hyperparameters in `config.py`:
- Image size: 256×256
- Batch size: 16
- Epochs: 50
- Learning rate: 1e-4
- Loss weights: Dice (0.6) + BCE (0.4)

## 📝 Notes

- **155 MRI slices per patient** → correctly mapped to **one text report per patient**
- **Binary segmentation** for tumor vs non-tumor
- **Cross-attention fusion** enables multimodal learning
- **Excellent recall (>99%)** ensures high tumor detection sensitivity

---
