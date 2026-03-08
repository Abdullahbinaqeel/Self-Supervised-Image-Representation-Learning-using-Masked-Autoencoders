# Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

## GenAI Assignment 02 | Spring 2026

---

## 📌 Overview

This project implements a **Masked Autoencoder (MAE)** from scratch using base **PyTorch**. The system learns visual representations by masking 75% of image patches and training a ViT-based encoder-decoder to reconstruct them.

## 🏗️ Architecture

| Component | Config | Details |
|-----------|--------|---------|
| **Encoder** | ViT-Base (B/16) | 768-dim, 12 layers, 12 heads, ~86M params |
| **Decoder** | ViT-Small (S/16) | 384-dim, 12 layers, 6 heads, ~22M params |
| **Patch Size** | 16×16 | 196 patches per 224×224 image |
| **Mask Ratio** | 75% | 49 visible, 147 masked patches |

## 📁 Project Structure

```
Assignment2/
├── mae_assignment.py     # Complete MAE implementation (notebook-friendly)
├── gradio_app.py         # Interactive Gradio deployment app
├── README.md             # This file
└── GenAI_Assignment02.docx  # Assignment specification
```

## 🚀 How to Run

### On Kaggle (Recommended)

1. Create a new **Kaggle Notebook**
2. Add the dataset: **akash2sharma/tiny-imagenet**
3. Set accelerator to **GPU T4 x2**
4. Copy the contents of `mae_assignment.py` into notebook cells (split at `# %%` markers)
5. Run all cells sequentially

### Gradio App

After training, add the `gradio_app.py` cell to your notebook or run standalone:

```bash
pip install gradio
python gradio_app.py
```

## 📊 Deliverables

- [x] Complete PyTorch implementation (no HuggingFace/timm)
- [x] Training loop with mixed precision, AdamW, Cosine LR
- [x] Loss vs. epochs plot
- [x] 5+ reconstruction visualizations
- [x] PSNR and SSIM evaluation
- [x] Gradio app with image upload + masking ratio slider

## 📝 Submission Checklist

- [ ] Rename notebook: `AI_ASS01_XXF_YYYY.ipynb`
- [ ] Push to GitHub
- [ ] Write Medium blog post
- [ ] Write LinkedIn post
- [ ] Submit Word doc with all links
# Self-Supervised-Image-Representation-Learning-using-Masked-Autoencoders
