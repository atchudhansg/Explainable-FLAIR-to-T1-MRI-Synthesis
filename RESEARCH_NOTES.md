# Research Notes — Explainable FLAIR-to-T1 MRI Synthesis
### Strengthening for ICCDM-2026 (Springer LNNS Series)
**Paper ID:** 83 | **Author:** Atchudhan Sreekanth et al., VIT Chennai

---

## 📌 Overview

This document consolidates everything done in the original paper and all actionable suggestions for the retraining on Google Cloud to produce a stronger camera-ready version.

- **Task:** Cross-modal MRI synthesis — FLAIR → T1-weighted
- **Framework:** Pix2Pix cGAN with ResNet-9 generator + PatchGAN discriminator
- **Key Innovation:** Grad-CAM integration for voxel-level interpretability
- **Publication Venue:** ICCDM-2026 (Springer LNNS · Scopus, EI Compendex, DBLP, SCImago)

---

## 🏗️ Model Architecture

### Generator — ResNet-9 (~3.1M parameters)

| Stage | Layer | Input → Output | Params |
|---|---|---|---|
| Input | — | 256×256 | — |
| Init Conv | 7×7 Conv + IN + ReLU | 256² | 1.7K |
| Down 1 | Conv + IN + ReLU (stride 2) | 64@256² → 128@128² | 74K |
| Down 2 | Conv + IN + ReLU (stride 2) | 128@128² → 256@64² | 295K |
| ResBlocks 1–9 | Residual blocks (×9) | 256@64² → 256@64² | 2.36M |
| Up 1 | TransposeConv + IN + ReLU | 256@64² → 128@128² | 295K |
| Up 2 | TransposeConv + IN + ReLU | 128@128² → 64@256² | 74K |
| Out Conv | 7×7 Conv + Tanh | 64@256² → 3@256² | 1.7K |

> Output range: **[-1, 1]** via Tanh activation

### Discriminator — PatchGAN (~2.7M parameters)

| Layer | Input → Output | Kernel/Stride |
|---|---|---|
| Conv1 + LReLU | 256² → 64@128² | 4×4 / 2 |
| Conv2 + IN + LReLU | 64@128² → 128@64² | 4×4 / 2 |
| Conv3 + IN + LReLU | 128@64² → 256@32² | 4×4 / 2 |
| Conv4 + IN + LReLU | 256@32² → 512@32² | 4×4 / 1 |
| Final Conv | 512@32² → 1@31² | 4×4 / 1 |

> Evaluates **local patches** (31×31 activation map) for fine-grained realism

---

## 📉 Loss Functions

### Discriminator Loss
```
L_D = -E[log D(x, y)] - E[log(1 - D(x, G(x)))]
```

### Generator Loss (3 components)
```
L_G = L_adv + λ₁·L_L1 + λ₂·L_SSIM
    = L_adv + 100·E[|y - G(x)|₁] + 10·(1 - SSIM(y, G(x)))
```

| Loss Component | Weight | Purpose |
|---|---|---|
| Adversarial (GAN) | 1 | Fool the discriminator |
| L1 Reconstruction | **λ₁ = 100** | Pixel-wise anatomical fidelity, prevent mode collapse |
| SSIM | **λ₂ = 10** | Perceptual quality + structural consistency |

---

## ⚙️ Training Parameters

| Parameter | Value |
|---|---|
| Optimizer | **Adam** |
| Learning Rate | **2 × 10⁻⁴** |
| β₁ | **0.5** |
| β₂ | **0.999** |
| Batch Size | **1** |
| Epochs | **100** |
| Mixed Precision | **FP16/FP32** (`torch.cuda.amp`) |
| Image Size | **256 × 256** |
| Normalization Range | **[-1, 1]** |

### Training Order (per iteration)
1. Generate `ŷ_T1 = G(x_FLAIR)`
2. **Update Discriminator** on (real T1, FLAIR) and (fake T1, FLAIR)
3. **Update Generator** minimizing `L_G = L_adv + 100·L_L1 + 10·L_SSIM`

---

## 💾 Hardware & Compute (Original Run)

| Metric | Value |
|---|---|
| Total Training Time | **27 hours** |
| Peak GPU Memory | **4.12 GB** |
| Allocated GPU Memory | **0.22 GB** |
| Mixed Precision | FP16/FP32 |

---

## 📊 Dataset Details

### BraTS 2021 (Training + Internal Validation)
- **Total volumes:** 1,252 paired FLAIR + T1-weighted MRI
- **Pathology:** Glioblastoma patients
- **Split:** 80/20 → ~1,000 train / **252 test subjects**
- **Slice extraction:** Central axial slice (`MONAI ExtractMidSlice`)
- **Format:** 2D slices resized to 256×256, normalized to [-1, 1]

### BraTS 2023 GLI Challenge (External Validation)
- **Total subjects:** **36**
- **Purpose:** Generalization to unseen acquisition protocols and patient cohorts

---

## 📈 Metrics Tracked

### During Training (per epoch)
- Generator Loss
- Discriminator Loss (real + fake)
- Validation Loss (MSE)

### Final Evaluation Metrics

| Metric | Description |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | Structural Similarity Index |
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **Precision** | Pixel-level lesion classification |
| **Recall** | Pixel-level lesion classification |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **Validation MSE** | Mean Squared Error on validation set |

### Statistical Robustness
- **95% Confidence Intervals** via bootstrap resampling (**1,000 iterations**)

### Grad-CAM Interpretability Metrics (n=20)
| Metric | Value |
|---|---|
| Entropy | 11.2511 ± 0.0357 |
| Spatial Variance | 0.0862 ± 0.0073 |
| Peak Activation | 1.0000 ± 0.0000 |
| High Activation Ratio | 0.0977 ± 0.0207 |
| Error Correlation | 0.3399 ± 0.0263 |
| Mean Activation | 0.1842 ± 0.0098 |
| Std Activation | 0.2933 ± 0.0124 |

---

## 📊 Results

### BraTS 2021 (Held-out test, 252 subjects)

| Metric | Value |
|---|---|
| **PSNR** | **22.33 dB** |
| **SSIM** | **0.8839** |
| **MAE** | **0.0281** |
| **RMSE** | **0.0602** |
| **Precision** | **0.5606** |
| **Recall** | **0.5575** |
| **F1 Score** | **0.5590** |
| **Validation MSE** | **0.0042** |

### BraTS 2023 (External validation, 36 subjects)

| Metric | Value |
|---|---|
| **PSNR** | **22.3254 dB** |
| **SSIM** | **0.8821** |
| **MAE** | **0.0294** |
| **RMSE** | **0.0618** |
| **Precision** | **0.5582** |
| **Recall** | **0.5561** |
| **F1 Score** | **0.5571** |
| **Validation MSE** | **0.0045** |

### Training Convergence
- Generator loss stabilized at **0.5–0.6** after initial drop
- Discriminator loss declined to **0.2–0.3**
- **No mode collapse** — smooth convergence

### Confusion Matrix (Pixel-level, BraTS 2021)
- True Negatives (healthy tissue): **6.03 million**
- True Positives (lesions): **446,897**
- False Negatives: **209,359** ← class imbalance challenge

---

## 🏆 Comparison with State-of-the-Art

> **NOTE:** In the original paper, these numbers were cited from other papers — NOT trained on the same data. This is the biggest weakness to fix.

| Method | Architecture | SSIM | MAE | Interpretability |
|---|---|---|---|---|
| CycleGAN | ResNet gen + PatchGAN disc | 0.812 | 0.034 | None |
| Pix2Pix (vanilla) | U-Net gen + PatchGAN disc | 0.850 | 0.029 | None |
| **Our Method** | ResNet-9 gen + PatchGAN disc | **0.8839** | **0.0281** | **Grad-CAM** |
| Attention-GAN | Attention-augmented GAN | 0.868 | 0.028 | None |
| MedGAN | DenseNet gen + PatchGAN disc | 0.875 | 0.027 | None |

---

## 🔧 Suggestions for Strengthening the Work (ICCDM-2026)

### 1. Train Comparison Models Yourself (MOST IMPORTANT)

Reviewers and conference attendees will challenge comparisons using numbers from different papers with different datasets/preprocessing. Training all baselines on the **same BraTS 2021 split with identical preprocessing** makes the comparison bulletproof.

| Model | Priority | Difficulty | Reason |
|---|---|---|---|
| **Pix2Pix (vanilla, U-Net gen)** | Must | Easy | Direct ablation — same framework, different generator |
| **CycleGAN** | Must | Medium | Most well-known competitor |
| **Attention-GAN** | Nice to have | Medium-Hard | Attention vs. Grad-CAM interpretability angle |
| **MedGAN** | Nice to have | Hard | DenseNet-based, high effort |

**Minimum recommendation:** Train Pix2Pix (vanilla) + CycleGAN yourself. This covers your 2 strongest baselines.

---

### 2. Improve Main Model Performance

| Suggestion | Expected Impact |
|---|---|
| Add **learning rate scheduler** (cosine annealing or linear decay after epoch 50) | More stable convergence, potentially better final metrics |
| Add **Perceptual Loss (VGG-based)** as a 4th loss component | Push SSIM from 0.88 → 0.90+ |
| Try **batch size 2 or 4** if GCP GPU has enough memory | Faster training, better gradient estimates |
| Increase to **150–200 epochs** | Better convergence |
| Add **data augmentation** (random flips, rotations, intensity jitter) | Improved robustness |

---

### 3. Add More Evaluation Metrics

| Metric | Why Add It |
|---|---|
| **FID (Fréchet Inception Distance)** | Standard GAN quality metric — reviewers expect it |
| **Perceptual Similarity (LPIPS)** | Measures perceptual realism beyond pixel-level |
| **Normalized Cross-Correlation (NCC)** | Common in medical image registration literature |
| **Bootstrap CIs on ALL metrics** | Report CIs in the final results table, not just mentioned in text |

---

### 4. Improve Grad-CAM Analysis

| Suggestion | Impact |
|---|---|
| Increase Grad-CAM sample size from **n=20 → n=100+** | More statistically significant interpretability analysis |
| Stratify Grad-CAM by **tumor vs. healthy region** | Show the model focuses on clinically relevant areas |
| Add **radiologist qualitative evaluation** (even 1–2 radiologists, even informal) | Huge credibility boost for clinical trustworthiness |

---

### 5. External Validation Improvements

| Suggestion | Impact |
|---|---|
| Increase BraTS 2023 validation from **36 → more subjects** if available | Stronger generalization claim |
| Stratify results by **tumor grade/type** | Shows robustness across pathology spectrum |
| Add a **third external dataset** (e.g., MSD, IXI) if available | Demonstrates true generalization |

---

### 6. Paper Writing Improvements

| Section | Suggestion |
|---|---|
| Table 5 (comparison) | Add a column for **training time** and **GPU memory** — your model wins here |
| Results | Add CIs to all reported numbers, not just bootstrap mention in text |
| Limitations | Already acknowledged well — consider adding a plan timeline for addressing them |
| Ablation Study | Add a table showing: no SSIM loss vs. with SSIM loss vs. no Grad-CAM — shows each component's contribution |

---

## 🚀 Retraining Checklist (GCP Instance)

- [ ] Set up GCP instance (recommend: T4 or V100 GPU, 16GB+ VRAM)
- [ ] Install dependencies: PyTorch, torchvision, MONAI, scikit-image, lpips, scipy, matplotlib
- [ ] Download BraTS 2021 dataset (register at Synapse)
- [ ] Download BraTS 2023 GLI dataset (for external validation)
- [ ] Implement data pipeline with `ExtractMidSlice`, resize to 256×256, normalize to [-1, 1]
- [ ] Fix random seed for reproducibility across all models
- [ ] Implement 80/20 split (same split for ALL models)
- [ ] Implement ResNet-9 generator + PatchGAN discriminator (your model)
- [ ] Implement vanilla Pix2Pix (U-Net generator) for baseline
- [ ] Implement CycleGAN for baseline
- [ ] Train all models under identical conditions (same data, same split, same preprocessing)
- [ ] Log: loss curves, PSNR, SSIM, MAE, RMSE, Precision, Recall, F1 per epoch
- [ ] Add FID + LPIPS to evaluation
- [ ] Run Grad-CAM on n=100 samples
- [ ] Compute 95% bootstrap CIs on all final metrics
- [ ] Run external validation on BraTS 2023

---

## ⚠️ Known Limitations (Acknowledged in Paper)

1. Only tested on **glioma** datasets — no stroke, dementia, etc.
2. **Cross-scanner robustness** untested (different MRI vendors/field strengths)
3. No evaluation on **low-SNR or motion-artifact** scans
4. No stratification by tumor size, location, or enhancement pattern
5. No formal **radiologist clinical evaluation**

---

*Last updated: March 28, 2026*
*Prepared for ICCDM-2026 camera-ready strengthening*
