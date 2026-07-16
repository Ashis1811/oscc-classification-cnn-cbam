<div align="center">

# 🔬 OSCC Classification using Convolutional Neural Networks
### Detecting Oral Squamous Cell Carcinoma from Histopathological Images

<p>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Medical%20Imaging-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/Best%20Test%20Accuracy-92.06%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Attention-CBAM-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

</div>

---

## 📖 Overview

This project performs **supervised image classification** on histopathological images to detect **Oral Squamous Cell Carcinoma (OSCC)**. It uses **lightweight, scalable CNN architectures** enhanced with the **Convolutional Block Attention Module (CBAM)**, combined with data augmentation and preprocessing, to classify tissue images into two categories:

> 🟢 **Normal**  &nbsp;|&nbsp;  🔴 **OSCC (Cancerous)**

---

## 📑 Table of Contents

- [Dataset](#-dataset)
- [Pipeline](#-image-classification-pipeline)
- [Model 1 — MobileNetV2](#-model-1--mobilenetv2)
- [Model 2 — EfficientNet](#-model-2--efficientnet)
- [Model 3 — Hybrid (MobileNetV2 + EfficientNet)](#-model-3--hybrid-mobilenetv2--efficientnet)
- [Performance Comparison](#-performance-comparison)
- [t-SNE Visualization](#-t-sne-visualization)

---

## 📂 Dataset

Histopathological images of oral tissue, captured via **H&E-stained microscopy** at multiple magnifications.

| Property | Detail |
|---|---|
| 🖼️ Format | JPG |
| 🔬 Source | Microscopy (H&E stained slides) |
| 🔍 Magnifications | 100x, 400x |
| 🏷️ Classes | Normal, OSCC |
| 📦 Split | Train / Validation / Test |
| 🎯 Task | Binary medical image classification |

---

## ⚙️ Image Classification Pipeline

Implemented end-to-end in **PyTorch**, with three model notebooks each following a consistent pipeline:

```
Init → Load Data → Preprocess/Augment → Build Model → Add CBAM → Train → Evaluate
```

---

## 🧠 Model 1 — MobileNetV2

📓 `mobilenetv2-oral-squamous-dataset.ipynb`

<details>
<summary><strong>View pipeline steps</strong></summary>

| Step | Description |
|---|---|
| 1️⃣ Initialization | Import libraries, set random seeds for reproducibility |
| 2️⃣ Load Dataset | `ImageFolder` + DataLoaders for train/val/test |
| 3️⃣ Preprocessing | Resize, normalize, random horizontal flip, random rotation |
| 4️⃣ Build Model | MobileNetV2: depthwise separable convolutions + inverted residual blocks + FC layer |
| 5️⃣ Add CBAM | Channel & spatial attention for stronger feature focus |
| 6️⃣ Train | Adam optimizer, CrossEntropyLoss |
| 7️⃣ Evaluate | Accuracy/loss curves across epochs |

</details>

<p align="center">
  <img src="image.png" width="45%" />
  <img src="image-1.png" width="45%" />
</p>

---

## 🧠 Model 2 — EfficientNet

📓 `efficientnet-oral-squamous-dataset.ipynb`

<details>
<summary><strong>View pipeline steps</strong></summary>

| Step | Description |
|---|---|
| 1️⃣ Initialization | Import libraries, set seeds |
| 2️⃣ Load & Transform | DataLoader + resize, normalize, augment |
| 3️⃣ Build Model | EfficientNet: compound scaling + MBConv blocks + SE modules + FC layer |
| 4️⃣ Train | Adam optimizer, CrossEntropyLoss |
| 5️⃣ Add CBAM | Channel & spatial attention |
| 6️⃣ Evaluate | Training/validation accuracy plotted |

</details>

<p align="center">
  <img src="image-2.png" width="45%" />
  <img src="image-3.png" width="45%" />
</p>

---

## 🧠 Model 3 — Hybrid (MobileNetV2 + EfficientNet)

📓 `mobilenet-efficientnet-oral-squamous-dataset.ipynb`

<details>
<summary><strong>View pipeline steps</strong></summary>

| Step | Description |
|---|---|
| 1️⃣ Initialization | Import libraries, set reproducibility seeds |
| 2️⃣ Feature Extraction | Extract deep features from MobileNetV2 & EfficientNet |
| 3️⃣ Feature Fusion | Concatenate features from both backbones |
| 4️⃣ Classification Layer | Fully connected layers on fused features |
| 5️⃣ Train | Adam optimizer, CrossEntropyLoss |
| 6️⃣ Add CBAM | Channel & spatial attention |
| 7️⃣ Evaluate | Compare accuracy/loss across all models |

</details>

<p align="center">
  <img src="image-4.png" width="45%" />
  <img src="image-5.png" width="45%" />
</p>

---

## 📊 Performance Comparison

<div align="center">

| Model | Type | Train Acc | Train F1 | Test Acc | Test F1 | Valid Acc | Valid F1 | Attention | Notes |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | Single | 0.9812 | 0.9812 | 0.8889 | 0.8503 | 0.9167 | 0.8835 | ❌ | Baseline |
| MobileNetV2 | Single | 0.9832 | 0.9832 | 0.8968 | 0.8624 | 0.9417 | 0.9174 | ✅ | Improved |
| EfficientNet | Single | 0.9863 | 0.9862 | 0.8810 | 0.8340 | 0.9250 | 0.8910 | ❌ | Baseline |
| EfficientNet | Single | 0.9883 | 0.9883 | 0.9048 | 0.8687 | 0.9083 | 0.8688 | ✅ | Improved |
| **MobileNetV2 + EfficientNet** | **Ensemble** | **0.9953** | **0.9953** | 🏆 **0.9206** | **0.8906** | 0.9083 | 0.8668 | ✅ Fusion | **Best performance** |

</div>

> 🏆 The **Hybrid Fusion model** achieves the highest test accuracy (**92.06%**) and F1 score, confirming that combining MobileNetV2 and EfficientNet features with CBAM attention yields the strongest performance.

---

## 🌌 t-SNE Visualization

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** projects high-dimensional feature embeddings into 2D space to visualize how well each model separates the **Normal** vs **OSCC** classes.

<p align="center">
  <img src="image-6.png" width="70%" />
</p>

Generated for all three models — MobileNetV2, EfficientNet, and the Hybrid model — to compare learned feature discriminability.

---

<div align="center">

*Built with PyTorch • CBAM Attention • Medical Image Analysis*

</div>
