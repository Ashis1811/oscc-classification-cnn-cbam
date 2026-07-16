Here's a cleaner, more professional, and GitHub-ready version of your project description.

---

# Convolutional Neural Networks for Oral Squamous Cell Carcinoma (OSCC) Classification

## Project Overview

This project focuses on the **supervised classification of histopathological images** for the detection of **Oral Squamous Cell Carcinoma (OSCC)** using deep learning techniques. The primary objective is to accurately distinguish between **Normal** and **Cancerous (OSCC)** tissue images by leveraging lightweight convolutional neural network (CNN) architectures enhanced with the **Convolutional Block Attention Module (CBAM)**.

The project investigates the performance of three different approaches:

* **MobileNetV2**
* **EfficientNet**
* **Hybrid Feature Fusion (MobileNetV2 + EfficientNet)**

To improve model generalization and robustness, several image preprocessing and data augmentation techniques are applied before training.

---

# Dataset

The dataset consists of **histopathological images of oral tissue** collected from publicly available research datasets. The images are stained using **Hematoxylin and Eosin (H&E)** and captured under different microscopic magnifications.

### Dataset Characteristics

* Image format: **JPG**
* Staining technique: **H&E Stained Histopathology Slides**
* Magnifications:

  * **100×**
  * **400×**
* Classification type: **Binary Classification**

### Classes

* **Normal**
* **Oral Squamous Cell Carcinoma (OSCC)**

### Dataset Split

The dataset is divided into three subsets:

* Training Set
* Validation Set
* Testing Set

---

# Project Workflow

The entire classification pipeline is implemented using **PyTorch** and follows a standard deep learning workflow.

## Step 1: Initialization

* Import required Python libraries
* Set random seeds for reproducibility
* Configure GPU/CPU device

---

## Step 2: Dataset Loading

* Load images using **PyTorch ImageFolder**
* Create DataLoaders for:

  * Training
  * Validation
  * Testing

---

## Step 3: Image Preprocessing

Image preprocessing and augmentation are performed using `torchvision.transforms.Compose`.

The preprocessing pipeline includes:

* Image resizing
* Normalization
* Random horizontal flipping
* Random rotation
* Tensor conversion

These techniques improve model robustness and reduce overfitting.

---

# Model 1: MobileNetV2

**Notebook:** `mobilenetv2-oral-squamous-dataset.ipynb`

### Architecture

MobileNetV2 is a lightweight CNN designed for efficient image classification. It consists of:

* Standard convolution layer
* Depthwise separable convolutions
* Inverted residual bottleneck blocks
* Global average pooling
* Fully connected classification layer

### CBAM Integration

The **Convolutional Block Attention Module (CBAM)** is integrated into the network to improve feature extraction by learning:

* Channel attention
* Spatial attention

This enables the model to focus on the most informative regions of histopathological images.

### Training Configuration

* Optimizer: **Adam**
* Loss Function: **CrossEntropyLoss**
* Batch Size: Experimental
* Epochs: Experimental

### Performance Evaluation

Training and validation accuracy and loss are monitored throughout the training process.

---

# Model 2: EfficientNet

**Notebook:** `efficientnet-oral-squamous-dataset.ipynb`

### Architecture

EfficientNet employs compound scaling to balance network depth, width, and image resolution efficiently.

Main components include:

* MBConv blocks
* Squeeze-and-Excitation (SE) modules
* Compound scaling strategy
* Fully connected classifier

### CBAM Integration

CBAM is incorporated to further enhance feature representation by emphasizing important spatial and channel-wise information.

### Training Configuration

* Optimizer: **Adam**
* Loss Function: **CrossEntropyLoss**

### Performance Evaluation

Training and validation accuracy and loss curves are analyzed to evaluate model performance.

---

# Model 3: Hybrid Model (MobileNetV2 + EfficientNet)

**Notebook:** `mobilenet-efficientnet-oral-squamous-dataset.ipynb`

### Architecture

The hybrid model combines complementary features extracted from both MobileNetV2 and EfficientNet.

The workflow includes:

1. Feature extraction using MobileNetV2
2. Feature extraction using EfficientNet
3. Feature-level concatenation
4. CBAM-based attention refinement
5. Fully connected classification layer

This fusion strategy enables the model to capture richer and more discriminative representations of histopathological images.

### Training Configuration

* Optimizer: **Adam**
* Loss Function: **CrossEntropyLoss**

### Performance Evaluation

The hybrid model is compared against the individual CNN architectures based on accuracy and F1-score.

---

# Performance Analysis

The models are evaluated using the following metrics:

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* F1 Score
* Training Loss
* Validation Loss

## Experimental Results

| Model                          | Architecture   | Attention | Train Accuracy | Validation Accuracy | Test Accuracy | Test F1 Score |
| ------------------------------ | -------------- | --------- | -------------: | ------------------: | ------------: | ------------: |
| MobileNetV2                    | Single CNN     | No        |         98.12% |              91.67% |        88.89% |        85.03% |
| MobileNetV2                    | Single CNN     | CBAM      |         98.32% |              94.17% |        89.68% |        86.24% |
| EfficientNet                   | Single CNN     | No        |         98.63% |              92.50% |        88.10% |        83.40% |
| EfficientNet                   | Single CNN     | CBAM      |         98.83% |              90.83% |        90.48% |        86.87% |
| **MobileNetV2 + EfficientNet** | **Hybrid CNN** | **CBAM**  |     **99.53%** |          **90.83%** |    **92.06%** |    **89.06%** |

### Key Observations

* CBAM consistently improves feature learning compared to the baseline models.
* MobileNetV2 benefits from attention by achieving higher validation and test accuracy.
* EfficientNet with CBAM provides improved test performance over the baseline.
* The hybrid feature fusion model achieves the **highest overall test accuracy (92.06%)** and **F1-score (89.06%)**, demonstrating the effectiveness of combining complementary deep features.

---

# t-SNE Visualization

To better understand the learned feature representations, **t-Distributed Stochastic Neighbor Embedding (t-SNE)** is used to project high-dimensional features into a two-dimensional space.

The visualization is generated for:

* MobileNetV2
* EfficientNet
* Hybrid Model

The t-SNE plots illustrate how well the models separate **Normal** and **OSCC** samples in the learned feature space. Improved cluster separation indicates stronger discriminative feature learning and better classification capability.

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV

---

# Conclusion

This project demonstrates the effectiveness of lightweight CNN architectures for automated **Oral Squamous Cell Carcinoma (OSCC)** detection from histopathological images. Integrating **CBAM attention** improves the models' ability to focus on diagnostically relevant features, leading to enhanced classification performance. Among the evaluated approaches, the **Hybrid MobileNetV2 + EfficientNet model with CBAM** achieved the best overall results, making it a promising solution for computer-aided oral cancer diagnosis.
