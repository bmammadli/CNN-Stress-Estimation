# CNN-Stress-Estimation

Official implementation of a **Convolutional Neural Network (CNN)** for stress state estimation from deformed surface images.

This work was presented at **PyTorch Conference Europe 2026**.

---

## Overview

Mechanical stress governs material deformation and failure but cannot be directly observed from conventional surface images. This project demonstrates that stress can be estimated directly from deformed surface images using a deep learning framework based on **ResNet18** and transfer learning. The proposed approach enables non-contact stress estimation and provides early insight into material behavior before visible damage appears.

---

## Data Preparation and Model Overview

Approximately **11,000 surface images** were collected from uniaxial tensile experiments on two independent miniaturized **316L stainless steel** specimens (Experiments A and B). Each image was paired with its corresponding stress value. The CNN was trained using **Experiment A** and evaluated on the independent **Experiment B** to assess its generalization capability.

The proposed model is based on a pretrained **ResNet18** with transfer learning and selective fine-tuning. A regression head predicts continuous stress values, while the **Huber loss** is used for robust training.

<p align="center">
  <img src="figures/model.png" width="900">
</p>

---

## Results

The model captures the stress evolution in the training data and achieves an **MAE** of approximately **20 MPa** on an independent experiment, which is **acceptable for engineering applications**. These results show that stress can be inferred directly from surface images, enabling early insight before visible damage appears.

<p align="center">
  <img src="figures/results.png" width="900">
</p>

---

## PyTorch Conference Europe 2026

This project was presented as a research poster at **PyTorch Conference Europe 2026**, demonstrating the application of deep learning for stress estimation directly from deformed surface images. The repository provides the official implementation accompanying the presented work.

---

## Code

The repository contains the complete implementation of the proposed CNN framework in a single Python script:

**`CNN_Stress.py`**

The code includes the complete workflow, including:

- dataset preparation and preprocessing,
- transfer learning with **ResNet18**,
- model training and selective fine-tuning,
- model evaluation on an independent experiment,
- saving the trained model,
- stress prediction using the trained network.

---

