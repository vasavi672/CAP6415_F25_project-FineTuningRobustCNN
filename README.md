# CAP6415_F25_project-FineTuningRobustCNN

## Fine-Tuning Compact CNNs for Robust Image Recognition on Corrupted Data

## Overview
This project is developed as part of **CAP6415 - Computer Vision** (Fall 2025).  
The goal is to enhance the robustness of small, efficient convolutional neural networks (CNNs) such as EfficientNet-B0 and MobileNetV3-Small when facing visually degraded images.  
We fine-tune pretrained models using corrupted datasets to improve classification performance under challenging conditions (noise, blur, brightness distortion, etc.).

## Motivation
Deep learning models trained on clean images often fail when tested on real-world data containing noise, blur, or other imperfections.  
This project aims to bridge that gap by exploring whether compact models—optimized for speed and efficiency—can be fine-tuned to handle corrupted data more effectively without significant computational overhead.

## Objectives
1. Establish baseline performance of pretrained models on clean and corrupted datasets.
2. Fine-tune models using a mixture of clean and corrupted images.
3. Evaluate improvements in robustness and generalization.
4. Visualize and report results for reproducibility.

## Datasets
- **CIFAR-10:** 60,000 images (32x32 pixels) across 10 object classes.
- **CIFAR-10-C:** Benchmark dataset introducing 15 corruption types (noise, blur, weather effects, digital distortions) at 5 severity levels.

Both datasets are publicly available:
- CIFAR-10: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-10-C: [https://zenodo.org/record/2535967](https://zenodo.org/record/2535967)

## Model Architecture
The following pretrained models (on ImageNet) will be evaluated and fine-tuned:
- **EfficientNet-B0**
- **MobileNetV3-Small**

Both models are compact, well-suited for resource-constrained environments, and available through `torchvision.models` or `timm`.

## Methodology
1. **Baseline Evaluation:** Measure pretrained model accuracy on clean CIFAR-10 and corrupted CIFAR-10-C.
2. **Fine-Tuning:** Retrain models on a balanced mix of clean and corrupted samples.
3. **Evaluation:** Test fine-tuned models on unseen corruption types and compare metrics (accuracy, confusion matrix, corruption-specific robustness).
4. **Visualization:** Generate plots illustrating improvements before and after fine-tuning.

## Project Structure
```
CAP6415_F25_project-FineTuningRobustCNN/
│
├── src/                # Model training and evaluation scripts
├── results/            # Plots, metrics, and qualitative results
├── logs/               # Weekly progress logs
├── notebooks/          # Experiments and visualization notebooks
├── README.md           # Project summary and documentation
└── Documentation.pdf    # Installation and reproducibility details
```

## Dependencies
To install all dependencies:
```bash
pip install -r requirements.txt
```
Required packages include:
- Python 3.10+
- PyTorch 2.1+
- Torchvision
- Albumentations
- Matplotlib
- NumPy
- tqdm

## Expected Results
- Robustness improvement on corrupted datasets.
- Visualization of classification performance across corruption types.
- Side-by-side comparison of clean vs. corrupted image predictions.

## Deliverables
| Component | Weight | Description |
|------------|---------|-------------|
| Development Log | 10% | Weekly logs documenting progress |
| Description | 10% | Project summary and structure in this README |
| Documentation | 20% | Environment setup, installation, and usage guide |
| Reproducibility | 30% | Code that runs seamlessly on TA's machine |
| Video Demo | 30% | 10–20 min narrated demonstration of project and results |


