# Tea Leaf Disease Classification
### Transfer Learning with VGG16 & ResNet50 | PyTorch

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
End-to-end deep learning pipeline classifying tea leaf diseases using transfer learning on two ImageNet-pretrained CNN architectures.

**Classes:** Algal Leaf · Brown Blight · White Spot  
**Dataset:** 368 images | Split: 70% train / 15% val / 15% test

## Results
| Model    | Test Accuracy | F1-Score (macro) | Params (trainable) |
|----------|:------------:|:----------------:|:------------------:|
| VGG16    | [XX.X%]      | [XX.X%]          | ~15M               |
| ResNet50 | [XX.X%]      | [XX.X%]          | ~24M               |

## Architecture
- **VGG16** — last 2 conv blocks fine-tuned + custom head: `25088→1024→256→3`  
- **ResNet50** — layer4 fine-tuned + custom head: `2048→512→3`

## Training Setup
- Optimiser: AdamW | LR: 1e-4 with CosineAnnealingLR
- Loss: CrossEntropy with label smoothing 0.1
- Augmentation: flip, rotation, colour jitter, affine transform
- Hardware: Google Colab T4 GPU

## Repository Structure
```
├── tea_disease_classification.ipynb   # main notebook
├── training_curves.png
├── confusion_matrices.png
├── comparison_bar.png
├── report.pdf
└── README.md
```

## How to Run
1. Open notebook in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Upload the dataset zip when prompted in Cell 2
4. Run all cells - total time ~20 min
