# Weedy Rice Multimodal Segmentation

## 1. Project Overview

This experiment investigates multimodal deep learning for Weedy Rice segmentation using RGB and multispectral imagery.

The objective is to compare single-modality models with early-fusion and late-fusion architectures for pixel-level segmentation.

## 2. Dataset

**Dataset:** WeedyRice-RGBMS-DB

| Split | Number of samples |
|---|---:|
| Training | 438 |
| Validation | 148 |
| Testing | 148 |
| **Total** | **734** |

The raw dataset is not included in this repository because of its size.

## 3. Models

Four architectures are evaluated:

1. **RGB U-Net:** Uses three-channel RGB imagery.
2. **Multispectral U-Net:** Uses four-channel multispectral imagery.
3. **Early Fusion U-Net:** Concatenates RGB and multispectral inputs into a seven-channel input.
4. **Late Fusion U-Net:** Uses separate RGB and multispectral encoders and combines their learned features.

## 4. Test Results

| Model | Dice / F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| RGB U-Net | 0.8465 | 0.7339 | 0.8195 | 0.8753 |
| Multispectral U-Net | 0.8690 | 0.7684 | 0.8596 | 0.8786 |
| Early Fusion U-Net | **0.8710** | **0.7714** | **0.9019** | 0.8421 |
| Late Fusion U-Net | 0.8616 | 0.7568 | 0.8170 | **0.9113** |

## 5. Initial Observations

- Early Fusion achieved the highest Dice score, IoU, and precision.
- Late Fusion achieved the highest recall.
- Multispectral U-Net outperformed RGB U-Net on the reported metrics.
- These results represent one training run and one test split. Further experiments are required to establish statistical significance.

## 6. Visualizations

Five qualitative comparison images are included in `results/visualizations/`.

Each comparison includes:
- RGB input
- Multispectral input
- Ground-truth mask
- RGB U-Net prediction
- Multispectral U-Net prediction
- Early Fusion U-Net prediction
- Late Fusion U-Net prediction

## 7. Project Structure

```text
weedy_rice/
├── README.md
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
└── results/
    ├── test_metrics.csv
    └── visualizations/
```

## 8. Implementation

Implemented using Python and PyTorch.

The project includes:
- Dataset loading and preprocessing
- U-Net-based segmentation architectures
- RGB and multispectral feature processing
- Early and late multimodal fusion
- Model training and evaluation
- Pixel-level segmentation metrics
- Qualitative prediction visualization

## 9. Reproducibility

Raw dataset files, model checkpoints, and training-history files are excluded from this repository because of their size.

The evaluation metrics and