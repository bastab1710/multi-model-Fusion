# Weedy Rice Multimodal Segmentation

## 1. Project Overview

This experiment investigates multimodal deep learning for pixel-level Weedy Rice segmentation using RGB and multispectral imagery.

The objective is to compare single-modality models with early-fusion and late-fusion architectures and evaluate whether combining RGB and multispectral information improves segmentation performance.

Four U-Net-based models are implemented and evaluated using the same dataset split.

## 2. Dataset

**Dataset:** WeedyRice-RGBMS-DB

| Split | Number of Samples |
|---|---:|
| Training | 438 |
| Validation | 148 |
| Testing | 148 |
| **Total** | **734** |

The dataset contains RGB images, multispectral imagery, and corresponding segmentation masks.

### Input Modalities

- **RGB:** Three-channel colour imagery.
- **Multispectral:** Four bands in the order Green (G), Red (R), Red Edge (RE), and Near Infrared (NIR).
- **Ground Truth:** Binary segmentation masks.

Images are resized to 192 × 256 pixels. Bilinear interpolation is used for image resizing, while nearest-neighbour interpolation is used for segmentation masks.

The inspected multispectral TIFF files use 8-bit image values, which are normalized by dividing by 255.

The raw dataset is excluded from this repository because of its size.

## 3. Model Architectures

Four models are implemented:

### 3.1 RGB U-Net
Uses three-channel RGB imagery as input for pixel-level segmentation.

### 3.2 Multispectral U-Net
Uses four-channel multispectral imagery (G, R, RE, NIR) to learn spectral features.

### 3.3 Early Fusion U-Net
RGB and multispectral inputs are concatenated along the channel dimension before being passed into a single U-Net.

The resulting input contains seven channels:
- 3 RGB channels
- 4 multispectral channels

This combines both modalities at the input level.

### 3.4 Late Fusion U-Net
Uses separate RGB and multispectral encoders. Their learned features are combined before the segmentation prediction.

This allows each modality to be processed separately before feature-level fusion.

## 4. Training Configuration

All four models are trained independently using the same training and validation splits.

The implementation includes:

- PyTorch-based model training.
- Adam optimizer.
- Combined Binary Cross-Entropy and Dice loss.
- Validation-based checkpoint selection.
- Fixed random seed of 42.
- Training and validation metric tracking.
- Saving of best model checkpoints and training histories.

Each model was trained for 20 epochs.

The best checkpoint for each model was selected using validation Dice.

### Best Checkpoint Information

| Model | Best Epoch | Best Validation Dice |
|---|---:|---:|
| RGB U-Net | 18 | 0.7986 |
| Multispectral U-Net | 11 | 0.7833 |
| Early Fusion U-Net | 20 | 0.8151 |
| Late Fusion U-Net | 13 | 0.8057 |

These validation Dice values are the values recorded during checkpoint selection. They are batch-averaged training-script metrics.

## 5. Test Evaluation

The four selected checkpoints are evaluated on the held-out test split.

Binary predictions are generated using a sigmoid probability threshold of 0.5.

Test metrics are calculated using globally accumulated pixel-level true positives, false positives, false negatives, and true negatives.

### Test Results

| Model | Dice / F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| RGB U-Net | 0.8465 | 0.7339 | 0.8195 | 0.8753 |
| Multispectral U-Net | 0.8690 | 0.7684 | 0.8596 | 0.8786 |
| Early Fusion U-Net | **0.8710** | **0.7714** | **0.9019** | 0.8421 |
| Late Fusion U-Net | 0.8616 | 0.7568 | 0.8170 | **0.9113** |

### Observations

- Early Fusion U-Net achieved the highest test Dice, IoU, and precision.
- Late Fusion U-Net achieved the highest recall.
- Multispectral U-Net outperformed RGB U-Net on the reported test metrics.
- Both multimodal approaches achieved higher Dice and IoU than the RGB-only baseline.
- Early fusion provided the strongest overall Dice and IoU results in this experiment.

Dice and F1 are equal here because both are calculated using the same binary segmentation counts.

These findings describe one training run and one held-out test split. They do not establish statistical significance or guarantee performance on other datasets.

## 6. Qualitative Visualization

Five qualitative comparison images are included in:

`results/visualizations/`

Each comparison contains seven panels:

1. RGB input.
2. Multispectral false-colour composite.
3. Ground-truth segmentation mask.
4. RGB U-Net prediction.
5. Multispectral U-Net prediction.
6. Early Fusion U-Net prediction.
7. Late Fusion U-Net prediction.

The multispectral false-colour composite uses NIR for red, Red for green, and Green for blue display channels.

These visualizations provide qualitative comparisons between model predictions and ground truth.

## 7. Project Structure

```text
weedy_rice/
|
|-- README.md
|
|-- src/
|   |-- dataset.py
|   |-- models.py
|   |-- metrics.py
|   |-- train.py
|   |-- evaluate.py
|   |-- visualize.py
|
|-- data/
|   |-- raw/                 # Local dataset; excluded from Git
|
|-- models/                  # Local checkpoints; excluded from Git
|
|-- results/
    |-- test_metrics.csv
    |
    |-- visualizations/
        |-- Five comparison images
```

Training histories are saved locally in the results directory and excluded from Git.

## 8. Implementation

The experiment is implemented using:

- Python
- PyTorch
- NumPy
- Pillow
- tifffile

The implementation is organized into six Python modules:

| File | Responsibility |
|---|---|
| `dataset.py` | Dataset loading, preprocessing, and modality handling |
| `models.py` | U-Net and multimodal fusion architectures |
| `metrics.py` | Dice loss and segmentation metrics |
| `train.py` | Training, validation, and checkpoint saving |
| `evaluate.py` | Test-set evaluation |
| `visualize.py` | Qualitative prediction comparisons |

## 9. Reproducibility and Limitations

The repository contains source code, test metrics, and qualitative visualizations.

The raw dataset, trained model checkpoints, and PyTorch training-history files are excluded from Git. These files must be available locally to reproduce evaluation and visualization.

Important experimental limitations:

- Results are based on one training run and one dataset split.
- Multiple independent runs are needed to assess variability and statistical significance.
- Only five test samples are included in the qualitative visualization set.
- Validation metrics used during training are batch-averaged, whereas final test metrics are calculated from globally accumulated pixel counts.
- TIFF channel equality was verified on sampled files, not every TIFF file in the dataset.

Further work could include repeated training runs, data augmentation experiments, and more extensive qualitative analysis.

## 10. Conclusion

This experiment compares RGB-only, multispectral-only, early-fusion, and late-fusion U-Net architectures for Weedy Rice segmentation.

Among the evaluated models, Early Fusion U-Net achieved the highest test Dice score and IoU, while Late Fusion U-Net achieved the highest recall.

The results indicate that combining RGB and multispectral imagery is promising for this segmentation task. Further experiments are required to establish the reliability and generalizability of these findings.