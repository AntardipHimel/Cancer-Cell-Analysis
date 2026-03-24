# Cancer Cell Analysis

Automated cancer cell classification from microscopy videos using deep learning. This repository contains the complete research pipeline — from raw video processing to a production-ready GUI application — developed for classifying cancer cells captured through microlens array imaging.

## OncoLens

OncoLens is the standalone desktop application built from this research. It provides a 6-step guided workflow for analyzing microscopy videos: drift correction, circle detection, lens cropping, deep learning classification, and result export.

**Key results:**
- ResNet+LSTM: **96.58%** test accuracy, AUC 0.988
- 3D CNN (ResNet3D-18): **94.01%** test accuracy, AUC 0.983

## Pipeline Overview

```
Microscopy Video
       |
       v
[1] Drift Correction -----> Phase correlation stabilization, 30 frame extraction
       |
       v
[2] Circle Detection -----> MATLAB Hough Transform, NMS, edge/overlap handling
       |
       v
[3] Lens Cropping ---------> Individual lens extraction (96x96), color preserved
       |
       v
[4] Classification --------> ResNet+LSTM or 3D CNN (contain_cell vs no_cell)
       |
       v
[5] Export -----------------> CSV, JSON, organized folders, videos, report
```

## Repository Structure

```
Cancer-Cell-Analysis/
|
|-- OncoLens/                          # Standalone GUI application
|   |-- oncolens_app.py                # Main PyQt5 application (6-step workflow)
|   |-- modules/
|   |   |-- m1_config.py               # Configuration and paths
|   |   |-- m2_video_utils.py          # Video reading and metadata
|   |   |-- m3_drift_correction.py     # Phase correlation drift correction
|   |   |-- m4_matlab_bridge.py        # Python-MATLAB interface
|   |   |-- m5_lens_cropping.py        # Lens extraction from frames
|   |   |-- m6_models.py               # ResNet+LSTM and 3D CNN architectures
|   |   |-- m7_classifier.py           # Inference and prediction
|   |   |-- m8_export_utils.py         # CSV, JSON, video, report export
|   |   +-- __init__.py
|   |-- matlab/
|   |   +-- detect_circles.m           # Hough Transform circle detection
|   |-- assets/                        # Application icons and logos
|   |-- models/                        # Trained model checkpoints (not in repo)
|   |-- input/                         # Input videos (not in repo)
|   +-- output/                        # Processing results (not in repo)
|
|-- scripts/                           # Research and development scripts
|   |-- 1_1_video_log.py               # Video metadata extraction
|   |-- 1_2_drift_correct_and_extract_frames.py
|   |-- 1_4_crop_lens.py               # Lens cropping (color)
|   |-- 1_5_crop_lens_gray.py          # Lens cropping (grayscale)
|   |-- 1_6_crop_lens_gray_pure.py     # Lens cropping (pure grayscale)
|   |-- 2_1_cell_classification_gui.py # Early classification GUI prototype
|   |-- 2_2_extract_features.py        # Feature extraction
|   |-- 2_3_frame_level_feature.py     # Frame-level analysis
|   |-- 2_4_temporal_gradient_analysis.py
|   |-- 2_5_temporal_analysis.py       # Temporal pattern analysis
|   |-- 2_5b_improved_gap_visuals.py
|   |-- 3_1_feature_analysis.py        # Statistical feature analysis
|   |-- 3_2_Edge_Gradient.py           # Edge gradient features
|   |-- 3_3_intensity_analysis.py      # Pixel intensity analysis
|   |-- 3_4_entropy_analysis.py        # Entropy-based features
|   |-- 3_5_glcm_analysis.py           # GLCM texture features
|   |-- 3_6_opticalflow_analysis.py    # Optical flow analysis
|   |-- 4_1_check_image_sizes.py       # Dataset validation
|   |-- 4_2_check_class_distribution.py
|   |-- 4_3_prepare_dataset.py         # Dataset preparation (multi-class)
|   |-- 4_4_dataset.py                 # PyTorch dataset class
|   |-- 4_5_model.py                   # Model architecture (multi-class)
|   |-- 4_6_train.py                   # Training script (multi-class)
|   |-- 4_7_evaluate.py                # Evaluation (multi-class)
|   |-- 5_1_prepare_dataset_2class.py  # 2-class dataset preparation
|   |-- 5_2_train_2class.py            # Training (contain_cell vs no_cell)
|   |-- 5_3_evaluate_2class.py         # Evaluation (2-class)
|   |-- 5_4_visualize_predictions.py   # Prediction visualization
|   |-- 6_1_train_3dcnn.py             # 3D CNN training
|   |-- 6_2_evaluate_3dcnn.py          # 3D CNN evaluation
|   |-- 6_3_visualize_predictions_3dcnn.py
|   |-- 7_1_prepare_dataset_good.py    # Good vs Not Good dataset
|   |-- 7_2_train_good.py              # Good vs Not Good training
|   |-- 7_3_evaluate_good.py           # Good vs Not Good evaluation
|   |-- 7_4_visualize_good.py
|   |-- cell_classifier_app.py         # Standalone classifier app
|   +-- step_1_3_detect_circles.m      # Original MATLAB circle detection
|
|-- .gitignore
|-- LICENSE
+-- setup_git.ps1
```

## Model Architectures

### ResNet+LSTM (Recommended)

The primary model combines spatial feature extraction with temporal sequence modeling:

- **Backbone:** ResNet-18 (modified for single-channel grayscale input)
- **Temporal:** Bidirectional LSTM (2 layers, 256 hidden units)
- **Input:** 30 frames per lens, 96x96 grayscale
- **Parameters:** ~12.4M
- **Accuracy:** 96.58% (test), AUC 0.988

### 3D CNN (ResNet3D-18)

An alternative architecture that processes spatial and temporal dimensions jointly:

- **Architecture:** Custom 3D ResNet-18 with residual blocks
- **Input:** 30 frames per lens, 96x96 grayscale (as 3D volume)
- **Parameters:** ~14.1M
- **Accuracy:** 94.01% (test), AUC 0.983

### Good vs Not Good Classifier

A secondary ResNet+LSTM model for quality assessment of cell-containing lenses.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, not required)
- MATLAB R2020a+ (for circle detection step)

### Setup

```bash
git clone https://github.com/AntardipHimel/Cancer-Cell-Analysis.git
cd Cancer-Cell-Analysis/OncoLens

pip install torch torchvision
pip install PyQt5
pip install opencv-python
pip install numpy
pip install matplotlib
pip install Pillow
```

### Model Weights

Trained model checkpoints are not included in this repository due to file size. Place them in `OncoLens/models/`:

```
OncoLens/models/
  |-- resnet_lstm_2class.pt          # ResNet+LSTM (96.58%)
  |-- cnn3d_2class.pt                # 3D CNN (94.01%)
  +-- resnet_lstm_good_vs_notgood.pt # Quality classifier
```

## Usage

### OncoLens GUI

```bash
cd OncoLens
python oncolens_app.py
```

The application guides you through 6 steps:

1. **Select Video** — Browse and preview microscopy video files
2. **Drift Correction** — Stabilize video using phase correlation, extract 30 frames
3. **Circle Detection** — Detect circular lenses using MATLAB Hough Transform
4. **Lens Cropping** — Crop individual lenses from all frames (96x96, color preserved)
5. **Classification** — Classify each lens as `contain_cell` or `no_cell`
6. **Export** — Save results as CSV/JSON, organize by class, generate report

### Output Structure

```
output/video_name_YYYYMMDD_HHMMSS/
  |-- 1_extracted_frames/    # Drift-corrected frames
  |-- 2_circle_detection/    # MATLAB detection results and annotated images
  |-- 3_cropped_lenses/      # Individual lens frames (color)
  |-- 4_classification/      # CSV and JSON results
  |-- 5_classified_lenses/   # Lenses organized by predicted class
  |-- 6_videos/              # Lens videos organized by class
  +-- report.txt             # Summary report
```

## Research Progression

The `scripts/` folder documents the full research journey:

| Phase | Scripts | Description |
|-------|---------|-------------|
| Data Processing | 1_1 to 1_6 | Video logging, drift correction, lens cropping |
| Feature Analysis | 2_1 to 2_5, 3_1 to 3_6 | Temporal gradients, intensity, entropy, GLCM, optical flow |
| Multi-class Model | 4_1 to 4_7 | Initial multi-class classification attempt |
| 2-Class Model | 5_1 to 5_4 | Refined binary classification (contain_cell vs no_cell) |
| 3D CNN | 6_1 to 6_3 | Alternative 3D convolutional architecture |
| Quality Model | 7_1 to 7_4 | Good vs Not Good secondary classifier |

## Technical Details

- **Drift Correction:** Sub-pixel phase correlation against a reference frame
- **Circle Detection:** Two-pass Hough Transform with non-max suppression, edge-cut filtering, and overlap shrinking
- **Preprocessing:** CLAHE enhancement, median filtering, binary thresholding
- **Training:** AdamW optimizer, cosine annealing LR schedule, early stopping
- **Inference:** Automatic grayscale conversion and resizing at prediction time

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- PyQt5 >= 5.15
- OpenCV >= 4.5
- NumPy
- Matplotlib
- Pillow
- MATLAB R2020a+ (circle detection only)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

**Antardip Himel**

- GitHub: [@AntardipHimel](https://github.com/AntardipHimel)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{himel2026oncolens,
  author = {Himel, Antardip},
  title = {OncoLens: Cancer Cell Classification from Microscopy Videos},
  year = {2026},
  url = {https://github.com/AntardipHimel/Cancer-Cell-Analysis}
}
```
