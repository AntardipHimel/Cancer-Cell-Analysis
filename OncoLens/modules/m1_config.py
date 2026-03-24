# -*- coding: utf-8 -*-
"""
m1_config.py

ONCOLENS - MODULE 1: CONFIGURATION
===================================
All paths, settings, and constants for the pipeline.

Folder Structure:
    OncoLens/
    ├── input/                  ← Default input folder
    ├── output/                 ← All outputs saved here
    ├── models/                 ← Trained model checkpoints
    │   ├── resnet_lstm_2class.pt
    │   ├── cnn3d_2class.pt
    │   └── resnet_lstm_good_vs_notgood.pt
    ├── modules/
    ├── matlab/
    └── assets/

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m1_config.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import torch

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# BASE PATHS
# =============================================================================

def get_app_root():
    """Get application root directory (works for script and exe)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OncoLens application directory
ONCOLENS_DIR = get_app_root()

# Default input/output directories
DEFAULT_INPUT_DIR = os.path.join(ONCOLENS_DIR, "input")
DEFAULT_OUTPUT_DIR = os.path.join(ONCOLENS_DIR, "output")

# =============================================================================
# MODEL PATHS (inside OncoLens/models/ for portability)
# =============================================================================

MODELS_DIR = os.path.join(ONCOLENS_DIR, "models")

# Model 1: ResNet+LSTM (2-class: contain_cell vs no_cell) - 96.58%
RESNET_LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "resnet_lstm_2class.pt")

# Model 2: 3D CNN (2-class: contain_cell vs no_cell) - 94.01%
CNN3D_MODEL_PATH = os.path.join(MODELS_DIR, "cnn3d_2class.pt")

# Model 3: ResNet+LSTM (Good vs Not Good)
GOOD_VS_NOTGOOD_MODEL_PATH = os.path.join(MODELS_DIR, "resnet_lstm_good_vs_notgood.pt")

# =============================================================================
# MATLAB CONFIGURATION
# =============================================================================

MATLAB_EXECUTABLE = "matlab"
MATLAB_SCRIPT_DIR = os.path.join(ONCOLENS_DIR, "matlab")

# Assets directory
ASSETS_DIR = os.path.join(ONCOLENS_DIR, "assets")

# =============================================================================
# OUTPUT FOLDER STRUCTURE
# =============================================================================

OUTPUT_FOLDERS = {
    'frames': '1_extracted_frames',
    'circles': '2_circle_detection',
    'lenses': '3_cropped_lenses',
    'classification': '4_classification',
    'classified': '5_classified_lenses',
    'videos': '6_videos',
}

# =============================================================================
# VIDEO PROCESSING SETTINGS
# =============================================================================

NUM_FRAMES = 30
VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm')

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

IMAGE_SIZE = (96, 96)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# =============================================================================
# MODEL SETTINGS
# =============================================================================

NUM_CLASSES = 2
CLASS_NAMES = ['no_cell', 'contain_cell']
CLASS_NAMES_DISPLAY = {
    'no_cell': 'No Cell',
    'contain_cell': 'Contains Cell'
}

RESNET_LSTM_CONFIG = {
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'num_classes': NUM_CLASSES
}

CNN3D_CONFIG = {
    'dropout': 0.5,
    'num_classes': NUM_CLASSES
}

GOOD_VS_NOTGOOD_CONFIG = {
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'num_classes': 2
}

GOOD_VS_NOTGOOD_CLASS_NAMES = ['not_good', 'good']

# =============================================================================
# CLASSIFICATION SETTINGS
# =============================================================================

CONFIDENCE_THRESHOLD = 0.8
BATCH_SIZE = 1

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

VIDEO_FPS = 10
VIDEO_CODEC = 'XVID'

# =============================================================================
# UI SETTINGS
# =============================================================================

APP_NAME = "OncoLens"
APP_VERSION = "1.0.0"
WINDOW_TITLE = "OncoLens - Cancer Cell Classification"
WINDOW_SIZE = "1300x850"
THEME = 'clam'

COLOR_PRIMARY = '#1976D2'
COLOR_PRIMARY_DARK = '#0D47A1'
COLOR_PRIMARY_LIGHT = '#64B5F6'
COLOR_SUCCESS = '#4CAF50'
COLOR_WARNING = '#FF9800'
COLOR_ERROR = '#F44336'
COLOR_BG = '#F5F5F5'
COLOR_TEXT = '#212121'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def setup_default_directories():
    """Create default directories if they don't exist."""
    ensure_dir(DEFAULT_INPUT_DIR)
    ensure_dir(DEFAULT_OUTPUT_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(MATLAB_SCRIPT_DIR)
    ensure_dir(ASSETS_DIR)


def create_output_structure(video_name, output_base=None, timestamp=True):
    """Create complete output folder structure for a video."""
    from datetime import datetime
    
    if output_base is None:
        output_base = DEFAULT_OUTPUT_DIR
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{video_name}_{ts}"
    else:
        folder_name = video_name
    
    main_output = os.path.join(output_base, folder_name)
    ensure_dir(main_output)
    
    paths = {'main': main_output}
    
    for key, subfolder in OUTPUT_FOLDERS.items():
        folder_path = os.path.join(main_output, subfolder)
        ensure_dir(folder_path)
        paths[key] = folder_path
    
    ensure_dir(os.path.join(paths['classified'], 'contain_cell'))
    ensure_dir(os.path.join(paths['classified'], 'no_cell'))
    ensure_dir(os.path.join(paths['videos'], 'contain_cell'))
    ensure_dir(os.path.join(paths['videos'], 'no_cell'))
    
    return paths


def get_model_path(model_type):
    """Get model checkpoint path based on type."""
    if model_type == 'resnet_lstm':
        return RESNET_LSTM_MODEL_PATH
    elif model_type == '3dcnn':
        return CNN3D_MODEL_PATH
    elif model_type == 'good_vs_notgood':
        return GOOD_VS_NOTGOOD_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def check_model_exists(model_type):
    """Check if a model checkpoint exists."""
    path = get_model_path(model_type)
    return os.path.exists(path)


def get_device_info():
    """Get information about the compute device."""
    info = {
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    return info


def validate_config():
    """Validate configuration on import."""
    warnings = []
    
    if not check_model_exists('resnet_lstm'):
        warnings.append(f"ResNet+LSTM model not found: {RESNET_LSTM_MODEL_PATH}")
    
    if not check_model_exists('3dcnn'):
        warnings.append(f"3D CNN model not found: {CNN3D_MODEL_PATH}")
    
    if not check_model_exists('good_vs_notgood'):
        warnings.append(f"Good vs NotGood model not found: {GOOD_VS_NOTGOOD_MODEL_PATH}")
    
    return warnings


# Setup default directories and run validation on import.
# Guarded by _ONCOLENS_SKIP_INIT env var for testing/import-only usage.
if not os.environ.get('_ONCOLENS_SKIP_INIT'):
    setup_default_directories()
    
    _config_warnings = validate_config()
    if _config_warnings:
        print("[!] OncoLens Configuration Warnings:")
        for w in _config_warnings:
            print(f"   - {w}")