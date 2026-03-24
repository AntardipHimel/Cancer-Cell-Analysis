# -*- coding: utf-8 -*-
"""
OncoLens Modules Package
=========================

Cancer Cell Classification Software

Modules:
    m1_config           - Configuration and settings
    m2_video_utils      - Video reading and processing
    m3_drift_correction - Drift correction and frame extraction
    m4_matlab_bridge    - MATLAB interface for circle detection
    m5_lens_cropping    - Crop lenses from frames
    m6_models           - Neural network models (ResNet+LSTM, 3D CNN)
    m7_classifier       - Classification functions
    m8_export_utils     - Export results, reports, videos

Usage:
    from modules import m1_config as config
    from modules import m7_classifier
    
    classifier = m7_classifier.LensClassifier('resnet_lstm')
    classifier.load()
    result = classifier.predict(frames)

Author: Antardip Himel
Date: March 2026
"""

# Import all modules for easy access
from . import m1_config
from . import m2_video_utils
from . import m3_drift_correction
from . import m4_matlab_bridge
from . import m5_lens_cropping
from . import m6_models
from . import m7_classifier
from . import m8_export_utils

# Version
__version__ = "1.0.0"
__author__ = "Antardip Himel"

# Convenient aliases
config = m1_config
video_utils = m2_video_utils
drift_correction = m3_drift_correction
matlab_bridge = m4_matlab_bridge
lens_cropping = m5_lens_cropping
models = m6_models
classifier = m7_classifier
export_utils = m8_export_utils