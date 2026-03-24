# -*- coding: utf-8 -*-
"""
m7_classifier.py

ONCOLENS - MODULE 7: CLASSIFIER
================================
Classification functions for lens prediction.

Features:
    - Load trained models
    - Preprocess lens frames
    - Predict single lens or batch
    - Get classification summary

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m7_classifier.py

Author: Antardip Himel
Date: March 2026
"""

import os
import torch
import numpy as np
from PIL import Image

from . import m1_config as config
from . import m6_models as models


class LensClassifier:
    """
    Classifier for lens images using trained models.
    
    Example:
        classifier = LensClassifier('resnet_lstm')
        classifier.load()
        
        result = classifier.predict(frames)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(self, model_type='resnet_lstm', model_path=None, device=None):
        """
        Initialize classifier.
        
        Args:
            model_type: 'resnet_lstm' or '3dcnn'
            model_path: Path to model checkpoint (default: from config)
            device: Torch device (default: from config)
        """
        self.model_type = model_type
        self.device = device or config.DEVICE
        
        if model_path is None:
            model_path = config.get_model_path(model_type)
        
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
    
    def load(self):
        """
        Load the model from checkpoint.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_loaded:
            return True
        
        try:
            self.model = models.load_model(self.model_type, self.model_path, self.device)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_frames(self, frames):
        """
        Preprocess frames for model input.
        
        Handles both color (BGR) and grayscale input frames.
        Converts to grayscale, resizes to 96x96, normalizes to [0, 1].
        
        Args:
            frames: List of frames (numpy arrays, BGR color or grayscale)
            
        Returns:
            Preprocessed tensor ready for model
        """
        import cv2
        
        # Ensure we have exactly NUM_FRAMES frames
        while len(frames) < config.NUM_FRAMES:
            frames.append(frames[-1])  # Pad by repeating last frame
        frames = frames[:config.NUM_FRAMES]
        
        # Convert to grayscale and resize if needed
        processed = []
        for frame in frames:
            # Convert color to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to expected size if needed
            h, w = frame.shape[:2]
            if (w, h) != config.IMAGE_SIZE:
                frame = cv2.resize(frame, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
            processed.append(frame)
        
        # Stack and normalize to [0, 1]
        frames_array = np.stack(processed, axis=0).astype(np.float32)
        frames_array = frames_array / 255.0
        
        if self.model_type == 'resnet_lstm':
            # Shape: (1, 30, 1, 96, 96) - batch, frames, channels, H, W
            frames_array = frames_array[:, np.newaxis, :, :]
            tensor = torch.from_numpy(frames_array).unsqueeze(0)
        
        elif self.model_type == '3dcnn':
            # Shape: (1, 1, 30, 96, 96) - batch, channels, depth, H, W
            frames_array = frames_array[np.newaxis, :, :, :]
            tensor = torch.from_numpy(frames_array).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, frames):
        """
        Predict class for a single lens.
        
        Args:
            frames: List of frames (numpy arrays, grayscale, 96x96)
            
        Returns:
            dict with:
                - prediction: class name ('contain_cell' or 'no_cell')
                - prediction_idx: class index
                - confidence: confidence score
                - probabilities: dict of class probabilities
        """
        if not self.is_loaded:
            if not self.load():
                return None
        
        # Preprocess
        tensor = self.preprocess_frames(frames)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(1).item()
            confidence = probs[pred_idx].item()
        
        return {
            'prediction': config.CLASS_NAMES[pred_idx],
            'prediction_idx': pred_idx,
            'confidence': confidence,
            'probabilities': {
                config.CLASS_NAMES[i]: probs[i].item() 
                for i in range(config.NUM_CLASSES)
            }
        }
    
    def predict_batch(self, lenses_data, progress_callback=None):
        """
        Predict classes for multiple lenses.
        
        Args:
            lenses_data: List of (lens_id, frames) tuples
            progress_callback: Optional callback(message, progress_pct)
            
        Returns:
            List of prediction dicts (each includes lens_id)
        """
        if not self.is_loaded:
            if not self.load():
                return []
        
        results = []
        total = len(lenses_data)
        
        for idx, (lens_id, frames) in enumerate(lenses_data):
            if progress_callback:
                pct = int((idx / total) * 100)
                progress_callback(f"Classifying {lens_id} ({idx+1}/{total})", pct)
            
            result = self.predict(frames)
            if result:
                result['lens_id'] = lens_id
                results.append(result)
        
        if progress_callback:
            progress_callback("Classification complete!", 100)
        
        return results


def classify_lens_directory(lens_dir, classifier):
    """
    Classify a lens from its directory.
    
    Loads frames in original color. The classifier's preprocess_frames()
    handles grayscale conversion and resizing internally.
    
    Args:
        lens_dir: Directory containing lens frames (frame_001.png, etc.)
        classifier: LensClassifier instance
        
    Returns:
        Prediction dict or None if failed
    """
    import cv2
    
    # Load frames in original color
    frame_files = sorted([
        f for f in os.listdir(lens_dir)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ])
    
    if not frame_files:
        return None
    
    frames = []
    for f in frame_files:
        frame = cv2.imread(os.path.join(lens_dir, f))
        if frame is not None:
            frames.append(frame)
    
    if not frames:
        return None
    
    return classifier.predict(frames)


def classify_all_lenses(lenses_dir, classifier, progress_callback=None):
    """
    Classify all lenses in a directory.
    
    Args:
        lenses_dir: Directory containing lens subdirectories
        classifier: LensClassifier instance
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        List of prediction dicts with lens_id and lens_path
    """
    # Find all lens directories
    lens_dirs = sorted([
        d for d in os.listdir(lenses_dir)
        if os.path.isdir(os.path.join(lenses_dir, d))
    ])
    
    if not lens_dirs:
        return []
    
    results = []
    total = len(lens_dirs)
    
    for idx, lens_id in enumerate(lens_dirs):
        if progress_callback:
            pct = int((idx / total) * 100)
            progress_callback(f"Classifying {lens_id} ({idx+1}/{total})", pct)
        
        lens_path = os.path.join(lenses_dir, lens_id)
        result = classify_lens_directory(lens_path, classifier)
        
        if result:
            result['lens_id'] = lens_id
            result['lens_path'] = lens_path
            results.append(result)
    
    if progress_callback:
        progress_callback("Classification complete!", 100)
    
    return results


def get_classification_summary(results):
    """
    Get summary statistics of classification results.
    
    Args:
        results: List of prediction dicts
        
    Returns:
        dict with:
            - total: total lenses
            - contain_cell: count
            - no_cell: count
            - contain_cell_pct: percentage
            - no_cell_pct: percentage
            - avg_confidence: average confidence
            - low_confidence: list of results below threshold
    """
    if not results:
        return {
            'total': 0,
            'contain_cell': 0,
            'no_cell': 0,
            'contain_cell_pct': 0,
            'no_cell_pct': 0,
            'avg_confidence': 0,
            'low_confidence': []
        }
    
    total = len(results)
    contain_cell = sum(1 for r in results if r['prediction'] == 'contain_cell')
    no_cell = total - contain_cell
    avg_conf = sum(r['confidence'] for r in results) / total
    
    # Find low confidence predictions
    low_conf = [r for r in results if r['confidence'] < config.CONFIDENCE_THRESHOLD]
    
    return {
        'total': total,
        'contain_cell': contain_cell,
        'no_cell': no_cell,
        'contain_cell_pct': contain_cell / total * 100,
        'no_cell_pct': no_cell / total * 100,
        'avg_confidence': avg_conf,
        'avg_confidence_pct': avg_conf * 100,
        'low_confidence': low_conf,
        'low_confidence_count': len(low_conf)
    }