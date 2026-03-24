# -*- coding: utf-8 -*-
"""
m5_lens_cropping.py

ONCOLENS - MODULE 5: LENS CROPPING
===================================
Crop individual lenses from frames based on detected circles.

Features:
    - Crop circular regions from frames
    - Process all frames for each lens
    - Skip edge-cut circles (only crop usable circles)
    - Convert to grayscale
    - Save cropped lenses in organized folders

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m5_lens_cropping.py

Author: Antardip Himel
Date: March 2026
"""

import os
import cv2
import numpy as np

from . import m1_config as config


def crop_circle(frame, center_x, center_y, radius, output_size=None, padding=5):
    """
    Crop a circular region from frame.
    
    Args:
        frame: Input frame (BGR or grayscale)
        center_x: Circle center X coordinate
        center_y: Circle center Y coordinate
        radius: Circle radius
        output_size: Optional (width, height) to resize output
        padding: Extra padding around circle
        
    Returns:
        Cropped image
    """
    h, w = frame.shape[:2]
    
    # Calculate bounding box with padding
    r = int(radius + padding)
    x1 = max(0, int(center_x - r))
    y1 = max(0, int(center_y - r))
    x2 = min(w, int(center_x + r))
    y2 = min(h, int(center_y + r))
    
    # Crop
    cropped = frame[y1:y2, x1:x2].copy()
    
    # Resize if needed
    if output_size is not None:
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)
    
    return cropped


def crop_lens_from_frames(frames, circle, output_size=None):
    """
    Crop a lens from all frames.
    
    Args:
        frames: List of frames (numpy arrays)
        circle: Circle dict with center_x, center_y, radius
        output_size: Optional output size (width, height)
        
    Returns:
        List of cropped frames
    """
    cropped_frames = []
    
    for frame in frames:
        cropped = crop_circle(
            frame,
            circle['center_x'],
            circle['center_y'],
            circle['radius'],
            output_size
        )
        cropped_frames.append(cropped)
    
    return cropped_frames


def crop_all_lenses(frames_dir, circles, output_dir, output_size=None, 
                    progress_callback=None):
    """
    Crop all lenses from all frames.
    
    IMPORTANT: Only crops USABLE circles (not edge-cut).
    Edge-cut circles are skipped as they would produce incomplete images.
    
    Saves crops in ORIGINAL COLOR. Grayscale conversion is handled
    by the classifier at prediction time.
    
    Args:
        frames_dir: Directory containing frame images
        circles: List of circle dicts from MATLAB detection
        output_dir: Output directory for cropped lenses
        output_size: Optional output size (width, height), default from config
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        dict with:
            - success: True/False
            - num_lenses: number of lenses cropped
            - lenses: list of lens info dicts
            - output_dir: path to output directory
            - error: error message if failed
    """
    if output_size is None:
        output_size = config.IMAGE_SIZE
    
    # Load all frames
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ])
    
    if not frame_files:
        return {
            'success': False,
            'error': 'No frames found in directory',
            'num_lenses': 0
        }
    
    if progress_callback:
        progress_callback(f"Loading {len(frame_files)} frames...", 5)
    
    # Read all frames
    frames = []
    for f in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, f))
        if frame is not None:
            frames.append(frame)
    
    if not frames:
        return {
            'success': False,
            'error': 'Failed to load frames',
            'num_lenses': 0
        }
    
    if len(frames) != config.NUM_FRAMES:
        if progress_callback:
            progress_callback(f"Warning: Expected {config.NUM_FRAMES} frames, got {len(frames)}", 8)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out edge-cut circles - only crop usable ones
    usable_circles = [c for c in circles if not c.get('is_edge', False)]
    total_circles = len(circles)
    num_edge_cut = total_circles - len(usable_circles)
    
    if progress_callback:
        progress_callback(f"Cropping {len(usable_circles)} usable circles (skipping {num_edge_cut} edge-cut)", 10)
    
    # Crop each usable lens
    cropped_lenses = []
    
    for idx, circle in enumerate(usable_circles):
        # Use circle_id from MATLAB detection (1-based, already sequential for usable)
        circle_id = circle.get('circle_id', idx + 1)
        lens_id = f"lens_{circle_id:03d}"
        
        if progress_callback:
            pct = 10 + int((idx / len(usable_circles)) * 85)
            progress_callback(f"Cropping {lens_id} ({idx+1}/{len(usable_circles)})", pct)
        
        # Create lens output directory
        lens_dir = os.path.join(output_dir, lens_id)
        os.makedirs(lens_dir, exist_ok=True)
        
        # Crop from all frames
        lens_frames = []
        for frame_idx, frame in enumerate(frames):
            cropped = crop_circle(
                frame,
                circle['center_x'],
                circle['center_y'],
                circle['radius'],
                output_size
            )
            
            lens_frames.append(cropped)
            
            # Save frame in ORIGINAL COLOR
            frame_path = os.path.join(lens_dir, f"frame_{frame_idx+1:03d}.png")
            cv2.imwrite(frame_path, cropped)
        
        cropped_lenses.append({
            'lens_id': lens_id,
            'circle_id': circle_id,
            'circle': circle,
            'frames': lens_frames,
            'output_dir': lens_dir,
            'num_frames': len(lens_frames)
        })
    
    if progress_callback:
        progress_callback(f"Cropped {len(cropped_lenses)} lenses!", 100)
    
    return {
        'success': True,
        'num_lenses': len(cropped_lenses),
        'num_skipped_edge': num_edge_cut,
        'lenses': cropped_lenses,
        'output_dir': output_dir
    }


def load_lens_frames(lens_dir):
    """
    Load all frames for a single lens.
    
    Args:
        lens_dir: Directory containing lens frames
        
    Returns:
        List of frames as numpy arrays (original color)
    """
    frame_files = sorted([
        f for f in os.listdir(lens_dir)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ])
    
    frames = []
    for f in frame_files:
        frame = cv2.imread(os.path.join(lens_dir, f))
        if frame is not None:
            frames.append(frame)
    
    return frames


def get_lens_thumbnail(lens_frames, frame_idx=0, size=(96, 96)):
    """
    Get thumbnail image of a lens (first frame).
    
    Args:
        lens_frames: List of frames
        frame_idx: Which frame to use as thumbnail
        size: Thumbnail size
        
    Returns:
        Thumbnail image (BGR for display)
    """
    if not lens_frames:
        return None
    
    frame = lens_frames[min(frame_idx, len(lens_frames) - 1)]
    
    # Convert to BGR for display if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def create_lens_montage(lens_frames, cols=6, frame_size=(64, 64)):
    """
    Create a montage image showing multiple frames from a lens.
    
    Args:
        lens_frames: List of frames
        cols: Number of columns
        frame_size: Size of each frame in montage
        
    Returns:
        Montage image (BGR)
    """
    if not lens_frames:
        return None
    
    n = len(lens_frames)
    rows = (n + cols - 1) // cols
    
    # Resize and convert frames
    resized = []
    for frame in lens_frames:
        f = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        resized.append(f)
    
    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
    
    # Create montage
    montage_rows = []
    for r in range(rows):
        row_frames = resized[r * cols:(r + 1) * cols]
        montage_rows.append(np.hstack(row_frames))
    
    return np.vstack(montage_rows)


def count_lenses_in_directory(lenses_dir):
    """
    Count number of lens folders in a directory.
    
    Args:
        lenses_dir: Directory containing lens subfolders
        
    Returns:
        Number of lens folders
    """
    if not os.path.exists(lenses_dir):
        return 0
    
    return len([
        d for d in os.listdir(lenses_dir)
        if os.path.isdir(os.path.join(lenses_dir, d))
    ])