# -*- coding: utf-8 -*-
"""
m3_drift_correction.py

ONCOLENS - MODULE 3: DRIFT CORRECTION
=======================================
Stabilizes video and extracts frames using phase correlation.

Features:
    - Compute drift relative to reference frame
    - Apply drift correction to frames
    - Extract evenly spaced corrected frames
    - Save drift log for debugging

Based on: 1_2_drift_correct_and_extract_frames.py

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m3_drift_correction.py

Author: Antardip Himel
Date: March 2026
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime

from . import m1_config as config


def compute_drift_to_reference(video_path, ref_frame_idx, progress_callback=None):
    """
    Compute drift for every frame directly against reference frame.
    Uses phase correlation for sub-pixel accuracy.
    
    Args:
        video_path: Path to video file
        ref_frame_idx: Index of reference frame
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        drift_data dict with:
            - cumulative_drift: (N, 2) array of (dx, dy) per frame
            - frame_drift: frame-to-frame drift
            - ref_frame_idx: reference frame index
            - video info (total_frames, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if progress_callback:
        progress_callback(f"Reading reference frame {ref_frame_idx}...", 5)
    
    # Read reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError(f"Cannot read reference frame {ref_frame_idx}")
    
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # Compute drift for all frames
    cumulative_drift = np.zeros((total_frames, 2))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i == ref_frame_idx:
            cumulative_drift[i] = [0, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            # Phase correlation for drift estimation
            shift, _ = cv2.phaseCorrelate(gray, ref_gray)
            cumulative_drift[i] = [shift[0], shift[1]]
        
        if progress_callback and i % 100 == 0:
            pct = 5 + int((i / total_frames) * 70)
            progress_callback(f"Computing drift: frame {i}/{total_frames}", pct)
    
    cap.release()
    
    # Compute frame-to-frame drift (for logging)
    frame_drift = np.zeros((total_frames, 2))
    frame_drift[0] = cumulative_drift[0]
    for i in range(1, total_frames):
        frame_drift[i] = cumulative_drift[i] - cumulative_drift[i-1]
    
    return {
        'cumulative_drift': cumulative_drift,
        'frame_drift': frame_drift,
        'ref_frame_idx': ref_frame_idx,
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height
    }


def extract_corrected_frames(video_path, drift_data, frame_indices, 
                             output_dir=None, progress_callback=None):
    """
    Extract and drift-correct specific frames.
    
    Args:
        video_path: Path to video
        drift_data: Output from compute_drift_to_reference
        frame_indices: List of frame indices to extract
        output_dir: Optional directory to save frames as PNG
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        List of corrected frames (numpy arrays)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    cumulative_drift = drift_data['cumulative_drift']
    width = drift_data['width']
    height = drift_data['height']
    
    corrected_frames = []
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Get drift for this frame
        dx, dy = cumulative_drift[frame_idx]
        
        # Create translation matrix to correct drift
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        
        # Apply correction
        corrected = cv2.warpAffine(frame, M, (width, height),
                                   borderMode=cv2.BORDER_REFLECT)
        
        corrected_frames.append(corrected)
        
        # Save if output_dir specified
        if output_dir:
            frame_path = os.path.join(output_dir, f"frame_{idx+1:03d}.png")
            cv2.imwrite(frame_path, corrected)
        
        if progress_callback:
            pct = 75 + int((idx / len(frame_indices)) * 20)
            progress_callback(f"Extracting frame {idx+1}/{len(frame_indices)}", pct)
    
    cap.release()
    
    return corrected_frames


def process_video(video_path, output_dir, num_frames=None, 
                  skip_first_last=True, progress_callback=None):
    """
    Full drift correction and frame extraction pipeline.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: from config)
        skip_first_last: Skip first and last video frames
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        dict with:
            - frames: list of corrected frames
            - frame_indices: list of original frame indices
            - output_dir: path where frames were saved
            - log_path: path to extraction log
            - drift_data: full drift data
            - num_frames: number of frames extracted
    """
    if num_frames is None:
        num_frames = config.NUM_FRAMES
    
    if progress_callback:
        progress_callback("Starting drift correction...", 0)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate frame indices (evenly spaced)
    if skip_first_last and total_frames > 2:
        start = 1
        end = total_frames - 1
    else:
        start = 0
        end = total_frames
    
    frame_indices = np.linspace(start, end - 1, num_frames, dtype=int).tolist()
    ref_frame_idx = frame_indices[0]  # First selected frame is reference
    
    if progress_callback:
        progress_callback(f"Reference frame: {ref_frame_idx}", 2)
    
    # Compute drift
    drift_data = compute_drift_to_reference(video_path, ref_frame_idx, progress_callback)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract corrected frames
    if progress_callback:
        progress_callback("Extracting corrected frames...", 75)
    
    frames = extract_corrected_frames(
        video_path, drift_data, frame_indices, 
        output_dir, progress_callback
    )
    
    # Save extraction log
    log_data = {
        'video_path': video_path,
        'video_name': os.path.basename(video_path),
        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_video_frames': drift_data['total_frames'],
        'extracted_frames': num_frames,
        'reference_frame': ref_frame_idx,
        'frame_indices': frame_indices,
        'fps': drift_data['fps'],
        'resolution': f"{drift_data['width']}x{drift_data['height']}",
        'drift_stats': {
            'max_x': float(np.max(np.abs(drift_data['cumulative_drift'][:, 0]))),
            'max_y': float(np.max(np.abs(drift_data['cumulative_drift'][:, 1]))),
            'mean_x': float(np.mean(np.abs(drift_data['cumulative_drift'][:, 0]))),
            'mean_y': float(np.mean(np.abs(drift_data['cumulative_drift'][:, 1]))),
        }
    }
    
    log_path = os.path.join(output_dir, "extraction_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    if progress_callback:
        progress_callback("Drift correction complete!", 100)
    
    return {
        'frames': frames,
        'frame_indices': frame_indices,
        'output_dir': output_dir,
        'log_path': log_path,
        'drift_data': drift_data,
        'num_frames': len(frames)
    }


def get_drift_summary(drift_data):
    """
    Get summary statistics of drift.
    
    Args:
        drift_data: Output from compute_drift_to_reference
        
    Returns:
        Summary string for display
    """
    cumulative = drift_data['cumulative_drift']
    
    max_dx = np.max(np.abs(cumulative[:, 0]))
    max_dy = np.max(np.abs(cumulative[:, 1]))
    mean_dx = np.mean(np.abs(cumulative[:, 0]))
    mean_dy = np.mean(np.abs(cumulative[:, 1]))
    
    return (
        f"Drift Summary:\n"
        f"  Max X drift: {max_dx:.2f} pixels\n"
        f"  Max Y drift: {max_dy:.2f} pixels\n"
        f"  Mean X drift: {mean_dx:.2f} pixels\n"
        f"  Mean Y drift: {mean_dy:.2f} pixels\n"
        f"  Reference frame: {drift_data['ref_frame_idx']}"
    )


def load_extraction_log(log_path):
    """
    Load extraction log from JSON file.
    
    Args:
        log_path: Path to extraction_log.json
        
    Returns:
        Log data as dictionary
    """
    with open(log_path, 'r') as f:
        return json.load(f)