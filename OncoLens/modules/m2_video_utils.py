# -*- coding: utf-8 -*-
"""
m2_video_utils.py

ONCOLENS - MODULE 2: VIDEO UTILITIES
=====================================
Functions for video information, reading, and basic processing.

Features:
    - Get video metadata (frames, fps, duration, resolution)
    - Read specific frames
    - Read evenly spaced frames
    - Convert frames to video
    - Resize and convert to grayscale

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m2_video_utils.py

Author: Antardip Himel
Date: March 2026
"""

import os
import cv2
import numpy as np


def get_video_info(video_path):
    """
    Extract video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict with video info or None if failed
        
    Example:
        info = get_video_info("video.avi")
        print(f"Frames: {info['total_frames']}, FPS: {info['fps']}")
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        if fps > 0:
            duration_sec = total_frames / fps
        else:
            duration_sec = 0
        
        # Format duration as string
        minutes = int(duration_sec // 60)
        seconds = duration_sec % 60
        duration_formatted = f"{minutes}m {seconds:.1f}s"
        
        # Get file size
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        return {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'name': os.path.splitext(os.path.basename(video_path))[0],
            'total_frames': total_frames,
            'fps': round(fps, 2),
            'duration_sec': round(duration_sec, 2),
            'duration_formatted': duration_formatted,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'file_size_mb': round(file_size_mb, 2)
        }
    finally:
        cap.release()


def read_frame(video_path, frame_idx):
    """
    Read a specific frame from video.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)
        
    Returns:
        Frame as numpy array (BGR) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def read_frames_range(video_path, start_idx, end_idx):
    """
    Read a range of frames from video.
    
    Args:
        video_path: Path to video file
        start_idx: Starting frame index (inclusive)
        end_idx: Ending frame index (exclusive)
        
    Returns:
        List of frames as numpy arrays (BGR)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(end_idx - start_idx):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    
    return frames


def read_evenly_spaced_frames(video_path, num_frames=30, skip_first_last=True):
    """
    Read evenly spaced frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        skip_first_last: Skip very first and last video frames (avoids edge artifacts)
        
    Returns:
        List of (frame_idx, frame) tuples
        
    Example:
        frames = read_evenly_spaced_frames("video.avi", num_frames=30)
        for idx, frame in frames:
            print(f"Frame {idx}: shape {frame.shape}")
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define range
        if skip_first_last and total_frames > 2:
            start = 1
            end = total_frames - 1
        else:
            start = 0
            end = total_frames
        
        # Calculate evenly spaced indices
        if end - start <= num_frames:
            indices = list(range(start, end))
        else:
            indices = np.linspace(start, end - 1, num_frames, dtype=int).tolist()
        
        # Read frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        
        return frames
    
    finally:
        cap.release()


def frames_to_video(frames, output_path, fps=10, codec='XVID'):
    """
    Convert list of frames to video file.
    
    Args:
        frames: List of frames (numpy arrays, BGR or grayscale)
        output_path: Output video path (.avi recommended)
        fps: Frames per second
        codec: Video codec (e.g., 'XVID', 'mp4v', 'MJPG')
        
    Returns:
        True if successful, False otherwise
        
    Example:
        frames = [frame1, frame2, frame3, ...]
        success = frames_to_video(frames, "output.avi", fps=10)
    """
    if not frames:
        return False
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Handle grayscale frames - convert to BGR
    processed_frames = []
    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        processed_frames.append(frame)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        return False
    
    try:
        for frame in processed_frames:
            out.write(frame)
        return True
    finally:
        out.release()


def resize_frame(frame, size):
    """
    Resize frame to specified size.
    
    Args:
        frame: Input frame (numpy array)
        size: (width, height) tuple
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)


def to_grayscale(frame):
    """
    Convert frame to grayscale.
    
    Args:
        frame: Input frame (BGR or already grayscale)
        
    Returns:
        Grayscale frame
    """
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        frame: Grayscale frame
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced frame
    """
    if len(frame.shape) == 3:
        frame = to_grayscale(frame)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(frame)


def format_video_info(info):
    """
    Format video info as string for display.
    
    Args:
        info: Video info dict from get_video_info()
        
    Returns:
        Formatted string
    """
    if info is None:
        return "Unable to read video"
    
    return (
        f"File: {info['filename']}\n"
        f"Resolution: {info['resolution']}\n"
        f"Frames: {info['total_frames']:,}\n"
        f"FPS: {info['fps']}\n"
        f"Duration: {info['duration_formatted']}\n"
        f"Size: {info['file_size_mb']} MB"
    )


def validate_video(video_path):
    """
    Validate that a video file can be opened and read.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    # Check extension
    ext = os.path.splitext(video_path)[1].lower()
    valid_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm')
    if ext not in valid_extensions:
        return False, f"Invalid extension: {ext}"
    
    # Try to open
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Cannot read frames from video"
    
    return True, "Valid"