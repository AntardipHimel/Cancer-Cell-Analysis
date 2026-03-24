# -*- coding: utf-8 -*-
"""
m8_export_utils.py

ONCOLENS - MODULE 8: EXPORT UTILITIES
======================================
Functions for saving results, exporting videos, and generating reports.

Features:
    - Save classification results (CSV, JSON)
    - Organize lenses by class
    - Export lens videos
    - Generate text reports
    - Create summary visualizations

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m8_export_utils.py

Author: Antardip Himel
Date: March 2026
"""

import os
import csv
import json
import shutil
import cv2
import numpy as np
from datetime import datetime

from . import m1_config as config


def save_classification_results(results, output_dir, video_name=None):
    """
    Save classification results to CSV and JSON.
    
    Args:
        results: List of prediction dicts
        output_dir: Output directory (4_classification folder)
        video_name: Optional video name for metadata
        
    Returns:
        dict with csv_path and json_path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "classification_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lens_id', 'prediction', 'confidence', 
                        'prob_no_cell', 'prob_contain_cell'])
        for r in results:
            writer.writerow([
                r['lens_id'],
                r['prediction'],
                f"{r['confidence']:.4f}",
                f"{r['probabilities']['no_cell']:.4f}",
                f"{r['probabilities']['contain_cell']:.4f}"
            ])
    
    # Calculate summary
    total = len(results)
    contain_cell = sum(1 for r in results if r['prediction'] == 'contain_cell')
    no_cell = total - contain_cell
    avg_conf = sum(r['confidence'] for r in results) / total if total > 0 else 0
    
    # Save JSON
    json_path = os.path.join(output_dir, "classification_results.json")
    summary = {
        'video_name': video_name,
        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_used': 'ResNet+LSTM' if config.check_model_exists('resnet_lstm') else '3D CNN',
        'summary': {
            'total_lenses': total,
            'contain_cell_count': contain_cell,
            'no_cell_count': no_cell,
            'contain_cell_pct': round(contain_cell / total * 100, 2) if total > 0 else 0,
            'no_cell_pct': round(no_cell / total * 100, 2) if total > 0 else 0,
            'avg_confidence': round(avg_conf, 4)
        },
        'results': results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return {
        'csv_path': csv_path,
        'json_path': json_path
    }


def organize_lenses_by_class(results, source_dir, output_dir, progress_callback=None):
    """
    Organize lenses into class folders (copy frames).
    
    Args:
        results: List of prediction dicts with lens_id
        source_dir: Source directory containing lens folders (3_cropped_lenses)
        output_dir: Output directory (5_classified_lenses)
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        dict with contain_cell_dir, no_cell_dir, counts
    """
    # Create class directories
    contain_dir = os.path.join(output_dir, "contain_cell")
    no_cell_dir = os.path.join(output_dir, "no_cell")
    os.makedirs(contain_dir, exist_ok=True)
    os.makedirs(no_cell_dir, exist_ok=True)
    
    total = len(results)
    contain_count = 0
    no_cell_count = 0
    
    for idx, result in enumerate(results):
        lens_id = result['lens_id']
        prediction = result['prediction']
        
        if progress_callback:
            pct = int((idx / total) * 100)
            progress_callback(f"Organizing {lens_id}", pct)
        
        # Source and destination
        src_path = os.path.join(source_dir, lens_id)
        
        if prediction == 'contain_cell':
            dst_path = os.path.join(contain_dir, lens_id)
            contain_count += 1
        else:
            dst_path = os.path.join(no_cell_dir, lens_id)
            no_cell_count += 1
        
        # Copy lens folder
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
    
    if progress_callback:
        progress_callback("Organization complete!", 100)
    
    return {
        'contain_cell_dir': contain_dir,
        'no_cell_dir': no_cell_dir,
        'contain_cell_count': contain_count,
        'no_cell_count': no_cell_count
    }


def export_lens_as_video(lens_dir, output_path, fps=None, codec=None):
    """
    Export a single lens (30 frames) as video.
    
    Args:
        lens_dir: Directory containing lens frames
        output_path: Output video path (.avi)
        fps: Frames per second (default: from config)
        codec: Video codec (default: from config)
        
    Returns:
        True if successful, False otherwise
    """
    if fps is None:
        fps = config.VIDEO_FPS
    if codec is None:
        codec = config.VIDEO_CODEC
    
    # Load frames
    frame_files = sorted([
        f for f in os.listdir(lens_dir)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ])
    
    if not frame_files:
        return False
    
    frames = []
    for f in frame_files:
        frame = cv2.imread(os.path.join(lens_dir, f))
        if frame is not None:
            frames.append(frame)
    
    if not frames:
        return False
    
    # Get dimensions
    h, w = frames[0].shape[:2]
    
    # Convert grayscale to BGR if needed
    processed = []
    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        processed.append(frame)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        return False
    
    try:
        for frame in processed:
            out.write(frame)
        return True
    finally:
        out.release()


def export_all_lenses_as_videos(results, source_dir, output_dir, 
                                fps=None, progress_callback=None):
    """
    Export all classified lenses as videos, organized by class.
    
    Args:
        results: List of prediction dicts
        source_dir: Source directory with lens folders (3_cropped_lenses)
        output_dir: Output directory (6_videos)
        fps: Frames per second
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        dict with paths and counts
    """
    # Create output directories
    contain_dir = os.path.join(output_dir, "contain_cell")
    no_cell_dir = os.path.join(output_dir, "no_cell")
    os.makedirs(contain_dir, exist_ok=True)
    os.makedirs(no_cell_dir, exist_ok=True)
    
    total = len(results)
    exported = 0
    failed = 0
    
    for idx, result in enumerate(results):
        lens_id = result['lens_id']
        prediction = result['prediction']
        
        if progress_callback:
            pct = int((idx / total) * 100)
            progress_callback(f"Exporting {lens_id} to video", pct)
        
        # Source lens directory
        lens_path = os.path.join(source_dir, lens_id)
        if not os.path.exists(lens_path):
            failed += 1
            continue
        
        # Output video path
        if prediction == 'contain_cell':
            video_path = os.path.join(contain_dir, f"{lens_id}.avi")
        else:
            video_path = os.path.join(no_cell_dir, f"{lens_id}.avi")
        
        # Export
        if export_lens_as_video(lens_path, video_path, fps):
            exported += 1
        else:
            failed += 1
    
    if progress_callback:
        progress_callback(f"Exported {exported} videos!", 100)
    
    return {
        'contain_cell_videos': contain_dir,
        'no_cell_videos': no_cell_dir,
        'total_exported': exported,
        'failed': failed
    }


def generate_report(results, output_path, video_name=None, model_type=None, 
                    processing_time=None):
    """
    Generate a text report of classification results.
    
    Args:
        results: List of prediction dicts
        output_path: Output file path (.txt)
        video_name: Optional video name
        model_type: Model used ('resnet_lstm' or '3dcnn')
        processing_time: Optional processing time string
        
    Returns:
        Path to report file
    """
    total = len(results)
    contain_cell = sum(1 for r in results if r['prediction'] == 'contain_cell')
    no_cell = total - contain_cell
    avg_conf = sum(r['confidence'] for r in results) / total if total > 0 else 0
    
    # Find low confidence predictions
    low_conf = [r for r in results if r['confidence'] < config.CONFIDENCE_THRESHOLD]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  ONCOLENS - CANCER CELL CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if video_name:
            f.write(f"Video: {video_name}\n")
        if model_type:
            model_name = "ResNet+LSTM (96.58%)" if model_type == 'resnet_lstm' else "3D CNN (94.01%)"
            f.write(f"Model: {model_name}\n")
        if processing_time:
            f.write(f"Processing Time: {processing_time}\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("  SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Lenses Analyzed: {total}\n\n")
        f.write(f"  [+] Contains Cell: {contain_cell} ({contain_cell/total*100:.1f}%)\n")
        f.write(f"  [-] No Cell:       {no_cell} ({no_cell/total*100:.1f}%)\n\n")
        f.write(f"Average Confidence: {avg_conf*100:.1f}%\n")
        f.write("\n")
        
        if low_conf:
            f.write("-" * 70 + "\n")
            f.write(f"  LOW CONFIDENCE PREDICTIONS (<{config.CONFIDENCE_THRESHOLD*100:.0f}%): {len(low_conf)}\n")
            f.write("-" * 70 + "\n")
            f.write("  These predictions may need manual review:\n\n")
            for r in sorted(low_conf, key=lambda x: x['confidence']):
                f.write(f"    {r['lens_id']}: {r['prediction']} ({r['confidence']*100:.1f}%)\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("  ALL PREDICTIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Lens ID':<15} {'Prediction':<15} {'Confidence':<12}\n")
        f.write("  " + "-" * 42 + "\n")
        for r in results:
            conf_str = f"{r['confidence']*100:.1f}%"
            flag = " (!)" if r['confidence'] < config.CONFIDENCE_THRESHOLD else ""
            f.write(f"  {r['lens_id']:<15} {r['prediction']:<15} {conf_str:<12}{flag}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    return output_path


def create_summary_image(results, output_path, title="Classification Results"):
    """
    Create a summary visualization image (pie chart + histogram).
    
    Args:
        results: List of prediction dicts
        output_path: Output image path (.png)
        title: Image title
        
    Returns:
        Path to image or None if matplotlib not available
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        total = len(results)
        contain_cell = sum(1 for r in results if r['prediction'] == 'contain_cell')
        no_cell = total - contain_cell
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        labels = ['Contains Cell', 'No Cell']
        sizes = [contain_cell, no_cell]
        colors = ['#4CAF50', '#F44336']
        explode = (0.05, 0)
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=explode, textprops={'fontsize': 12})
        axes[0].set_title('Classification Distribution', fontsize=14, fontweight='bold')
        
        # Confidence histogram
        confidences = [r['confidence'] for r in results]
        axes[1].hist(confidences, bins=20, color='#1976D2', edgecolor='white', alpha=0.7)
        axes[1].axvline(x=config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', 
                       linewidth=2, label=f'{config.CONFIDENCE_THRESHOLD*100:.0f}% threshold')
        axes[1].set_xlabel('Confidence', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    except ImportError:
        return None