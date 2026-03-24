# -*- coding: utf-8 -*-
"""
3_1_feature_analysis.py

EDGE GRADIENT FOCUSED ANALYSIS
Analyzes Edge Gradient (Mean & Std) differences between frames separated by various gaps.

For each lens (30 frames), computes differences between frame pairs:
  - Gap 1:  (F1,F2), (F2,F3), ... (F29,F30) → 29 pairs
  - Gap 3:  (F1,F4), (F2,F5), ... (F27,F30) → 27 pairs
  - Gap 5:  (F1,F6), (F2,F7), ... (F25,F30) → 25 pairs
  - Gap 7:  (F1,F8), (F2,F9), ... (F23,F30) → 23 pairs
  - Gap 10: (F1,F11), (F2,F12), ... (F20,F30) → 20 pairs

Each lens is INDEPENDENT - no averaging across lenses!
Each frame pair is a separate data point.

Outputs:
  1. Data CSVs (pair-level, lens-level, case-level)
  2. Confusion matrices for each gap
  3. Histograms for each gap (separate files)
  4. Case-based analysis (Max gradient values per lens)

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\output\\

Author: Antardip Himel
Date: February 2026
"""

import os
import sys
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HIGH-QUALITY PLOT SETTINGS
# =============================================================================
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Professional color palette
COLORS = {
    'cell': '#E63946',      # Red
    'no_cell': '#457B9D',   # Blue
    'accent1': '#2A9D8F',   # Teal
    'accent2': '#E9C46A',   # Yellow
    'accent3': '#F4A261',   # Orange
    'background': '#F8F9FA',
}

CLASS_COLORS = {0: COLORS['no_cell'], 1: COLORS['cell']}
CLASS_NAMES = {0: 'No Cell', 1: 'Cell'}

GAP_COLORS = {
    1: '#264653',
    3: '#2A9D8F',
    5: '#E9C46A',
    7: '#F4A261',
    10: '#E76F51'
}

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\output"

# Gaps to analyze
GAPS = [1, 3, 5, 7, 10]

# Number of top cases to analyze (max gradient values)
NUM_CASES = 10

# Class mapping
CLASS_MAP = {
    'contain_cell': 1,
    'no_cell': 0,
}


# =============================================================================
# FEATURE EXTRACTION (Edge Gradient Only)
# =============================================================================

def extract_edge_gradient(img):
    """Extract edge gradient features from a single grayscale frame."""
    if img is None:
        return None, None
    
    # Sobel gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    edge_mean = np.mean(gradient_mag)
    edge_std = np.std(gradient_mag)
    
    return edge_mean, edge_std


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_edge_gradient_data(input_dir, gaps):
    """Collect edge gradient data for all lenses and gaps."""
    
    pair_level_data = []      # Each frame pair = 1 row
    lens_level_data = []      # Each lens = 1 row (aggregated)
    lens_frame_data = []      # All frames per lens (for case analysis)
    
    video_folders = sorted([f for f in os.listdir(input_dir) 
                           if os.path.isdir(os.path.join(input_dir, f))])
    
    total_lenses = 0
    total_pairs = 0
    
    print(f"\n  Found {len(video_folders)} videos to process", flush=True)
    print(f"  Gaps to analyze: {gaps}", flush=True)
    sys.stdout.flush()
    
    for vi, video_name in enumerate(video_folders):
        video_path = os.path.join(input_dir, video_name)
        print(f"\n  [{vi+1}/{len(video_folders)}] {video_name}", flush=True)
        sys.stdout.flush()
        
        for category in ['contain_cell', 'no_cell']:
            if category not in CLASS_MAP:
                continue
                
            category_path = os.path.join(video_path, category)
            if not os.path.exists(category_path):
                continue
            
            lens_folders = sorted([f for f in os.listdir(category_path) 
                                  if os.path.isdir(os.path.join(category_path, f))])
            
            print(f"      {category}: {len(lens_folders)} lenses ", end='', flush=True)
            sys.stdout.flush()
            
            for lens_idx, lens_name in enumerate(lens_folders):
                lens_path = os.path.join(category_path, lens_name)
                frame_files = sorted([f for f in os.listdir(lens_path) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if len(frame_files) < 15:  # Need enough frames for gap=10
                    continue
                
                # Extract edge gradient from ALL frames
                frame_features = []
                for frame_idx, frame_file in enumerate(frame_files):
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    edge_mean, edge_std = extract_edge_gradient(img)
                    
                    if edge_mean is not None:
                        frame_features.append({
                            'frame_idx': frame_idx + 1,
                            'frame_file': frame_file,
                            'edge_gradient_mean': edge_mean,
                            'edge_gradient_std': edge_std
                        })
                
                if len(frame_features) < 15:
                    continue
                
                total_lenses += 1
                
                # Store all frame data for this lens (for case analysis)
                lens_frame_data.append({
                    'video': video_name,
                    'lens': lens_name,
                    'category': category,
                    'class': CLASS_MAP[category],
                    'num_frames': len(frame_features),
                    'frames': frame_features
                })
                
                # === PAIR-LEVEL DATA ===
                lens_pair_features = {gap: [] for gap in gaps}
                
                for gap in gaps:
                    for i in range(len(frame_features) - gap):
                        j = i + gap
                        
                        # Edge Gradient Mean difference
                        mean_i = frame_features[i]['edge_gradient_mean']
                        mean_j = frame_features[j]['edge_gradient_mean']
                        
                        # Edge Gradient Std difference
                        std_i = frame_features[i]['edge_gradient_std']
                        std_j = frame_features[j]['edge_gradient_std']
                        
                        pair_feat = {
                            'video': video_name,
                            'lens': lens_name,
                            'category': category,
                            'class': CLASS_MAP[category],
                            'gap': gap,
                            'frame_i': i + 1,
                            'frame_j': j + 1,
                            # Mean features
                            'edge_gradient_mean_i': mean_i,
                            'edge_gradient_mean_j': mean_j,
                            'edge_gradient_mean_diff': mean_j - mean_i,
                            'edge_gradient_mean_abs_diff': abs(mean_j - mean_i),
                            # Std features
                            'edge_gradient_std_i': std_i,
                            'edge_gradient_std_j': std_j,
                            'edge_gradient_std_diff': std_j - std_i,
                            'edge_gradient_std_abs_diff': abs(std_j - std_i),
                        }
                        
                        pair_level_data.append(pair_feat)
                        lens_pair_features[gap].append(pair_feat)
                        total_pairs += 1
                
                # === LENS-LEVEL DATA ===
                lens_stats = {
                    'video': video_name,
                    'lens': lens_name,
                    'category': category,
                    'class': CLASS_MAP[category],
                    'num_frames': len(frame_features),
                }
                
                # Per-gap aggregated stats
                for gap in gaps:
                    if not lens_pair_features[gap]:
                        continue
                    
                    gap_pairs = lens_pair_features[gap]
                    
                    # Edge Gradient Mean stats
                    mean_diffs = [p['edge_gradient_mean_abs_diff'] for p in gap_pairs]
                    lens_stats[f'edge_mean_gap{gap}_avg'] = np.mean(mean_diffs)
                    lens_stats[f'edge_mean_gap{gap}_std'] = np.std(mean_diffs)
                    lens_stats[f'edge_mean_gap{gap}_max'] = np.max(mean_diffs)
                    lens_stats[f'edge_mean_gap{gap}_min'] = np.min(mean_diffs)
                    
                    # Edge Gradient Std stats
                    std_diffs = [p['edge_gradient_std_abs_diff'] for p in gap_pairs]
                    lens_stats[f'edge_std_gap{gap}_avg'] = np.mean(std_diffs)
                    lens_stats[f'edge_std_gap{gap}_std'] = np.std(std_diffs)
                    lens_stats[f'edge_std_gap{gap}_max'] = np.max(std_diffs)
                
                # Overall motion energy (average of all absolute diffs)
                all_mean_diffs = [p['edge_gradient_mean_abs_diff'] for p in pair_level_data 
                                  if p['lens'] == lens_name and p['video'] == video_name]
                lens_stats['motion_energy'] = np.mean(all_mean_diffs) if all_mean_diffs else 0
                lens_stats['max_jump'] = np.max(all_mean_diffs) if all_mean_diffs else 0
                
                lens_level_data.append(lens_stats)
                
                # Progress
                if (lens_idx + 1) % 10 == 0:
                    print(".", end='', flush=True)
                    sys.stdout.flush()
                if (lens_idx + 1) % 50 == 0:
                    print(f"[{lens_idx+1}]", end='', flush=True)
                    sys.stdout.flush()
            
            print(f" done", flush=True)
            sys.stdout.flush()
        
        print(f"      Running total: {total_lenses} lenses, {total_pairs:,} pairs", flush=True)
        sys.stdout.flush()
    
    print(f"\n  " + "=" * 50, flush=True)
    print(f"  COLLECTION COMPLETE!", flush=True)
    print(f"  Total lenses: {total_lenses}", flush=True)
    print(f"  Total pairs: {total_pairs:,}", flush=True)
    print(f"  " + "=" * 50, flush=True)
    sys.stdout.flush()
    
    return pair_level_data, lens_level_data, lens_frame_data


# =============================================================================
# CASE-BASED ANALYSIS
# =============================================================================

def compute_case_data(lens_frame_data, gaps, num_cases=10):
    """
    Compute case-based analysis: Top N maximum gradient differences per lens.
    
    Case 1 = 1 Maximum gradient value (single highest)
    Case 2 = 2 Maximum gradient values (sum of top 2)
    Case 3 = 3 Maximum gradient values (sum of top 3)
    ...
    Case N = N Maximum gradient values (sum of top N)
    
    For each lens, we aggregate the TOP N highest differences.
    This captures how much "extreme activity" exists in each lens.
    
    Cell lenses should have HIGHER sums (more big changes)
    No Cell lenses should have LOWER sums (mostly small changes)
    """
    
    case_data = []
    
    for lens_info in lens_frame_data:
        frames = lens_info['frames']
        
        for gap in gaps:
            # Compute all pair differences for this gap
            pair_diffs_mean = []
            pair_diffs_std = []
            
            for i in range(len(frames) - gap):
                j = i + gap
                
                mean_diff = abs(frames[j]['edge_gradient_mean'] - frames[i]['edge_gradient_mean'])
                std_diff = abs(frames[j]['edge_gradient_std'] - frames[i]['edge_gradient_std'])
                
                pair_diffs_mean.append(mean_diff)
                pair_diffs_std.append(std_diff)
            
            # Sort descending to get top values
            sorted_mean_diffs = sorted(pair_diffs_mean, reverse=True)
            sorted_std_diffs = sorted(pair_diffs_std, reverse=True)
            
            # For each case (1 to N), compute SUM and AVERAGE of top N values
            for case_num in range(1, min(num_cases + 1, len(sorted_mean_diffs) + 1)):
                # Get top N values
                top_n_mean = sorted_mean_diffs[:case_num]
                top_n_std = sorted_std_diffs[:case_num]
                
                case_data.append({
                    'video': lens_info['video'],
                    'lens': lens_info['lens'],
                    'category': lens_info['category'],
                    'class': lens_info['class'],
                    'gap': gap,
                    'case': case_num,
                    'num_values': case_num,
                    # Sum of top N
                    'edge_mean_sum_top_n': np.sum(top_n_mean),
                    'edge_std_sum_top_n': np.sum(top_n_std),
                    # Average of top N
                    'edge_mean_avg_top_n': np.mean(top_n_mean),
                    'edge_std_avg_top_n': np.mean(top_n_std),
                    # Std of top N values (variation among the top values)
                    'edge_mean_std_top_n': np.std(top_n_mean) if case_num > 1 else 0,
                    'edge_std_std_top_n': np.std(top_n_std) if case_num > 1 else 0,
                    # Individual top values (for reference)
                    'edge_mean_max_1': sorted_mean_diffs[0] if len(sorted_mean_diffs) > 0 else 0,
                    'edge_mean_max_n': sorted_mean_diffs[case_num-1] if case_num <= len(sorted_mean_diffs) else 0,
                })
    
    return case_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_histogram_single_gap(pair_data, gap, output_path, feature='edge_gradient_mean_abs_diff'):
    """Create histogram for a single gap."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cls in [0, 1]:
        vals = [p[feature] for p in pair_data if p['class'] == cls and p['gap'] == gap]
        n = len(vals)
        
        ax.hist(vals, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
               label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('|Δ Edge Gradient Mean|', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Edge Gradient Mean Difference Distribution - Gap {gap}\n'
                f'(Each bar = count of frame pairs with that difference value)', 
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # Add statistics text
    cell_vals = [p[feature] for p in pair_data if p['class'] == 1 and p['gap'] == gap]
    nocell_vals = [p[feature] for p in pair_data if p['class'] == 0 and p['gap'] == gap]
    
    stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                  f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_histogram_std_single_gap(pair_data, gap, output_path):
    """Create histogram for edge gradient STD for a single gap."""
    
    feature = 'edge_gradient_std_abs_diff'
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cls in [0, 1]:
        vals = [p[feature] for p in pair_data if p['class'] == cls and p['gap'] == gap]
        n = len(vals)
        
        ax.hist(vals, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
               label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('|Δ Edge Gradient Std|', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Edge Gradient Std Difference Distribution - Gap {gap}\n'
                f'(Each bar = count of frame pairs with that difference value)', 
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # Add statistics text
    cell_vals = [p[feature] for p in pair_data if p['class'] == 1 and p['gap'] == gap]
    nocell_vals = [p[feature] for p in pair_data if p['class'] == 0 and p['gap'] == gap]
    
    stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                  f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_case_histogram(case_data, gap, case_num, output_dir):
    """
    Create histograms for a specific case (Top N maximum gradients).
    Creates THREE histograms:
      1. Sum of top N values
      2. Mean of top N values
      3. Std of top N values (only for case >= 2)
    
    Case 1 = 1 max value
    Case 2 = top 2 max values (sum, mean, std)
    Case N = top N max values (sum, mean, std)
    """
    
    # ─── HISTOGRAM 1: Sum of Top N ───
    fig, ax = plt.subplots(figsize=(12, 7))
    
    feature = 'edge_mean_sum_top_n'
    
    for cls in [0, 1]:
        vals = [c[feature] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    # Create label
    if case_num == 1:
        case_label = "1 Maximum Value"
    else:
        case_label = f"Sum of Top {case_num} Maximum Values"
    
    ax.set_xlabel(f'Sum of Top {case_num} |Δ Edge Gradient Mean|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: {case_label} per Lens - Gap {gap}\n'
                f'(Each lens contributes 1 data point = sum of its top {case_num} largest differences)', 
                fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # Add statistics
    cell_vals = [c[feature] for c in case_data 
                if c['class'] == 1 and c['gap'] == gap and c['case'] == case_num]
    nocell_vals = [c[feature] for c in case_data 
                  if c['class'] == 0 and c['gap'] == gap and c['case'] == case_num]
    
    if cell_vals and nocell_vals:
        mean_diff = np.mean(cell_vals) - np.mean(nocell_vals)
        stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                      f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}\n'
                      f'Δ Mean (Cell - NoCell): {mean_diff:.2f}')
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    sum_path = os.path.join(output_dir, f'case_{case_num}_gap_{gap}_SUM.png')
    plt.savefig(sum_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ─── HISTOGRAM 2: Mean of Top N ───
    fig, ax = plt.subplots(figsize=(12, 7))
    
    feature = 'edge_mean_avg_top_n'
    
    for cls in [0, 1]:
        vals = [c[feature] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    # Create label
    if case_num == 1:
        case_label = "1 Maximum Value"
    else:
        case_label = f"Mean of Top {case_num} Maximum Values"
    
    ax.set_xlabel(f'Mean of Top {case_num} |Δ Edge Gradient Mean|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: {case_label} per Lens - Gap {gap}\n'
                f'(Each lens contributes 1 data point = mean of its top {case_num} largest differences)', 
                fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # Add statistics
    cell_vals = [c[feature] for c in case_data 
                if c['class'] == 1 and c['gap'] == gap and c['case'] == case_num]
    nocell_vals = [c[feature] for c in case_data 
                  if c['class'] == 0 and c['gap'] == gap and c['case'] == case_num]
    
    if cell_vals and nocell_vals:
        mean_diff = np.mean(cell_vals) - np.mean(nocell_vals)
        stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                      f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}\n'
                      f'Δ Mean (Cell - NoCell): {mean_diff:.2f}')
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    mean_path = os.path.join(output_dir, f'case_{case_num}_gap_{gap}_MEAN.png')
    plt.savefig(mean_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ─── HISTOGRAM 3: Std of Top N (only for case >= 2) ───
    std_path = None
    if case_num >= 2:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        feature = 'edge_mean_std_top_n'
        
        for cls in [0, 1]:
            vals = [c[feature] for c in case_data 
                   if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
            n = len(vals)
            
            if vals:
                ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                       label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
        
        case_label = f"Std of Top {case_num} Maximum Values"
        
        ax.set_xlabel(f'Std of Top {case_num} |Δ Edge Gradient Mean|', fontsize=12)
        ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
        ax.set_title(f'Case {case_num}: {case_label} per Lens - Gap {gap}\n'
                    f'(Each lens contributes 1 data point = std of its top {case_num} largest differences)', 
                    fontweight='bold', fontsize=13)
        ax.legend(loc='upper right', fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
        
        # Add statistics
        cell_vals = [c[feature] for c in case_data 
                    if c['class'] == 1 and c['gap'] == gap and c['case'] == case_num]
        nocell_vals = [c[feature] for c in case_data 
                      if c['class'] == 0 and c['gap'] == gap and c['case'] == case_num]
        
        if cell_vals and nocell_vals:
            mean_diff = np.mean(cell_vals) - np.mean(nocell_vals)
            stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                          f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}\n'
                          f'Δ Mean (Cell - NoCell): {mean_diff:.2f}')
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        std_path = os.path.join(output_dir, f'case_{case_num}_gap_{gap}_STD.png')
        plt.savefig(std_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return sum_path, mean_path, std_path


def create_confusion_matrix_plot(pair_data, gap, output_path, feature='edge_gradient_mean_abs_diff'):
    """Create confusion matrix for a single gap."""
    
    gap_data = [p for p in pair_data if p['gap'] == gap]
    vals = np.array([p[feature] for p in gap_data])
    labels = np.array([p['class'] for p in gap_data])
    
    n_total = len(gap_data)
    n_cell = sum(labels == 1)
    n_nocell = sum(labels == 0)
    
    # Find optimal threshold
    thresholds = np.linspace(vals.min(), vals.max(), 100)
    best_acc = 0
    best_pred = None
    best_thresh = 0
    
    for thresh in thresholds:
        pred_high = (vals > thresh).astype(int)
        pred_low = (vals < thresh).astype(int)
        
        acc_high = accuracy_score(labels, pred_high)
        acc_low = accuracy_score(labels, pred_low)
        
        if acc_high > best_acc:
            best_acc = acc_high
            best_pred = pred_high
            best_thresh = thresh
        if acc_low > best_acc:
            best_acc = acc_low
            best_pred = pred_low
            best_thresh = thresh
    
    cm = confusion_matrix(labels, best_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Cell', 'Cell'], fontsize=12)
    ax.set_yticklabels(['No Cell', 'Cell'], fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    
    # Add counts and percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            row_total = cm[i, :].sum()
            pct = count / row_total * 100 if row_total > 0 else 0
            color = 'white' if count > cm.max()/2 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                   fontsize=14, fontweight='bold', color=color)
    
    ax.set_title(f'Confusion Matrix - Gap {gap} (Edge Gradient Mean)\n'
                f'Accuracy: {best_acc:.1%} | Threshold: {best_thresh:.2f}\n'
                f'(n={n_total:,} | Cell: {n_cell:,} | No Cell: {n_nocell:,})', 
                fontweight='bold', fontsize=13)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return best_acc, best_thresh


def create_all_gaps_confusion_matrix(pair_data, output_path):
    """Create confusion matrices for all gaps in one figure."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    feature = 'edge_gradient_mean_abs_diff'
    results = []
    
    for idx, gap in enumerate(GAPS):
        ax = axes[idx]
        
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = np.array([p[feature] for p in gap_data])
        labels = np.array([p['class'] for p in gap_data])
        
        n_total = len(gap_data)
        n_cell = sum(labels == 1)
        n_nocell = sum(labels == 0)
        
        # Find optimal threshold
        thresholds = np.linspace(vals.min(), vals.max(), 100)
        best_acc = 0
        best_pred = None
        
        for thresh in thresholds:
            pred_high = (vals > thresh).astype(int)
            pred_low = (vals < thresh).astype(int)
            
            acc_high = accuracy_score(labels, pred_high)
            acc_low = accuracy_score(labels, pred_low)
            
            if acc_high > best_acc:
                best_acc = acc_high
                best_pred = pred_high
            if acc_low > best_acc:
                best_acc = acc_low
                best_pred = pred_low
        
        cm = confusion_matrix(labels, best_pred)
        
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'], fontsize=11)
        ax.set_yticklabels(['No Cell', 'Cell'], fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                row_total = cm[i, :].sum()
                pct = count / row_total * 100 if row_total > 0 else 0
                color = 'white' if count > cm.max()/2 else 'black'
                ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=color)
        
        ax.set_title(f'Gap {gap}: Acc={best_acc:.1%}\n'
                    f'(n={n_total:,} | Cell:{n_cell:,} | NoCell:{n_nocell:,})', 
                    fontweight='bold', fontsize=11)
        
        results.append({'gap': gap, 'accuracy': best_acc, 'n': n_total})
    
    # Empty 6th subplot - use for summary
    ax = axes[5]
    ax.axis('off')
    
    summary_text = "SUMMARY\n" + "="*30 + "\n\n"
    for r in results:
        summary_text += f"Gap {r['gap']}: {r['accuracy']:.1%} (n={r['n']:,})\n"
    
    best_result = max(results, key=lambda x: x['accuracy'])
    summary_text += f"\nBest: Gap {best_result['gap']} ({best_result['accuracy']:.1%})"
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='center', horizontalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Confusion Matrices: Edge Gradient Mean by Gap\n'
                '(Optimal threshold classification)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return results


def create_case_comparison_plot(case_data, gap, output_path, num_cases=10):
    """
    Create plot comparing all cases for a given gap.
    Shows how sum of top N values separates Cell vs No Cell.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cases = list(range(1, num_cases + 1))
    
    # Left plot: Sum of top N
    ax1 = axes[0]
    for cls in [0, 1]:
        means = []
        stds = []
        
        for case_num in cases:
            vals = [c['edge_mean_sum_top_n'] for c in case_data 
                   if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) / np.sqrt(len(vals)))  # SEM
            else:
                means.append(0)
                stds.append(0)
        
        ax1.errorbar(cases, means, yerr=stds, marker='o', markersize=8,
                    linewidth=2.5, capsize=5, capthick=2,
                    color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    
    ax1.set_xlabel('Case Number (N = number of top values summed)', fontsize=12)
    ax1.set_ylabel('Mean Sum of Top N |Δ Edge Gradient|', fontsize=12)
    ax1.set_title(f'Sum of Top N Maximum Values - Gap {gap}\n'
                 f'(Higher = more extreme changes)', 
                 fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left', fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(cases)
    ax1.set_facecolor(COLORS['background'])
    
    # Right plot: Average of top N (to see if pattern holds)
    ax2 = axes[1]
    for cls in [0, 1]:
        means = []
        stds = []
        
        for case_num in cases:
            vals = [c['edge_mean_avg_top_n'] for c in case_data 
                   if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(0)
                stds.append(0)
        
        ax2.errorbar(cases, means, yerr=stds, marker='s', markersize=8,
                    linewidth=2.5, capsize=5, capthick=2,
                    color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    
    ax2.set_xlabel('Case Number (N = number of top values averaged)', fontsize=12)
    ax2.set_ylabel('Mean Average of Top N |Δ Edge Gradient|', fontsize=12)
    ax2.set_title(f'Average of Top N Maximum Values - Gap {gap}\n'
                 f'(Should decrease as N increases - includes smaller values)', 
                 fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(cases)
    ax2.set_facecolor(COLORS['background'])
    
    plt.suptitle(f'Case Analysis: Top N Maximum Gradient Values per Lens - Gap {gap}', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_gap_scaling_plot(pair_data, output_path):
    """Create gap scaling analysis plot."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Edge Gradient Mean
    ax1 = axes[0]
    for cls in [0, 1]:
        means = []
        sems = []
        counts = []
        
        for gap in GAPS:
            vals = [p['edge_gradient_mean_abs_diff'] for p in pair_data 
                   if p['class'] == cls and p['gap'] == gap]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
            counts.append(len(vals))
        
        ax1.errorbar(GAPS, means, yerr=sems, marker='o', markersize=10,
                    linewidth=2.5, capsize=5, capthick=2,
                    color=CLASS_COLORS[cls], label=f'{CLASS_NAMES[cls]}')
        
        # Add count labels
        for i, (gap, mean, count) in enumerate(zip(GAPS, means, counts)):
            offset = 0.15 if cls == 1 else -0.15
            ax1.annotate(f'n={count:,}', xy=(gap, mean), 
                        xytext=(gap + offset, mean),
                        fontsize=8, color=CLASS_COLORS[cls], alpha=0.8)
    
    ax1.set_xlabel('Frame Gap', fontsize=13)
    ax1.set_ylabel('Mean |Δ Edge Gradient Mean|', fontsize=13)
    ax1.set_title('Edge Gradient Mean: Gap Scaling', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(GAPS)
    ax1.set_facecolor(COLORS['background'])
    
    # Edge Gradient Std
    ax2 = axes[1]
    for cls in [0, 1]:
        means = []
        sems = []
        
        for gap in GAPS:
            vals = [p['edge_gradient_std_abs_diff'] for p in pair_data 
                   if p['class'] == cls and p['gap'] == gap]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        ax2.errorbar(GAPS, means, yerr=sems, marker='s', markersize=10,
                    linewidth=2.5, capsize=5, capthick=2,
                    color=CLASS_COLORS[cls], label=f'{CLASS_NAMES[cls]}')
    
    ax2.set_xlabel('Frame Gap', fontsize=13)
    ax2.set_ylabel('Mean |Δ Edge Gradient Std|', fontsize=13)
    ax2.set_title('Edge Gradient Std: Gap Scaling', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper left', fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(GAPS)
    ax2.set_facecolor(COLORS['background'])
    
    plt.suptitle('Gap Scaling Analysis: How Differences Increase with Frame Gap', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_roc_curves(pair_data, output_path):
    """Create ROC curves for all gaps."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    feature = 'edge_gradient_mean_abs_diff'
    
    for gap in GAPS:
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = np.array([p[feature] for p in gap_data])
        labels = np.array([p['class'] for p in gap_data])
        
        n = len(gap_data)
        
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=GAP_COLORS[gap], linewidth=2.5,
               label=f'Gap {gap} (n={n:,}, AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves: Edge Gradient Mean by Gap', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_data_summary_table(pair_data, lens_data, output_path):
    """Create data summary table."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Prepare data
    summary_data = []
    
    # Header
    summary_data.append(['Gap', 'Total Pairs', 'Cell Pairs', 'No Cell Pairs', 
                        'Cell Mean', 'Cell Std', 'NoCell Mean', 'NoCell Std', 'Δ Mean'])
    
    for gap in GAPS:
        gap_pairs = [p for p in pair_data if p['gap'] == gap]
        cell_pairs = [p for p in gap_pairs if p['class'] == 1]
        nocell_pairs = [p for p in gap_pairs if p['class'] == 0]
        
        cell_vals = [p['edge_gradient_mean_abs_diff'] for p in cell_pairs]
        nocell_vals = [p['edge_gradient_mean_abs_diff'] for p in nocell_pairs]
        
        summary_data.append([
            f'Gap {gap}',
            f'{len(gap_pairs):,}',
            f'{len(cell_pairs):,}',
            f'{len(nocell_pairs):,}',
            f'{np.mean(cell_vals):.3f}',
            f'{np.std(cell_vals):.3f}',
            f'{np.mean(nocell_vals):.3f}',
            f'{np.std(nocell_vals):.3f}',
            f'{np.mean(cell_vals) - np.mean(nocell_vals):.3f}'
        ])
    
    # Add totals
    summary_data.append([
        'TOTAL',
        f'{len(pair_data):,}',
        f'{len([p for p in pair_data if p["class"]==1]):,}',
        f'{len([p for p in pair_data if p["class"]==0]):,}',
        '-', '-', '-', '-', '-'
    ])
    
    # Add lens-level summary
    summary_data.append(['', '', '', '', '', '', '', '', ''])
    summary_data.append(['LENS-LEVEL', 'Total', 'Cell', 'No Cell', '', '', '', '', ''])
    summary_data.append([
        '',
        f'{len(lens_data):,}',
        f'{len([l for l in lens_data if l["class"]==1]):,}',
        f'{len([l for l in lens_data if l["class"]==0]):,}',
        '', '', '', '', ''
    ])
    
    # Create table
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                    colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Style header
    for j in range(9):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style totals row
    for j in range(9):
        table[(6, j)].set_facecolor('#E8E8E8')
        table[(6, j)].set_text_props(fontweight='bold')
        table[(8, j)].set_facecolor('#D5E8D4')
        table[(8, j)].set_text_props(fontweight='bold')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    plt.title('DATA SUMMARY: Edge Gradient Analysis\n'
              'Pair-Level = Frame-to-Frame Differences | Lens-Level = Aggregated per Lens',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("  EDGE GRADIENT FEATURE ANALYSIS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    histograms_dir = os.path.join(visuals_dir, 'histograms')
    histograms_mean_dir = os.path.join(histograms_dir, 'edge_gradient_mean')
    histograms_std_dir = os.path.join(histograms_dir, 'edge_gradient_std')
    histograms_cases_dir = os.path.join(histograms_dir, 'cases')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(histograms_dir, exist_ok=True)
    os.makedirs(histograms_mean_dir, exist_ok=True)
    os.makedirs(histograms_std_dir, exist_ok=True)
    os.makedirs(histograms_cases_dir, exist_ok=True)
    
    # Create case folders for each gap
    for gap in GAPS:
        os.makedirs(os.path.join(histograms_cases_dir, f'gap_{gap}'), exist_ok=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"  Gaps:   {GAPS}", flush=True)
    
    # ─── Step 1: Collect data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting edge gradient features...", flush=True)
    print("  (This may take 15-20 minutes)", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    pair_data, lens_data, lens_frame_data = collect_edge_gradient_data(INPUT_DIR, GAPS)
    
    cell_pairs = sum(1 for p in pair_data if p['class'] == 1)
    nocell_pairs = sum(1 for p in pair_data if p['class'] == 0)
    cell_lenses = sum(1 for l in lens_data if l['class'] == 1)
    nocell_lenses = sum(1 for l in lens_data if l['class'] == 0)
    
    print(f"\n  PAIR-LEVEL: {len(pair_data):,} samples", flush=True)
    print(f"    Cell: {cell_pairs:,} | No Cell: {nocell_pairs:,}", flush=True)
    print(f"\n  LENS-LEVEL: {len(lens_data):,} samples", flush=True)
    print(f"    Cell: {cell_lenses:,} | No Cell: {nocell_lenses:,}", flush=True)
    sys.stdout.flush()
    
    # ─── Step 2: Compute case data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 2: Computing case-based analysis (Top N max gradients per lens)...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    case_data = compute_case_data(lens_frame_data, GAPS, NUM_CASES)
    print(f"    Case data: {len(case_data):,} entries", flush=True)
    
    # ─── Step 3: Create visualizations ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 3: Creating visualizations...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    # 3.1 Gap scaling plot
    print("    [1] Gap scaling analysis...", flush=True)
    create_gap_scaling_plot(pair_data, os.path.join(visuals_dir, '01_gap_scaling_analysis.png'))
    print("        ✓ Done", flush=True)
    
    # 3.2 ROC curves
    print("    [2] ROC curves...", flush=True)
    create_roc_curves(pair_data, os.path.join(visuals_dir, '02_roc_curves.png'))
    print("        ✓ Done", flush=True)
    
    # 3.3 All gaps confusion matrix
    print("    [3] Confusion matrices (all gaps)...", flush=True)
    cm_results = create_all_gaps_confusion_matrix(pair_data, os.path.join(visuals_dir, '03_confusion_matrices_all_gaps.png'))
    print("        ✓ Done", flush=True)
    
    # 3.4 Individual confusion matrices
    print("    [4] Individual confusion matrices...", flush=True)
    for gap in GAPS:
        acc, thresh = create_confusion_matrix_plot(pair_data, gap, 
                                                   os.path.join(visuals_dir, f'04_confusion_matrix_gap_{gap}.png'))
        print(f"        Gap {gap}: Accuracy={acc:.1%}", flush=True)
    print("        ✓ Done", flush=True)
    
    # 3.5 Data summary table
    print("    [5] Data summary table...", flush=True)
    create_data_summary_table(pair_data, lens_data, os.path.join(visuals_dir, '05_data_summary_table.png'))
    print("        ✓ Done", flush=True)
    
    # 3.6 Histograms - Edge Gradient Mean
    print("    [6] Histograms - Edge Gradient Mean (per gap)...", flush=True)
    for gap in GAPS:
        create_histogram_single_gap(pair_data, gap, 
                                   os.path.join(histograms_mean_dir, f'histogram_edge_mean_gap_{gap}.png'))
        print(f"        Gap {gap} ✓", flush=True)
    print("        ✓ Done", flush=True)
    
    # 3.7 Histograms - Edge Gradient Std
    print("    [7] Histograms - Edge Gradient Std (per gap)...", flush=True)
    for gap in GAPS:
        create_histogram_std_single_gap(pair_data, gap, 
                                       os.path.join(histograms_std_dir, f'histogram_edge_std_gap_{gap}.png'))
        print(f"        Gap {gap} ✓", flush=True)
    print("        ✓ Done", flush=True)
    
    # 3.8 Case histograms
    print("    [8] Case histograms (Top N max per lens - SUM and MEAN)...", flush=True)
    for gap in GAPS:
        print(f"        Gap {gap}:", end=' ', flush=True)
        gap_case_dir = os.path.join(histograms_cases_dir, f'gap_{gap}')
        for case_num in range(1, NUM_CASES + 1):
            create_case_histogram(case_data, gap, case_num, gap_case_dir)
            print(f"C{case_num}", end=' ', flush=True)
        print("✓", flush=True)
    print("        ✓ Done", flush=True)
    
    # 3.9 Case comparison plots
    print("    [9] Case comparison plots...", flush=True)
    for gap in GAPS:
        create_case_comparison_plot(case_data, gap, 
                                   os.path.join(visuals_dir, f'06_case_comparison_gap_{gap}.png'), NUM_CASES)
        print(f"        Gap {gap} ✓", flush=True)
    print("        ✓ Done", flush=True)
    
    # ─── Step 4: Save data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data files...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    # Pair-level CSV
    print("    Saving pair-level CSV...", flush=True)
    pair_csv = os.path.join(data_dir, 'pair_level_edge_gradient.csv')
    if pair_data:
        fieldnames = list(pair_data[0].keys())
        with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_data)
        print(f"    ✓ Saved: {pair_csv} ({len(pair_data):,} rows)", flush=True)
    
    # Lens-level CSV
    print("    Saving lens-level CSV...", flush=True)
    lens_csv = os.path.join(data_dir, 'lens_level_edge_gradient.csv')
    if lens_data:
        fieldnames = list(lens_data[0].keys())
        with open(lens_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lens_data)
        print(f"    ✓ Saved: {lens_csv} ({len(lens_data):,} rows)", flush=True)
    
    # Case-level CSV
    print("    Saving case-level CSV...", flush=True)
    case_csv = os.path.join(data_dir, 'case_level_edge_gradient.csv')
    if case_data:
        fieldnames = list(case_data[0].keys())
        with open(case_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(case_data)
        print(f"    ✓ Saved: {case_csv} ({len(case_data):,} rows)", flush=True)
    
    # Summary report
    print("    Saving summary report...", flush=True)
    report_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  EDGE GRADIENT FEATURE ANALYSIS - SUMMARY REPORT\n")
        f.write("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pair-level samples: {len(pair_data):,}\n")
        f.write(f"  Cell pairs: {cell_pairs:,}\n")
        f.write(f"  No Cell pairs: {nocell_pairs:,}\n\n")
        f.write(f"Lens-level samples: {len(lens_data):,}\n")
        f.write(f"  Cell lenses: {cell_lenses:,}\n")
        f.write(f"  No Cell lenses: {nocell_lenses:,}\n\n")
        f.write(f"Case-level samples: {len(case_data):,}\n\n")
        
        f.write("CONFUSION MATRIX RESULTS\n")
        f.write("-" * 40 + "\n")
        for r in cm_results:
            f.write(f"Gap {r['gap']}: Accuracy={r['accuracy']:.1%} (n={r['n']:,})\n")
        
        best = max(cm_results, key=lambda x: x['accuracy'])
        f.write(f"\nBest: Gap {best['gap']} ({best['accuracy']:.1%})\n")
        
        f.write("\n\nOUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Visuals: {visuals_dir}\n")
        f.write(f"Histograms: {histograms_dir}\n")
    
    print(f"    ✓ Saved: {report_path}", flush=True)
    sys.stdout.flush()
    
    # ─── Summary ───
    print("\n" + "=" * 80, flush=True)
    print("  ✅ EDGE GRADIENT ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    
    best = max(cm_results, key=lambda x: x['accuracy'])
    print(f"\n  🏆 BEST GAP: {best['gap']}", flush=True)
    print(f"     Accuracy: {best['accuracy']:.1%}", flush=True)
    
    print(f"\n  📊 OUTPUTS:", flush=True)
    print(f"     Data:       {data_dir}", flush=True)
    print(f"     Visuals:    {visuals_dir}", flush=True)
    print(f"     Histograms: {histograms_dir}", flush=True)
    print(f"       - Edge Mean: {histograms_mean_dir}", flush=True)
    print(f"       - Edge Std:  {histograms_std_dir}", flush=True)
    print(f"       - Cases:     {histograms_cases_dir}", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()