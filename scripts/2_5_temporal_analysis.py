# -*- coding: utf-8 -*-
"""
2_5_gap_temporal_analysis.py

GAP-BASED TEMPORAL ANALYSIS
Analyzes feature differences between frames separated by various gaps

For each lens (30 frames), computes differences between frame pairs:
  - Gap 1:  (F1,F2), (F2,F3), ... (F29,F30) → 29 pairs
  - Gap 3:  (F1,F4), (F2,F5), ... (F27,F30) → 27 pairs
  - Gap 5:  (F1,F6), (F2,F7), ... (F25,F30) → 25 pairs
  - Gap 7:  (F1,F8), (F2,F9), ... (F23,F30) → 23 pairs
  - Gap 10: (F1,F11), (F2,F12), ... (F20,F30) → 20 pairs

Each lens is INDEPENDENT - no averaging across lenses!

Output:
  1. Pair-level data (~400K rows)
  2. Lens-level aggregated stats (~3.2K rows)
  3. High-quality visualizations (10 plots)

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\gap_analysis\\

Author: Antardip Himel
Date: February 2026
"""

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from collections import defaultdict
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

# Professional color palette (colorblind-friendly)
COLORS = {
    'cell': '#E63946',      # Red
    'no_cell': '#457B9D',   # Blue
    'accent1': '#2A9D8F',   # Teal
    'accent2': '#E9C46A',   # Yellow
    'accent3': '#F4A261',   # Orange
    'accent4': '#9B59B6',   # Purple
    'background': '#F8F9FA',
    'grid': '#E0E0E0'
}

CLASS_COLORS = {
    0: COLORS['no_cell'],
    1: COLORS['cell']
}

CLASS_NAMES = {
    0: 'No Cell',
    1: 'Cell'
}

# Gap colors for multi-gap plots
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
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\gap_analysis"

# Gaps to analyze
GAPS = [1, 3, 5, 7, 10]

# Class mapping
CLASS_MAP = {
    'contain_cell': 1,
    'no_cell': 0,
}


# =============================================================================
# FEATURE EXTRACTION (Single Frame)
# =============================================================================

def extract_frame_features(img):
    """Extract features from a single grayscale frame."""
    if img is None:
        return None
    
    features = {}
    pixels = img.flatten().astype(np.float64)
    
    # Intensity Features
    features['intensity_mean'] = np.mean(pixels)
    features['intensity_std'] = np.std(pixels)
    features['intensity_entropy'] = stats.entropy(np.histogram(pixels, bins=256)[0] + 1e-10)
    
    # Edge Features (Sobel gradient)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    features['edge_gradient_mean'] = np.mean(gradient_mag)
    features['edge_gradient_std'] = np.std(gradient_mag)
    
    # Edge density (Canny)
    edges = cv2.Canny(img, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # GLCM Texture
    try:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        img_reduced = (img_norm // 4).astype(np.uint8)
        glcm = graycomatrix(img_reduced, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
    except:
        features['glcm_contrast'] = 0
        features['glcm_homogeneity'] = 0
    
    return features


def compute_pair_features(feat_i, feat_j, gap):
    """Compute difference features between two frames."""
    pair_features = {'gap': gap}
    
    base_features = ['edge_gradient_mean', 'intensity_mean', 'intensity_entropy', 
                     'edge_gradient_std', 'edge_density', 'glcm_contrast']
    
    for feat in base_features:
        if feat not in feat_i or feat not in feat_j:
            continue
            
        val_i = feat_i[feat]
        val_j = feat_j[feat]
        
        # Difference (signed)
        pair_features[f'{feat}_diff'] = val_j - val_i
        
        # Absolute difference
        pair_features[f'{feat}_abs_diff'] = abs(val_j - val_i)
        
        # Ratio (with safety for division by zero)
        if abs(val_i) > 1e-10:
            pair_features[f'{feat}_ratio'] = val_j / val_i
            pair_features[f'{feat}_pct_change'] = ((val_j - val_i) / abs(val_i)) * 100
        else:
            pair_features[f'{feat}_ratio'] = 1.0
            pair_features[f'{feat}_pct_change'] = 0.0
    
    return pair_features


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_gap_data(input_dir, gaps):
    """Collect pair-level and lens-level features for all gaps."""
    import sys
    
    pair_level_data = []  # Each frame pair = 1 row
    lens_level_data = []  # Each lens = 1 row (aggregated)
    lens_frame_series = []  # For trajectory plotting
    
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
                
                # Extract features from ALL frames first
                frame_features = []
                for frame_file in frame_files:
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    features = extract_frame_features(img)
                    if features:
                        features['frame_idx'] = len(frame_features) + 1
                        frame_features.append(features)
                
                if len(frame_features) < 15:
                    continue
                
                total_lenses += 1
                lens_pairs_count = 0
                
                # Store frame series for trajectory plotting (sample)
                if len(lens_frame_series) < 300:
                    lens_frame_series.append({
                        'video': video_name,
                        'lens': lens_name,
                        'class': CLASS_MAP[category],
                        'frames': frame_features
                    })
                
                # === PAIR-LEVEL DATA ===
                # Compute differences for each gap
                lens_pair_features = {gap: [] for gap in gaps}
                
                for gap in gaps:
                    for i in range(len(frame_features) - gap):
                        j = i + gap
                        
                        pair_feat = compute_pair_features(frame_features[i], frame_features[j], gap)
                        pair_feat['video'] = video_name
                        pair_feat['lens'] = lens_name
                        pair_feat['frame_i'] = i + 1
                        pair_feat['frame_j'] = j + 1
                        pair_feat['category'] = category
                        pair_feat['class'] = CLASS_MAP[category]
                        
                        pair_level_data.append(pair_feat)
                        lens_pair_features[gap].append(pair_feat)
                        lens_pairs_count += 1
                        total_pairs += 1
                
                # === LENS-LEVEL DATA ===
                # Aggregate stats for this lens (no mixing with other lenses!)
                lens_stats = {
                    'video': video_name,
                    'lens': lens_name,
                    'category': category,
                    'class': CLASS_MAP[category],
                    'num_frames': len(frame_features)
                }
                
                # Cumulative drift (F_last - F_first)
                lens_stats['edge_drift'] = frame_features[-1]['edge_gradient_mean'] - frame_features[0]['edge_gradient_mean']
                lens_stats['intensity_drift'] = frame_features[-1]['intensity_mean'] - frame_features[0]['intensity_mean']
                lens_stats['entropy_drift'] = frame_features[-1]['intensity_entropy'] - frame_features[0]['intensity_entropy']
                
                # Per-gap aggregated stats
                for gap in gaps:
                    if not lens_pair_features[gap]:
                        continue
                    
                    gap_pairs = lens_pair_features[gap]
                    
                    for base_feat in ['edge_gradient_mean', 'intensity_mean', 'intensity_entropy']:
                        diff_key = f'{base_feat}_abs_diff'
                        
                        if diff_key not in gap_pairs[0]:
                            continue
                        
                        diffs = [p[diff_key] for p in gap_pairs]
                        
                        # Stats for this gap
                        lens_stats[f'{base_feat}_gap{gap}_mean_diff'] = np.mean(diffs)
                        lens_stats[f'{base_feat}_gap{gap}_std_diff'] = np.std(diffs)
                        lens_stats[f'{base_feat}_gap{gap}_max_diff'] = np.max(diffs)
                
                # Motion Energy (sum of all absolute edge diffs across all gaps)
                all_edge_diffs = [p['edge_gradient_mean_abs_diff'] for p in pair_level_data 
                                  if p['lens'] == lens_name and p['video'] == video_name]
                lens_stats['motion_energy'] = np.sum(all_edge_diffs) / len(all_edge_diffs) if all_edge_diffs else 0
                
                # Temporal Variability Index (TVI)
                if all_edge_diffs and np.mean(all_edge_diffs) > 1e-10:
                    lens_stats['tvi'] = np.std(all_edge_diffs) / np.mean(all_edge_diffs)
                else:
                    lens_stats['tvi'] = 0
                
                # Max Jump
                lens_stats['max_jump'] = np.max(all_edge_diffs) if all_edge_diffs else 0
                
                lens_level_data.append(lens_stats)
                
                # Progress - show dots every 10 lenses
                if (lens_idx + 1) % 10 == 0:
                    print(".", end='', flush=True)
                    sys.stdout.flush()
                if (lens_idx + 1) % 50 == 0:
                    print(f"[{lens_idx+1}]", end='', flush=True)
                    sys.stdout.flush()
            
            print(f" done ({lens_pairs_count if 'lens_pairs_count' in dir() else 0} pairs)", flush=True)
            sys.stdout.flush()
        
        # Show running total after each video
        print(f"      Running total: {total_lenses} lenses, {total_pairs:,} pairs", flush=True)
        sys.stdout.flush()
    
    print(f"\n  " + "=" * 50, flush=True)
    print(f"  COLLECTION COMPLETE!", flush=True)
    print(f"  Total lenses processed: {total_lenses}", flush=True)
    print(f"  Total pair-level samples: {total_pairs:,}", flush=True)
    print(f"  " + "=" * 50, flush=True)
    sys.stdout.flush()
    
    return pair_level_data, lens_level_data, lens_frame_series


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_01_gap_scaling_analysis(pair_data, output_path):
    """
    Plot 1: How does mean |difference| scale with gap size?
    Shows if cells have increasing differences with larger gaps (motion).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    features_to_plot = ['edge_gradient_mean', 'intensity_mean', 'intensity_entropy']
    feature_labels = ['Edge Gradient', 'Intensity', 'Entropy']
    
    for ax, feat, label in zip(axes, features_to_plot, feature_labels):
        diff_key = f'{feat}_abs_diff'
        
        for cls in [0, 1]:
            means = []
            stds = []
            
            for gap in GAPS:
                vals = [p[diff_key] for p in pair_data 
                       if p['class'] == cls and p['gap'] == gap and diff_key in p]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) / np.sqrt(len(vals)))  # SEM
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.errorbar(GAPS, means, yerr=stds, marker='o', markersize=8,
                       linewidth=2.5, capsize=5, capthick=2,
                       color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
        
        ax.set_xlabel('Frame Gap', fontsize=12)
        ax.set_ylabel(f'Mean |Δ{label}|', fontsize=12)
        ax.set_title(f'{label} Change vs Gap', fontweight='bold', fontsize=13)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(GAPS)
        
        # Highlight the gap region
        ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Gap Scaling Analysis: Does Difference Increase with Gap?', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_02_violin_by_gap(pair_data, output_path):
    """
    Plot 2: Violin plots showing distribution of differences at each gap.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for row, cls in enumerate([0, 1]):
        for col, gap in enumerate(GAPS[:3]):  # First 3 gaps in top row
            ax = axes[row, col]
            
            vals = [p[feat] for p in pair_data if p['class'] == cls and p['gap'] == gap]
            
            if vals:
                parts = ax.violinplot([vals], positions=[1], showmeans=True, showmedians=True)
                parts['bodies'][0].set_facecolor(CLASS_COLORS[cls])
                parts['bodies'][0].set_alpha(0.7)
                parts['bodies'][0].set_edgecolor('black')
            
            ax.set_title(f'{CLASS_NAMES[cls]} - Gap {gap}', fontweight='bold')
            ax.set_ylabel('|Edge Gradient Diff|')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor(COLORS['background'])
    
    # Add remaining gaps in a different layout
    plt.suptitle('Distribution of Edge Gradient Changes by Gap and Class', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_03_heatmap_accuracy(pair_data, output_path):
    """
    Plot 3: Heatmap showing classification accuracy for each gap × feature combo.
    """
    features = ['edge_gradient_mean_abs_diff', 'intensity_mean_abs_diff', 
                'intensity_entropy_abs_diff', 'edge_gradient_std_abs_diff',
                'edge_density_abs_diff', 'glcm_contrast_abs_diff']
    
    feature_labels = ['Edge Grad', 'Intensity', 'Entropy', 'Edge Std', 'Edge Density', 'GLCM']
    
    accuracy_matrix = np.zeros((len(GAPS), len(features)))
    
    for i, gap in enumerate(GAPS):
        gap_data = [p for p in pair_data if p['gap'] == gap]
        
        for j, feat in enumerate(features):
            if not gap_data or feat not in gap_data[0]:
                continue
            
            vals = [p[feat] for p in gap_data]
            labels = [p['class'] for p in gap_data]
            
            # Find optimal threshold
            thresholds = np.linspace(min(vals), max(vals), 30)
            best_acc = 0
            
            for thresh in thresholds:
                pred = [1 if v > thresh else 0 for v in vals]
                acc = accuracy_score(labels, pred)
                best_acc = max(best_acc, acc, 1 - acc)
            
            accuracy_matrix[i, j] = best_acc
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
    
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(GAPS)))
    ax.set_yticklabels([f'Gap {g}' for g in GAPS], fontsize=11)
    
    # Add text annotations
    for i in range(len(GAPS)):
        for j in range(len(features)):
            val = accuracy_matrix[i, j]
            color = 'white' if val > 0.75 else 'black'
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Classification Accuracy', fontsize=12)
    
    ax.set_xlabel('Feature', fontsize=13)
    ax.set_ylabel('Frame Gap', fontsize=13)
    ax.set_title('Classification Accuracy: Gap × Feature', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return accuracy_matrix


def plot_04_motion_energy_distribution(lens_data, output_path):
    """
    Plot 4: Distribution of motion energy by class.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Motion Energy
    ax1 = axes[0]
    for cls in [0, 1]:
        vals = [d['motion_energy'] for d in lens_data if d['class'] == cls]
        ax1.hist(vals, bins=40, alpha=0.7, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Motion Energy', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Motion Energy Distribution', fontweight='bold', fontsize=13)
    ax1.legend(frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor(COLORS['background'])
    
    # TVI
    ax2 = axes[1]
    for cls in [0, 1]:
        vals = [d['tvi'] for d in lens_data if d['class'] == cls and d['tvi'] < 5]  # Filter outliers
        ax2.hist(vals, bins=40, alpha=0.7, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Temporal Variability Index', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('TVI Distribution', fontweight='bold', fontsize=13)
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor(COLORS['background'])
    
    # Max Jump
    ax3 = axes[2]
    for cls in [0, 1]:
        vals = [d['max_jump'] for d in lens_data if d['class'] == cls]
        ax3.hist(vals, bins=40, alpha=0.7, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Max Jump (Largest Single Change)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Max Jump Distribution', fontweight='bold', fontsize=13)
    ax3.legend(frameon=True, fancybox=True)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor(COLORS['background'])
    
    plt.suptitle('Lens-Level Motion Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_05_scatter_tvi_vs_motion(lens_data, output_path):
    """
    Plot 5: 2D scatter of TVI vs Motion Energy (clustering view).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in [0, 1]:
        x = [d['motion_energy'] for d in lens_data if d['class'] == cls]
        y = [d['tvi'] for d in lens_data if d['class'] == cls and d['tvi'] < 5]
        
        # Match lengths
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        ax.scatter(x, y, c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                  alpha=0.5, s=40, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Motion Energy', fontsize=13)
    ax.set_ylabel('Temporal Variability Index (TVI)', fontsize=13)
    ax.set_title('Motion Energy vs TVI: Class Separation', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_06_sample_trajectories(frame_series, output_path, n_samples=20):
    """
    Plot 6: Sample lens trajectories showing feature values across frames.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: Edge Gradient Mean
    # Bottom row: Intensity Mean
    
    class_0_samples = [s for s in frame_series if s['class'] == 0][:n_samples]
    class_1_samples = [s for s in frame_series if s['class'] == 1][:n_samples]
    
    # No Cell - Edge
    ax = axes[0, 0]
    for sample in class_0_samples:
        vals = [f['edge_gradient_mean'] for f in sample['frames']]
        ax.plot(range(1, len(vals)+1), vals, alpha=0.4, color=CLASS_COLORS[0], linewidth=1.5)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Edge Gradient Mean')
    ax.set_title('No Cell - Edge Gradient Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    # Cell - Edge
    ax = axes[0, 1]
    for sample in class_1_samples:
        vals = [f['edge_gradient_mean'] for f in sample['frames']]
        ax.plot(range(1, len(vals)+1), vals, alpha=0.4, color=CLASS_COLORS[1], linewidth=1.5)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Edge Gradient Mean')
    ax.set_title('Cell - Edge Gradient Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    # No Cell - Intensity
    ax = axes[1, 0]
    for sample in class_0_samples:
        vals = [f['intensity_mean'] for f in sample['frames']]
        ax.plot(range(1, len(vals)+1), vals, alpha=0.4, color=CLASS_COLORS[0], linewidth=1.5)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Intensity Mean')
    ax.set_title('No Cell - Intensity Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    # Cell - Intensity
    ax = axes[1, 1]
    for sample in class_1_samples:
        vals = [f['intensity_mean'] for f in sample['frames']]
        ax.plot(range(1, len(vals)+1), vals, alpha=0.4, color=CLASS_COLORS[1], linewidth=1.5)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Intensity Mean')
    ax.set_title('Cell - Intensity Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    plt.suptitle(f'Individual Lens Trajectories (n={n_samples} per class)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_07_confusion_matrices(pair_data, lens_data, output_path):
    """
    Plot 7: Confusion matrices for best features at different gaps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Best feature: edge_gradient_mean_abs_diff at each gap
    feat = 'edge_gradient_mean_abs_diff'
    
    for idx, gap in enumerate(GAPS):
        if idx >= 5:
            break
            
        ax = axes[idx]
        
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = [p[feat] for p in gap_data]
        labels = [p['class'] for p in gap_data]
        
        # Find optimal threshold
        thresholds = np.linspace(min(vals), max(vals), 50)
        best_acc = 0
        best_pred = None
        
        for thresh in thresholds:
            pred_g = [1 if v > thresh else 0 for v in vals]
            pred_l = [1 if v < thresh else 0 for v in vals]
            
            acc_g = accuracy_score(labels, pred_g)
            acc_l = accuracy_score(labels, pred_l)
            
            if acc_g > best_acc:
                best_acc = acc_g
                best_pred = pred_g
            if acc_l > best_acc:
                best_acc = acc_l
                best_pred = pred_l
        
        cm = confusion_matrix(labels, best_pred)
        
        # Plot
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_yticklabels(['No Cell', 'Cell'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=color)
        
        ax.set_title(f'Gap {gap}: Acc={best_acc:.1%}', fontweight='bold', fontsize=12)
    
    # Last subplot: Lens-level with motion energy
    ax = axes[5]
    vals = [d['motion_energy'] for d in lens_data]
    labels = [d['class'] for d in lens_data]
    
    thresholds = np.linspace(min(vals), max(vals), 50)
    best_acc = 0
    best_pred = None
    
    for thresh in thresholds:
        pred_g = [1 if v > thresh else 0 for v in vals]
        pred_l = [1 if v < thresh else 0 for v in vals]
        
        acc_g = accuracy_score(labels, pred_g)
        acc_l = accuracy_score(labels, pred_l)
        
        if acc_g > best_acc:
            best_acc = acc_g
            best_pred = pred_g
        if acc_l > best_acc:
            best_acc = acc_l
            best_pred = pred_l
    
    cm = confusion_matrix(labels, best_pred)
    
    im = ax.imshow(cm, cmap='Greens')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Cell', 'Cell'])
    ax.set_yticklabels(['No Cell', 'Cell'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)
    
    ax.set_title(f'Motion Energy: Acc={best_acc:.1%}', fontweight='bold', fontsize=12)
    
    plt.suptitle('Confusion Matrices by Gap (Edge Gradient)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_08_roc_curves(pair_data, lens_data, output_path):
    """
    Plot 8: ROC curves comparing different gaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    # Left: ROC by gap
    ax1 = axes[0]
    for gap in GAPS:
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = np.array([p[feat] for p in gap_data])
        labels = np.array([p['class'] for p in gap_data])
        
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color=GAP_COLORS[gap], linewidth=2.5,
                label=f'Gap {gap} (AUC={roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves by Gap', fontweight='bold', fontsize=13)
    ax1.legend(loc='lower right', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(COLORS['background'])
    
    # Right: ROC for lens-level metrics
    ax2 = axes[1]
    
    metrics = ['motion_energy', 'tvi', 'max_jump']
    colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]
    
    for metric, color in zip(metrics, colors):
        vals = np.array([d[metric] for d in lens_data])
        labels = np.array([d['class'] for d in lens_data])
        
        # Handle NaN/Inf
        mask = np.isfinite(vals)
        vals = vals[mask]
        labels = labels[mask]
        
        if len(vals) > 0:
            fpr, tpr, _ = roc_curve(labels, vals)
            roc_auc = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{metric.replace("_", " ").title()} (AUC={roc_auc:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves: Lens-Level Metrics', fontweight='bold', fontsize=13)
    ax2.legend(loc='lower right', frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(COLORS['background'])
    
    plt.suptitle('ROC Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_09_boxplots_by_gap(pair_data, output_path):
    """
    Plot 9: Box plots comparing classes at each gap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    features = ['edge_gradient_mean_abs_diff', 'intensity_mean_abs_diff', 'intensity_entropy_abs_diff']
    titles = ['Edge Gradient', 'Intensity', 'Entropy']
    
    for ax, feat, title in zip(axes, features, titles):
        # Prepare data for boxplot
        data_to_plot = []
        positions = []
        colors_list = []
        
        for gi, gap in enumerate(GAPS):
            for cls in [0, 1]:
                vals = [p[feat] for p in pair_data if p['gap'] == gap and p['class'] == cls]
                
                # Subsample if too large
                if len(vals) > 5000:
                    vals = np.random.choice(vals, 5000, replace=False)
                
                data_to_plot.append(vals)
                positions.append(gi * 3 + cls + 1)
                colors_list.append(CLASS_COLORS[cls])
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.8, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks([gi * 3 + 1.5 for gi in range(len(GAPS))])
        ax.set_xticklabels([f'Gap {g}' for g in GAPS])
        ax.set_ylabel(f'|Δ{title}|')
        ax.set_title(f'{title} Difference by Gap', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=CLASS_COLORS[0], alpha=0.7, label='No Cell'),
                          Patch(facecolor=CLASS_COLORS[1], alpha=0.7, label='Cell')]
        ax.legend(handles=legend_elements, loc='upper left')
    
    plt.suptitle('Box Plots: Feature Differences by Gap and Class', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_10_cumulative_drift(lens_data, output_path):
    """
    Plot 10: Cumulative drift (F_last - F_first) distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['edge_drift', 'intensity_drift', 'entropy_drift']
    titles = ['Edge Gradient Drift', 'Intensity Drift', 'Entropy Drift']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for cls in [0, 1]:
            vals = [d[metric] for d in lens_data if d['class'] == cls and metric in d]
            ax.hist(vals, bins=40, alpha=0.7, color=CLASS_COLORS[cls],
                   label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel(f'{title} (F₃₀ - F₁)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Cumulative Drift: Total Change from First to Last Frame', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def compute_gap_statistics(pair_data, lens_data):
    """Compute summary statistics for each gap."""
    results = []
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for gap in GAPS:
        gap_data = [p for p in pair_data if p['gap'] == gap]
        
        vals = [p[feat] for p in gap_data]
        labels = [p['class'] for p in gap_data]
        
        vals_0 = [v for v, l in zip(vals, labels) if l == 0]
        vals_1 = [v for v, l in zip(vals, labels) if l == 1]
        
        # T-test
        t_stat, t_pval = ttest_ind(vals_0, vals_1)
        
        # Effect size
        pooled_std = np.sqrt(((len(vals_0)-1)*np.var(vals_0) + (len(vals_1)-1)*np.var(vals_1)) / 
                            (len(vals_0) + len(vals_1) - 2))
        cohens_d = (np.mean(vals_1) - np.mean(vals_0)) / (pooled_std + 1e-10)
        
        # Best accuracy
        thresholds = np.linspace(min(vals), max(vals), 50)
        best_acc = 0
        for thresh in thresholds:
            pred = [1 if v > thresh else 0 for v in vals]
            acc = accuracy_score(labels, pred)
            best_acc = max(best_acc, acc, 1 - acc)
        
        results.append({
            'gap': gap,
            'n_pairs': len(gap_data),
            'nocell_mean': np.mean(vals_0),
            'cell_mean': np.mean(vals_1),
            'cohens_d': cohens_d,
            'ttest_pval': t_pval,
            'best_accuracy': best_acc
        })
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    
    print("=" * 80, flush=True)
    print("  GAP-BASED TEMPORAL ANALYSIS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"  Gaps:   {GAPS}", flush=True)
    
    # ─── Step 1: Collect data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting features and computing pair differences...", flush=True)
    print("  (This may take 15-20 minutes)", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    pair_data, lens_data, frame_series = collect_gap_data(INPUT_DIR, GAPS)
    
    cell_pairs = sum(1 for p in pair_data if p['class'] == 1)
    nocell_pairs = sum(1 for p in pair_data if p['class'] == 0)
    cell_lenses = sum(1 for d in lens_data if d['class'] == 1)
    nocell_lenses = sum(1 for d in lens_data if d['class'] == 0)
    
    print(f"\n  PAIR-LEVEL: {len(pair_data):,} samples", flush=True)
    print(f"    Cell: {cell_pairs:,} | No Cell: {nocell_pairs:,}", flush=True)
    print(f"\n  LENS-LEVEL: {len(lens_data):,} samples", flush=True)
    print(f"    Cell: {cell_lenses:,} | No Cell: {nocell_lenses:,}", flush=True)
    sys.stdout.flush()
    
    # ─── Step 2: Compute statistics ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 2: Computing gap statistics...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    gap_stats = compute_gap_statistics(pair_data, lens_data)
    
    print("\n  Gap Analysis Summary (Edge Gradient):", flush=True)
    print("  " + "-" * 60, flush=True)
    print(f"  {'Gap':<6} {'Accuracy':<12} {'Cohen d':<12} {'Cell Mean':<12} {'NoCell Mean':<12}", flush=True)
    print("  " + "-" * 60, flush=True)
    for gs in gap_stats:
        print(f"  {gs['gap']:<6} {gs['best_accuracy']:.1%}        {gs['cohens_d']:.3f}        {gs['cell_mean']:.2f}        {gs['nocell_mean']:.2f}", flush=True)
    sys.stdout.flush()
    
    # ─── Step 3: Create visualizations ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 3: Creating high-quality visualizations...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    print("    [1/10] Gap scaling analysis...", flush=True)
    sys.stdout.flush()
    plot_01_gap_scaling_analysis(pair_data, os.path.join(visuals_dir, '01_gap_scaling_analysis.png'))
    print("           ✓ Done", flush=True)
    
    print("    [2/10] Violin plots by gap...", flush=True)
    sys.stdout.flush()
    plot_02_violin_by_gap(pair_data, os.path.join(visuals_dir, '02_violin_by_gap.png'))
    print("           ✓ Done", flush=True)
    
    print("    [3/10] Accuracy heatmap...", flush=True)
    sys.stdout.flush()
    plot_03_heatmap_accuracy(pair_data, os.path.join(visuals_dir, '03_heatmap_gap_feature_accuracy.png'))
    print("           ✓ Done", flush=True)
    
    print("    [4/10] Motion energy distribution...", flush=True)
    sys.stdout.flush()
    plot_04_motion_energy_distribution(lens_data, os.path.join(visuals_dir, '04_motion_energy_distribution.png'))
    print("           ✓ Done", flush=True)
    
    print("    [5/10] TVI vs Motion scatter...", flush=True)
    sys.stdout.flush()
    plot_05_scatter_tvi_vs_motion(lens_data, os.path.join(visuals_dir, '05_scatter_tvi_vs_motion.png'))
    print("           ✓ Done", flush=True)
    
    print("    [6/10] Sample trajectories...", flush=True)
    sys.stdout.flush()
    plot_06_sample_trajectories(frame_series, os.path.join(visuals_dir, '06_sample_lens_trajectories.png'))
    print("           ✓ Done", flush=True)
    
    print("    [7/10] Confusion matrices...", flush=True)
    sys.stdout.flush()
    plot_07_confusion_matrices(pair_data, lens_data, os.path.join(visuals_dir, '07_confusion_matrices_by_gap.png'))
    print("           ✓ Done", flush=True)
    
    print("    [8/10] ROC curves...", flush=True)
    sys.stdout.flush()
    plot_08_roc_curves(pair_data, lens_data, os.path.join(visuals_dir, '08_roc_curves.png'))
    print("           ✓ Done", flush=True)
    
    print("    [9/10] Box plots...", flush=True)
    sys.stdout.flush()
    plot_09_boxplots_by_gap(pair_data, os.path.join(visuals_dir, '09_boxplots_by_gap.png'))
    print("           ✓ Done", flush=True)
    
    print("    [10/10] Cumulative drift...", flush=True)
    sys.stdout.flush()
    plot_10_cumulative_drift(lens_data, os.path.join(visuals_dir, '10_cumulative_drift.png'))
    print("           ✓ Done", flush=True)
    
    print("\n    All visualizations complete!", flush=True)
    sys.stdout.flush()
    
    # ─── Step 4: Save data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data files...", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()
    
    # Pair-level CSV
    print("    Saving pair-level CSV...", flush=True)
    sys.stdout.flush()
    pair_csv = os.path.join(data_dir, 'pair_level_features.csv')
    if pair_data:
        fieldnames = list(pair_data[0].keys())
        with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_data)
        print(f"    ✓ Saved: {pair_csv} ({len(pair_data):,} rows)", flush=True)
    
    # Lens-level CSV
    print("    Saving lens-level CSV...", flush=True)
    sys.stdout.flush()
    lens_csv = os.path.join(data_dir, 'lens_level_features.csv')
    if lens_data:
        fieldnames = list(lens_data[0].keys())
        with open(lens_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lens_data)
        print(f"    ✓ Saved: {lens_csv} ({len(lens_data):,} rows)", flush=True)
    
    # Gap statistics CSV
    print("    Saving gap statistics...", flush=True)
    sys.stdout.flush()
    stats_csv = os.path.join(data_dir, 'gap_accuracy_summary.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_stats[0].keys()))
        writer.writeheader()
        writer.writerows(gap_stats)
    print(f"    ✓ Saved: {stats_csv}", flush=True)
    
    # Summary report
    print("    Saving summary report...", flush=True)
    sys.stdout.flush()
    report_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  GAP-BASED TEMPORAL ANALYSIS - SUMMARY REPORT\n")
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
        
        f.write("GAP ANALYSIS RESULTS (Edge Gradient)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Gap':<8} {'Accuracy':<12} {'Cohen d':<12}\n")
        for gs in gap_stats:
            f.write(f"{gs['gap']:<8} {gs['best_accuracy']:.1%}        {gs['cohens_d']:.3f}\n")
        
        best_gap = max(gap_stats, key=lambda x: x['best_accuracy'])
        f.write(f"\nBest Gap: {best_gap['gap']} (Accuracy: {best_gap['best_accuracy']:.1%})\n")
        
        f.write("\n\nOUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Visuals: {visuals_dir}\n")
        f.write(f"Data: {data_dir}\n")
    
    print(f"    ✓ Saved: {report_path}", flush=True)
    sys.stdout.flush()
    
    # ─── Summary ───
    print("\n" + "=" * 80, flush=True)
    print("  ✅ GAP ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    best_gap = max(gap_stats, key=lambda x: x['best_accuracy'])
    print(f"\n  🏆 BEST GAP: {best_gap['gap']}", flush=True)
    print(f"     Accuracy: {best_gap['best_accuracy']:.1%}", flush=True)
    print(f"     Effect size (Cohen's d): {best_gap['cohens_d']:.3f}", flush=True)
    
    print(f"\n  📊 OUTPUTS:", flush=True)
    print(f"     Visuals: {visuals_dir}", flush=True)
    print(f"     Data:    {data_dir}", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()