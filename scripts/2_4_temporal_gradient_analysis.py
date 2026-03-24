# -*- coding: utf-8 -*-
"""
2_4_temporal_gradient_analysis.py

TEMPORAL/GRADIENT ANALYSIS
Analyzes how features change across 30 frames per lens

For each lens (30 frames), computes:
  1. Trend (slope) - Is feature increasing/decreasing over time?
  2. Frame-to-frame gradient - How much change between consecutive frames?
  3. Stability (std) - How much variation across 30 frames?
  4. Range (max-min) - Total spread
  5. First-Last difference - Change from frame 1 to frame 30

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\temporal_analysis\\

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
from scipy.stats import ttest_ind, linregress
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\temporal_analysis"

# Feature extraction settings
FFT_CUTOFF_FRACTION = 1/6

# Class mapping
CLASS_MAP = {
    'contain_cell': 1,
    'no_cell': 0,
}

CLASS_COLORS = {
    0: '#3498db',  # Blue - No Cell
    1: '#e74c3c',  # Red - Cell
}

CLASS_NAMES = {
    0: 'No Cell',
    1: 'Cell'
}


# =============================================================================
# FEATURE EXTRACTION (Single Frame)
# =============================================================================

def create_circular_mask(h, w):
    """Create circular mask for FFT analysis."""
    center = (w / 2, h / 2)
    radius = min(h, w) / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist <= radius


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
    
    # Edge Features
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    edges = cv2.Canny(img, 50, 150)
    
    features['edge_gradient_mean'] = np.mean(gradient_mag)
    features['edge_gradient_std'] = np.std(gradient_mag)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # FFT Features
    h, w = img.shape
    mask = create_circular_mask(h, w)
    masked_img = img.astype(np.float64).copy()
    masked_img[~mask] = 0
    
    F = np.fft.fftshift(np.fft.fft2(masked_img))
    P = np.abs(F) ** 2
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    freq_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    cutoff = min(h, w) * FFT_CUTOFF_FRACTION
    low_mask = freq_dist <= cutoff
    high_mask = freq_dist > cutoff
    
    lf_energy = np.sum(P[low_mask])
    hf_energy = np.sum(P[high_mask])
    total_energy = lf_energy + hf_energy
    
    features['fft_hf_ratio'] = hf_energy / total_energy if total_energy > 0 else 0
    
    # GLCM
    try:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        img_reduced = (img_norm // 4).astype(np.uint8)
        glcm = graycomatrix(img_reduced, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
    except:
        features['glcm_contrast'] = 0
    
    return features


# =============================================================================
# TEMPORAL FEATURE COMPUTATION
# =============================================================================

def compute_temporal_features(frame_features_list):
    """
    Compute temporal features from a list of frame features (30 frames).
    
    For each base feature, computes:
    - trend_slope: Linear regression slope over 30 frames
    - gradient_mean: Mean of frame-to-frame differences
    - gradient_std: Std of frame-to-frame differences
    - stability: Coefficient of variation (std/mean)
    - range: max - min
    - first_last_diff: frame30 - frame1
    """
    if not frame_features_list or len(frame_features_list) < 2:
        return None
    
    base_features = list(frame_features_list[0].keys())
    temporal_features = {}
    
    for feat in base_features:
        values = [f[feat] for f in frame_features_list]
        values = np.array(values)
        
        # Skip if all same
        if np.std(values) < 1e-10:
            temporal_features[f'{feat}_trend_slope'] = 0
            temporal_features[f'{feat}_gradient_mean'] = 0
            temporal_features[f'{feat}_gradient_std'] = 0
            temporal_features[f'{feat}_stability'] = 0
            temporal_features[f'{feat}_range'] = 0
            temporal_features[f'{feat}_first_last_diff'] = 0
            continue
        
        # Trend (linear regression slope)
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = linregress(x, values)
        temporal_features[f'{feat}_trend_slope'] = slope
        
        # Frame-to-frame gradient
        gradients = np.diff(values)
        temporal_features[f'{feat}_gradient_mean'] = np.mean(gradients)
        temporal_features[f'{feat}_gradient_std'] = np.std(gradients)
        temporal_features[f'{feat}_gradient_max'] = np.max(np.abs(gradients))
        
        # Stability (coefficient of variation)
        mean_val = np.mean(values)
        std_val = np.std(values)
        temporal_features[f'{feat}_stability'] = std_val / (np.abs(mean_val) + 1e-10)
        
        # Range
        temporal_features[f'{feat}_range'] = np.max(values) - np.min(values)
        
        # First-last difference
        temporal_features[f'{feat}_first_last_diff'] = values[-1] - values[0]
        
        # Also keep mean and std of the base feature
        temporal_features[f'{feat}_mean'] = mean_val
        temporal_features[f'{feat}_std'] = std_val
    
    return temporal_features


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_temporal_data(input_dir):
    """Collect temporal features from all lenses."""
    import sys
    
    all_data = []
    all_frame_series = []  # For plotting individual lens trajectories
    
    video_folders = sorted([f for f in os.listdir(input_dir) 
                           if os.path.isdir(os.path.join(input_dir, f))])
    
    for vi, video_name in enumerate(video_folders):
        video_path = os.path.join(input_dir, video_name)
        print(f"\n  [{vi+1}/{len(video_folders)}] {video_name}", flush=True)
        
        for category in ['contain_cell', 'no_cell']:
            if category not in CLASS_MAP:
                continue
                
            category_path = os.path.join(video_path, category)
            if not os.path.exists(category_path):
                continue
            
            lens_folders = sorted([f for f in os.listdir(category_path) 
                                  if os.path.isdir(os.path.join(category_path, f))])
            
            print(f"      {category}: {len(lens_folders)} lenses ... ", end='', flush=True)
            
            for lens_idx, lens_name in enumerate(lens_folders):
                lens_path = os.path.join(category_path, lens_name)
                frame_files = sorted([f for f in os.listdir(lens_path) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                # Extract features from all frames
                frame_features_list = []
                for frame_file in frame_files:
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    features = extract_frame_features(img)
                    if features:
                        frame_features_list.append(features)
                
                if len(frame_features_list) < 5:
                    continue
                
                # Compute temporal features
                temporal_features = compute_temporal_features(frame_features_list)
                if temporal_features is None:
                    continue
                
                # Add metadata
                temporal_features['video'] = video_name
                temporal_features['lens'] = lens_name
                temporal_features['category'] = category
                temporal_features['class'] = CLASS_MAP[category]
                temporal_features['num_frames'] = len(frame_features_list)
                
                all_data.append(temporal_features)
                
                # Save frame series for trajectory plotting (sample)
                if len(all_frame_series) < 200:  # Keep 200 samples for plotting
                    all_frame_series.append({
                        'video': video_name,
                        'lens': lens_name,
                        'class': CLASS_MAP[category],
                        'frames': frame_features_list
                    })
                
                # Progress
                if (lens_idx + 1) % 50 == 0:
                    print(f"{lens_idx+1}...", end='', flush=True)
            
            print("done", flush=True)
    
    return all_data, all_frame_series


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_temporal_trajectories(frame_series, feature_name, output_path, n_samples=50):
    """Plot feature values across 30 frames for sample lenses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate by class
    class_0_samples = [s for s in frame_series if s['class'] == 0][:n_samples]
    class_1_samples = [s for s in frame_series if s['class'] == 1][:n_samples]
    
    # Plot No Cell
    ax1 = axes[0]
    for sample in class_0_samples:
        values = [f[feature_name] for f in sample['frames']]
        ax1.plot(range(1, len(values)+1), values, alpha=0.3, color=CLASS_COLORS[0], linewidth=1)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel(feature_name.replace('_', ' ').title())
    ax1.set_title(f'No Cell (n={len(class_0_samples)})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot Cell
    ax2 = axes[1]
    for sample in class_1_samples:
        values = [f[feature_name] for f in sample['frames']]
        ax2.plot(range(1, len(values)+1), values, alpha=0.3, color=CLASS_COLORS[1], linewidth=1)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel(feature_name.replace('_', ' ').title())
    ax2.set_title(f'Cell (n={len(class_1_samples)})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Temporal Trajectories: {feature_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mean_trajectories(frame_series, feature_name, output_path):
    """Plot mean ± std trajectory for each class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in [0, 1]:
        samples = [s for s in frame_series if s['class'] == cls]
        
        # Get all trajectories, pad to same length
        max_len = max(len(s['frames']) for s in samples)
        all_values = []
        
        for sample in samples:
            values = [f[feature_name] for f in sample['frames']]
            # Pad with NaN if shorter
            if len(values) < max_len:
                values = values + [np.nan] * (max_len - len(values))
            all_values.append(values)
        
        all_values = np.array(all_values)
        mean_values = np.nanmean(all_values, axis=0)
        std_values = np.nanstd(all_values, axis=0)
        
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean_values, color=CLASS_COLORS[cls], label=CLASS_NAMES[cls], linewidth=2)
        ax.fill_between(x, mean_values - std_values, mean_values + std_values, 
                       color=CLASS_COLORS[cls], alpha=0.2)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel(feature_name.replace('_', ' ').title())
    ax.set_title(f'Mean Trajectory: {feature_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gradient_distribution(data, feature_name, output_path):
    """Plot distribution of frame-to-frame gradients."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    gradient_feat = f'{feature_name}_gradient_mean'
    slope_feat = f'{feature_name}_trend_slope'
    stability_feat = f'{feature_name}_stability'
    
    # Gradient mean
    ax1 = axes[0]
    for cls in [0, 1]:
        vals = [d[gradient_feat] for d in data if d['class'] == cls and gradient_feat in d]
        ax1.hist(vals, bins=30, alpha=0.6, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Gradient Mean (frame-to-frame change)')
    ax1.set_ylabel('Count')
    ax1.set_title('Frame-to-Frame Gradient', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trend slope
    ax2 = axes[1]
    for cls in [0, 1]:
        vals = [d[slope_feat] for d in data if d['class'] == cls and slope_feat in d]
        ax2.hist(vals, bins=30, alpha=0.6, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Trend Slope (overall direction)')
    ax2.set_ylabel('Count')
    ax2.set_title('Trend Over 30 Frames', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Stability
    ax3 = axes[2]
    for cls in [0, 1]:
        vals = [d[stability_feat] for d in data if d['class'] == cls and stability_feat in d]
        ax3.hist(vals, bins=30, alpha=0.6, color=CLASS_COLORS[cls], 
                label=CLASS_NAMES[cls], edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Stability (coefficient of variation)')
    ax3.set_ylabel('Count')
    ax3.set_title('Temporal Stability', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Temporal Features: {feature_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_temporal_feature_comparison(data, output_path):
    """Plot comparison of all temporal features."""
    # Get temporal feature names (gradient, slope, stability)
    temporal_suffixes = ['_gradient_mean', '_trend_slope', '_stability', '_range']
    base_features = ['edge_gradient_mean', 'intensity_mean', 'fft_hf_ratio', 'glcm_contrast']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    
    for i, base in enumerate(base_features):
        for j, suffix in enumerate(temporal_suffixes):
            ax = axes[i, j]
            feat_name = f'{base}{suffix}'
            
            if feat_name not in data[0]:
                ax.set_visible(False)
                continue
            
            for cls in [0, 1]:
                vals = [d[feat_name] for d in data if d['class'] == cls]
                ax.hist(vals, bins=25, alpha=0.6, color=CLASS_COLORS[cls], 
                       label=CLASS_NAMES[cls] if i == 0 else '', edgecolor='black', linewidth=0.3)
            
            ax.set_title(f'{base[:15]}...\n{suffix[1:]}', fontsize=8, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=7)
    
    plt.suptitle('Temporal Feature Distributions: Cell vs No Cell', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices_temporal(data, output_path):
    """Create confusion matrices for temporal features."""
    # Select key temporal features
    temporal_features = [
        'edge_gradient_mean_stability',
        'edge_gradient_mean_gradient_std',
        'intensity_mean_trend_slope',
        'edge_gradient_mean_range',
        'fft_hf_ratio_stability',
        'intensity_mean_stability',
        'edge_gradient_std_stability',
        'glcm_contrast_gradient_mean'
    ]
    
    # Filter to features that exist
    temporal_features = [f for f in temporal_features if f in data[0]][:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    results = []
    
    for idx, feat in enumerate(temporal_features):
        vals = [d[feat] for d in data]
        labels = [d['class'] for d in data]
        
        # Find optimal threshold
        min_val, max_val = min(vals), max(vals)
        thresholds = np.linspace(min_val, max_val, 50)
        
        best_acc = 0
        best_thresh = 0
        best_dir = 'greater'
        
        for thresh in thresholds:
            pred_g = [1 if v > thresh else 0 for v in vals]
            pred_l = [1 if v < thresh else 0 for v in vals]
            
            acc_g = accuracy_score(labels, pred_g)
            acc_l = accuracy_score(labels, pred_l)
            
            if acc_g > best_acc:
                best_acc = acc_g
                best_thresh = thresh
                best_dir = 'greater'
            if acc_l > best_acc:
                best_acc = acc_l
                best_thresh = thresh
                best_dir = 'less'
        
        # Get predictions at best threshold
        if best_dir == 'greater':
            predictions = [1 if v > best_thresh else 0 for v in vals]
        else:
            predictions = [1 if v < best_thresh else 0 for v in vals]
        
        cm = confusion_matrix(labels, predictions)
        results.append({'feature': feat, 'accuracy': best_acc, 'cm': cm})
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_yticklabels(['No Cell', 'Cell'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=11, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        feat_short = feat[:20] + '...' if len(feat) > 20 else feat
        ax.set_title(f"{feat_short}\nAcc: {best_acc:.1%}", fontsize=9, fontweight='bold')
    
    plt.suptitle('Confusion Matrices: Temporal Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def plot_violin_temporal(data, output_path):
    """Violin plot for temporal features."""
    temporal_features = [
        'edge_gradient_mean_stability',
        'edge_gradient_mean_gradient_std',
        'intensity_mean_stability',
        'fft_hf_ratio_stability',
        'edge_gradient_mean_range',
        'intensity_mean_trend_slope'
    ]
    
    temporal_features = [f for f in temporal_features if f in data[0]][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(temporal_features):
        ax = axes[idx]
        
        vals_0 = [d[feat] for d in data if d['class'] == 0]
        vals_1 = [d[feat] for d in data if d['class'] == 1]
        
        parts = ax.violinplot([vals_0, vals_1], positions=[1, 2], showmeans=True, showmedians=True)
        
        parts['bodies'][0].set_facecolor(CLASS_COLORS[0])
        parts['bodies'][1].set_facecolor(CLASS_COLORS[1])
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_title(feat.replace('_', ' ').title()[:35], fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Temporal Feature Distributions (Violin Plots)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_temporal_statistics(data):
    """Compute statistics for temporal features."""
    # Get all temporal feature names
    temporal_suffixes = ['_gradient_mean', '_gradient_std', '_trend_slope', '_stability', '_range', '_first_last_diff']
    
    feature_cols = [k for k in data[0].keys() 
                   if any(k.endswith(s) for s in temporal_suffixes)]
    
    results = []
    
    for feat in feature_cols:
        vals_0 = [d[feat] for d in data if d['class'] == 0]
        vals_1 = [d[feat] for d in data if d['class'] == 1]
        
        if len(vals_0) < 3 or len(vals_1) < 3:
            continue
        
        t_stat, t_pval = ttest_ind(vals_0, vals_1)
        
        pooled_std = np.sqrt(((len(vals_0)-1)*np.var(vals_0) + (len(vals_1)-1)*np.var(vals_1)) / 
                            (len(vals_0) + len(vals_1) - 2))
        cohens_d = (np.mean(vals_1) - np.mean(vals_0)) / (pooled_std + 1e-10)
        
        results.append({
            'feature': feat,
            'nocell_mean': np.mean(vals_0),
            'cell_mean': np.mean(vals_1),
            'ttest_pval': t_pval,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
        })
    
    return sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("  TEMPORAL/GRADIENT ANALYSIS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    
    # ─── Step 1: Collect temporal data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting temporal features from lenses...", flush=True)
    print("  (Analyzing how features change across 30 frames per lens)", flush=True)
    print("-" * 80, flush=True)
    
    data, frame_series = collect_temporal_data(INPUT_DIR)
    
    print(f"\n  Total lenses analyzed: {len(data)}", flush=True)
    
    cell_count = sum(1 for d in data if d['class'] == 1)
    nocell_count = sum(1 for d in data if d['class'] == 0)
    print(f"  Cell: {cell_count} | No Cell: {nocell_count}", flush=True)
    
    # ─── Step 2: Statistical analysis ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 2: Computing temporal statistics...", flush=True)
    print("-" * 80, flush=True)
    
    stats_results = compute_temporal_statistics(data)
    
    print(f"  Top 5 temporal features by effect size:", flush=True)
    for i, r in enumerate(stats_results[:5]):
        print(f"    {i+1}. {r['feature']}: Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']})", flush=True)
    
    # ─── Step 3: Create visualizations ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 3: Creating visualizations...", flush=True)
    print("-" * 80, flush=True)
    
    # 1. Temporal trajectories for key features
    print("    Creating temporal trajectory plots...", flush=True)
    for feat in ['edge_gradient_mean', 'intensity_mean', 'fft_hf_ratio']:
        plot_temporal_trajectories(frame_series, feat, 
                                   os.path.join(visuals_dir, f'trajectories_{feat}.png'))
        plot_mean_trajectories(frame_series, feat,
                              os.path.join(visuals_dir, f'mean_trajectory_{feat}.png'))
    
    # 2. Gradient distributions
    print("    Creating gradient distribution plots...", flush=True)
    for feat in ['edge_gradient_mean', 'intensity_mean']:
        plot_gradient_distribution(data, feat,
                                   os.path.join(visuals_dir, f'gradient_dist_{feat}.png'))
    
    # 3. Temporal feature comparison
    print("    Creating temporal feature comparison...", flush=True)
    plot_temporal_feature_comparison(data, os.path.join(visuals_dir, 'temporal_feature_comparison.png'))
    
    # 4. Confusion matrices
    print("    Creating confusion matrices...", flush=True)
    cm_results = plot_confusion_matrices_temporal(data, os.path.join(visuals_dir, 'confusion_matrices_temporal.png'))
    
    # 5. Violin plots
    print("    Creating violin plots...", flush=True)
    plot_violin_temporal(data, os.path.join(visuals_dir, 'violin_temporal.png'))
    
    # ─── Step 4: Save data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data...", flush=True)
    print("-" * 80, flush=True)
    
    # Get all feature columns
    feature_cols = [k for k in data[0].keys() if k not in ['video', 'lens', 'category', 'class', 'num_frames']]
    
    # Save temporal features
    csv_path = os.path.join(data_dir, 'temporal_features.csv')
    fieldnames = ['video', 'lens', 'category', 'class', 'num_frames'] + feature_cols
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow({k: d.get(k, '') for k in fieldnames})
    print(f"    Saved: {csv_path}", flush=True)
    
    # Save statistics
    stats_csv = os.path.join(data_dir, 'temporal_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_results[0].keys()))
        writer.writeheader()
        writer.writerows(stats_results)
    print(f"    Saved: {stats_csv}", flush=True)
    
    # ─── Summary ───
    print("\n" + "=" * 80, flush=True)
    print("  TEMPORAL ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  Total lenses: {len(data)}", flush=True)
    print(f"  Cell: {cell_count} | No Cell: {nocell_count}", flush=True)
    print(f"\n  TOP TEMPORAL FEATURES:", flush=True)
    for i, r in enumerate(stats_results[:3]):
        print(f"    {i+1}. {r['feature']}", flush=True)
        print(f"       Effect size: {r['cohens_d']:.3f} ({r['effect_size']})", flush=True)
    if cm_results:
        best_cm = max(cm_results, key=lambda x: x['accuracy'])
        print(f"\n  Best classification accuracy: {best_cm['accuracy']:.1%}", flush=True)
        print(f"    Feature: {best_cm['feature']}", flush=True)
    print(f"\n  OUTPUTS:", flush=True)
    print(f"    Visuals: {visuals_dir}", flush=True)
    print(f"    Data:    {data_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()