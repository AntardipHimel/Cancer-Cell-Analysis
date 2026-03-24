# -*- coding: utf-8 -*-
"""
2_3_frame_level_feature_analysis.py

FRAME-BY-FRAME FEATURE ANALYSIS
Analyzes each individual frame (not aggregated per lens)

Each lens has 30 frames → Each frame = 1 data point
Total: ~135,000 samples (4,517 lenses × 30 frames)

Outputs:
  1. Scatter plots (Feature 1 vs Feature 2, colored by class)
  2. Violin plots (distribution shape per feature)
  3. Strip/Swarm plots (individual points)
  4. Confusion matrices (per feature threshold classification)
  5. PCA scatter plot
  6. Statistical analysis

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\frame_analysis\\

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
from scipy.stats import ttest_ind, mannwhitneyu
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(['#3498db', '#e74c3c'])  # Blue for class 0, Red for class 1


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\frame_analysis"

# Feature extraction settings
FFT_CUTOFF_FRACTION = 1/6

# Class mapping
CLASS_MAP = {
    'contain_cell': 1,  # Class 1 (Red)
    'no_cell': 0,       # Class 0 (Blue)
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
    """Extract all features from a single grayscale frame."""
    if img is None:
        return None
    
    features = {}
    pixels = img.flatten().astype(np.float64)
    
    # --- Intensity Features ---
    features['intensity_mean'] = np.mean(pixels)
    features['intensity_std'] = np.std(pixels)
    features['intensity_skewness'] = stats.skew(pixels)
    features['intensity_kurtosis'] = stats.kurtosis(pixels)
    features['intensity_entropy'] = stats.entropy(np.histogram(pixels, bins=256)[0] + 1e-10)
    
    # --- Edge Features ---
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    edges = cv2.Canny(img, 50, 150)
    
    features['edge_gradient_mean'] = np.mean(gradient_mag)
    features['edge_gradient_std'] = np.std(gradient_mag)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # --- FFT Features ---
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
    features['fft_total_energy'] = np.log1p(total_energy)
    
    # --- GLCM Texture Features ---
    try:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        img_reduced = (img_norm // 4).astype(np.uint8)
        
        glcm = graycomatrix(img_reduced, distances=[1, 3], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=64, symmetric=True, normed=True)
        
        features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
        features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
        features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    except:
        features['glcm_contrast'] = 0
        features['glcm_homogeneity'] = 0
        features['glcm_energy'] = 0
        features['glcm_correlation'] = 0
    
    return features


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_frame_data(input_dir, max_samples=None):
    """Collect features from all individual frames."""
    import sys
    
    all_data = []
    
    video_folders = sorted([f for f in os.listdir(input_dir) 
                           if os.path.isdir(os.path.join(input_dir, f))])
    
    sample_count = 0
    total_frames = 0
    
    for vi, video_name in enumerate(video_folders):
        video_path = os.path.join(input_dir, video_name)
        print(f"\n  [{vi+1}/{len(video_folders)}] {video_name}", flush=True)
        
        video_frames = 0
        
        for category in ['contain_cell', 'no_cell']:  # Skip uncertain for clean analysis
            if category not in CLASS_MAP:
                continue
                
            category_path = os.path.join(video_path, category)
            if not os.path.exists(category_path):
                continue
            
            lens_folders = sorted([f for f in os.listdir(category_path) 
                                  if os.path.isdir(os.path.join(category_path, f))])
            
            print(f"      {category}: {len(lens_folders)} lenses ... ", end='', flush=True)
            cat_frames = 0
            
            for lens_idx, lens_name in enumerate(lens_folders):
                lens_path = os.path.join(category_path, lens_name)
                frame_files = sorted([f for f in os.listdir(lens_path) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                for frame_idx, frame_file in enumerate(frame_files):
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    features = extract_frame_features(img)
                    if features is None:
                        continue
                    
                    # Add metadata
                    features['video'] = video_name
                    features['lens'] = lens_name
                    features['frame'] = frame_idx + 1
                    features['category'] = category
                    features['class'] = CLASS_MAP[category]
                    
                    all_data.append(features)
                    sample_count += 1
                    cat_frames += 1
                    video_frames += 1
                    
                    if max_samples and sample_count >= max_samples:
                        print(f"{cat_frames} frames", flush=True)
                        print(f"\n  Reached max samples: {max_samples}", flush=True)
                        return all_data
                
                # Show progress every 20 lenses
                if (lens_idx + 1) % 20 == 0:
                    print(f"{lens_idx+1}...", end='', flush=True)
            
            print(f"{cat_frames} frames", flush=True)
        
        total_frames += video_frames
        print(f"      Video total: {video_frames} frames (Running total: {total_frames})", flush=True)
    
    return all_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_scatter_plot_2class(data, feature1, feature2, output_path):
    """Create scatter plot of two features colored by class (like your Image 1)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in [0, 1]:
        mask = [d['class'] == cls for d in data]
        x = [d[feature1] for d, m in zip(data, mask) if m]
        y = [d[feature2] for d, m in zip(data, mask) if m]
        
        ax.scatter(x, y, c=CLASS_COLORS[cls], label=f'class {cls}', 
                  alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel(feature1.replace('_', ' ').title())
    ax.set_ylabel(feature2.replace('_', ' ').title())
    ax.set_title('Scatter plot of data set with two classes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_pca_scatter_plot(data, feature_cols, output_path):
    """Create PCA scatter plot colored by class."""
    X = np.array([[d[f] for f in feature_cols] for d in data])
    y = np.array([d['class'] for d in data])
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=CLASS_COLORS[cls], label=f'class {cls}',
                  alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel(f'feature 1')
    ax.set_ylabel(f'feature 2')
    ax.set_title('Scatter plot of data set with two classes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pca.explained_variance_ratio_


def create_violin_plot(data, feature_cols, output_path):
    """Create violin plot for multiple features (like your Image 2 left)."""
    # Standardize features for comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for violin plot
    plot_data = []
    positions = []
    colors = []
    
    for i, feat in enumerate(feature_cols[:4]):  # Top 4 features
        vals = [d[feat] for d in data]
        
        # Standardize
        vals = np.array(vals)
        vals = (vals - np.mean(vals)) / (np.std(vals) + 1e-10)
        
        plot_data.append(vals)
        positions.append(i + 1)
    
    # Create violin plot
    parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True)
    
    # Color the violins
    colors_list = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'std_columns{i+1}' for i in range(len(feature_cols[:4]))])
    ax.set_ylabel('Standardized Value')
    ax.set_title('Violin Plot of Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_strip_plot(data, feature_cols, output_path):
    """Create strip/swarm plot (like your Image 2 right)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_list = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    
    for i, feat in enumerate(feature_cols[:4]):
        vals = [d[feat] for d in data]
        
        # Standardize
        vals = np.array(vals)
        vals = (vals - np.mean(vals)) / (np.std(vals) + 1e-10)
        
        # Add jitter
        x = np.random.normal(i + 1, 0.1, size=len(vals))
        
        # Subsample for visibility (max 100 points per feature)
        if len(vals) > 100:
            idx = np.random.choice(len(vals), 100, replace=False)
            x = x[idx]
            vals = vals[idx]
        
        ax.scatter(x, vals, c=colors_list[i], alpha=0.6, s=30, edgecolors='none')
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f'std_columns{i+1}' for i in range(4)])
    ax.set_ylabel('Standardized Value')
    ax.set_title('Strip Plot of Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_violin_strip_combined(data, feature_cols, output_path):
    """Create combined violin + strip plot side by side (like your Image 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_list = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    top_features = feature_cols[:4]
    
    # Prepare standardized data
    std_data = []
    for feat in top_features:
        vals = np.array([d[feat] for d in data])
        vals = (vals - np.mean(vals)) / (np.std(vals) + 1e-10)
        std_data.append(vals)
    
    # --- Left: Violin Plot ---
    ax1 = axes[0]
    parts = ax1.violinplot(std_data, positions=[1, 2, 3, 4], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels([f'std_columns{i+1}' for i in range(4)])
    ax1.set_ylabel('Standardized Value')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- Right: Strip Plot ---
    ax2 = axes[1]
    
    for i, vals in enumerate(std_data):
        # Subsample for visibility
        if len(vals) > 100:
            idx = np.random.choice(len(vals), 100, replace=False)
            vals_sub = vals[idx]
        else:
            vals_sub = vals
        
        x = np.random.normal(i + 1, 0.1, size=len(vals_sub))
        ax2.scatter(x, vals_sub, c=colors_list[i], alpha=0.6, s=30, edgecolors='none')
    
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels([f'std_columns{i+1}' for i in range(4)])
    ax2.set_ylabel('Standardized Value')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distribution: Violin Plot (left) vs Strip Plot (right)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_class_violin_plot(data, feature_cols, output_path):
    """Create violin plot comparing classes for each feature."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(feature_cols[:6]):
        ax = axes[idx]
        
        # Get values per class
        vals_0 = [d[feat] for d in data if d['class'] == 0]
        vals_1 = [d[feat] for d in data if d['class'] == 1]
        
        # Create violin plot
        parts = ax.violinplot([vals_0, vals_1], positions=[1, 2], showmeans=True, showmedians=True)
        
        # Color by class
        parts['bodies'][0].set_facecolor(CLASS_COLORS[0])
        parts['bodies'][1].set_facecolor(CLASS_COLORS[1])
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_title(feat.replace('_', ' ').title()[:25], fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distributions by Class (Violin Plots)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_class_strip_plot(data, feature_cols, output_path):
    """Create strip plot comparing classes for each feature."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(feature_cols[:6]):
        ax = axes[idx]
        
        for cls in [0, 1]:
            vals = [d[feat] for d in data if d['class'] == cls]
            
            # Subsample
            if len(vals) > 200:
                vals = np.random.choice(vals, 200, replace=False)
            
            x = np.random.normal(cls + 1, 0.15, size=len(vals))
            ax.scatter(x, vals, c=CLASS_COLORS[cls], alpha=0.5, s=20, 
                      edgecolors='none', label=CLASS_NAMES[cls] if idx == 0 else '')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_title(feat.replace('_', ' ').title()[:25], fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend to first subplot
    axes[0].legend(loc='upper right')
    
    plt.suptitle('Feature Distributions by Class (Strip Plots)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_confusion_matrices(data, feature_cols, output_path):
    """Create confusion matrices for top features."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    results = []
    
    for feat in feature_cols:
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
        results.append({
            'feature': feat,
            'accuracy': best_acc,
            'threshold': best_thresh,
            'direction': best_dir,
            'cm': cm
        })
    
    # Sort by accuracy and plot top 8
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:8]
    
    for idx, res in enumerate(results):
        ax = axes[idx]
        cm = res['cm']
        
        # Plot
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_yticklabels(['No Cell', 'Cell'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Add text
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        feat_short = res['feature'][:15] + '...' if len(res['feature']) > 15 else res['feature']
        ax.set_title(f"{feat_short}\nAcc: {res['accuracy']:.1%}", fontsize=9, fontweight='bold')
    
    plt.suptitle('Confusion Matrices: Frame-Level Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def create_feature_histograms(data, feature_cols, output_path):
    """Create histogram comparison for top features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(feature_cols[:6]):
        ax = axes[idx]
        
        vals_0 = [d[feat] for d in data if d['class'] == 0]
        vals_1 = [d[feat] for d in data if d['class'] == 1]
        
        ax.hist(vals_0, bins=50, alpha=0.6, color=CLASS_COLORS[0], 
               label='No Cell', edgecolor='black', linewidth=0.5)
        ax.hist(vals_1, bins=50, alpha=0.6, color=CLASS_COLORS[1], 
               label='Cell', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(feat.replace('_', ' ').title()[:25])
        ax.set_ylabel('Count')
        ax.set_title(feat[:30], fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions: Cell vs No Cell (Histograms)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_frame_statistics(data, feature_cols):
    """Compute statistical tests for frame-level data."""
    results = []
    
    for feat in feature_cols:
        vals_0 = [d[feat] for d in data if d['class'] == 0]
        vals_1 = [d[feat] for d in data if d['class'] == 1]
        
        if len(vals_0) < 3 or len(vals_1) < 3:
            continue
        
        # T-test
        t_stat, t_pval = ttest_ind(vals_0, vals_1)
        
        # Effect size
        pooled_std = np.sqrt(((len(vals_0)-1)*np.var(vals_0) + (len(vals_1)-1)*np.var(vals_1)) / 
                            (len(vals_0) + len(vals_1) - 2))
        cohens_d = (np.mean(vals_1) - np.mean(vals_0)) / (pooled_std + 1e-10)
        
        results.append({
            'feature': feat,
            'nocell_mean': np.mean(vals_0),
            'nocell_std': np.std(vals_0),
            'cell_mean': np.mean(vals_1),
            'cell_std': np.std(vals_1),
            'ttest_pval': t_pval,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
        })
    
    return sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    
    print("=" * 80, flush=True)
    print("  FRAME-BY-FRAME FEATURE ANALYSIS", flush=True)
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
    
    # ─── Step 1: Collect frame-level data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting features from individual frames...", flush=True)
    print("  (This may take 10-20 minutes for ~135,000 frames)", flush=True)
    print("-" * 80, flush=True)
    
    data = collect_frame_data(INPUT_DIR)
    
    print(f"\n  Total frames analyzed: {len(data)}", flush=True)
    
    cell_count = sum(1 for d in data if d['class'] == 1)
    nocell_count = sum(1 for d in data if d['class'] == 0)
    print(f"  Cell frames: {cell_count}", flush=True)
    print(f"  No Cell frames: {nocell_count}", flush=True)
    
    # Get feature columns
    feature_cols = [k for k in data[0].keys() 
                   if k not in ['video', 'lens', 'frame', 'category', 'class']]
    print(f"  Features extracted: {len(feature_cols)}", flush=True)
    
    # ─── Step 2: Statistical analysis ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 2: Computing statistics...", flush=True)
    print("-" * 80, flush=True)
    
    stats_results = compute_frame_statistics(data, feature_cols)
    
    print(f"  Top 3 features by effect size:", flush=True)
    for i, r in enumerate(stats_results[:3]):
        print(f"    {i+1}. {r['feature']}: Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']})", flush=True)
    
    # Sort features by effect size for plotting
    sorted_features = [r['feature'] for r in stats_results]
    
    # ─── Step 3: Create visualizations ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 3: Creating visualizations...", flush=True)
    print("-" * 80, flush=True)
    
    # 1. PCA Scatter Plot (like Image 1)
    print("    Creating PCA scatter plot...", flush=True)
    create_pca_scatter_plot(data, feature_cols, os.path.join(visuals_dir, 'scatter_pca_2class.png'))
    
    # 2. Two-feature scatter plot
    print("    Creating feature scatter plot...", flush=True)
    if len(sorted_features) >= 2:
        create_scatter_plot_2class(data, sorted_features[0], sorted_features[1], 
                                   os.path.join(visuals_dir, 'scatter_top2_features.png'))
    
    # 3. Violin + Strip combined (like Image 2)
    print("    Creating violin + strip combined plot...", flush=True)
    create_violin_strip_combined(data, sorted_features, os.path.join(visuals_dir, 'violin_strip_combined.png'))
    
    # 4. Class-separated violin plot
    print("    Creating class violin plots...", flush=True)
    create_class_violin_plot(data, sorted_features, os.path.join(visuals_dir, 'violin_by_class.png'))
    
    # 5. Class-separated strip plot
    print("    Creating class strip plots...", flush=True)
    create_class_strip_plot(data, sorted_features, os.path.join(visuals_dir, 'strip_by_class.png'))
    
    # 6. Histograms
    print("    Creating histograms...", flush=True)
    create_feature_histograms(data, sorted_features, os.path.join(visuals_dir, 'histograms_by_class.png'))
    
    # 7. Confusion matrices
    print("    Creating confusion matrices...", flush=True)
    cm_results = create_confusion_matrices(data, feature_cols, os.path.join(visuals_dir, 'confusion_matrices.png'))
    
    # ─── Step 4: Save data ───
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data...", flush=True)
    print("-" * 80, flush=True)
    
    # Save frame-level features
    csv_path = os.path.join(data_dir, 'frame_features.csv')
    fieldnames = ['video', 'lens', 'frame', 'category', 'class'] + feature_cols
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow({k: d.get(k, '') for k in fieldnames})
    print(f"    Saved: {csv_path}", flush=True)
    
    # Save statistics
    stats_csv = os.path.join(data_dir, 'frame_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_results[0].keys()))
        writer.writeheader()
        writer.writerows(stats_results)
    print(f"    Saved: {stats_csv}", flush=True)
    
    # ─── Summary ───
    print("\n" + "=" * 80, flush=True)
    print("  FRAME-LEVEL ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  Total frames: {len(data)}", flush=True)
    print(f"  Cell: {cell_count} | No Cell: {nocell_count}", flush=True)
    print(f"\n  TOP FINDINGS (Frame-Level):", flush=True)
    print(f"    Best feature: {stats_results[0]['feature']}", flush=True)
    print(f"      Effect size: {stats_results[0]['cohens_d']:.3f} ({stats_results[0]['effect_size']})", flush=True)
    print(f"      P-value: {stats_results[0]['ttest_pval']:.2e}", flush=True)
    print(f"\n    Best classification accuracy: {cm_results[0]['accuracy']:.1%}", flush=True)
    print(f"      Feature: {cm_results[0]['feature']}", flush=True)
    print(f"\n  OUTPUTS:", flush=True)
    print(f"    Visuals: {visuals_dir}", flush=True)
    print(f"    Data:    {data_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()





    


