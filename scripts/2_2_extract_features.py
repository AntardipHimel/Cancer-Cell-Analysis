# -*- coding: utf-8 -*-
"""
2_2_feature_analysis_toolkit.py

COMPREHENSIVE FEATURE ANALYSIS TOOLKIT
For lensless microscopy cancer cell classification

This toolkit provides:
  1. Per-video analysis (separate reports for each video)
  2. Combined analysis (all videos pooled)
  3. Cross-video comparison (feature consistency across videos)
  4. Statistical tests (p-values, effect sizes)
  5. Feature ranking (which features best separate classes)
  6. Threshold suggestions (optimal cutoffs)
  7. Confusion matrix preview (accuracy with simple thresholds)
  8. HTML dashboard report (easy navigation)

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\feature_analysis\\
        - visuals/           (all plots)
        - reports/           (HTML reports)
        - data/              (CSV exports)

Author: Antardip Himel
Date: February 2026
"""

import os
import cv2
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\feature_analysis"

# FFT frequency cutoff
FFT_CUTOFF_FRACTION = 1/6

# Colors for classes
CLASS_COLORS = {
    'contain_cell': '#27ae60',
    'no_cell': '#e74c3c',
    'uncertain_cell': '#f39c12'
}

CLASS_LABELS = {
    'contain_cell': 'Cell',
    'no_cell': 'No Cell', 
    'uncertain_cell': 'Uncertain'
}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def create_circular_mask(h, w):
    """Create circular mask for FFT analysis."""
    center = (w / 2, h / 2)
    radius = min(h, w) / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist <= radius


def compute_fft_features(gray_img):
    """Compute FFT-based features."""
    h, w = gray_img.shape
    mask = create_circular_mask(h, w)
    masked_img = gray_img.astype(np.float64).copy()
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
    
    hf_ratio = hf_energy / total_energy if total_energy > 0 else 0
    
    return {
        'fft_hf_ratio': hf_ratio,
        'fft_total_energy': total_energy,
        'power_spectrum': np.log1p(P)
    }


def compute_glcm_features(gray_img):
    """Compute GLCM texture features."""
    img_norm = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-10) * 255).astype(np.uint8)
    img_reduced = (img_norm // 4).astype(np.uint8)
    
    try:
        glcm = graycomatrix(img_reduced, distances=[1, 3], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=64, symmetric=True, normed=True)
        
        features = {}
        for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
            features[f'glcm_{prop}'] = np.mean(graycoprops(glcm, prop))
        return features
    except:
        return {f'glcm_{p}': 0 for p in ['contrast', 'homogeneity', 'energy', 'correlation']}


def compute_intensity_features(gray_img):
    """Compute intensity statistics."""
    pixels = gray_img.flatten().astype(np.float64)
    
    return {
        'intensity_mean': np.mean(pixels),
        'intensity_std': np.std(pixels),
        'intensity_skewness': stats.skew(pixels),
        'intensity_kurtosis': stats.kurtosis(pixels),
        'intensity_entropy': stats.entropy(np.histogram(pixels, bins=256)[0] + 1e-10),
    }


def compute_edge_features(gray_img):
    """Compute edge/gradient features."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    edges = cv2.Canny(gray_img, 50, 150)
    
    return {
        'edge_gradient_mean': np.mean(gradient_mag),
        'edge_gradient_std': np.std(gradient_mag),
        'edge_density': np.sum(edges > 0) / edges.size,
    }


def extract_lens_features(lens_path):
    """Extract all features from a lens folder (all 30 frames)."""
    frame_files = sorted([f for f in os.listdir(lens_path) if f.endswith(('.png', '.jpg'))])
    if not frame_files:
        return None
    
    all_frame_features = []
    
    for fname in frame_files:
        img = cv2.imread(os.path.join(lens_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        features = {}
        features.update(compute_intensity_features(img))
        features.update(compute_glcm_features(img))
        features.update(compute_edge_features(img))
        fft_result = compute_fft_features(img)
        features['fft_hf_ratio'] = fft_result['fft_hf_ratio']
        features['fft_total_energy'] = fft_result['fft_total_energy']
        
        all_frame_features.append(features)
    
    if not all_frame_features:
        return None
    
    # Aggregate: mean and std across frames
    aggregated = {}
    for key in all_frame_features[0].keys():
        values = [f[key] for f in all_frame_features]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    # Temporal features
    hf_ratios = [f['fft_hf_ratio'] for f in all_frame_features]
    aggregated['temporal_hf_stability'] = np.std(hf_ratios) / (np.mean(hf_ratios) + 1e-10)
    aggregated['temporal_hf_trend'] = hf_ratios[-1] - hf_ratios[0] if len(hf_ratios) > 1 else 0
    
    return aggregated


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistical_tests(data, feature_name):
    """Compute statistical tests between classes for a feature."""
    cell_vals = [d[feature_name] for d in data if d['category'] == 'contain_cell' and feature_name in d]
    nocell_vals = [d[feature_name] for d in data if d['category'] == 'no_cell' and feature_name in d]
    
    results = {
        'feature': feature_name,
        'cell_mean': np.mean(cell_vals) if cell_vals else 0,
        'cell_std': np.std(cell_vals) if cell_vals else 0,
        'cell_n': len(cell_vals),
        'nocell_mean': np.mean(nocell_vals) if nocell_vals else 0,
        'nocell_std': np.std(nocell_vals) if nocell_vals else 0,
        'nocell_n': len(nocell_vals),
    }
    
    if len(cell_vals) >= 3 and len(nocell_vals) >= 3:
        # T-test
        t_stat, t_pval = ttest_ind(cell_vals, nocell_vals)
        results['ttest_pval'] = t_pval
        results['ttest_significant'] = t_pval < 0.05
        
        # Mann-Whitney U (non-parametric)
        u_stat, u_pval = mannwhitneyu(cell_vals, nocell_vals, alternative='two-sided')
        results['mannwhitney_pval'] = u_pval
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(cell_vals)-1)*np.var(cell_vals) + (len(nocell_vals)-1)*np.var(nocell_vals)) / 
                            (len(cell_vals) + len(nocell_vals) - 2))
        cohens_d = (np.mean(cell_vals) - np.mean(nocell_vals)) / (pooled_std + 1e-10)
        results['cohens_d'] = cohens_d
        results['effect_size'] = 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
        
        # Separation score (0-1, higher = better separation)
        overlap = compute_distribution_overlap(cell_vals, nocell_vals)
        results['separation_score'] = 1 - overlap
    else:
        results['ttest_pval'] = 1.0
        results['ttest_significant'] = False
        results['mannwhitney_pval'] = 1.0
        results['cohens_d'] = 0
        results['effect_size'] = 'insufficient_data'
        results['separation_score'] = 0
    
    return results


def compute_distribution_overlap(vals1, vals2):
    """Compute overlap between two distributions (0 = no overlap, 1 = complete overlap)."""
    if not vals1 or not vals2:
        return 1.0
    
    min_val = min(min(vals1), min(vals2))
    max_val = max(max(vals1), max(vals2))
    
    bins = np.linspace(min_val, max_val, 50)
    hist1, _ = np.histogram(vals1, bins=bins, density=True)
    hist2, _ = np.histogram(vals2, bins=bins, density=True)
    
    # Overlap coefficient
    overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
    return min(overlap, 1.0)


def find_optimal_threshold(data, feature_name):
    """Find optimal threshold to separate classes."""
    cell_vals = [d[feature_name] for d in data if d['category'] == 'contain_cell' and feature_name in d]
    nocell_vals = [d[feature_name] for d in data if d['category'] == 'no_cell' and feature_name in d]
    
    if not cell_vals or not nocell_vals:
        return None
    
    all_vals = cell_vals + nocell_vals
    all_labels = [1] * len(cell_vals) + [0] * len(nocell_vals)
    
    # Try different thresholds
    min_val, max_val = min(all_vals), max(all_vals)
    thresholds = np.linspace(min_val, max_val, 100)
    
    best_threshold = None
    best_accuracy = 0
    best_direction = 'greater'  # cell > threshold
    
    for thresh in thresholds:
        # Try: cell > threshold
        pred_greater = [1 if v > thresh else 0 for v in all_vals]
        acc_greater = accuracy_score(all_labels, pred_greater)
        
        # Try: cell < threshold
        pred_less = [1 if v < thresh else 0 for v in all_vals]
        acc_less = accuracy_score(all_labels, pred_less)
        
        if acc_greater > best_accuracy:
            best_accuracy = acc_greater
            best_threshold = thresh
            best_direction = 'greater'
        
        if acc_less > best_accuracy:
            best_accuracy = acc_less
            best_threshold = thresh
            best_direction = 'less'
    
    # Compute confusion matrix at best threshold
    if best_direction == 'greater':
        predictions = [1 if v > best_threshold else 0 for v in all_vals]
    else:
        predictions = [1 if v < best_threshold else 0 for v in all_vals]
    
    cm = confusion_matrix(all_labels, predictions)
    
    return {
        'threshold': best_threshold,
        'direction': best_direction,
        'accuracy': best_accuracy,
        'confusion_matrix': cm,
        'rule': f"Cell if {feature_name} {'>' if best_direction == 'greater' else '<'} {best_threshold:.4f}"
    }


def compute_feature_importance(data, features_list):
    """Compute feature importance using Random Forest."""
    X = []
    y = []
    
    for d in data:
        if d['category'] in ['contain_cell', 'no_cell']:
            row = [d.get(f, 0) for f in features_list]
            X.append(row)
            y.append(1 if d['category'] == 'contain_cell' else 0)
    
    if len(X) < 10:
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X, y, cv=min(5, len(y)//2))
    
    importance = dict(zip(features_list, rf.feature_importances_))
    
    return {
        'importance': importance,
        'cv_accuracy_mean': np.mean(cv_scores),
        'cv_accuracy_std': np.std(cv_scores)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_feature_ranking_plot(stats_results, output_path):
    """Create feature ranking visualization."""
    # Sort by separation score
    sorted_results = sorted(stats_results, key=lambda x: x.get('separation_score', 0), reverse=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Plot 1: Separation Score
    ax1 = axes[0]
    features = [r['feature'][:25] for r in sorted_results[:15]]
    scores = [r.get('separation_score', 0) for r in sorted_results[:15]]
    colors = ['#27ae60' if s > 0.5 else '#f39c12' if s > 0.3 else '#e74c3c' for s in scores]
    
    bars = ax1.barh(range(len(features)), scores, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Separation Score (0-1)')
    ax1.set_title('Feature Ranking by Class Separation', fontweight='bold')
    ax1.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='Good threshold')
    ax1.axvline(0.3, color='orange', linestyle='--', alpha=0.7, label='Fair threshold')
    ax1.legend(loc='lower right')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    
    # Plot 2: P-values
    ax2 = axes[1]
    pvals = [-np.log10(r.get('ttest_pval', 1) + 1e-10) for r in sorted_results[:15]]
    colors2 = ['#27ae60' if p > -np.log10(0.05) else '#e74c3c' for p in pvals]
    
    ax2.barh(range(len(features)), pvals, color=colors2, edgecolor='black')
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features)
    ax2.set_xlabel('-log10(p-value)')
    ax2.set_title('Statistical Significance (T-test)', fontweight='bold')
    ax2.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax2.axvline(-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01')
    ax2.legend(loc='lower right')
    ax2.invert_yaxis()
    
    # Plot 3: Effect Size
    ax3 = axes[2]
    effect_sizes = [abs(r.get('cohens_d', 0)) for r in sorted_results[:15]]
    colors3 = ['#27ae60' if e > 0.8 else '#f39c12' if e > 0.5 else '#e74c3c' for e in effect_sizes]
    
    ax3.barh(range(len(features)), effect_sizes, color=colors3, edgecolor='black')
    ax3.set_yticks(range(len(features)))
    ax3.set_yticklabels(features)
    ax3.set_xlabel("Cohen's d (Effect Size)")
    ax3.set_title('Effect Size Magnitude', fontweight='bold')
    ax3.axvline(0.8, color='green', linestyle='--', label='Large (0.8)')
    ax3.axvline(0.5, color='orange', linestyle='--', label='Medium (0.5)')
    ax3.legend(loc='lower right')
    ax3.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_threshold_analysis_plot(threshold_results, output_path):
    """Create threshold analysis visualization."""
    # Sort by accuracy
    sorted_results = sorted(threshold_results, key=lambda x: x['accuracy'], reverse=True)[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, result in enumerate(sorted_results):
        ax = axes[idx]
        cm = result['confusion_matrix']
        
        # Plot confusion matrix
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_yticklabels(['No Cell', 'Cell'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                       fontsize=14, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        feature_short = result['feature'][:20] + '...' if len(result['feature']) > 20 else result['feature']
        ax.set_title(f"{feature_short}\nAcc: {result['accuracy']:.1%}", fontsize=9, fontweight='bold')
    
    plt.suptitle('Top 10 Single-Feature Classification Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_class_distribution_plot(data, features_list, output_path):
    """Create distribution comparison for top features."""
    # Get top 6 features by separation
    top_features = features_list[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        for cat in ['contain_cell', 'no_cell']:
            vals = [d[feature] for d in data if d['category'] == cat and feature in d]
            if vals:
                ax.hist(vals, bins=25, alpha=0.6, label=CLASS_LABELS[cat], 
                       color=CLASS_COLORS[cat], edgecolor='black')
        
        ax.set_xlabel(feature.replace('_', ' ').title()[:30])
        ax.set_ylabel('Count')
        ax.set_title(feature[:35], fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions: Cell vs No Cell', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_cross_video_comparison(data, feature_name, output_path):
    """Create cross-video comparison for a feature."""
    # Group by video
    video_data = defaultdict(lambda: {'contain_cell': [], 'no_cell': []})
    
    for d in data:
        if d['category'] in ['contain_cell', 'no_cell'] and feature_name in d:
            video_data[d['video']][d['category']].append(d[feature_name])
    
    videos = sorted(video_data.keys())
    
    if len(videos) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(videos)*0.8), 6))
    
    x = np.arange(len(videos))
    width = 0.35
    
    cell_means = [np.mean(video_data[v]['contain_cell']) if video_data[v]['contain_cell'] else 0 for v in videos]
    cell_stds = [np.std(video_data[v]['contain_cell']) if video_data[v]['contain_cell'] else 0 for v in videos]
    nocell_means = [np.mean(video_data[v]['no_cell']) if video_data[v]['no_cell'] else 0 for v in videos]
    nocell_stds = [np.std(video_data[v]['no_cell']) if video_data[v]['no_cell'] else 0 for v in videos]
    
    bars1 = ax.bar(x - width/2, cell_means, width, yerr=cell_stds, label='Cell', 
                   color=CLASS_COLORS['contain_cell'], capsize=3, edgecolor='black')
    bars2 = ax.bar(x + width/2, nocell_means, width, yerr=nocell_stds, label='No Cell',
                   color=CLASS_COLORS['no_cell'], capsize=3, edgecolor='black')
    
    ax.set_xlabel('Video')
    ax.set_ylabel(feature_name.replace('_', ' ').title())
    ax.set_title(f'Cross-Video Comparison: {feature_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([v[:15] + '...' if len(v) > 15 else v for v in videos], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_pca_plot(data, features_list, output_path):
    """Create PCA visualization."""
    X = []
    y = []
    
    for d in data:
        if d['category'] in ['contain_cell', 'no_cell']:
            row = [d.get(f, 0) for f in features_list]
            X.append(row)
            y.append(d['category'])
    
    if len(X) < 10:
        return
    
    X = np.array(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cat in ['contain_cell', 'no_cell']:
        mask = [yi == cat for yi in y]
        ax.scatter(X_pca[np.array(mask), 0], X_pca[np.array(mask), 1],
                  c=CLASS_COLORS[cat], label=CLASS_LABELS[cat], alpha=0.7, s=60, edgecolors='black')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Projection of All Features', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pca.explained_variance_ratio_


def create_feature_importance_plot(importance_dict, output_path):
    """Create feature importance visualization."""
    if importance_dict is None:
        return
    
    importance = importance_dict['importance']
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = [x[0][:30] for x in sorted_imp]
    scores = [x[1] for x in sorted_imp]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]
    
    bars = ax.barh(range(len(features)), scores, color=colors, edgecolor='black')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Random Forest Feature Importance\n(CV Accuracy: {importance_dict["cv_accuracy_mean"]:.1%} +/- {importance_dict["cv_accuracy_std"]:.1%})',
                fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(stats_results, threshold_results, importance_result, 
                        per_video_stats, output_dir, total_data):
    """Generate comprehensive HTML report."""
    
    html_path = os.path.join(output_dir, 'reports', 'analysis_report.html')
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    # Count samples
    cell_count = sum(1 for d in total_data if d['category'] == 'contain_cell')
    nocell_count = sum(1 for d in total_data if d['category'] == 'no_cell')
    uncertain_count = sum(1 for d in total_data if d['category'] == 'uncertain_cell')
    video_count = len(set(d['video'] for d in total_data))
    
    # Sort results
    sorted_stats = sorted(stats_results, key=lambda x: x.get('separation_score', 0), reverse=True)
    sorted_thresh = sorted(threshold_results, key=lambda x: x['accuracy'], reverse=True)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Analysis Report - Cancer Cell Classification</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .stat-card h3 {{ color: #667eea; font-size: 2em; }}
        .stat-card p {{ color: #7f8c8d; font-size: 0.9em; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .poor {{ color: #e74c3c; font-weight: bold; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .image-card {{ text-align: center; }}
        .image-card img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .image-card p {{ margin-top: 10px; font-weight: bold; color: #7f8c8d; }}
        nav {{ background: white; padding: 15px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        nav a {{ color: #667eea; text-decoration: none; margin-right: 20px; font-weight: bold; }}
        nav a:hover {{ text-decoration: underline; }}
        .highlight {{ background: #fff9c4; padding: 2px 6px; border-radius: 3px; }}
        .recommendation {{ background: #e8f5e9; border-left: 4px solid #27ae60; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .warning {{ background: #fff3e0; border-left: 4px solid #f39c12; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Feature Analysis Report</h1>
            <p>Cancer Cell Classification - Lensless Microscopy</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <nav>
            <a href="#overview">Overview</a>
            <a href="#ranking">Feature Ranking</a>
            <a href="#thresholds">Thresholds</a>
            <a href="#importance">ML Importance</a>
            <a href="#visuals">Visualizations</a>
            <a href="#recommendations">Recommendations</a>
        </nav>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{cell_count}</h3>
                <p>Cell Samples</p>
            </div>
            <div class="stat-card">
                <h3>{nocell_count}</h3>
                <p>No Cell Samples</p>
            </div>
            <div class="stat-card">
                <h3>{uncertain_count}</h3>
                <p>Uncertain Samples</p>
            </div>
            <div class="stat-card">
                <h3>{video_count}</h3>
                <p>Videos Analyzed</p>
            </div>
        </div>
        
        <section class="section" id="overview">
            <h2>Dataset Overview</h2>
            <p>This report analyzes <strong>{len(total_data)}</strong> labeled lens samples across <strong>{video_count}</strong> videos.</p>
            <p>The goal is to identify which features best distinguish <span class="good">cells</span> from <span class="poor">no cells</span>.</p>
        </section>
        
        <section class="section" id="ranking">
            <h2>Feature Ranking (Top 15)</h2>
            <p>Features ranked by their ability to separate Cell vs No Cell classes.</p>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Separation Score</th>
                    <th>P-value</th>
                    <th>Effect Size</th>
                    <th>Cell Mean</th>
                    <th>No Cell Mean</th>
                </tr>
    """
    
    for i, r in enumerate(sorted_stats[:15]):
        sep_class = 'good' if r.get('separation_score', 0) > 0.5 else ('medium' if r.get('separation_score', 0) > 0.3 else 'poor')
        pval = r.get('ttest_pval', 1)
        pval_class = 'good' if pval < 0.01 else ('medium' if pval < 0.05 else 'poor')
        effect_class = 'good' if r.get('effect_size') == 'large' else ('medium' if r.get('effect_size') == 'medium' else 'poor')
        
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{r['feature']}</td>
                    <td class="{sep_class}">{r.get('separation_score', 0):.3f}</td>
                    <td class="{pval_class}">{pval:.2e}</td>
                    <td class="{effect_class}">{r.get('effect_size', 'N/A')}</td>
                    <td>{r.get('cell_mean', 0):.4f}</td>
                    <td>{r.get('nocell_mean', 0):.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </section>
        
        <section class="section" id="thresholds">
            <h2>Single-Feature Classification Thresholds</h2>
            <p>Optimal thresholds for classifying with a single feature.</p>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Threshold</th>
                    <th>Rule</th>
                    <th>Accuracy</th>
                </tr>
    """
    
    for i, r in enumerate(sorted_thresh[:10]):
        acc_class = 'good' if r['accuracy'] > 0.8 else ('medium' if r['accuracy'] > 0.7 else 'poor')
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{r['feature'][:40]}</td>
                    <td>{r['threshold']:.4f}</td>
                    <td>{r['rule'][:50]}</td>
                    <td class="{acc_class}">{r['accuracy']:.1%}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </section>
    """
    
    if importance_result:
        html_content += f"""
        <section class="section" id="importance">
            <h2>Machine Learning Feature Importance</h2>
            <p>Random Forest classifier with cross-validation accuracy: <strong class="good">{importance_result['cv_accuracy_mean']:.1%}</strong> (+/- {importance_result['cv_accuracy_std']:.1%})</p>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance Score</th>
                </tr>
        """
        
        sorted_imp = sorted(importance_result['importance'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feat, score) in enumerate(sorted_imp):
            html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{feat}</td>
                    <td>{score:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </section>
        """
    
    html_content += """
        <section class="section" id="visuals">
            <h2>Visualizations</h2>
            <div class="image-grid">
                <div class="image-card">
                    <img src="../visuals/feature_ranking.png" alt="Feature Ranking">
                    <p>Feature Ranking by Separation, P-value, Effect Size</p>
                </div>
                <div class="image-card">
                    <img src="../visuals/threshold_analysis.png" alt="Threshold Analysis">
                    <p>Confusion Matrices for Top Features</p>
                </div>
                <div class="image-card">
                    <img src="../visuals/class_distributions.png" alt="Class Distributions">
                    <p>Feature Distributions by Class</p>
                </div>
                <div class="image-card">
                    <img src="../visuals/pca_projection.png" alt="PCA Projection">
                    <p>PCA Projection of All Features</p>
                </div>
                <div class="image-card">
                    <img src="../visuals/feature_importance.png" alt="Feature Importance">
                    <p>Random Forest Feature Importance</p>
                </div>
                <div class="image-card">
                    <img src="../visuals/cross_video_hf_ratio.png" alt="Cross-Video Comparison">
                    <p>HF Ratio Across Videos</p>
                </div>
            </div>
        </section>
        
        <section class="section" id="recommendations">
            <h2>Recommendations</h2>
    """
    
    # Add recommendations based on results
    top_feature = sorted_stats[0] if sorted_stats else None
    top_thresh = sorted_thresh[0] if sorted_thresh else None
    
    if top_feature and top_feature.get('separation_score', 0) > 0.5:
        html_content += f"""
            <div class="recommendation">
                <strong>Best Feature:</strong> <span class="highlight">{top_feature['feature']}</span> shows strong separation (score: {top_feature.get('separation_score', 0):.3f}).
                This is a good candidate for classification.
            </div>
        """
    
    if top_thresh and top_thresh['accuracy'] > 0.75:
        html_content += f"""
            <div class="recommendation">
                <strong>Simple Threshold Rule:</strong> Using {top_thresh['rule']} achieves <span class="highlight">{top_thresh['accuracy']:.1%}</span> accuracy.
                This could be a quick baseline before training a CNN.
            </div>
        """
    
    if importance_result and importance_result['cv_accuracy_mean'] > 0.8:
        html_content += f"""
            <div class="recommendation">
                <strong>ML Baseline:</strong> A Random Forest classifier achieves <span class="highlight">{importance_result['cv_accuracy_mean']:.1%}</span> cross-validation accuracy.
                This suggests the features are discriminative and a CNN should perform even better.
            </div>
        """
    
    html_content += """
            <div class="warning">
                <strong>Note:</strong> These results are based on your labeled data. Continue labeling more samples to improve statistical reliability.
            </div>
        </section>
        
        <footer style="text-align: center; padding: 20px; color: #7f8c8d;">
            <p>Generated by Feature Analysis Toolkit | Cancer Cell Classification Pipeline</p>
        </footer>
    </div>
</body>
</html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  COMPREHENSIVE FEATURE ANALYSIS TOOLKIT")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)
    
    # Create output directories
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    reports_dir = os.path.join(OUTPUT_DIR, 'reports')
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    
    for d in [visuals_dir, reports_dir, data_dir]:
        os.makedirs(d, exist_ok=True)
    
    print("\n  Input:  " + INPUT_DIR)
    print("  Output: " + OUTPUT_DIR)
    
    if not os.path.exists(INPUT_DIR):
        print("\nERROR: Input directory not found!")
        return
    
    # ─── Step 1: Extract features from all labeled data ───
    print("\n" + "-" * 80)
    print("  STEP 1: Extracting features from labeled data...")
    print("-" * 80)
    
    all_data = []
    video_folders = sorted([f for f in os.listdir(INPUT_DIR) 
                           if os.path.isdir(os.path.join(INPUT_DIR, f))])
    
    for vi, video_name in enumerate(video_folders):
        video_path = os.path.join(INPUT_DIR, video_name)
        print(f"\n  [{vi+1}/{len(video_folders)}] {video_name}")
        
        for category in ['contain_cell', 'no_cell', 'uncertain_cell']:
            category_path = os.path.join(video_path, category)
            if not os.path.exists(category_path):
                continue
            
            lens_folders = [f for f in os.listdir(category_path) 
                          if os.path.isdir(os.path.join(category_path, f))]
            
            print(f"      {category}: {len(lens_folders)} lenses")
            
            for lens_name in lens_folders:
                lens_path = os.path.join(category_path, lens_name)
                features = extract_lens_features(lens_path)
                
                if features:
                    features['video'] = video_name
                    features['lens'] = lens_name
                    features['category'] = category
                    all_data.append(features)
    
    print(f"\n  Total samples extracted: {len(all_data)}")
    
    if len(all_data) < 10:
        print("\nERROR: Not enough labeled data. Please label more samples.")
        return
    
    # Get feature list
    feature_cols = [k for k in all_data[0].keys() if k not in ['video', 'lens', 'category']]
    
    # ─── Step 2: Statistical analysis ───
    print("\n" + "-" * 80)
    print("  STEP 2: Computing statistical tests...")
    print("-" * 80)
    
    stats_results = []
    threshold_results = []
    
    for feat in feature_cols:
        stat_result = compute_statistical_tests(all_data, feat)
        stats_results.append(stat_result)
        
        thresh_result = find_optimal_threshold(all_data, feat)
        if thresh_result:
            thresh_result['feature'] = feat
            threshold_results.append(thresh_result)
    
    # Sort by separation score
    stats_results = sorted(stats_results, key=lambda x: x.get('separation_score', 0), reverse=True)
    threshold_results = sorted(threshold_results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"  Analyzed {len(feature_cols)} features")
    print(f"  Top feature by separation: {stats_results[0]['feature']} (score: {stats_results[0].get('separation_score', 0):.3f})")
    print(f"  Top feature by threshold accuracy: {threshold_results[0]['feature']} ({threshold_results[0]['accuracy']:.1%})")
    
    # ─── Step 3: ML feature importance ───
    print("\n" + "-" * 80)
    print("  STEP 3: Computing ML feature importance...")
    print("-" * 80)
    
    importance_result = compute_feature_importance(all_data, feature_cols)
    
    if importance_result:
        print(f"  Random Forest CV Accuracy: {importance_result['cv_accuracy_mean']:.1%} (+/- {importance_result['cv_accuracy_std']:.1%})")
        top_imp = sorted(importance_result['importance'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3 important features: {', '.join([f[0] for f in top_imp])}")
    
    # ─── Step 4: Create visualizations ───
    print("\n" + "-" * 80)
    print("  STEP 4: Creating visualizations...")
    print("-" * 80)
    
    # Feature ranking plot
    print("    Creating feature ranking plot...")
    create_feature_ranking_plot(stats_results, os.path.join(visuals_dir, 'feature_ranking.png'))
    
    # Threshold analysis plot
    print("    Creating threshold analysis plot...")
    create_threshold_analysis_plot(threshold_results, os.path.join(visuals_dir, 'threshold_analysis.png'))
    
    # Class distribution plot
    print("    Creating class distribution plot...")
    top_features = [r['feature'] for r in stats_results[:6]]
    create_class_distribution_plot(all_data, top_features, os.path.join(visuals_dir, 'class_distributions.png'))
    
    # PCA plot
    print("    Creating PCA plot...")
    create_pca_plot(all_data, feature_cols, os.path.join(visuals_dir, 'pca_projection.png'))
    
    # Feature importance plot
    print("    Creating feature importance plot...")
    create_feature_importance_plot(importance_result, os.path.join(visuals_dir, 'feature_importance.png'))
    
    # Cross-video comparison
    print("    Creating cross-video comparison...")
    create_cross_video_comparison(all_data, 'fft_hf_ratio_mean', os.path.join(visuals_dir, 'cross_video_hf_ratio.png'))
    
    # ─── Step 5: Save data exports ───
    print("\n" + "-" * 80)
    print("  STEP 5: Saving data exports...")
    print("-" * 80)
    
    # Save all features CSV
    csv_path = os.path.join(data_dir, 'all_features.csv')
    fieldnames = ['video', 'lens', 'category'] + feature_cols
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in all_data:
            writer.writerow({k: d.get(k, '') for k in fieldnames})
    print(f"    Saved: {csv_path}")
    
    # Save stats results CSV
    stats_csv = os.path.join(data_dir, 'feature_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_results[0].keys()))
        writer.writeheader()
        writer.writerows(stats_results)
    print(f"    Saved: {stats_csv}")
    
    # Save threshold results CSV
    thresh_csv = os.path.join(data_dir, 'threshold_results.csv')
    thresh_export = [{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in threshold_results]
    with open(thresh_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(thresh_export[0].keys()))
        writer.writeheader()
        writer.writerows(thresh_export)
    print(f"    Saved: {thresh_csv}")
    
    # ─── Step 6: Generate HTML report ───
    print("\n" + "-" * 80)
    print("  STEP 6: Generating HTML report...")
    print("-" * 80)
    
    html_path = generate_html_report(
        stats_results, threshold_results, importance_result,
        {}, visuals_dir, all_data
    )
    print(f"    Saved: {html_path}")
    
    # ─── Final Summary ───
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n  Samples analyzed: {len(all_data)}")
    print(f"  Features analyzed: {len(feature_cols)}")
    print(f"\n  OUTPUTS:")
    print(f"    HTML Report:  {html_path}")
    print(f"    Visuals:      {visuals_dir}")
    print(f"    Data:         {data_dir}")
    print(f"\n  TOP FINDINGS:")
    print(f"    Best separating feature: {stats_results[0]['feature']}")
    print(f"      - Separation score: {stats_results[0].get('separation_score', 0):.3f}")
    print(f"      - P-value: {stats_results[0].get('ttest_pval', 1):.2e}")
    print(f"    Best threshold rule: {threshold_results[0]['rule']}")
    print(f"      - Accuracy: {threshold_results[0]['accuracy']:.1%}")
    if importance_result:
        print(f"    ML baseline accuracy: {importance_result['cv_accuracy_mean']:.1%}")
    print("\n  Open the HTML report in a browser for the full interactive dashboard!")
    print("=" * 80)


if __name__ == "__main__":
    main()