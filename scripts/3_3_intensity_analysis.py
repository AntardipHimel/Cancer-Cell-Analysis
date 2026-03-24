# -*- coding: utf-8 -*-
"""
3_2_intensity_analysis.py

INTENSITY FOCUSED ANALYSIS
Analyzes Intensity (Mean & Std) differences between frames separated by various gaps.

For each lens (30 frames), computes differences between frame pairs:
  - Gap 1:  (F1,F2), (F2,F3), ... (F29,F30) → 29 pairs
  - Gap 3:  (F1,F4), (F2,F5), ... (F27,F30) → 27 pairs
  - Gap 5:  (F1,F6), (F2,F7), ... (F25,F30) → 25 pairs
  - Gap 7:  (F1,F8), (F2,F9), ... (F23,F30) → 23 pairs
  - Gap 10: (F1,F11), (F2,F12), ... (F20,F30) → 20 pairs
  - Gap 15: (F1,F16), (F2,F17), ... (F15,F30) → 15 pairs
  - Gap 17: (F1,F18), (F2,F19), ... (F13,F30) → 13 pairs
  - Gap 20: (F1,F21), (F2,F22), ... (F10,F30) → 10 pairs
  - Gap 23: (F1,F24), (F2,F25), ... (F7,F30)  →  7 pairs
  - Gap 25: (F1,F26), (F2,F27), ... (F5,F30)  →  5 pairs

Intensity = Average pixel brightness in the image

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\output\\Intensity\\

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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PLOT SETTINGS
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

COLORS = {
    'cell': '#E63946',
    'no_cell': '#457B9D',
    'background': '#F8F9FA',
}

CLASS_COLORS = {0: COLORS['no_cell'], 1: COLORS['cell']}
CLASS_NAMES = {0: 'No Cell', 1: 'Cell'}

GAP_COLORS = {
    1: '#264653', 3: '#2A9D8F', 5: '#E9C46A', 7: '#F4A261', 10: '#E76F51',
    15: '#9B59B6', 17: '#3498DB', 20: '#1ABC9C', 23: '#E74C3C', 25: '#8E44AD'
}

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\output\Intensity"

GAPS = [1, 3, 5, 7, 10, 15, 17, 20, 23, 25]
NUM_CASES = 10

CLASS_MAP = {'contain_cell': 1, 'no_cell': 0}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_intensity(img):
    """Extract intensity features from a single grayscale frame."""
    if img is None:
        return None, None
    
    pixels = img.flatten().astype(np.float64)
    intensity_mean = np.mean(pixels)
    intensity_std = np.std(pixels)
    
    return intensity_mean, intensity_std


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_data(input_dir, gaps):
    """Collect intensity data for all lenses and gaps."""
    
    pair_level_data = []
    lens_level_data = []
    lens_frame_data = []
    
    video_folders = sorted([f for f in os.listdir(input_dir) 
                           if os.path.isdir(os.path.join(input_dir, f))])
    
    total_lenses = 0
    total_pairs = 0
    
    print(f"\n  Found {len(video_folders)} videos to process", flush=True)
    print(f"  Gaps to analyze: {gaps}", flush=True)
    
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
            
            print(f"      {category}: {len(lens_folders)} lenses ", end='', flush=True)
            
            for lens_idx, lens_name in enumerate(lens_folders):
                lens_path = os.path.join(category_path, lens_name)
                frame_files = sorted([f for f in os.listdir(lens_path) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if len(frame_files) < 15:
                    continue
                
                # Extract features from ALL frames
                frame_features = []
                for frame_idx, frame_file in enumerate(frame_files):
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    feat_mean, feat_std = extract_intensity(img)
                    
                    if feat_mean is not None:
                        frame_features.append({
                            'frame_idx': frame_idx + 1,
                            'intensity_mean': feat_mean,
                            'intensity_std': feat_std
                        })
                
                if len(frame_features) < 15:
                    continue
                
                total_lenses += 1
                
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
                        
                        mean_i = frame_features[i]['intensity_mean']
                        mean_j = frame_features[j]['intensity_mean']
                        std_i = frame_features[i]['intensity_std']
                        std_j = frame_features[j]['intensity_std']
                        
                        pair_feat = {
                            'video': video_name,
                            'lens': lens_name,
                            'category': category,
                            'class': CLASS_MAP[category],
                            'gap': gap,
                            'frame_i': i + 1,
                            'frame_j': j + 1,
                            'intensity_mean_i': mean_i,
                            'intensity_mean_j': mean_j,
                            'intensity_mean_diff': mean_j - mean_i,
                            'intensity_mean_abs_diff': abs(mean_j - mean_i),
                            'intensity_std_i': std_i,
                            'intensity_std_j': std_j,
                            'intensity_std_diff': std_j - std_i,
                            'intensity_std_abs_diff': abs(std_j - std_i),
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
                
                for gap in gaps:
                    if not lens_pair_features[gap]:
                        continue
                    
                    gap_pairs = lens_pair_features[gap]
                    mean_diffs = [p['intensity_mean_abs_diff'] for p in gap_pairs]
                    
                    lens_stats[f'intensity_mean_gap{gap}_avg'] = np.mean(mean_diffs)
                    lens_stats[f'intensity_mean_gap{gap}_std'] = np.std(mean_diffs)
                    lens_stats[f'intensity_mean_gap{gap}_max'] = np.max(mean_diffs)
                
                lens_level_data.append(lens_stats)
                
                if (lens_idx + 1) % 10 == 0:
                    print(".", end='', flush=True)
                if (lens_idx + 1) % 50 == 0:
                    print(f"[{lens_idx+1}]", end='', flush=True)
            
            print(f" done", flush=True)
        
        print(f"      Running total: {total_lenses} lenses, {total_pairs:,} pairs", flush=True)
    
    print(f"\n  {'='*50}", flush=True)
    print(f"  COLLECTION COMPLETE!", flush=True)
    print(f"  Total lenses: {total_lenses}", flush=True)
    print(f"  Total pairs: {total_pairs:,}", flush=True)
    print(f"  {'='*50}", flush=True)
    
    return pair_level_data, lens_level_data, lens_frame_data


# =============================================================================
# CASE-BASED ANALYSIS
# =============================================================================

def compute_case_data(lens_frame_data, gaps, num_cases=10):
    """Compute case-based analysis: Top N maximum differences per lens."""
    
    case_data = []
    
    for lens_info in lens_frame_data:
        frames = lens_info['frames']
        
        for gap in gaps:
            pair_diffs_mean = []
            pair_diffs_std = []
            
            for i in range(len(frames) - gap):
                j = i + gap
                mean_diff = abs(frames[j]['intensity_mean'] - frames[i]['intensity_mean'])
                std_diff = abs(frames[j]['intensity_std'] - frames[i]['intensity_std'])
                pair_diffs_mean.append(mean_diff)
                pair_diffs_std.append(std_diff)
            
            sorted_mean_diffs = sorted(pair_diffs_mean, reverse=True)
            sorted_std_diffs = sorted(pair_diffs_std, reverse=True)
            
            for case_num in range(1, min(num_cases + 1, len(sorted_mean_diffs) + 1)):
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
                    'intensity_mean_sum_top_n': np.sum(top_n_mean),
                    'intensity_std_sum_top_n': np.sum(top_n_std),
                    'intensity_mean_avg_top_n': np.mean(top_n_mean),
                    'intensity_std_avg_top_n': np.mean(top_n_std),
                    'intensity_mean_std_top_n': np.std(top_n_mean) if case_num > 1 else 0,
                })
    
    return case_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_histogram(pair_data, gap, output_path, feature='intensity_mean_abs_diff'):
    """Create histogram for a single gap."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cls in [0, 1]:
        vals = [p[feature] for p in pair_data if p['class'] == cls and p['gap'] == gap]
        n = len(vals)
        ax.hist(vals, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
               label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('|Δ Intensity Mean|', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Intensity Mean Difference Distribution - Gap {gap}', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
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
    """Create histograms for a specific case."""
    
    # SUM histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    feature = 'intensity_mean_sum_top_n'
    
    for cls in [0, 1]:
        vals = [c[feature] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    case_label = "1 Maximum Value" if case_num == 1 else f"Sum of Top {case_num} Maximum Values"
    ax.set_xlabel(f'Sum of Top {case_num} |Δ Intensity Mean|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: {case_label} per Lens - Gap {gap}', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    cell_vals = [c[feature] for c in case_data if c['class'] == 1 and c['gap'] == gap and c['case'] == case_num]
    nocell_vals = [c[feature] for c in case_data if c['class'] == 0 and c['gap'] == gap and c['case'] == case_num]
    
    if cell_vals and nocell_vals:
        mean_diff = np.mean(cell_vals) - np.mean(nocell_vals)
        stats_text = (f'Cell: Mean={np.mean(cell_vals):.2f}, Std={np.std(cell_vals):.2f}\n'
                      f'No Cell: Mean={np.mean(nocell_vals):.2f}, Std={np.std(nocell_vals):.2f}\n'
                      f'Δ Mean: {mean_diff:.2f}')
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'case_{case_num}_gap_{gap}_SUM.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # MEAN histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    feature = 'intensity_mean_avg_top_n'
    
    for cls in [0, 1]:
        vals = [c[feature] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    case_label = "1 Maximum Value" if case_num == 1 else f"Mean of Top {case_num} Maximum Values"
    ax.set_xlabel(f'Mean of Top {case_num} |Δ Intensity Mean|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: {case_label} per Lens - Gap {gap}', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'case_{case_num}_gap_{gap}_MEAN.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # STD histogram (only for case >= 2)
    if case_num >= 2:
        fig, ax = plt.subplots(figsize=(12, 7))
        feature = 'intensity_mean_std_top_n'
        
        for cls in [0, 1]:
            vals = [c[feature] for c in case_data 
                   if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
            n = len(vals)
            if vals:
                ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                       label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel(f'Std of Top {case_num} |Δ Intensity Mean|', fontsize=12)
        ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
        ax.set_title(f'Case {case_num}: Std of Top {case_num} Maximum Values - Gap {gap}', fontweight='bold', fontsize=13)
        ax.legend(loc='upper right', fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'case_{case_num}_gap_{gap}_STD.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def create_confusion_matrix(pair_data, gap, output_path, feature='intensity_mean_abs_diff'):
    """Create confusion matrix for a single gap."""
    
    gap_data = [p for p in pair_data if p['gap'] == gap]
    vals = np.array([p[feature] for p in gap_data])
    labels = np.array([p['class'] for p in gap_data])
    
    n_total = len(gap_data)
    n_cell = sum(labels == 1)
    n_nocell = sum(labels == 0)
    
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
    
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            row_total = cm[i, :].sum()
            pct = count / row_total * 100 if row_total > 0 else 0
            color = 'white' if count > cm.max()/2 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                   fontsize=14, fontweight='bold', color=color)
    
    ax.set_title(f'Confusion Matrix - Gap {gap} (Intensity Mean)\n'
                f'Accuracy: {best_acc:.1%} | Threshold: {best_thresh:.2f}\n'
                f'(n={n_total:,} | Cell: {n_cell:,} | No Cell: {n_nocell:,})', 
                fontweight='bold', fontsize=13)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return best_acc, best_thresh


def create_all_gaps_confusion_matrix(pair_data, output_path):
    """Create confusion matrices for all gaps."""
    
    n_gaps = len(GAPS)
    n_cols = 4
    n_rows = (n_gaps + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    feature = 'intensity_mean_abs_diff'
    results = []
    
    for idx, gap in enumerate(GAPS):
        ax = axes[idx]
        
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = np.array([p[feature] for p in gap_data])
        labels = np.array([p['class'] for p in gap_data])
        
        n_total = len(gap_data)
        
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
        
        im = ax.imshow(cm, cmap='Greens')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'], fontsize=10)
        ax.set_yticklabels(['No Cell', 'Cell'], fontsize=10)
        
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                row_total = cm[i, :].sum()
                pct = count / row_total * 100 if row_total > 0 else 0
                color = 'white' if count > cm.max()/2 else 'black'
                ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)
        
        ax.set_title(f'Gap {gap}: Acc={best_acc:.1%}\n(n={n_total:,})', fontweight='bold', fontsize=10)
        results.append({'gap': gap, 'accuracy': best_acc, 'n': n_total})
    
    # Summary panel
    ax = axes[len(GAPS)]
    ax.axis('off')
    summary_text = "SUMMARY\n" + "="*30 + "\n\n"
    for r in results:
        summary_text += f"Gap {r['gap']:2d}: {r['accuracy']:.1%} (n={r['n']:,})\n"
    best_result = max(results, key=lambda x: x['accuracy'])
    summary_text += f"\nBest: Gap {best_result['gap']} ({best_result['accuracy']:.1%})"
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='center', horizontalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    for idx in range(len(GAPS) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices: Intensity Mean by Gap', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return results


def create_roc_curves(pair_data, output_path):
    """Create ROC curves for all gaps."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    feature = 'intensity_mean_abs_diff'
    
    for gap in GAPS:
        gap_data = [p for p in pair_data if p['gap'] == gap]
        vals = np.array([p[feature] for p in gap_data])
        labels = np.array([p['class'] for p in gap_data])
        
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=GAP_COLORS[gap], linewidth=2,
               label=f'Gap {gap:2d} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves: Intensity Mean by Gap', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_gap_scaling_plot(pair_data, output_path):
    """Create gap scaling analysis plot."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    for cls in [0, 1]:
        means = []
        sems = []
        
        for gap in GAPS:
            vals = [p['intensity_mean_abs_diff'] for p in pair_data 
                   if p['class'] == cls and p['gap'] == gap]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        ax1.errorbar(GAPS, means, yerr=sems, marker='o', markersize=8,
                    linewidth=2.5, capsize=5, color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    
    ax1.set_xlabel('Frame Gap', fontsize=13)
    ax1.set_ylabel('Mean |Δ Intensity Mean|', fontsize=13)
    ax1.set_title('Intensity Mean: Gap Scaling', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(GAPS)
    ax1.set_facecolor(COLORS['background'])
    
    ax2 = axes[1]
    for cls in [0, 1]:
        means = []
        sems = []
        
        for gap in GAPS:
            vals = [p['intensity_std_abs_diff'] for p in pair_data 
                   if p['class'] == cls and p['gap'] == gap]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        ax2.errorbar(GAPS, means, yerr=sems, marker='s', markersize=8,
                    linewidth=2.5, capsize=5, color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    
    ax2.set_xlabel('Frame Gap', fontsize=13)
    ax2.set_ylabel('Mean |Δ Intensity Std|', fontsize=13)
    ax2.set_title('Intensity Std: Gap Scaling', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper left', fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(GAPS)
    ax2.set_facecolor(COLORS['background'])
    
    plt.suptitle('Gap Scaling Analysis: How Intensity Differences Increase with Frame Gap', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("  INTENSITY FEATURE ANALYSIS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    histograms_dir = os.path.join(visuals_dir, 'histograms')
    histograms_mean_dir = os.path.join(histograms_dir, 'intensity_mean')
    histograms_cases_dir = os.path.join(histograms_dir, 'cases')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(histograms_dir, exist_ok=True)
    os.makedirs(histograms_mean_dir, exist_ok=True)
    os.makedirs(histograms_cases_dir, exist_ok=True)
    
    for gap in GAPS:
        os.makedirs(os.path.join(histograms_cases_dir, f'gap_{gap}'), exist_ok=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    
    # Step 1: Collect data
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting intensity features...", flush=True)
    print("-" * 80, flush=True)
    
    pair_data, lens_data, lens_frame_data = collect_data(INPUT_DIR, GAPS)
    
    # Step 2: Case analysis
    print("\n" + "-" * 80, flush=True)
    print("  STEP 2: Computing case-based analysis...", flush=True)
    print("-" * 80, flush=True)
    
    case_data = compute_case_data(lens_frame_data, GAPS, NUM_CASES)
    print(f"    Case data: {len(case_data):,} entries", flush=True)
    
    # Step 3: Visualizations
    print("\n" + "-" * 80, flush=True)
    print("  STEP 3: Creating visualizations...", flush=True)
    print("-" * 80, flush=True)
    
    print("    [1] Gap scaling analysis...", flush=True)
    create_gap_scaling_plot(pair_data, os.path.join(visuals_dir, '01_gap_scaling_analysis.png'))
    
    print("    [2] ROC curves...", flush=True)
    create_roc_curves(pair_data, os.path.join(visuals_dir, '02_roc_curves.png'))
    
    print("    [3] Confusion matrices...", flush=True)
    cm_results = create_all_gaps_confusion_matrix(pair_data, os.path.join(visuals_dir, '03_confusion_matrices_all_gaps.png'))
    
    print("    [4] Individual confusion matrices...", flush=True)
    for gap in GAPS:
        acc, _ = create_confusion_matrix(pair_data, gap, os.path.join(visuals_dir, f'04_confusion_matrix_gap_{gap}.png'))
        print(f"        Gap {gap}: Accuracy={acc:.1%}", flush=True)
    
    print("    [5] Histograms...", flush=True)
    for gap in GAPS:
        create_histogram(pair_data, gap, os.path.join(histograms_mean_dir, f'histogram_intensity_mean_gap_{gap}.png'))
    
    print("    [6] Case histograms...", flush=True)
    for gap in GAPS:
        print(f"        Gap {gap}:", end=' ', flush=True)
        gap_case_dir = os.path.join(histograms_cases_dir, f'gap_{gap}')
        for case_num in range(1, NUM_CASES + 1):
            create_case_histogram(case_data, gap, case_num, gap_case_dir)
            print(f"C{case_num}", end=' ', flush=True)
        print("✓", flush=True)
    
    # Step 4: Save data
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data files...", flush=True)
    print("-" * 80, flush=True)
    
    pair_csv = os.path.join(data_dir, 'pair_level_intensity.csv')
    if pair_data:
        fieldnames = list(pair_data[0].keys())
        with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_data)
        print(f"    ✓ Saved: {pair_csv} ({len(pair_data):,} rows)", flush=True)
    
    lens_csv = os.path.join(data_dir, 'lens_level_intensity.csv')
    if lens_data:
        fieldnames = list(lens_data[0].keys())
        with open(lens_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lens_data)
        print(f"    ✓ Saved: {lens_csv} ({len(lens_data):,} rows)", flush=True)
    
    case_csv = os.path.join(data_dir, 'case_level_intensity.csv')
    if case_data:
        fieldnames = list(case_data[0].keys())
        with open(case_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(case_data)
        print(f"    ✓ Saved: {case_csv} ({len(case_data):,} rows)", flush=True)
    
    # Summary
    print("\n" + "=" * 80, flush=True)
    print("  ✅ INTENSITY ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    
    best = max(cm_results, key=lambda x: x['accuracy'])
    print(f"\n  🏆 BEST GAP: {best['gap']} (Accuracy: {best['accuracy']:.1%})", flush=True)
    print(f"\n  📊 Output: {OUTPUT_DIR}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()