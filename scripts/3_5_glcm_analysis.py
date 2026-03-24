# -*- coding: utf-8 -*-
"""
3_4_glcm_analysis.py

GLCM (Gray-Level Co-occurrence Matrix) TEXTURE ANALYSIS
Analyzes texture features (Contrast, Homogeneity, Energy, Correlation) 
differences between frames separated by various gaps.

GLCM captures spatial relationships between pixels - useful for texture analysis.

Input:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
Output: D:\\Research\\Cancer_Cell_Analysis\\output\\GLCM\\

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
from skimage.feature import graycomatrix, graycoprops
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
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\output\GLCM"

GAPS = [1, 3, 5, 7, 10, 15, 17, 20, 23, 25]
NUM_CASES = 10

CLASS_MAP = {'contain_cell': 1, 'no_cell': 0}

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_glcm_features(img):
    """
    Extract GLCM texture features from a single grayscale frame.
    
    Features:
    - Contrast: Measures local variations
    - Homogeneity: Measures closeness of element distribution
    - Energy: Measures uniformity (sum of squared elements)
    - Correlation: Measures linear dependency of gray levels
    """
    if img is None:
        return None
    
    # Reduce levels to speed up computation (256 -> 64 levels)
    img_reduced = (img // 4).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(img_reduced, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=64, symmetric=True, normed=True)
    
    # Extract properties (average over all angles)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_data(input_dir, gaps):
    """Collect GLCM data for all lenses and gaps."""
    
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
                
                frame_features = []
                for frame_idx, frame_file in enumerate(frame_files):
                    img_path = os.path.join(lens_path, frame_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    glcm_feats = extract_glcm_features(img)
                    
                    if glcm_feats is not None:
                        frame_features.append({
                            'frame_idx': frame_idx + 1,
                            **glcm_feats
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
                        
                        pair_feat = {
                            'video': video_name,
                            'lens': lens_name,
                            'category': category,
                            'class': CLASS_MAP[category],
                            'gap': gap,
                            'frame_i': i + 1,
                            'frame_j': j + 1,
                        }
                        
                        # Add all GLCM features
                        for feat_name in ['contrast', 'homogeneity', 'energy', 'correlation']:
                            val_i = frame_features[i][feat_name]
                            val_j = frame_features[j][feat_name]
                            pair_feat[f'{feat_name}_i'] = val_i
                            pair_feat[f'{feat_name}_j'] = val_j
                            pair_feat[f'{feat_name}_diff'] = val_j - val_i
                            pair_feat[f'{feat_name}_abs_diff'] = abs(val_j - val_i)
                        
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
                    for feat_name in ['contrast', 'homogeneity', 'energy', 'correlation']:
                        diffs = [p[f'{feat_name}_abs_diff'] for p in gap_pairs]
                        lens_stats[f'{feat_name}_gap{gap}_avg'] = np.mean(diffs)
                        lens_stats[f'{feat_name}_gap{gap}_max'] = np.max(diffs)
                
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
            # Collect diffs for each feature
            pair_diffs = {feat: [] for feat in ['contrast', 'homogeneity', 'energy', 'correlation']}
            
            for i in range(len(frames) - gap):
                j = i + gap
                for feat_name in pair_diffs.keys():
                    diff = abs(frames[j][feat_name] - frames[i][feat_name])
                    pair_diffs[feat_name].append(diff)
            
            # Sort each feature's diffs
            sorted_diffs = {feat: sorted(diffs, reverse=True) for feat, diffs in pair_diffs.items()}
            
            for case_num in range(1, min(num_cases + 1, len(sorted_diffs['contrast']) + 1)):
                case_entry = {
                    'video': lens_info['video'],
                    'lens': lens_info['lens'],
                    'category': lens_info['category'],
                    'class': lens_info['class'],
                    'gap': gap,
                    'case': case_num,
                    'num_values': case_num,
                }
                
                for feat_name in ['contrast', 'homogeneity', 'energy', 'correlation']:
                    top_n = sorted_diffs[feat_name][:case_num]
                    case_entry[f'{feat_name}_sum_top_n'] = np.sum(top_n)
                    case_entry[f'{feat_name}_avg_top_n'] = np.mean(top_n)
                    case_entry[f'{feat_name}_std_top_n'] = np.std(top_n) if case_num > 1 else 0
                
                case_data.append(case_entry)
    
    return case_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_histogram(pair_data, gap, output_path, feature='contrast_abs_diff'):
    """Create histogram for a single gap and feature."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cls in [0, 1]:
        vals = [p[feature] for p in pair_data if p['class'] == cls and p['gap'] == gap]
        n = len(vals)
        ax.hist(vals, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
               label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    feat_name = feature.replace('_abs_diff', '').title()
    ax.set_xlabel(f'|Δ {feat_name}|', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'{feat_name} Difference Distribution - Gap {gap}', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    cell_vals = [p[feature] for p in pair_data if p['class'] == 1 and p['gap'] == gap]
    nocell_vals = [p[feature] for p in pair_data if p['class'] == 0 and p['gap'] == gap]
    
    stats_text = (f'Cell: Mean={np.mean(cell_vals):.4f}\n'
                  f'No Cell: Mean={np.mean(nocell_vals):.4f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_case_histogram(case_data, gap, case_num, output_dir, feature='contrast'):
    """Create histograms for a specific case and feature."""
    
    # SUM histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    feat_col = f'{feature}_sum_top_n'
    
    for cls in [0, 1]:
        vals = [c[feat_col] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(f'Sum of Top {case_num} |Δ {feature.title()}|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: Sum of Top {case_num} {feature.title()} - Gap {gap}', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'case_{case_num}_gap_{gap}_{feature}_SUM.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # MEAN histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    feat_col = f'{feature}_avg_top_n'
    
    for cls in [0, 1]:
        vals = [c[feat_col] for c in case_data 
               if c['class'] == cls and c['gap'] == gap and c['case'] == case_num]
        n = len(vals)
        if vals:
            ax.hist(vals, bins=40, alpha=0.6, color=CLASS_COLORS[cls],
                   label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(f'Mean of Top {case_num} |Δ {feature.title()}|', fontsize=12)
    ax.set_ylabel('Count (Number of Lenses)', fontsize=13)
    ax.set_title(f'Case {case_num}: Mean of Top {case_num} {feature.title()} - Gap {gap}', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'case_{case_num}_gap_{gap}_{feature}_MEAN.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_all_gaps_confusion_matrix(pair_data, output_path, feature='contrast_abs_diff'):
    """Create confusion matrices for all gaps."""
    
    n_gaps = len(GAPS)
    n_cols = 4
    n_rows = (n_gaps + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
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
        
        im = ax.imshow(cm, cmap='Oranges')
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
        summary_text += f"Gap {r['gap']:2d}: {r['accuracy']:.1%}\n"
    best_result = max(results, key=lambda x: x['accuracy'])
    summary_text += f"\nBest: Gap {best_result['gap']} ({best_result['accuracy']:.1%})"
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='center', horizontalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    for idx in range(len(GAPS) + 1, len(axes)):
        axes[idx].axis('off')
    
    feat_name = feature.replace('_abs_diff', '').title()
    plt.suptitle(f'Confusion Matrices: GLCM {feat_name} by Gap', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return results


def create_roc_curves(pair_data, output_path, feature='contrast_abs_diff'):
    """Create ROC curves for all gaps."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    feat_name = feature.replace('_abs_diff', '').title()
    ax.set_title(f'ROC Curves: GLCM {feat_name} by Gap', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_gap_scaling_plot(pair_data, output_path):
    """Create gap scaling analysis plot for all GLCM features."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    features = ['contrast', 'homogeneity', 'energy', 'correlation']
    
    for ax_idx, feat_name in enumerate(features):
        ax = axes[ax_idx]
        feature = f'{feat_name}_abs_diff'
        
        for cls in [0, 1]:
            means = []
            sems = []
            
            for gap in GAPS:
                vals = [p[feature] for p in pair_data 
                       if p['class'] == cls and p['gap'] == gap]
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            
            ax.errorbar(GAPS, means, yerr=sems, marker='o', markersize=8,
                       linewidth=2.5, capsize=5, color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
        
        ax.set_xlabel('Frame Gap', fontsize=12)
        ax.set_ylabel(f'Mean |Δ {feat_name.title()}|', fontsize=12)
        ax.set_title(f'GLCM {feat_name.title()}: Gap Scaling', fontweight='bold', fontsize=13)
        ax.legend(loc='upper left', fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(GAPS)
        ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Gap Scaling Analysis: GLCM Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("  GLCM TEXTURE FEATURE ANALYSIS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    visuals_dir = os.path.join(OUTPUT_DIR, 'visuals')
    histograms_dir = os.path.join(visuals_dir, 'histograms')
    histograms_cases_dir = os.path.join(histograms_dir, 'cases')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(histograms_dir, exist_ok=True)
    os.makedirs(histograms_cases_dir, exist_ok=True)
    
    # Create feature-specific histogram folders
    for feat in ['contrast', 'homogeneity', 'energy', 'correlation']:
        os.makedirs(os.path.join(histograms_dir, feat), exist_ok=True)
    
    for gap in GAPS:
        os.makedirs(os.path.join(histograms_cases_dir, f'gap_{gap}'), exist_ok=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    
    # Step 1: Collect data
    print("\n" + "-" * 80, flush=True)
    print("  STEP 1: Extracting GLCM features...", flush=True)
    print("  (This may take 20-30 minutes due to GLCM computation)", flush=True)
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
    
    # For each GLCM feature
    all_results = {}
    for feat_name in ['contrast', 'homogeneity', 'energy', 'correlation']:
        print(f"    [{feat_name.upper()}]", flush=True)
        feature = f'{feat_name}_abs_diff'
        
        print(f"        ROC curves...", flush=True)
        create_roc_curves(pair_data, os.path.join(visuals_dir, f'02_roc_curves_{feat_name}.png'), feature)
        
        print(f"        Confusion matrices...", flush=True)
        results = create_all_gaps_confusion_matrix(pair_data, 
                                                   os.path.join(visuals_dir, f'03_confusion_matrices_{feat_name}.png'), 
                                                   feature)
        all_results[feat_name] = results
        
        print(f"        Histograms...", flush=True)
        for gap in GAPS:
            create_histogram(pair_data, gap, 
                           os.path.join(histograms_dir, feat_name, f'histogram_{feat_name}_gap_{gap}.png'),
                           feature)
    
    print("    [5] Case histograms (contrast only)...", flush=True)
    for gap in GAPS:
        print(f"        Gap {gap}:", end=' ', flush=True)
        gap_case_dir = os.path.join(histograms_cases_dir, f'gap_{gap}')
        for case_num in range(1, NUM_CASES + 1):
            create_case_histogram(case_data, gap, case_num, gap_case_dir, 'contrast')
            print(f"C{case_num}", end=' ', flush=True)
        print("✓", flush=True)
    
    # Step 4: Save data
    print("\n" + "-" * 80, flush=True)
    print("  STEP 4: Saving data files...", flush=True)
    print("-" * 80, flush=True)
    
    pair_csv = os.path.join(data_dir, 'pair_level_glcm.csv')
    if pair_data:
        fieldnames = list(pair_data[0].keys())
        with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_data)
        print(f"    ✓ Saved: {pair_csv} ({len(pair_data):,} rows)", flush=True)
    
    lens_csv = os.path.join(data_dir, 'lens_level_glcm.csv')
    if lens_data:
        fieldnames = list(lens_data[0].keys())
        with open(lens_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lens_data)
        print(f"    ✓ Saved: {lens_csv} ({len(lens_data):,} rows)", flush=True)
    
    case_csv = os.path.join(data_dir, 'case_level_glcm.csv')
    if case_data:
        fieldnames = list(case_data[0].keys())
        with open(case_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(case_data)
        print(f"    ✓ Saved: {case_csv} ({len(case_data):,} rows)", flush=True)
    
    # Summary
    print("\n" + "=" * 80, flush=True)
    print("  ✅ GLCM ANALYSIS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    
    print("\n  🏆 BEST RESULTS BY FEATURE:", flush=True)
    for feat_name, results in all_results.items():
        best = max(results, key=lambda x: x['accuracy'])
        print(f"      {feat_name.upper()}: Gap {best['gap']} (Accuracy: {best['accuracy']:.1%})", flush=True)
    
    print(f"\n  📊 Output: {OUTPUT_DIR}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()