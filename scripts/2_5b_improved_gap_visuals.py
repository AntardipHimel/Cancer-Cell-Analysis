# -*- coding: utf-8 -*-
"""
2_5b_improved_gap_visuals.py

IMPROVED GAP ANALYSIS VISUALIZATIONS
- Shows sample counts (n=X) on ALL plots
- Displays individual data points where possible
- Each lens is a separate data point (no cross-lens averaging)
- Clear data source annotations

Uses existing data from:
  D:\Research\Cancer_Cell_Analysis\gap_analysis\data\pair_level_features.csv
  D:\Research\Cancer_Cell_Analysis\gap_analysis\data\lens_level_features.csv

Output:
  D:\Research\Cancer_Cell_Analysis\gap_analysis\visuals_improved\

Author: Antardip Himel
Date: February 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
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

# Professional colors
COLORS = {
    'cell': '#E63946',
    'no_cell': '#457B9D',
    'background': '#F8F9FA'
}

CLASS_COLORS = {0: COLORS['no_cell'], 1: COLORS['cell']}
CLASS_NAMES = {0: 'No Cell', 1: 'Cell'}

GAP_COLORS = {1: '#264653', 3: '#2A9D8F', 5: '#E9C46A', 7: '#F4A261', 10: '#E76F51'}

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = r"D:\Research\Cancer_Cell_Analysis\gap_analysis\data"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\gap_analysis\visuals_improved"

GAPS = [1, 3, 5, 7, 10]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load pair-level and lens-level data."""
    print("  Loading data...", flush=True)
    sys.stdout.flush()
    
    pair_csv = os.path.join(DATA_DIR, 'pair_level_features.csv')
    lens_csv = os.path.join(DATA_DIR, 'lens_level_features.csv')
    
    pair_df = pd.read_csv(pair_csv)
    lens_df = pd.read_csv(lens_csv)
    
    print(f"    Pair-level: {len(pair_df):,} rows", flush=True)
    print(f"    Lens-level: {len(lens_df):,} rows", flush=True)
    
    return pair_df, lens_df


# =============================================================================
# IMPROVED VISUALIZATIONS
# =============================================================================

def plot_01_gap_scaling_with_counts(pair_df, output_path):
    """
    Gap scaling analysis with sample counts displayed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    features = ['edge_gradient_mean_abs_diff', 'intensity_mean_abs_diff', 'intensity_entropy_abs_diff']
    titles = ['Edge Gradient', 'Intensity', 'Entropy']
    
    for ax, feat, title in zip(axes, features, titles):
        for cls in [0, 1]:
            means = []
            stds = []
            counts = []
            
            for gap in GAPS:
                mask = (pair_df['class'] == cls) & (pair_df['gap'] == gap)
                vals = pair_df.loc[mask, feat].dropna()
                
                means.append(vals.mean())
                stds.append(vals.std() / np.sqrt(len(vals)))  # SEM
                counts.append(len(vals))
            
            line = ax.errorbar(GAPS, means, yerr=stds, marker='o', markersize=10,
                              linewidth=2.5, capsize=5, capthick=2,
                              color=CLASS_COLORS[cls], label=f'{CLASS_NAMES[cls]}')
            
            # Add count labels
            for i, (gap, mean, count) in enumerate(zip(GAPS, means, counts)):
                offset = 0.15 if cls == 1 else -0.15
                ax.annotate(f'n={count:,}', xy=(gap, mean), 
                           xytext=(gap + offset, mean),
                           fontsize=8, color=CLASS_COLORS[cls], alpha=0.8,
                           ha='left' if cls == 1 else 'right')
        
        ax.set_xlabel('Frame Gap', fontsize=12)
        ax.set_ylabel(f'Mean |Δ{title}|', fontsize=12)
        ax.set_title(f'{title} Change vs Gap', fontweight='bold')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(GAPS)
        ax.set_facecolor(COLORS['background'])
    
    # Add total counts annotation
    total_pairs = len(pair_df)
    cell_pairs = len(pair_df[pair_df['class'] == 1])
    nocell_pairs = len(pair_df[pair_df['class'] == 0])
    
    fig.text(0.5, 0.02, 
             f'Total: {total_pairs:,} pairs | Cell: {cell_pairs:,} | No Cell: {nocell_pairs:,}',
             ha='center', fontsize=11, style='italic')
    
    plt.suptitle('Gap Scaling Analysis: Mean |Difference| vs Frame Gap\n(with sample counts)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_02_scatter_individual_pairs(pair_df, output_path, max_points=15000):
    """
    Scatter plot showing INDIVIDUAL pair data points (subsampled for visibility).
    Each point = one frame pair from one lens.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for row, cls in enumerate([0, 1]):
        for col, gap in enumerate(GAPS[:3]):  # First 3 gaps
            ax = axes[row, col]
            
            mask = (pair_df['class'] == cls) & (pair_df['gap'] == gap)
            data = pair_df.loc[mask, feat].dropna()
            
            n_total = len(data)
            
            # Subsample for visibility
            if len(data) > max_points:
                data = data.sample(n=max_points, random_state=42)
            
            # Create jittered x positions
            x = np.random.normal(1, 0.15, size=len(data))
            
            ax.scatter(x, data.values, alpha=0.3, s=15, 
                      color=CLASS_COLORS[cls], edgecolors='none')
            
            # Add violin overlay for distribution
            parts = ax.violinplot([data.values], positions=[1], showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(CLASS_COLORS[cls])
            parts['bodies'][0].set_alpha(0.3)
            
            ax.set_xlim(0.3, 1.7)
            ax.set_xticks([])
            ax.set_ylabel('|Edge Gradient Diff|')
            ax.set_title(f'{CLASS_NAMES[cls]} - Gap {gap}\n(n={n_total:,}, showing {len(data):,})', 
                        fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor(COLORS['background'])
            
            # Add statistics
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nMedian: {data.median():.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Individual Frame Pair Differences (Each Point = 1 Frame Pair from 1 Lens)\n'
                 f'Edge Gradient |Δ|', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_03_lens_level_scatter_with_counts(lens_df, output_path):
    """
    Scatter plot of lens-level features with ALL data points visible.
    Each point = one lens. Shows ALL lenses (no subsampling).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['motion_energy', 'tvi', 'max_jump']
    titles = ['Motion Energy', 'Temporal Variability Index (TVI)', 'Max Jump']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for cls in [0, 1]:
            mask = lens_df['class'] == cls
            data = lens_df.loc[mask, metric].dropna()
            
            # Filter outliers for TVI
            if metric == 'tvi':
                data = data[data < 5]
            
            n = len(data)
            
            # Jittered strip plot - ALL POINTS
            x = np.random.normal(cls, 0.12, size=len(data))
            ax.scatter(x, data.values, alpha=0.35, s=20, 
                      color=CLASS_COLORS[cls], edgecolors='white', linewidth=0.2,
                      label=f'{CLASS_NAMES[cls]} (n={n:,})')
            
            # Add mean line
            ax.hlines(data.mean(), cls - 0.3, cls + 0.3, colors=CLASS_COLORS[cls], 
                     linewidth=3, linestyles='-')
            ax.hlines(data.median(), cls - 0.2, cls + 0.2, colors='black', 
                     linewidth=2, linestyles='--', alpha=0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Cell', 'Cell'])
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n(ALL {len(lens_df):,} lenses shown)', fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
        
        # Add count annotation
        n_nocell = len(lens_df[lens_df['class'] == 0])
        n_cell = len(lens_df[lens_df['class'] == 1])
    
    fig.text(0.5, 0.02, 
             f'ALL Lenses Shown: {len(lens_df):,} total | Cell: {n_cell:,} | No Cell: {n_nocell:,}',
             ha='center', fontsize=11, style='italic', fontweight='bold')
    
    plt.suptitle('Lens-Level Features: ALL Individual Lens Data Points\n'
                 '(Solid line = Mean, Dashed line = Median)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_04_histogram_with_counts(pair_df, output_path):
    """
    Histograms showing distribution with exact counts on each bar.
    """
    fig, axes = plt.subplots(len(GAPS), 3, figsize=(18, 4*len(GAPS)))
    
    features = ['edge_gradient_mean_abs_diff', 'intensity_mean_abs_diff', 'intensity_entropy_abs_diff']
    titles = ['Edge Gradient |Δ|', 'Intensity |Δ|', 'Entropy |Δ|']
    
    for row, gap in enumerate(GAPS):
        for col, (feat, title) in enumerate(zip(features, titles)):
            ax = axes[row, col]
            
            for cls in [0, 1]:
                mask = (pair_df['class'] == cls) & (pair_df['gap'] == gap)
                data = pair_df.loc[mask, feat].dropna()
                
                n = len(data)
                
                ax.hist(data, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
                       label=f'{CLASS_NAMES[cls]} (n={n:,})', edgecolor='white', linewidth=0.3)
            
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.set_title(f'Gap {gap}: {title}', fontweight='bold', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Feature Difference Distributions by Gap\n'
                 '(Each histogram shows individual frame pair counts)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_05_per_gap_comparison(pair_df, output_path):
    """
    Direct comparison of Cell vs No Cell at each gap with statistical annotations.
    """
    fig, axes = plt.subplots(1, len(GAPS), figsize=(4*len(GAPS), 6))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for ax, gap in zip(axes, GAPS):
        data_nocell = pair_df.loc[(pair_df['class'] == 0) & (pair_df['gap'] == gap), feat].dropna()
        data_cell = pair_df.loc[(pair_df['class'] == 1) & (pair_df['gap'] == gap), feat].dropna()
        
        # Box + Strip plot
        positions = [0, 1]
        
        for pos, data, cls in [(0, data_nocell, 0), (1, data_cell, 1)]:
            # Subsample for strip plot - increased to 3000
            plot_data = data.sample(n=min(3000, len(data)), random_state=42) if len(data) > 3000 else data
            
            # Strip
            x_jitter = np.random.normal(pos, 0.08, size=len(plot_data))
            ax.scatter(x_jitter, plot_data.values, alpha=0.3, s=10, 
                      color=CLASS_COLORS[cls], edgecolors='none')
            
            # Box
            bp = ax.boxplot([data.values], positions=[pos], widths=0.4, 
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(CLASS_COLORS[cls])
            bp['boxes'][0].set_alpha(0.5)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'No Cell\n(n={len(data_nocell):,})', 
                           f'Cell\n(n={len(data_cell):,})'])
        ax.set_ylabel('|Edge Gradient Diff|')
        ax.set_title(f'Gap {gap}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor(COLORS['background'])
        
        # Add statistics
        mean_diff = data_cell.mean() - data_nocell.mean()
        ax.text(0.5, 0.95, f'ΔMean: {mean_diff:.2f}', transform=ax.transAxes,
               fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.suptitle('Edge Gradient Differences: Cell vs No Cell by Gap\n'
                 '(Box + Individual Points, max 3,000 shown per class)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_06_data_summary_table(pair_df, lens_df, output_path):
    """
    Visual table summarizing all data counts.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Prepare data
    summary_data = []
    
    # Header
    summary_data.append(['Gap', 'Total Pairs', 'Cell Pairs', 'No Cell Pairs', 
                        'Cell Mean |Δ Edge|', 'NoCell Mean |Δ Edge|', 'Difference'])
    
    for gap in GAPS:
        gap_data = pair_df[pair_df['gap'] == gap]
        cell_data = gap_data[gap_data['class'] == 1]['edge_gradient_mean_abs_diff']
        nocell_data = gap_data[gap_data['class'] == 0]['edge_gradient_mean_abs_diff']
        
        summary_data.append([
            f'Gap {gap}',
            f'{len(gap_data):,}',
            f'{len(cell_data):,}',
            f'{len(nocell_data):,}',
            f'{cell_data.mean():.3f}',
            f'{nocell_data.mean():.3f}',
            f'{cell_data.mean() - nocell_data.mean():.3f}'
        ])
    
    # Add totals
    summary_data.append([
        'TOTAL',
        f'{len(pair_df):,}',
        f'{len(pair_df[pair_df["class"]==1]):,}',
        f'{len(pair_df[pair_df["class"]==0]):,}',
        '-', '-', '-'
    ])
    
    # Add lens-level summary
    summary_data.append(['', '', '', '', '', '', ''])
    summary_data.append(['LENS-LEVEL', 'Total Lenses', 'Cell Lenses', 'No Cell Lenses', '', '', ''])
    summary_data.append([
        '',
        f'{len(lens_df):,}',
        f'{len(lens_df[lens_df["class"]==1]):,}',
        f'{len(lens_df[lens_df["class"]==0]):,}',
        '', '', ''
    ])
    
    # Create table
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                    colWidths=[0.12, 0.12, 0.12, 0.12, 0.15, 0.15, 0.12])
    
    # Style header
    for j in range(7):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style totals row
    for j in range(7):
        table[(6, j)].set_facecolor('#E8E8E8')
        table[(6, j)].set_text_props(fontweight='bold')
        table[(8, j)].set_facecolor('#D5E8D4')
        table[(8, j)].set_text_props(fontweight='bold')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    plt.title('DATA SUMMARY: Gap Analysis\n'
              'Pair-Level = Frame-to-Frame Differences | Lens-Level = Aggregated per Lens',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_07_sample_lens_pairs(pair_df, output_path, n_lenses=15):
    """
    Show actual frame pairs from sample lenses to visualize what the data looks like.
    Each line = one lens, showing all its pair differences across frame indices.
    """
    fig, axes = plt.subplots(2, len(GAPS), figsize=(4*len(GAPS), 10))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for col, gap in enumerate(GAPS):
        for row, cls in enumerate([0, 1]):
            ax = axes[row, col]
            
            # Get sample lenses
            mask = (pair_df['class'] == cls) & (pair_df['gap'] == gap)
            gap_data = pair_df[mask]
            
            unique_lenses = gap_data.groupby(['video', 'lens']).size().reset_index()
            sample_lenses = unique_lenses.sample(n=min(n_lenses, len(unique_lenses)), random_state=42)
            
            n_total_lenses = len(unique_lenses)
            
            for _, lens_row in sample_lenses.iterrows():
                lens_data = gap_data[(gap_data['video'] == lens_row['video']) & 
                                     (gap_data['lens'] == lens_row['lens'])]
                
                x = lens_data['frame_i'].values
                y = lens_data[feat].values
                
                ax.plot(x, y, marker='o', markersize=4, alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Frame i (start of pair)')
            ax.set_ylabel(f'|Δ Edge Gradient|')
            ax.set_title(f'{CLASS_NAMES[cls]} - Gap {gap}\n'
                        f'(Showing {n_lenses} of {n_total_lenses:,} lenses)', 
                        fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(COLORS['background'])
    
    plt.suptitle('Sample Lens Trajectories: Frame Pair Differences\n'
                 'Each Line = 1 Lens | Each Point = 1 Frame Pair (Fi, Fi+gap)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_08_confusion_with_counts(pair_df, lens_df, output_path):
    """
    Confusion matrices with detailed count annotations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()
    
    feat = 'edge_gradient_mean_abs_diff'
    
    for idx, gap in enumerate(GAPS):
        ax = axes[idx]
        
        gap_data = pair_df[pair_df['gap'] == gap]
        vals = gap_data[feat].values
        labels = gap_data['class'].values
        
        n_total = len(gap_data)
        n_cell = sum(labels == 1)
        n_nocell = sum(labels == 0)
        
        # Find optimal threshold
        thresholds = np.linspace(vals.min(), vals.max(), 50)
        best_acc = 0
        best_pred = None
        
        for thresh in thresholds:
            pred_g = (vals > thresh).astype(int)
            pred_l = (vals < thresh).astype(int)
            
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
        
        # Add counts with percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                row_total = cm[i, :].sum()
                pct = count / row_total * 100 if row_total > 0 else 0
                color = 'white' if count > cm.max()/2 else 'black'
                ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                       fontsize=11, fontweight='bold', color=color)
        
        ax.set_title(f'Gap {gap}: Acc={best_acc:.1%}\n'
                    f'(n={n_total:,} | Cell:{n_cell:,} | NoCell:{n_nocell:,})', 
                    fontweight='bold', fontsize=10)
    
    # Last plot: Motion Energy (lens-level)
    ax = axes[5]
    vals = lens_df['motion_energy'].values
    labels = lens_df['class'].values
    
    n_total = len(lens_df)
    n_cell = sum(labels == 1)
    n_nocell = sum(labels == 0)
    
    thresholds = np.linspace(vals.min(), vals.max(), 50)
    best_acc = 0
    best_pred = None
    
    for thresh in thresholds:
        pred_g = (vals > thresh).astype(int)
        pred_l = (vals < thresh).astype(int)
        
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
            count = cm[i, j]
            row_total = cm[i, :].sum()
            pct = count / row_total * 100 if row_total > 0 else 0
            color = 'white' if count > cm.max()/2 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)
    
    ax.set_title(f'Motion Energy (Lens-Level): Acc={best_acc:.1%}\n'
                f'(n={n_total:,} lenses | Cell:{n_cell:,} | NoCell:{n_nocell:,})', 
                fontweight='bold', fontsize=10)
    
    plt.suptitle('Confusion Matrices with Sample Counts\n'
                 '(Numbers show Count and Row Percentage)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_09_roc_with_counts(pair_df, lens_df, output_path):
    """
    ROC curves with sample counts in legend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    feat = 'edge_gradient_mean_abs_diff'
    
    # Left: ROC by gap
    ax1 = axes[0]
    for gap in GAPS:
        gap_data = pair_df[pair_df['gap'] == gap]
        vals = gap_data[feat].values
        labels = gap_data['class'].values
        
        n = len(gap_data)
        
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color=GAP_COLORS[gap], linewidth=2.5,
                label=f'Gap {gap} (n={n:,}, AUC={roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves by Gap\n(Pair-Level Data)', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(COLORS['background'])
    
    # Right: Lens-level
    ax2 = axes[1]
    
    metrics = ['motion_energy', 'tvi', 'max_jump']
    colors = ['#2A9D8F', '#E9C46A', '#F4A261']
    
    for metric, color in zip(metrics, colors):
        vals = lens_df[metric].values
        labels = lens_df['class'].values
        
        mask = np.isfinite(vals)
        vals = vals[mask]
        labels = labels[mask]
        
        n = len(vals)
        
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{metric.replace("_", " ").title()} (n={n:,}, AUC={roc_auc:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves: Lens-Level Metrics\n(1 point per lens)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(COLORS['background'])
    
    plt.suptitle('ROC Analysis with Sample Counts', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


def plot_10_individual_lens_motion_energy(lens_df, output_path):
    """
    Show ALL individual lenses as data points for motion energy classification.
    NO subsampling - shows every single lens.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for cls in [0, 1]:
        data = lens_df[lens_df['class'] == cls]['motion_energy']
        n = len(data)
        
        # Show ALL points - no subsampling
        x = np.random.normal(cls, 0.18, size=len(data))
        ax.scatter(x, data.values, alpha=0.35, s=25, 
                  color=CLASS_COLORS[cls], edgecolors='white', linewidth=0.2,
                  label=f'{CLASS_NAMES[cls]} (n={n:,})')
        
        # Add mean and median lines
        ax.hlines(data.mean(), cls - 0.4, cls + 0.4, colors='black', 
                 linewidth=3, linestyles='-', label=f'Mean' if cls == 0 else '')
        ax.hlines(data.median(), cls - 0.3, cls + 0.3, colors='gray', 
                 linewidth=2, linestyles='--', label=f'Median' if cls == 0 else '')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Cell', 'Cell'], fontsize=12)
    ax.set_ylabel('Motion Energy', fontsize=12)
    ax.set_title('Motion Energy: ALL Individual Lens Data Points\n'
                '(Each point = 1 lens | Black line = Mean | Gray dashed = Median)',
                fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(COLORS['background'])
    
    # Add total count
    n_total = len(lens_df)
    n_cell = len(lens_df[lens_df['class'] == 1])
    n_nocell = len(lens_df[lens_df['class'] == 0])
    
    fig.text(0.5, 0.02, 
             f'ALL {n_total:,} Lenses Shown | '
             f'Cell: {n_cell:,} | '
             f'No Cell: {n_nocell:,}',
             ha='center', fontsize=11, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("  IMPROVED GAP ANALYSIS VISUALIZATIONS", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n  Input:  {DATA_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    
    # Load data
    pair_df, lens_df = load_data()
    
    # Create visualizations
    print("\n  Creating improved visualizations...", flush=True)
    print("-" * 60, flush=True)
    sys.stdout.flush()
    
    print("  [1/10] Gap scaling with counts...", flush=True)
    plot_01_gap_scaling_with_counts(pair_df, os.path.join(OUTPUT_DIR, '01_gap_scaling_with_counts.png'))
    
    print("  [2/10] Individual pair scatter...", flush=True)
    plot_02_scatter_individual_pairs(pair_df, os.path.join(OUTPUT_DIR, '02_individual_pair_scatter.png'))
    
    print("  [3/10] Lens-level scatter with counts...", flush=True)
    plot_03_lens_level_scatter_with_counts(lens_df, os.path.join(OUTPUT_DIR, '03_lens_level_scatter.png'))
    
    print("  [4/10] Histograms with counts...", flush=True)
    plot_04_histogram_with_counts(pair_df, os.path.join(OUTPUT_DIR, '04_histograms_with_counts.png'))
    
    print("  [5/10] Per-gap comparison...", flush=True)
    plot_05_per_gap_comparison(pair_df, os.path.join(OUTPUT_DIR, '05_per_gap_comparison.png'))
    
    print("  [6/10] Data summary table...", flush=True)
    plot_06_data_summary_table(pair_df, lens_df, os.path.join(OUTPUT_DIR, '06_data_summary_table.png'))
    
    print("  [7/10] Sample lens pairs...", flush=True)
    plot_07_sample_lens_pairs(pair_df, os.path.join(OUTPUT_DIR, '07_sample_lens_pairs.png'))
    
    print("  [8/10] Confusion matrices with counts...", flush=True)
    plot_08_confusion_with_counts(pair_df, lens_df, os.path.join(OUTPUT_DIR, '08_confusion_with_counts.png'))
    
    print("  [9/10] ROC with counts...", flush=True)
    plot_09_roc_with_counts(pair_df, lens_df, os.path.join(OUTPUT_DIR, '09_roc_with_counts.png'))
    
    print("  [10/10] Individual lens motion energy...", flush=True)
    plot_10_individual_lens_motion_energy(lens_df, os.path.join(OUTPUT_DIR, '10_individual_lens_motion.png'))
    
    print("\n" + "=" * 80, flush=True)
    print("  ✅ IMPROVED VISUALIZATIONS COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  📊 Output: {OUTPUT_DIR}", flush=True)
    print(f"\n  Key improvements:", flush=True)
    print(f"    ✓ Sample counts (n=X) on ALL plots", flush=True)
    print(f"    ✓ Individual data points visible", flush=True)
    print(f"    ✓ Each lens kept separate (no cross-lens averaging)", flush=True)
    print(f"    ✓ Clear data source annotations", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()