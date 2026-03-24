# -*- coding: utf-8 -*-
"""
6_2_evaluate_3dcnn.py

EVALUATION SCRIPT FOR 3D CNN MODEL
===================================

Evaluates the trained 3D CNN on test set.

INPUT:  D:/Research/Cancer_Cell_Analysis/dataset_2class/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dl_output_3dcnn/evaluation/

Run: python -u scripts/6_2_evaluate_3dcnn.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_curve, auc)
from datetime import datetime
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_3dcnn"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")

BATCH_SIZE = 4
NUM_FRAMES = 30
NUM_CLASSES = 2
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['no_cell', 'contain_cell']


# =============================================================================
# DATASET CLASS
# =============================================================================

class LensSequenceDataset3D(Dataset):
    """PyTorch Dataset for 3D CNN."""
    
    def __init__(self, root_dir, max_frames=NUM_FRAMES):
        self.root_dir = root_dir
        self.max_frames = max_frames
        
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            for lens_name in os.listdir(class_dir):
                lens_path = os.path.join(class_dir, lens_name)
                
                if os.path.isdir(lens_path):
                    self.samples.append({
                        'path': lens_path,
                        'class': class_name,
                        'label': self.class_to_idx[class_name]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        lens_path = sample['path']
        label = sample['label']
        
        frame_files = sorted([f for f in os.listdir(lens_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        frames = []
        for i, frame_file in enumerate(frame_files[:self.max_frames]):
            frame_path = os.path.join(lens_path, frame_file)
            
            with Image.open(frame_path) as img:
                img = img.convert('L')
                frame = np.array(img, dtype=np.float32)
            
            frame = frame / 255.0
            frames.append(frame)
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        
        # (C, D, H, W) = (1, 30, 96, 96)
        frames = np.stack(frames, axis=0)
        frames = frames[np.newaxis, :, :, :]
        
        frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, label


# =============================================================================
# 3D CNN MODEL (same as training)
# =============================================================================

class ResBlock3D(nn.Module):
    """3D Residual Block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D18(nn.Module):
    """3D ResNet-18 for video classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super(ResNet3D18, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), 
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix - 3D CNN Test Set', fontsize=15, fontweight='bold')
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.75, f'({cm_norm[i, j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path):
    """Plot ROC curve for binary classification."""
    y_score = y_probs[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#27ae60', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - 3D CNN', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Plot per-class precision, recall, F1."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#27ae60', '#2980b9', '#c0392b']
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i*width, values, width, label=metric.title(), color=colors[i])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Per-Class Metrics (3D CNN)', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return report


def plot_sample_predictions(test_dataset, model, device, output_path, num_samples=12):
    """Plot sample predictions with actual frames."""
    model.eval()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for idx, (ax, sample_idx) in enumerate(zip(axes, indices)):
        frames, label = test_dataset[sample_idx]
        
        with torch.no_grad():
            frames_batch = frames.unsqueeze(0).to(device)
            output = model(frames_batch)
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()
        
        # Get middle frame (index 15 in depth dimension)
        middle_frame = frames[0, 15].numpy()
        ax.imshow(middle_frame, cmap='gray')
        
        actual = CLASS_NAMES[label]
        predicted = CLASS_NAMES[pred]
        conf = probs[pred].item() * 100
        
        color = 'green' if label == pred else 'red'
        ax.set_title(f'Actual: {actual}\nPred: {predicted} ({conf:.1f}%)', 
                    fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions - 3D CNN (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_3dcnn, output_path):
    """Plot comparison between ResNet+LSTM and 3D CNN."""
    
    models = ['ResNet+LSTM', '3D CNN']
    
    # ResNet+LSTM results (from previous evaluation)
    resnet_results = {
        'accuracy': 96.58,
        'auc': 0.988,
        'no_cell_f1': 0.93,
        'contain_cell_f1': 0.98
    }
    
    # 3D CNN results
    cnn3d_results = {
        'accuracy': results_3dcnn['test_accuracy'],
        'auc': results_3dcnn['roc_auc'],
        'no_cell_f1': results_3dcnn['classification_report']['no_cell']['f1-score'],
        'contain_cell_f1': results_3dcnn['classification_report']['contain_cell']['f1-score']
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy & AUC comparison
    x = np.arange(2)
    width = 0.35
    
    acc_vals = [resnet_results['accuracy'], cnn3d_results['accuracy']]
    auc_vals = [resnet_results['auc'] * 100, cnn3d_results['auc'] * 100]
    
    bars1 = axes[0].bar(x - width/2, acc_vals, width, label='Accuracy', color='#3498db')
    bars2 = axes[0].bar(x + width/2, auc_vals, width, label='AUC × 100', color='#2ecc71')
    
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('Accuracy & AUC Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=10)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=10)
    
    # F1 Score comparison
    f1_no_cell = [resnet_results['no_cell_f1'], cnn3d_results['no_cell_f1']]
    f1_contain = [resnet_results['contain_cell_f1'], cnn3d_results['contain_cell_f1']]
    
    bars3 = axes[1].bar(x - width/2, f1_no_cell, width, label='no_cell F1', color='#e74c3c')
    bars4 = axes[1].bar(x + width/2, f1_contain, width, label='contain_cell F1', color='#9b59b6')
    
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', fontsize=10)
    for bar in bars4:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EVALUATION: 3D CNN (ResNet3D-18)")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"  Device: {DEVICE}", flush=True)
    
    # Load test dataset
    print("\n  Loading test dataset...", flush=True)
    test_dataset = LensSequenceDataset3D(os.path.join(DATASET_DIR, 'test'))
    print(f"    Test samples: {len(test_dataset)}", flush=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Load model
    print("\n  Loading trained model...", flush=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  ERROR: Checkpoint not found: {CHECKPOINT_PATH}", flush=True)
        print("  Please run 6_1_train_3dcnn.py first!", flush=True)
        sys.exit(1)
    
    model = ResNet3D18(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded checkpoint from epoch {checkpoint['epoch']+1}", flush=True)
    print(f"    Validation accuracy was: {checkpoint['val_acc']:.2f}%", flush=True)
    
    # Evaluate
    print("\n  Evaluating on test set...", flush=True)
    y_pred, y_true, y_probs = evaluate(model, test_loader, DEVICE)
    
    # Calculate metrics
    test_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Test Accuracy: {test_acc:.2f}%", flush=True)
    
    # Per-class accuracy
    print("\n  Per-class accuracy:", flush=True)
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            print(f"    {cls}: {cls_acc:.2f}%", flush=True)
    
    # Create evaluation directory
    eval_dir = os.path.join(OUTPUT_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print("\n  [1] Confusion matrix...", flush=True)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, 
                         os.path.join(eval_dir, 'confusion_matrix.png'))
    
    print("  [2] ROC curve...", flush=True)
    roc_auc = plot_roc_curve(y_true, y_probs,
                             os.path.join(eval_dir, 'roc_curve.png'))
    print(f"      AUC = {roc_auc:.3f}", flush=True)
    
    print("  [3] Per-class metrics...", flush=True)
    report = plot_per_class_metrics(y_true, y_pred, CLASS_NAMES,
                                    os.path.join(eval_dir, 'per_class_metrics.png'))
    
    print("  [4] Sample predictions...", flush=True)
    plot_sample_predictions(test_dataset, model, DEVICE,
                           os.path.join(eval_dir, 'sample_predictions.png'))
    
    # Save classification report
    print("  [5] Saving classification report...", flush=True)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"3D CNN EVALUATION\n")
        f.write(f"=================\n\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.3f}\n\n")
        f.write(report_text)
    
    # Save results JSON
    results = {
        'model': '3D CNN (ResNet3D-18)',
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'test_accuracy': test_acc,
        'roc_auc': roc_auc,
        'per_class_accuracy': {
            cls: float((y_pred[y_true == i] == y_true[y_true == i]).mean() * 100)
            for i, cls in enumerate(CLASS_NAMES) if (y_true == i).sum() > 0
        },
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'num_test_samples': len(test_dataset),
        'checkpoint_epoch': checkpoint['epoch'] + 1,
    }
    
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Model comparison plot
    print("  [6] Model comparison chart...", flush=True)
    plot_model_comparison(results, os.path.join(eval_dir, 'model_comparison.png'))
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ EVALUATION COMPLETE (3D CNN)!")
    print("=" * 70)
    
    print(f"\n  📊 Test Results:")
    print(f"      Overall Accuracy: {test_acc:.2f}%")
    print(f"      ROC AUC: {roc_auc:.3f}")
    print(f"\n      Per-class:")
    for cls in CLASS_NAMES:
        if cls in report:
            prec = report[cls]['precision']
            rec = report[cls]['recall']
            f1 = report[cls]['f1-score']
            print(f"        {cls}: P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")
    
    print(f"\n  📁 Output: {eval_dir}")
    print(f"      - confusion_matrix.png")
    print(f"      - roc_curve.png")
    print(f"      - per_class_metrics.png")
    print(f"      - sample_predictions.png")
    print(f"      - classification_report.txt")
    print(f"      - evaluation_results.json")
    print(f"      - model_comparison.png")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: ResNet+LSTM vs 3D CNN")
    print("=" * 70)
    print(f"\n  {'Metric':<20} {'ResNet+LSTM':<15} {'3D CNN':<15} {'Winner':<10}")
    print(f"  {'-'*60}")
    
    resnet_acc = 96.58
    resnet_auc = 0.988
    
    acc_winner = "3D CNN" if test_acc > resnet_acc else "ResNet+LSTM" if test_acc < resnet_acc else "Tie"
    auc_winner = "3D CNN" if roc_auc > resnet_auc else "ResNet+LSTM" if roc_auc < resnet_auc else "Tie"
    
    print(f"  {'Accuracy':<20} {resnet_acc:<15.2f} {test_acc:<15.2f} {acc_winner:<10}")
    print(f"  {'ROC AUC':<20} {resnet_auc:<15.3f} {roc_auc:<15.3f} {auc_winner:<10}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()