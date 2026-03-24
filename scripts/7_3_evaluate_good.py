# -*- coding: utf-8 -*-
"""
7_3_evaluate_good.py

EVALUATION: GOOD vs NOT_GOOD
=============================

Evaluates the trained ResNet+LSTM on test set.

Run: python -u scripts/7_3_evaluate_good.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from datetime import datetime
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_good_vs_notgood"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_good_vs_notgood"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")

BATCH_SIZE = 8
NUM_FRAMES = 30
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['not_good', 'good']


# =============================================================================
# DATASET & MODEL
# =============================================================================

class LensSequenceDataset(Dataset):
    
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
            frame = frame[np.newaxis, :, :]
            frames.append(frame)
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        
        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames), label


class ResNetLSTM(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2, dropout=0.5):
        super(ResNetLSTM, self).__init__()
        
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnet(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        h_combined = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        return self.classifier(h_combined)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
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
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix\n(good vs not_good)', fontsize=15, fontweight='bold')
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.75, f'({cm_norm[i, j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path):
    y_score = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#8e44ad', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve (good vs not_good)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#9b59b6', '#3498db', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i*width, values, width, label=metric.title(), color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Per-Class Metrics (good vs not_good)', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EVALUATION: GOOD vs NOT_GOOD")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"    good     = contain_cell")
    print(f"    not_good = no_cell + uncertain_cell")
    print(f"\n  Device: {DEVICE}")
    
    # Load test dataset
    print("\n  Loading test dataset...", flush=True)
    test_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'test'))
    print(f"    Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Load model
    print("\n  Loading trained model...", flush=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  ERROR: Checkpoint not found!")
        sys.exit(1)
    
    model = ResNetLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded from epoch {checkpoint['epoch']+1}")
    print(f"    Val accuracy was: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    print("\n  Evaluating on test set...", flush=True)
    y_pred, y_true, y_probs = evaluate(model, test_loader, DEVICE)
    
    test_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Test Accuracy: {test_acc:.2f}%")
    
    print("\n  Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            print(f"    {cls}: {cls_acc:.2f}%")
    
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
    roc_auc = plot_roc_curve(y_true, y_probs, os.path.join(eval_dir, 'roc_curve.png'))
    print(f"      AUC = {roc_auc:.3f}")
    
    print("  [3] Per-class metrics...", flush=True)
    report = plot_per_class_metrics(y_true, y_pred, CLASS_NAMES,
                                   os.path.join(eval_dir, 'per_class_metrics.png'))
    
    # Save results
    print("  [4] Saving results...", flush=True)
    
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"GOOD vs NOT_GOOD EVALUATION\n")
        f.write(f"============================\n\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.3f}\n\n")
        f.write(report_text)
    
    results = {
        'experiment': 'good_vs_notgood',
        'class_names': CLASS_NAMES,
        'test_accuracy': test_acc,
        'roc_auc': roc_auc,
        'per_class_accuracy': {
            cls: float((y_pred[y_true == i] == y_true[y_true == i]).mean() * 100)
            for i, cls in enumerate(CLASS_NAMES) if (y_true == i).sum() > 0
        },
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📊 Test Results (good vs not_good):")
    print(f"      Overall Accuracy: {test_acc:.2f}%")
    print(f"      ROC AUC: {roc_auc:.3f}")
    print(f"\n      Per-class:")
    for cls in CLASS_NAMES:
        prec = report[cls]['precision']
        rec = report[cls]['recall']
        f1 = report[cls]['f1-score']
        print(f"        {cls}: P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")
    
    print(f"\n  📁 Output: {eval_dir}")
    
    # Comparison with previous experiments
    print("\n" + "=" * 70)
    print("  COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("=" * 70)
    print(f"\n  {'Experiment':<30} {'Test Acc':<12} {'AUC':<10}")
    print(f"  {'-'*52}")
    print(f"  {'contain vs no (2-class)':<30} {'96.58%':<12} {'0.988':<10}")
    print(f"  {'3D CNN (2-class)':<30} {'94.01%':<12} {'0.983':<10}")
    print(f"  {'good vs not_good':<30} {f'{test_acc:.2f}%':<12} {f'{roc_auc:.3f}':<10}")
    print("=" * 70)
    
    print("\n  Next: Run 7_4_visualize_good.py")
    print("=" * 70)


if __name__ == "__main__":
    main()