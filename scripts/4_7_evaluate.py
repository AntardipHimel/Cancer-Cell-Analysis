# -*- coding: utf-8 -*-
"""
4_7_evaluate.py

EVALUATION SCRIPT FOR TRAINED MODEL
====================================

All-in-one evaluation script (no external imports needed).

Run: python -u scripts/4_7_evaluate.py

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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")

BATCH_SIZE = 8
NUM_FRAMES = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['no_cell', 'contain_cell', 'uncertain_cell']


# =============================================================================
# DATASET CLASS
# =============================================================================

class LensSequenceDataset(Dataset):
    """PyTorch Dataset for cell lens sequences."""
    
    def __init__(self, root_dir, max_frames=NUM_FRAMES):
        self.root_dir = root_dir
        self.max_frames = max_frames
        
        self.classes = ['no_cell', 'contain_cell', 'uncertain_cell']
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
        frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, label


# =============================================================================
# MODEL
# =============================================================================

class ResNetLSTM(nn.Module):
    """ResNet + LSTM for video classification."""
    
    def __init__(self, num_classes=3, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=False):
        super(ResNetLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ResNet backbone
        resnet = models.resnet18(weights=None)
        
        # Modify for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        lstm_output_size = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
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
        
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        logits = self.classifier(h_combined)
        return logits


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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix - Test Set', fontsize=15, fontweight='bold')
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Plot per-class precision, recall, F1."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i*width, values, width, label=metric.title(), color=colors[i])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Per-Class Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
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
        
        middle_frame = frames[15, 0].numpy()
        ax.imshow(middle_frame, cmap='gray')
        
        actual = CLASS_NAMES[label]
        predicted = CLASS_NAMES[pred]
        conf = probs[pred].item() * 100
        
        color = 'green' if label == pred else 'red'
        ax.set_title(f'Actual: {actual}\nPred: {predicted} ({conf:.1f}%)', 
                    fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EVALUATION: ResNet-18 + LSTM")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    print(f"\n  Device: {DEVICE}", flush=True)
    
    # Load test dataset
    print("\n  Loading test dataset...", flush=True)
    test_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'test'))
    print(f"    Test samples: {len(test_dataset)}", flush=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Load model
    print("\n  Loading trained model...", flush=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  ERROR: Checkpoint not found: {CHECKPOINT_PATH}", flush=True)
        print("  Please run 4_6_train.py first!", flush=True)
        sys.exit(1)
    
    model = ResNetLSTM(num_classes=3, pretrained=False).to(DEVICE)
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
    
    print("  [2] Per-class metrics...", flush=True)
    report = plot_per_class_metrics(y_true, y_pred, CLASS_NAMES,
                                    os.path.join(eval_dir, 'per_class_metrics.png'))
    
    print("  [3] Sample predictions...", flush=True)
    plot_sample_predictions(test_dataset, model, DEVICE,
                           os.path.join(eval_dir, 'sample_predictions.png'))
    
    # Save classification report
    print("  [4] Saving classification report...", flush=True)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(report_text)
    
    # Save results JSON
    results = {
        'test_accuracy': test_acc,
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
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📊 Test Results:")
    print(f"      Overall Accuracy: {test_acc:.2f}%")
    print(f"\n      Per-class:")
    for cls in CLASS_NAMES:
        if cls in report:
            prec = report[cls]['precision']
            rec = report[cls]['recall']
            f1 = report[cls]['f1-score']
            print(f"        {cls}: P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")
    
    print(f"\n  📁 Output: {eval_dir}")
    print(f"      - confusion_matrix.png")
    print(f"      - per_class_metrics.png")
    print(f"      - sample_predictions.png")
    print(f"      - classification_report.txt")
    print(f"      - evaluation_results.json")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()


"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> & "C:\Users\Antardip Himel\.conda\envs\slowfast\python.exe" d:/Research/Cancer_Cell_Analysis/scripts/4_7_evaluate.py
======================================================================
  EVALUATION: ResNet-18 + LSTM
  2026-03-12 05:12:29
======================================================================

  Device: cuda

  Loading test dataset...
    Test samples: 904

  Loading trained model...
    Loaded checkpoint from epoch 15
    Validation accuracy was: 75.80%

  Evaluating on test set...

  Test Accuracy: 71.68%

  Per-class accuracy:
    no_cell: 72.67%
    contain_cell: 80.18%
    uncertain_cell: 59.69%

======================================================================
  GENERATING VISUALIZATIONS
======================================================================

  [1] Confusion matrix...
  [2] Per-class metrics...
  [3] Sample predictions...
  [4] Saving classification report...

======================================================================
  ✅ EVALUATION COMPLETE!
======================================================================

  📊 Test Results:
      Overall Accuracy: 71.68%

      Per-class:
        no_cell: P=0.65, R=0.73, F1=0.69
        contain_cell: P=0.82, R=0.80, F1=0.81
        uncertain_cell: P=0.61, R=0.60, F1=0.60

  📁 Output: D:/Research/Cancer_Cell_Analysis/dl_output\evaluation
      - confusion_matrix.png
      - per_class_metrics.png
      - sample_predictions.png
      - classification_report.txt
      - evaluation_results.json

======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""