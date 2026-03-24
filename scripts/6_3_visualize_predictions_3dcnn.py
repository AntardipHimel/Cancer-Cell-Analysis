# -*- coding: utf-8 -*-
"""
6_3_visualize_predictions_3dcnn.py

PREDICTION VISUALIZATION FOR 3D CNN - SEPARATE BY CLASS
=========================================================

Creates separate visualizations for:
1. contain_cell predictions (correct & wrong)
2. no_cell predictions (correct & wrong)

Test set only.

Run: python -u scripts/6_3_visualize_predictions_3dcnn.py

Author: Antardip Himel
Date: March 2026
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_3dcnn"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

NUM_FRAMES = 30
NUM_CLASSES = 2
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['no_cell', 'contain_cell']


# =============================================================================
# DATASET CLASS (3D format)
# =============================================================================

class LensSequenceDataset3D(Dataset):
    
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
                        'lens_name': lens_name,
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
        
        # 3D format: (C, D, H, W) = (1, 30, 96, 96)
        frames = np.stack(frames, axis=0)
        frames = frames[np.newaxis, :, :, :]
        
        return torch.from_numpy(frames), label, idx


# =============================================================================
# 3D CNN MODEL
# =============================================================================

class ResBlock3D(nn.Module):
    
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D18(nn.Module):
    
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_predictions(model, dataset, device):
    """Get all predictions."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            frames, label, _ = dataset[idx]
            output = model(frames.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()
            
            results.append({
                'idx': idx,
                'label': label,
                'pred': pred,
                'conf': probs[pred].item(),
                'correct': label == pred
            })
    
    return results


def plot_class_samples(dataset, results, class_idx, class_name, output_path):
    """
    Plot samples for ONE class showing correct and wrong predictions.
    Shows 6 correct + 6 wrong (or as many as available).
    Each sample shows 6 frames to visualize motion.
    """
    
    # Filter for this class
    class_results = [r for r in results if r['label'] == class_idx]
    correct = [r for r in class_results if r['correct']]
    wrong = [r for r in class_results if not r['correct']]
    
    random.seed(42)
    correct_samples = random.sample(correct, min(6, len(correct)))
    wrong_samples = random.sample(wrong, min(6, len(wrong)))
    
    # Frames to show
    frame_indices = [0, 6, 12, 18, 24, 29]
    
    # Create figure
    n_correct = len(correct_samples)
    n_wrong = len(wrong_samples)
    total_rows = n_correct + n_wrong + 2  # +2 for headers
    
    fig = plt.figure(figsize=(18, 2.5 * (n_correct + n_wrong + 1)))
    
    row = 0
    
    # Header for correct
    ax_header = fig.add_subplot(total_rows, 1, row + 1)
    ax_header.text(0.5, 0.5, f"✓ CORRECT PREDICTIONS ({len(correct)} total)", 
                   ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    ax_header.axis('off')
    row += 1
    
    # Plot correct samples
    for i, sample in enumerate(correct_samples):
        ax = fig.add_subplot(total_rows, 1, row + 1)
        frames, _, _ = dataset[sample['idx']]
        
        # 3D format: (1, 30, 96, 96) - extract frames
        composite = np.hstack([frames[0, fi].numpy() for fi in frame_indices])
        ax.imshow(composite, cmap='gray')
        ax.set_title(f"Predicted: {CLASS_NAMES[sample['pred']]} | Confidence: {sample['conf']*100:.1f}%",
                    fontsize=11, color='green')
        ax.set_ylabel(f"#{i+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1
    
    # Header for wrong
    ax_header = fig.add_subplot(total_rows, 1, row + 1)
    if n_wrong > 0:
        ax_header.text(0.5, 0.5, f"✗ WRONG PREDICTIONS ({len(wrong)} total)", 
                      ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    else:
        ax_header.text(0.5, 0.5, f"✓ NO WRONG PREDICTIONS!", 
                      ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    ax_header.axis('off')
    row += 1
    
    # Plot wrong samples
    for i, sample in enumerate(wrong_samples):
        ax = fig.add_subplot(total_rows, 1, row + 1)
        frames, _, _ = dataset[sample['idx']]
        
        composite = np.hstack([frames[0, fi].numpy() for fi in frame_indices])
        ax.imshow(composite, cmap='gray')
        ax.set_title(f"Predicted: {CLASS_NAMES[sample['pred']]} (WRONG!) | Confidence: {sample['conf']*100:.1f}%",
                    fontsize=11, color='red')
        ax.set_ylabel(f"#{i+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1
    
    # Main title
    accuracy = len(correct) / len(class_results) * 100 if class_results else 0
    plt.suptitle(f"3D CNN - CLASS: {class_name.upper()}\n"
                 f"Total: {len(class_results)} | Correct: {len(correct)} | Wrong: {len(wrong)} | "
                 f"Accuracy: {accuracy:.1f}%\n"
                 f"(Showing frames 1, 7, 13, 19, 25, 30 per sample)",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return len(correct), len(wrong)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  3D CNN - PREDICTION VISUALIZATION - SEPARATE BY CLASS")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"\n  Device: {DEVICE}")
    print(f"  Output: {VIS_DIR}")
    
    # Load test dataset
    print("\n  Loading test dataset...", flush=True)
    test_dataset = LensSequenceDataset3D(os.path.join(DATASET_DIR, 'test'))
    print(f"    Test samples: {len(test_dataset)}")
    
    # Load model
    print("\n  Loading 3D CNN model...", flush=True)
    model = ResNet3D18(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded from epoch {checkpoint['epoch']+1}")
    
    # Get predictions
    print("\n  Getting predictions...", flush=True)
    results = get_predictions(model, test_dataset, DEVICE)
    
    correct_total = sum(1 for r in results if r['correct'])
    print(f"    Total: {len(results)}, Correct: {correct_total}, Wrong: {len(results) - correct_total}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. contain_cell predictions
    print("\n  [1] contain_cell predictions...", flush=True)
    c_correct, c_wrong = plot_class_samples(
        test_dataset, results, 
        class_idx=1, class_name='contain_cell',
        output_path=os.path.join(VIS_DIR, 'contain_cell_predictions.png')
    )
    print(f"      Correct: {c_correct}, Wrong: {c_wrong}")
    
    # 2. no_cell predictions
    print("\n  [2] no_cell predictions...", flush=True)
    n_correct, n_wrong = plot_class_samples(
        test_dataset, results,
        class_idx=0, class_name='no_cell',
        output_path=os.path.join(VIS_DIR, 'no_cell_predictions.png')
    )
    print(f"      Correct: {n_correct}, Wrong: {n_wrong}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ 3D CNN VISUALIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📁 Output: {VIS_DIR}")
    print(f"      - contain_cell_predictions.png")
    print(f"      - no_cell_predictions.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()