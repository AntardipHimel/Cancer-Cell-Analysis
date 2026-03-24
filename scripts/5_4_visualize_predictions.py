# -*- coding: utf-8 -*-
"""
5_4_visualize_predictions.py

PREDICTION VISUALIZATION - SEPARATE BY CLASS
==============================================

Creates separate visualizations for:
1. contain_cell predictions (correct & wrong)
2. no_cell predictions (correct & wrong)

Test set only.

Run: python -u scripts/5_4_visualize_predictions.py

Author: Antardip Himel
Date: March 2026
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_2class"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

NUM_FRAMES = 30
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['no_cell', 'contain_cell']


# =============================================================================
# DATASET CLASS
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
            frame = frame[np.newaxis, :, :]
            frames.append(frame)
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        
        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames), label, idx


# =============================================================================
# MODEL
# =============================================================================

class ResNetLSTM(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=False):
        super(ResNetLSTM, self).__init__()
        
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
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
        
        composite = np.hstack([frames[fi, 0].numpy() for fi in frame_indices])
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
        
        composite = np.hstack([frames[fi, 0].numpy() for fi in frame_indices])
        ax.imshow(composite, cmap='gray')
        ax.set_title(f"Predicted: {CLASS_NAMES[sample['pred']]} (WRONG!) | Confidence: {sample['conf']*100:.1f}%",
                    fontsize=11, color='red')
        ax.set_ylabel(f"#{i+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1
    
    # Main title
    accuracy = len(correct) / len(class_results) * 100 if class_results else 0
    plt.suptitle(f"CLASS: {class_name.upper()}\n"
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
    print("  PREDICTION VISUALIZATION - SEPARATE BY CLASS")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"\n  Device: {DEVICE}")
    print(f"  Output: {VIS_DIR}")
    
    # Load test dataset
    print("\n  Loading test dataset...", flush=True)
    test_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'test'))
    print(f"    Test samples: {len(test_dataset)}")
    
    # Load model
    print("\n  Loading model...", flush=True)
    model = ResNetLSTM(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
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
    print("  ✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📁 Output: {VIS_DIR}")
    print(f"      - contain_cell_predictions.png")
    print(f"      - no_cell_predictions.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()