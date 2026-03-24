# -*- coding: utf-8 -*-
"""
6_1_train_3dcnn.py

3D CNN TRAINING FOR 2-CLASS CLASSIFICATION
===========================================

Uses 3D ResNet-18 which learns spatial and temporal features together.

Architecture:
- Input: (batch, 1, 30, 96, 96) - 1 channel, 30 frames, 96x96
- 3D Convolutions throughout
- Output: 2 classes (no_cell, contain_cell)

INPUT:  D:/Research/Cancer_Cell_Analysis/dataset_2class/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dl_output_3dcnn/

Run: python -u scripts/6_1_train_3dcnn.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_3dcnn"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Training hyperparameters
BATCH_SIZE = 4          # Smaller batch size (3D CNN uses more memory)
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Early stopping
PATIENCE = 10

# Model parameters
NUM_FRAMES = 30
NUM_CLASSES = 2
DROPOUT = 0.5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
CLASS_NAMES = ['no_cell', 'contain_cell']


# =============================================================================
# DATASET CLASS
# =============================================================================

class LensSequenceDataset3D(Dataset):
    """
    PyTorch Dataset for 3D CNN.
    
    Returns frames in shape (C, D, H, W) = (1, 30, 96, 96)
    where D (depth) is the temporal dimension.
    """
    
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
        
        # Pad if needed
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        
        # Stack: (D, H, W) = (30, 96, 96)
        frames = np.stack(frames, axis=0)
        
        # Add channel dimension: (C, D, H, W) = (1, 30, 96, 96)
        frames = frames[np.newaxis, :, :, :]
        
        frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, label


# =============================================================================
# 3D CNN MODEL
# =============================================================================

class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


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
    """
    3D ResNet-18 for video classification.
    
    Architecture:
    ┌─────────────────────────────────────────────┐
    │  INPUT: (batch, 1, 30, 96, 96)              │
    │  (channels, depth/time, height, width)      │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  Conv3D: 1 → 64 channels                    │
    │  kernel: (3, 7, 7), stride: (1, 2, 2)       │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  4 Residual Layers (64 → 128 → 256 → 512)   │
    │  Progressive spatial and temporal pooling   │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  Global Average Pooling 3D                  │
    │  (batch, 512, T, H, W) → (batch, 512)       │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  FC: 512 → 256 → num_classes                │
    └─────────────────────────────────────────────┘
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super(ResNet3D18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        # Input: (1, 30, 96, 96) → Output: (64, 30, 48, 48)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), 
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling: (64, 30, 48, 48) → (64, 15, 24, 24)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)   # (64, 15, 24, 24)
        self.layer2 = self._make_layer(128, 2, stride=2)  # (128, 8, 12, 12)
        self.layer3 = self._make_layer(256, 2, stride=2)  # (256, 4, 6, 6)
        self.layer4 = self._make_layer(512, 2, stride=2)  # (512, 2, 3, 3)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, 1, 30, 96, 96)
        
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
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (frames, labels) in enumerate(train_loader):
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"      Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}", flush=True)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss (3D CNN)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy (3D CNN)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  TRAINING: 3D CNN (ResNet3D-18)")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n  Device: {DEVICE}", flush=True)
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    
    # Load datasets
    print("\n  Loading datasets...", flush=True)
    train_dataset = LensSequenceDataset3D(os.path.join(DATASET_DIR, 'train'))
    val_dataset = LensSequenceDataset3D(os.path.join(DATASET_DIR, 'val'))
    
    print(f"    Train: {len(train_dataset)} samples", flush=True)
    print(f"    Val:   {len(val_dataset)} samples", flush=True)
    
    # Create dataloaders (smaller batch size for 3D CNN)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"    Train batches: {len(train_loader)}", flush=True)
    print(f"    Val batches:   {len(val_loader)}", flush=True)
    
    # Load class weights
    print("\n  Loading class weights...", flush=True)
    weights_path = os.path.join(DATASET_DIR, 'class_weights.pt')
    class_weights = torch.load(weights_path, weights_only=True).to(DEVICE)
    print(f"    Weights: {class_weights.cpu().numpy()}", flush=True)
    
    # Create model
    print("\n  Creating 3D CNN model...", flush=True)
    model = ResNet3D18(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    
    total_params = count_parameters(model)
    print(f"    Total parameters: {total_params:,}", flush=True)
    
    # Test forward pass
    print("    Testing forward pass...", flush=True)
    with torch.no_grad():
        dummy = torch.randn(1, 1, 30, 96, 96).to(DEVICE)
        out = model(dummy)
        print(f"    Input: {dummy.shape} → Output: {out.shape}", flush=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING (3D CNN)")
    print("=" * 70)
    print(f"\n  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {PATIENCE}")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}", flush=True)
        print(f"  {'-'*50}", flush=True)
        
        # Train
        print("    Training...", flush=True)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        print("    Validating...", flush=True)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%", flush=True)
        print(f"    Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%", flush=True)
        print(f"    Time: {epoch_time:.1f}s", flush=True)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch,
                val_acc, os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            )
            print(f"    ✓ New best model saved! (Val Acc: {val_acc:.2f}%)", flush=True)
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{PATIENCE})", flush=True)
        
        # Save latest model
        save_checkpoint(
            model, optimizer, epoch,
            val_acc, os.path.join(CHECKPOINT_DIR, 'latest_model.pt')
        )
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n  ⚠️ Early stopping at epoch {epoch+1}", flush=True)
            break
    
    total_time = time.time() - start_time
    
    # Save results
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(OUTPUT_DIR, 'training_curves.png')
    )
    print(f"\n  ✓ Saved: training_curves.png", flush=True)
    
    history = {
        'model': '3D CNN (ResNet3D-18)',
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'total_parameters': total_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'total_time_seconds': total_time,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Saved: training_history.json", flush=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ TRAINING COMPLETE (3D CNN)!")
    print("=" * 70)
    
    print(f"\n  📊 Results:")
    print(f"      Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"      Best Epoch: {best_epoch}")
    print(f"      Total Epochs: {len(train_losses)}")
    print(f"      Total Time: {total_time/60:.1f} minutes")
    print(f"      Parameters: {total_params:,}")
    
    print(f"\n  📁 Output: {OUTPUT_DIR}")
    print(f"  💾 Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON: ResNet+LSTM vs 3D CNN")
    print("=" * 70)
    print(f"\n      ResNet+LSTM Val Accuracy: 97.08%")
    print(f"      3D CNN Val Accuracy:      {best_val_acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("  Next: Run 6_2_evaluate_3dcnn.py")
    print("=" * 70)


if __name__ == "__main__":
    main()