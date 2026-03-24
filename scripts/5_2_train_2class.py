# -*- coding: utf-8 -*-
"""
5_2_train_2class.py

TRAINING SCRIPT FOR 2-CLASS CLASSIFICATION
===========================================

ResNet-18 + LSTM for binary classification:
  0 = no_cell
  1 = contain_cell

INPUT:  D:/Research/Cancer_Cell_Analysis/dataset_2class/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dl_output_2class/

Run: python -u scripts/5_2_train_2class.py

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
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_2class"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Early stopping
PATIENCE = 10

# Model parameters
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_FRAMES = 30
NUM_CLASSES = 2  # Changed from 3 to 2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes (2 only)
CLASS_NAMES = ['no_cell', 'contain_cell']


# =============================================================================
# DATASET CLASS
# =============================================================================

class LensSequenceDataset(Dataset):
    """PyTorch Dataset for 2-class cell lens sequences."""
    
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
        frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, label


# =============================================================================
# MODEL
# =============================================================================

class ResNetLSTM(nn.Module):
    """ResNet + LSTM for video classification (2 classes)."""
    
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=True):
        super(ResNetLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ResNet backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # Modify for grayscale
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
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
        
        # Classifier (2 classes)
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
    axes[0].set_title('Training & Validation Loss (2-Class)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy (2-Class)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  TRAINING: ResNet-18 + LSTM (2-CLASS)")
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
    train_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'train'))
    val_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'val'))
    
    print(f"    Train: {len(train_dataset)} samples", flush=True)
    print(f"    Val:   {len(val_dataset)} samples", flush=True)
    
    # Create dataloaders
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
    print("\n  Creating model...", flush=True)
    model = ResNetLSTM(
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}", flush=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING (2-CLASS)")
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
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
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
    print("  ✅ TRAINING COMPLETE (2-CLASS)!")
    print("=" * 70)
    
    print(f"\n  📊 Results:")
    print(f"      Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"      Best Epoch: {best_epoch}")
    print(f"      Total Epochs: {len(train_losses)}")
    print(f"      Total Time: {total_time/60:.1f} minutes")
    
    print(f"\n  📁 Output: {OUTPUT_DIR}")
    print(f"  💾 Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    
    print("\n" + "=" * 70)
    print("  Next: Run 5_3_evaluate_2class.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
python -u scripts/5_2_train_2class.pyAnalysis>
======================================================================
  TRAINING: ResNet-18 + LSTM (2-CLASS)
  2026-03-13 04:53:50
======================================================================

  Classes: ['no_cell', 'contain_cell']
  Dataset: D:/Research/Cancer_Cell_Analysis/dataset_2class
  Output:  D:/Research/Cancer_Cell_Analysis/dl_output_2class

  Device: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  VRAM: 8.0 GB

  Loading datasets...
    Train: 2715 samples
    Val:   582 samples
    Train batches: 339
    Val batches:   73

  Loading class weights...
    Weights: [1.9522133 0.672151 ]

  Creating model...
    Total parameters: 14,456,002
C:\Users\Antardip Himel\.conda\envs\slowfast\lib\site-packages\torch\optim\lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

======================================================================
  TRAINING (2-CLASS)
======================================================================

  Epochs: 50
  Batch size: 8
  Learning rate: 0.0001
  Early stopping patience: 10

  Epoch 1/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.3929
      Batch 100/339, Loss: 0.0842
      Batch 150/339, Loss: 0.2702
      Batch 200/339, Loss: 0.3407
      Batch 250/339, Loss: 0.3245
      Batch 300/339, Loss: 0.0807
    Validating...

    Train Loss: 0.3126, Train Acc: 88.35%
    Val Loss:   0.2418, Val Acc:   90.72%
    Time: 198.1s
    ✓ New best model saved! (Val Acc: 90.72%)

  Epoch 2/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.2895
      Batch 100/339, Loss: 0.1546
      Batch 150/339, Loss: 0.0708
      Batch 200/339, Loss: 0.3119
      Batch 250/339, Loss: 0.0371
      Batch 300/339, Loss: 0.4045
    Validating...

    Train Loss: 0.1815, Train Acc: 93.29%
    Val Loss:   0.1050, Val Acc:   95.88%
    Time: 198.9s
    ✓ New best model saved! (Val Acc: 95.88%)

  Epoch 3/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0447
      Batch 100/339, Loss: 0.0275
      Batch 150/339, Loss: 0.0766
      Batch 200/339, Loss: 0.0293
      Batch 250/339, Loss: 0.0245
      Batch 300/339, Loss: 0.0187
    Validating...

    Train Loss: 0.1469, Train Acc: 95.10%
    Val Loss:   0.2958, Val Acc:   91.07%
    Time: 163.9s
    No improvement (1/10)

  Epoch 4/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0318
      Batch 100/339, Loss: 0.0507
      Batch 150/339, Loss: 0.1063
      Batch 200/339, Loss: 0.5432
      Batch 250/339, Loss: 0.1334
      Batch 300/339, Loss: 0.0299
    Validating...

    Train Loss: 0.1010, Train Acc: 96.72%
    Val Loss:   0.1350, Val Acc:   95.19%
    Time: 67.8s
    No improvement (2/10)

  Epoch 5/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0255
      Batch 100/339, Loss: 0.0054
      Batch 150/339, Loss: 0.0140
      Batch 200/339, Loss: 0.0063
      Batch 250/339, Loss: 0.0150
      Batch 300/339, Loss: 0.0168
    Validating...

    Train Loss: 0.0890, Train Acc: 97.49%
    Val Loss:   0.2941, Val Acc:   92.61%
    Time: 68.0s
    No improvement (3/10)

  Epoch 6/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.2821
      Batch 100/339, Loss: 0.0083
      Batch 150/339, Loss: 0.0052
      Batch 200/339, Loss: 0.0338
      Batch 250/339, Loss: 0.0149
      Batch 300/339, Loss: 0.0151
    Validating...

    Train Loss: 0.0823, Train Acc: 97.64%
    Val Loss:   0.2279, Val Acc:   94.16%
    Time: 68.1s
    No improvement (4/10)

  Epoch 7/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0266
      Batch 100/339, Loss: 0.0083
      Batch 150/339, Loss: 0.0164
      Batch 200/339, Loss: 0.1958
      Batch 250/339, Loss: 0.0150
      Batch 300/339, Loss: 0.0051
    Validating...

    Train Loss: 0.0663, Train Acc: 97.68%
    Val Loss:   0.0904, Val Acc:   97.08%
    Time: 67.8s
    ✓ New best model saved! (Val Acc: 97.08%)

  Epoch 8/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0021
      Batch 100/339, Loss: 0.3074
      Batch 150/339, Loss: 0.0020
      Batch 200/339, Loss: 0.0940
      Batch 250/339, Loss: 0.0009
      Batch 300/339, Loss: 0.1654
    Validating...

    Train Loss: 0.0463, Train Acc: 98.23%
    Val Loss:   0.1706, Val Acc:   93.47%
    Time: 68.1s
    No improvement (1/10)

  Epoch 9/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0319
      Batch 100/339, Loss: 0.0276
      Batch 150/339, Loss: 0.0028
      Batch 200/339, Loss: 0.0027
      Batch 250/339, Loss: 0.0048
      Batch 300/339, Loss: 0.0102
    Validating...

    Train Loss: 0.0735, Train Acc: 97.64%
    Val Loss:   0.2083, Val Acc:   94.67%
    Time: 68.1s
    No improvement (2/10)

  Epoch 10/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0048
      Batch 100/339, Loss: 0.0035
      Batch 150/339, Loss: 0.0047
      Batch 200/339, Loss: 0.0028
      Batch 250/339, Loss: 0.0183
      Batch 300/339, Loss: 0.0045
    Validating...

    Train Loss: 0.0439, Train Acc: 98.75%
    Val Loss:   0.1581, Val Acc:   94.16%
    Time: 67.8s
    No improvement (3/10)

  Epoch 11/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0007
      Batch 100/339, Loss: 0.0009
      Batch 150/339, Loss: 0.0084
      Batch 200/339, Loss: 0.0097
      Batch 250/339, Loss: 0.0007
      Batch 300/339, Loss: 0.0130
    Validating...

    Train Loss: 0.0639, Train Acc: 98.30%
    Val Loss:   0.2484, Val Acc:   94.67%
    Time: 67.8s
    No improvement (4/10)

  Epoch 12/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0913
      Batch 100/339, Loss: 0.4334
      Batch 150/339, Loss: 0.0010
      Batch 200/339, Loss: 0.0191
      Batch 250/339, Loss: 0.0153
      Batch 300/339, Loss: 0.0047
    Validating...

    Train Loss: 0.0485, Train Acc: 98.30%
    Val Loss:   0.1347, Val Acc:   94.85%
    Time: 67.8s
    No improvement (5/10)

  Epoch 13/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0017
      Batch 100/339, Loss: 0.0012
      Batch 150/339, Loss: 0.0020
      Batch 200/339, Loss: 0.0053
      Batch 250/339, Loss: 0.0361
      Batch 300/339, Loss: 0.0009
    Validating...

    Train Loss: 0.0322, Train Acc: 98.97%
    Val Loss:   0.1954, Val Acc:   95.19%
    Time: 68.2s
    No improvement (6/10)

  Epoch 14/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0198
      Batch 100/339, Loss: 0.0015
      Batch 150/339, Loss: 0.0032
      Batch 200/339, Loss: 0.0020
      Batch 250/339, Loss: 0.0014
      Batch 300/339, Loss: 0.0004
    Validating...

    Train Loss: 0.0213, Train Acc: 99.37%
    Val Loss:   0.1695, Val Acc:   95.19%
    Time: 67.9s
    No improvement (7/10)

  Epoch 15/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0617
      Batch 100/339, Loss: 0.0008
      Batch 150/339, Loss: 0.0005
      Batch 200/339, Loss: 0.0001
      Batch 250/339, Loss: 0.0007
      Batch 300/339, Loss: 0.0024
    Validating...

    Train Loss: 0.0091, Train Acc: 99.78%
    Val Loss:   0.1868, Val Acc:   95.19%
    Time: 67.9s
    No improvement (8/10)

  Epoch 16/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0119
      Batch 100/339, Loss: 0.0015
      Batch 150/339, Loss: 0.0023
      Batch 200/339, Loss: 0.0187
      Batch 250/339, Loss: 0.0005
      Batch 300/339, Loss: 0.0340
    Validating...

    Train Loss: 0.0092, Train Acc: 99.78%
    Val Loss:   0.1852, Val Acc:   95.53%
    Time: 70.2s
    No improvement (9/10)

  Epoch 17/50
  --------------------------------------------------
    Training...
      Batch 50/339, Loss: 0.0001
      Batch 100/339, Loss: 0.0004
      Batch 150/339, Loss: 0.0001
      Batch 200/339, Loss: 0.0001
      Batch 250/339, Loss: 0.0208
      Batch 300/339, Loss: 0.0002
    Validating...

    Train Loss: 0.0042, Train Acc: 99.89%
    Val Loss:   0.2606, Val Acc:   95.19%
    Time: 68.0s
    No improvement (10/10)

  ⚠️ Early stopping at epoch 17

======================================================================
  SAVING RESULTS
======================================================================

  ✓ Saved: training_curves.png
  ✓ Saved: training_history.json

======================================================================
  ✅ TRAINING COMPLETE (2-CLASS)!
======================================================================

  📊 Results:
      Best Val Accuracy: 97.08%
      Best Epoch: 7
      Total Epochs: 17
      Total Time: 25.5 minutes

  📁 Output: D:/Research/Cancer_Cell_Analysis/dl_output_2class
  💾 Best model: D:/Research/Cancer_Cell_Analysis/dl_output_2class\checkpoints\best_model.pt

======================================================================
  Next: Run 5_3_evaluate_2class.py
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> """