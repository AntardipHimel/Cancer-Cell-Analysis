# -*- coding: utf-8 -*-
"""
4_6_train.py

TRAINING SCRIPT FOR RESNET + LSTM
=================================

All-in-one training script (no external imports needed).

Run: python -u scripts/4_6_train.py

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

# Paths - use raw strings or forward slashes
DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output"
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

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
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
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
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
    print("  TRAINING: ResNet-18 + LSTM")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
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
        num_classes=3,
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
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
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
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📊 Results:")
    print(f"      Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"      Best Epoch: {best_epoch}")
    print(f"      Total Epochs: {len(train_losses)}")
    print(f"      Total Time: {total_time/60:.1f} minutes")
    
    print(f"\n  📁 Output: {OUTPUT_DIR}")
    print(f"  💾 Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    
    print("\n" + "=" * 70)
    print("  Next: Run 4_7_evaluate.py to evaluate on test set")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> & "C:\Users\Antardip Himel\.conda\envs\slowfast\python.exe" d:/Research/Cancer_Cell_Analysis/scripts/4_6_train.py
======================================================================
  TRAINING: ResNet-18 + LSTM
  2026-03-12 04:08:01
======================================================================

  Device: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  VRAM: 8.0 GB

  Loading datasets...
    Train: 4203 samples
    Val:   901 samples
    Train batches: 525
    Val batches:   113

  Loading class weights...
    Weights: [2.0147552  0.69368434 0.9415452 ]

  Creating model...
    Total parameters: 14,456,259
C:\Users\Antardip Himel\.conda\envs\slowfast\lib\site-packages\torch\optim\lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

======================================================================
  TRAINING
======================================================================

  Epochs: 50
  Batch size: 8
  Learning rate: 0.0001
  Early stopping patience: 10

  Epoch 1/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 1.0331
      Batch 100/525, Loss: 0.6037
      Batch 150/525, Loss: 0.7459
      Batch 200/525, Loss: 0.8865
      Batch 250/525, Loss: 0.8570
      Batch 300/525, Loss: 0.5146
      Batch 350/525, Loss: 0.6173
      Batch 400/525, Loss: 1.2632
      Batch 450/525, Loss: 0.8172
      Batch 500/525, Loss: 0.6413
    Validating...

    Train Loss: 0.7752, Train Acc: 63.90%
    Val Loss:   0.6244, Val Acc:   69.92%
    Time: 381.6s
    ✓ New best model saved! (Val Acc: 69.92%)

  Epoch 2/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 1.1653
      Batch 100/525, Loss: 0.8122
      Batch 150/525, Loss: 0.4597
      Batch 200/525, Loss: 0.7631
      Batch 250/525, Loss: 1.0978
      Batch 300/525, Loss: 0.8659
      Batch 350/525, Loss: 0.4425
      Batch 400/525, Loss: 0.5991
      Batch 450/525, Loss: 0.4715
      Batch 500/525, Loss: 0.4566
    Validating...

    Train Loss: 0.6501, Train Acc: 70.50%
    Val Loss:   0.6614, Val Acc:   70.37%
    Time: 104.2s
    ✓ New best model saved! (Val Acc: 70.37%)

  Epoch 3/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.3364
      Batch 100/525, Loss: 0.3819
      Batch 150/525, Loss: 0.5523
      Batch 200/525, Loss: 0.4706
      Batch 250/525, Loss: 0.4932
      Batch 300/525, Loss: 0.6514
      Batch 350/525, Loss: 0.3104
      Batch 400/525, Loss: 0.7342
      Batch 450/525, Loss: 0.6922
      Batch 500/525, Loss: 0.6857
    Validating...

    Train Loss: 0.5979, Train Acc: 73.07%
    Val Loss:   0.5794, Val Acc:   74.58%
    Time: 81.6s
    ✓ New best model saved! (Val Acc: 74.58%)

  Epoch 4/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.4064
      Batch 100/525, Loss: 0.8087
      Batch 150/525, Loss: 0.8401
      Batch 200/525, Loss: 0.3048
      Batch 250/525, Loss: 0.6663
      Batch 300/525, Loss: 0.6233
      Batch 350/525, Loss: 0.4671
      Batch 400/525, Loss: 0.4061
      Batch 450/525, Loss: 0.3114
      Batch 500/525, Loss: 0.4064
    Validating...

    Train Loss: 0.5446, Train Acc: 75.26%
    Val Loss:   0.6749, Val Acc:   69.59%
    Time: 81.4s
    No improvement (1/10)

  Epoch 5/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.8950
      Batch 100/525, Loss: 0.2220
      Batch 150/525, Loss: 0.8656
      Batch 200/525, Loss: 0.3993
      Batch 250/525, Loss: 0.4551
      Batch 300/525, Loss: 0.6693
      Batch 350/525, Loss: 0.7721
      Batch 400/525, Loss: 0.4034
      Batch 450/525, Loss: 0.4812
      Batch 500/525, Loss: 0.4460
    Validating...

    Train Loss: 0.4987, Train Acc: 78.10%
    Val Loss:   0.6267, Val Acc:   72.81%
    Time: 81.3s
    No improvement (2/10)

  Epoch 6/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.3958
      Batch 100/525, Loss: 0.6237
      Batch 150/525, Loss: 0.2690
      Batch 200/525, Loss: 0.5348
      Batch 250/525, Loss: 1.1676
      Batch 300/525, Loss: 1.1813
      Batch 350/525, Loss: 0.6012
      Batch 400/525, Loss: 0.5546
      Batch 450/525, Loss: 2.4512
      Batch 500/525, Loss: 0.2303
    Validating...

    Train Loss: 0.4729, Train Acc: 79.50%
    Val Loss:   0.5987, Val Acc:   75.14%
    Time: 81.0s
    ✓ New best model saved! (Val Acc: 75.14%)

  Epoch 7/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.1416
      Batch 100/525, Loss: 0.1132
      Batch 150/525, Loss: 0.6184
      Batch 200/525, Loss: 0.2562
      Batch 250/525, Loss: 0.8781
      Batch 300/525, Loss: 0.4550
      Batch 350/525, Loss: 0.5474
      Batch 400/525, Loss: 0.4167
      Batch 450/525, Loss: 0.6097
      Batch 500/525, Loss: 0.3135
    Validating...

    Train Loss: 0.4177, Train Acc: 81.74%
    Val Loss:   0.6619, Val Acc:   73.25%
    Time: 81.1s
    No improvement (1/10)

  Epoch 8/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.4176
      Batch 100/525, Loss: 0.1403
      Batch 150/525, Loss: 0.8460
      Batch 200/525, Loss: 0.2646
      Batch 250/525, Loss: 0.5086
      Batch 300/525, Loss: 0.6483
      Batch 350/525, Loss: 0.4999
      Batch 400/525, Loss: 0.1351
      Batch 450/525, Loss: 1.0573
      Batch 500/525, Loss: 0.1646
    Validating...

    Train Loss: 0.3557, Train Acc: 85.50%
    Val Loss:   0.6236, Val Acc:   74.14%
    Time: 81.0s
    No improvement (2/10)

  Epoch 9/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.3899
      Batch 100/525, Loss: 0.0625
      Batch 150/525, Loss: 0.0880
      Batch 200/525, Loss: 0.2273
      Batch 250/525, Loss: 0.0835
      Batch 300/525, Loss: 0.1547
      Batch 350/525, Loss: 0.1944
      Batch 400/525, Loss: 0.3053
      Batch 450/525, Loss: 0.3339
      Batch 500/525, Loss: 0.3755
    Validating...

    Train Loss: 0.3255, Train Acc: 86.14%
    Val Loss:   0.7211, Val Acc:   71.59%
    Time: 81.6s
    No improvement (3/10)

  Epoch 10/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.1770
      Batch 100/525, Loss: 0.3364
      Batch 150/525, Loss: 0.5684
      Batch 200/525, Loss: 0.2186
      Batch 250/525, Loss: 0.5598
      Batch 300/525, Loss: 0.4412
      Batch 350/525, Loss: 0.3686
      Batch 400/525, Loss: 0.5215
      Batch 450/525, Loss: 0.1562
      Batch 500/525, Loss: 0.3518
    Validating...

    Train Loss: 0.1951, Train Acc: 92.10%
    Val Loss:   0.8613, Val Acc:   73.03%
    Time: 81.6s
    No improvement (4/10)

  Epoch 11/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0068
      Batch 100/525, Loss: 0.0705
      Batch 150/525, Loss: 0.5741
      Batch 200/525, Loss: 0.0913
      Batch 250/525, Loss: 0.0317
      Batch 300/525, Loss: 0.1297
      Batch 350/525, Loss: 0.0216
      Batch 400/525, Loss: 0.0701
      Batch 450/525, Loss: 0.1210
      Batch 500/525, Loss: 0.4699
    Validating...

    Train Loss: 0.1410, Train Acc: 94.57%
    Val Loss:   0.9047, Val Acc:   75.36%
    Time: 81.7s
    ✓ New best model saved! (Val Acc: 75.36%)

  Epoch 12/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0043
      Batch 100/525, Loss: 0.5769
      Batch 150/525, Loss: 0.0375
      Batch 200/525, Loss: 0.1678
      Batch 250/525, Loss: 0.0264
      Batch 300/525, Loss: 0.0407
      Batch 350/525, Loss: 0.0313
      Batch 400/525, Loss: 0.0093
      Batch 450/525, Loss: 0.1769
      Batch 500/525, Loss: 0.0566
    Validating...

    Train Loss: 0.1154, Train Acc: 95.79%
    Val Loss:   0.9438, Val Acc:   74.81%
    Time: 81.7s
    No improvement (1/10)

  Epoch 13/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0743
      Batch 100/525, Loss: 0.0120
      Batch 150/525, Loss: 0.0175
      Batch 200/525, Loss: 0.1330
      Batch 250/525, Loss: 0.0054
      Batch 300/525, Loss: 0.1449
      Batch 350/525, Loss: 0.0246
      Batch 400/525, Loss: 0.0375
      Batch 450/525, Loss: 0.0142
      Batch 500/525, Loss: 0.0283
    Validating...

    Train Loss: 0.1108, Train Acc: 96.43%
    Val Loss:   1.0374, Val Acc:   72.92%
    Time: 81.7s
    No improvement (2/10)

  Epoch 14/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.2057
      Batch 100/525, Loss: 0.1540
      Batch 150/525, Loss: 0.0166
      Batch 200/525, Loss: 0.0164
      Batch 250/525, Loss: 0.1319
      Batch 300/525, Loss: 0.0105
      Batch 350/525, Loss: 0.1952
      Batch 400/525, Loss: 0.0178
      Batch 450/525, Loss: 0.0876
      Batch 500/525, Loss: 0.0300
    Validating...

    Train Loss: 0.0838, Train Acc: 97.00%
    Val Loss:   1.0668, Val Acc:   72.14%
    Time: 81.2s
    No improvement (3/10)

  Epoch 15/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0184
      Batch 100/525, Loss: 0.0235
      Batch 150/525, Loss: 0.0053
      Batch 200/525, Loss: 0.0173
      Batch 250/525, Loss: 0.2082
      Batch 300/525, Loss: 0.0957
      Batch 350/525, Loss: 0.0359
      Batch 400/525, Loss: 0.0302
      Batch 450/525, Loss: 0.0599
      Batch 500/525, Loss: 0.1366
    Validating...

    Train Loss: 0.0754, Train Acc: 97.26%
    Val Loss:   1.1198, Val Acc:   75.80%
    Time: 81.2s
    ✓ New best model saved! (Val Acc: 75.80%)

  Epoch 16/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.1336
      Batch 100/525, Loss: 0.0053
      Batch 150/525, Loss: 0.1902
      Batch 200/525, Loss: 0.0247
      Batch 250/525, Loss: 0.0046
      Batch 300/525, Loss: 0.0055
      Batch 350/525, Loss: 0.5252
      Batch 400/525, Loss: 0.2352
      Batch 450/525, Loss: 0.0074
      Batch 500/525, Loss: 0.0021
    Validating...

    Train Loss: 0.0497, Train Acc: 98.24%
    Val Loss:   1.1588, Val Acc:   74.69%
    Time: 81.4s
    No improvement (1/10)

  Epoch 17/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0041
      Batch 100/525, Loss: 0.0034
      Batch 150/525, Loss: 0.0051
      Batch 200/525, Loss: 0.0552
      Batch 250/525, Loss: 0.1112
      Batch 300/525, Loss: 0.0023
      Batch 350/525, Loss: 0.0105
      Batch 400/525, Loss: 0.0200
      Batch 450/525, Loss: 0.0017
      Batch 500/525, Loss: 0.1670
    Validating...

    Train Loss: 0.0425, Train Acc: 98.57%
    Val Loss:   1.2506, Val Acc:   75.14%
    Time: 81.4s
    No improvement (2/10)

  Epoch 18/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0046
      Batch 100/525, Loss: 0.0017
      Batch 150/525, Loss: 0.0496
      Batch 200/525, Loss: 0.0078
      Batch 250/525, Loss: 0.0262
      Batch 300/525, Loss: 0.1651
      Batch 350/525, Loss: 0.0079
      Batch 400/525, Loss: 0.0034
      Batch 450/525, Loss: 0.0014
      Batch 500/525, Loss: 0.0845
    Validating...

    Train Loss: 0.0392, Train Acc: 98.52%
    Val Loss:   1.2899, Val Acc:   73.58%
    Time: 81.6s
    No improvement (3/10)

  Epoch 19/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0019
      Batch 100/525, Loss: 0.0020
      Batch 150/525, Loss: 0.0008
      Batch 200/525, Loss: 0.0047
      Batch 250/525, Loss: 0.0034
      Batch 300/525, Loss: 0.0028
      Batch 350/525, Loss: 0.0028
      Batch 400/525, Loss: 0.0039
      Batch 450/525, Loss: 0.0050
      Batch 500/525, Loss: 0.0026
    Validating...

    Train Loss: 0.0311, Train Acc: 98.86%
    Val Loss:   1.2126, Val Acc:   75.36%
    Time: 81.6s
    No improvement (4/10)

  Epoch 20/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0017
      Batch 100/525, Loss: 0.0145
      Batch 150/525, Loss: 0.0014
      Batch 200/525, Loss: 0.0081
      Batch 250/525, Loss: 0.0018
      Batch 300/525, Loss: 0.3041
      Batch 350/525, Loss: 0.2232
      Batch 400/525, Loss: 0.0053
      Batch 450/525, Loss: 0.0052
      Batch 500/525, Loss: 0.0019
    Validating...

    Train Loss: 0.0338, Train Acc: 98.64%
    Val Loss:   1.2747, Val Acc:   74.69%
    Time: 81.6s
    No improvement (5/10)

  Epoch 21/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0112
      Batch 100/525, Loss: 0.0032
      Batch 150/525, Loss: 0.0021
      Batch 200/525, Loss: 0.0049
      Batch 250/525, Loss: 0.0016
      Batch 300/525, Loss: 0.0036
      Batch 350/525, Loss: 0.0154
      Batch 400/525, Loss: 0.3522
      Batch 450/525, Loss: 0.0036
      Batch 500/525, Loss: 0.0611
    Validating...

    Train Loss: 0.0445, Train Acc: 98.36%
    Val Loss:   1.2546, Val Acc:   74.69%
    Time: 81.7s
    No improvement (6/10)

  Epoch 22/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0179
      Batch 100/525, Loss: 0.0009
      Batch 150/525, Loss: 0.0767
      Batch 200/525, Loss: 0.0022
      Batch 250/525, Loss: 0.0039
      Batch 300/525, Loss: 0.0444
      Batch 350/525, Loss: 0.0054
      Batch 400/525, Loss: 0.0362
      Batch 450/525, Loss: 0.0144
      Batch 500/525, Loss: 0.0064
    Validating...

    Train Loss: 0.0314, Train Acc: 98.74%
    Val Loss:   1.3152, Val Acc:   73.58%
    Time: 81.7s
    No improvement (7/10)

  Epoch 23/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0026
      Batch 100/525, Loss: 0.0192
      Batch 150/525, Loss: 0.0005
      Batch 200/525, Loss: 0.0023
      Batch 250/525, Loss: 0.1202
      Batch 300/525, Loss: 0.0066
      Batch 350/525, Loss: 0.0012
      Batch 400/525, Loss: 0.0019
      Batch 450/525, Loss: 0.0023
      Batch 500/525, Loss: 0.0089
    Validating...

    Train Loss: 0.0252, Train Acc: 98.93%
    Val Loss:   1.3028, Val Acc:   74.47%
    Time: 81.2s
    No improvement (8/10)

  Epoch 24/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0017
      Batch 100/525, Loss: 0.0036
      Batch 150/525, Loss: 0.0034
      Batch 200/525, Loss: 0.0308
      Batch 250/525, Loss: 0.0005
      Batch 300/525, Loss: 0.0018
      Batch 350/525, Loss: 0.0059
      Batch 400/525, Loss: 0.0023
      Batch 450/525, Loss: 0.0348
      Batch 500/525, Loss: 0.1410
    Validating...

    Train Loss: 0.0224, Train Acc: 98.95%
    Val Loss:   1.3692, Val Acc:   73.36%
    Time: 81.4s
    No improvement (9/10)

  Epoch 25/50
  --------------------------------------------------
    Training...
      Batch 50/525, Loss: 0.0006
      Batch 100/525, Loss: 0.1144
      Batch 150/525, Loss: 0.0008
      Batch 200/525, Loss: 0.0006
      Batch 250/525, Loss: 0.0043
      Batch 300/525, Loss: 0.0025
      Batch 350/525, Loss: 0.0005
      Batch 400/525, Loss: 0.0034
      Batch 450/525, Loss: 0.0022
      Batch 500/525, Loss: 0.1814
    Validating...

    Train Loss: 0.0250, Train Acc: 98.98%
    Val Loss:   1.3505, Val Acc:   74.69%
    Time: 81.3s
    No improvement (10/10)

  ⚠️ Early stopping at epoch 25

======================================================================
  SAVING RESULTS
======================================================================

  ✓ Saved: training_curves.png
  ✓ Saved: training_history.json

======================================================================
  ✅ TRAINING COMPLETE!
======================================================================

  📊 Results:
      Best Val Accuracy: 75.80%
      Best Epoch: 15
      Total Epochs: 25
      Total Time: 39.7 minutes

  📁 Output: D:/Research/Cancer_Cell_Analysis/dl_output
  💾 Best model: D:/Research/Cancer_Cell_Analysis/dl_output\checkpoints\best_model.pt

======================================================================
  Next: Run 4_7_evaluate.py to evaluate on test set
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""