# -*- coding: utf-8 -*-
"""
7_2_train_good.py

TRAINING: GOOD vs NOT_GOOD
===========================

ResNet+LSTM for 2-class classification:
  - good (1):     contain_cell
  - not_good (0): no_cell + uncertain_cell

INPUT:  D:/Research/Cancer_Cell_Analysis/dataset_good_vs_notgood/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dl_output_good_vs_notgood/

Run: python -u scripts/7_2_train_good.py

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

DATASET_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_good_vs_notgood"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dl_output_good_vs_notgood"
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
NUM_CLASSES = 2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
CLASS_NAMES = ['not_good', 'good']


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


# =============================================================================
# MODEL
# =============================================================================

class ResNetLSTM(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=True):
        super(ResNetLSTM, self).__init__()
        
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
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
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
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
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
            print(f"      Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
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
    
    return running_loss / len(val_loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss\n(good vs not_good)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy\n(good vs not_good)', fontsize=14, fontweight='bold')
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
    print("  TRAINING: GOOD vs NOT_GOOD (ResNet+LSTM)")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"    good     = contain_cell")
    print(f"    not_good = no_cell + uncertain_cell")
    print(f"\n  Dataset: {DATASET_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n  Device: {DEVICE}", flush=True)
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    # Load datasets
    print("\n  Loading datasets...", flush=True)
    train_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'train'))
    val_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'val'))
    
    print(f"    Train: {len(train_dataset)} samples", flush=True)
    print(f"    Val:   {len(val_dataset)} samples", flush=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"    Train batches: {len(train_loader)}", flush=True)
    print(f"    Val batches:   {len(val_loader)}", flush=True)
    
    # Load class weights
    print("\n  Loading class weights...", flush=True)
    weights_path = os.path.join(DATASET_DIR, 'class_weights.pt')
    class_weights = torch.load(weights_path, weights_only=True).to(DEVICE)
    print(f"    Weights: {class_weights.cpu().numpy()}", flush=True)
    
    # Create model
    print("\n  Creating model...", flush=True)
    model = ResNetLSTM(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE,
                       num_layers=NUM_LAYERS, dropout=DROPOUT, pretrained=True).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}", flush=True)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)
    print(f"\n  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}", flush=True)
        print(f"  {'-'*50}", flush=True)
        
        print("    Training...", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        print("    Validating...", flush=True)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\n    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%", flush=True)
        print(f"    Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, 
                          os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"    ✓ New best model saved! (Val Acc: {val_acc:.2f}%)", flush=True)
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{PATIENCE})", flush=True)
        
        save_checkpoint(model, optimizer, epoch, val_acc,
                       os.path.join(CHECKPOINT_DIR, 'latest_model.pt'))
        
        if patience_counter >= PATIENCE:
            print(f"\n  ⚠️ Early stopping at epoch {epoch+1}", flush=True)
            break
    
    total_time = time.time() - start_time
    
    # Save results
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        os.path.join(OUTPUT_DIR, 'training_curves.png'))
    print(f"\n  ✓ Saved: training_curves.png", flush=True)
    
    history = {
        'experiment': 'good_vs_notgood',
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
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n  📊 Results (good vs not_good):")
    print(f"      Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"      Best Epoch: {best_epoch}")
    print(f"      Total Time: {total_time/60:.1f} minutes")
    
    print(f"\n  📁 Output: {OUTPUT_DIR}")
    print(f"  💾 Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    
    print("\n" + "=" * 70)
    print("  Next: Run 7_3_evaluate_good.py")
    print("=" * 70)


if __name__ == "__main__":
    main()