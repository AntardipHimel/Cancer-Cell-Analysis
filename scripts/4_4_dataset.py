# -*- coding: utf-8 -*-
"""
4_4_dataset.py

PYTORCH DATASET FOR CELL CLASSIFICATION
=======================================

This module provides:
- LensSequenceDataset: Loads 30 frames per lens as a sequence
- get_dataloaders(): Returns train/val/test dataloaders

Each sample:
- Input:  (30, 1, 96, 96) tensor - 30 grayscale frames
- Output: class label (0=no_cell, 1=contain_cell, 2=uncertain_cell)

Usage:
    from dataset import LensSequenceDataset, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

Author: Antardip Himel
Date: March 2026
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = r"D:\Research\Cancer_Cell_Analysis\dataset"
NUM_FRAMES = 30
IMAGE_SIZE = (96, 96)


# =============================================================================
# DATASET CLASS
# =============================================================================

class LensSequenceDataset(Dataset):
    """
    PyTorch Dataset for cell lens sequences.
    
    Each lens contains 30 frames that form a temporal sequence.
    The model will learn to classify based on motion patterns.
    
    Args:
        root_dir: Path to dataset split (train/val/test folder)
        transform: Optional transform to apply to each frame
        max_frames: Maximum number of frames to load per lens (default: 30)
    
    Returns:
        frames: Tensor of shape (num_frames, 1, H, W)
        label: Integer class label
    """
    
    def __init__(self, root_dir, transform=None, max_frames=NUM_FRAMES):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        
        # Class mapping
        self.classes = ['no_cell', 'contain_cell', 'uncertain_cell']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all lens paths
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
        
        # Get all frame files (sorted)
        frame_files = sorted([f for f in os.listdir(lens_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Load frames
        frames = []
        for i, frame_file in enumerate(frame_files[:self.max_frames]):
            frame_path = os.path.join(lens_path, frame_file)
            
            # Load as grayscale
            with Image.open(frame_path) as img:
                img = img.convert('L')  # Ensure grayscale
                frame = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            frame = frame / 255.0
            
            # Add channel dimension: (H, W) -> (1, H, W)
            frame = frame[np.newaxis, :, :]
            
            frames.append(frame)
        
        # Pad if fewer than max_frames (shouldn't happen, but just in case)
        while len(frames) < self.max_frames:
            frames.append(frames[-1])  # Repeat last frame
        
        # Stack frames: list of (1, H, W) -> (num_frames, 1, H, W)
        frames = np.stack(frames, axis=0)
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, label
    
    def get_class_counts(self):
        """Return count per class."""
        counts = {cls: 0 for cls in self.classes}
        for sample in self.samples:
            counts[sample['class']] += 1
        return counts


# =============================================================================
# DATALOADER HELPER
# =============================================================================

def get_dataloaders(dataset_dir=DATASET_DIR, batch_size=8, num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_dir: Path to dataset root
        batch_size: Batch size for training
        num_workers: Number of worker processes for loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = LensSequenceDataset(os.path.join(dataset_dir, 'train'))
    val_dataset = LensSequenceDataset(os.path.join(dataset_dir, 'val'))
    test_dataset = LensSequenceDataset(os.path.join(dataset_dir, 'test'))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for training stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_class_weights(dataset_dir=DATASET_DIR):
    """Load precomputed class weights."""
    weights_path = os.path.join(dataset_dir, 'class_weights.pt')
    return torch.load(weights_path)


def load_dataset_info(dataset_dir=DATASET_DIR):
    """Load dataset metadata."""
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'r') as f:
        return json.load(f)


# =============================================================================
# TEST DATASET
# =============================================================================

def test_dataset():
    """Test the dataset loading."""
    
    print("=" * 70)
    print("  TESTING DATASET")
    print("=" * 70)
    
    # Load dataset info
    info = load_dataset_info()
    print(f"\n  Dataset info loaded:")
    print(f"    Total lenses: {info['total_lenses']}")
    print(f"    Image size: {info['image_size']}")
    print(f"    Classes: {info['classes']}")
    
    # Test train dataset
    print(f"\n  Loading train dataset...")
    train_dataset = LensSequenceDataset(os.path.join(DATASET_DIR, 'train'))
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Class counts: {train_dataset.get_class_counts()}")
    
    # Load one sample
    print(f"\n  Loading sample 0...")
    frames, label = train_dataset[0]
    print(f"    Frames shape: {frames.shape}")
    print(f"    Frames dtype: {frames.dtype}")
    print(f"    Frames min/max: {frames.min():.3f} / {frames.max():.3f}")
    print(f"    Label: {label} ({train_dataset.classes[label]})")
    
    # Test dataloader
    print(f"\n  Testing dataloader (batch_size=4)...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    batch_frames, batch_labels = next(iter(train_loader))
    print(f"    Batch frames shape: {batch_frames.shape}")
    print(f"    Batch labels: {batch_labels.tolist()}")
    
    # Load class weights
    print(f"\n  Loading class weights...")
    weights = load_class_weights()
    print(f"    Weights: {weights}")
    
    print("\n" + "=" * 70)
    print("  ✅ DATASET TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_dataset()
"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> & "C:\Users\Antardip Himel\.conda\envs\slowfast\python.exe" d:/Research/Cancer_Cell_Analysis/scripts/4_4_dataset.py
======================================================================
  TESTING DATASET
======================================================================

  Dataset info loaded:
    Total lenses: 6008
    Image size: [96, 96]
    Classes: ['no_cell', 'contain_cell', 'uncertain_cell']

  Loading train dataset...
    Train samples: 4203
    Class counts: {'no_cell': 695, 'contain_cell': 2020, 'uncertain_cell': 1488}

  Loading sample 0...
    Frames shape: torch.Size([30, 1, 96, 96])
    Frames dtype: torch.float32
    Frames min/max: 0.435 / 0.773
    Label: 0 (no_cell)

  Testing dataloader (batch_size=4)...
    Batch frames shape: torch.Size([4, 30, 1, 96, 96])
    Batch labels: [1, 1, 2, 2]

  Loading class weights...
d:/Research/Cancer_Cell_Analysis/scripts/4_4_dataset.py:193: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(weights_path)
    Weights: tensor([2.0148, 0.6937, 0.9415])

======================================================================
  ✅ DATASET TEST PASSED!
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""