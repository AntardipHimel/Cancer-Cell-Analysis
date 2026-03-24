# -*- coding: utf-8 -*-
"""
5_1_prepare_dataset_2class.py

DATASET PREPARATION FOR 2-CLASS CLASSIFICATION
===============================================

This script prepares a 2-class dataset (contain_cell vs no_cell only).
Excludes uncertain_cell entirely.

Classes:
  0 = no_cell
  1 = contain_cell

INPUT:  D:/Research/Cancer_Cell_Analysis/cell_classification/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dataset_2class/

Run: python -u scripts/5_1_prepare_dataset_2class.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import json
import shutil
import random
from datetime import datetime
from collections import defaultdict
from PIL import Image
import torch
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "D:/Research/Cancer_Cell_Analysis/cell_classification"
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_2class"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image size
TARGET_SIZE = (96, 96)

# Random seed
RANDOM_SEED = 42

# 2 Classes only (no uncertain_cell)
CLASS_NAMES = ['no_cell', 'contain_cell']
CLASS_TO_IDX = {'no_cell': 0, 'contain_cell': 1}


# =============================================================================
# STEP 1: SCAN DATA
# =============================================================================

def scan_dataset(input_dir):
    """Scan all lenses (only no_cell and contain_cell)."""
    
    print("=" * 70, flush=True)
    print("  STEP 1: SCANNING DATASET (2-CLASS)", flush=True)
    print("=" * 70, flush=True)
    
    all_lenses = []
    class_counts = defaultdict(int)
    skipped_uncertain = 0
    
    videos = sorted([v for v in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, v))])
    
    print(f"\n  Found {len(videos)} videos", flush=True)
    print("  Scanning lenses (skipping uncertain_cell)...\n", flush=True)
    
    for video in videos:
        video_path = os.path.join(input_dir, video)
        
        for class_name in ['no_cell', 'contain_cell', 'uncertain_cell']:
            class_path = os.path.join(video_path, class_name)
            
            if not os.path.exists(class_path):
                continue
            
            lenses = sorted([l for l in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, l))])
            
            for lens_name in lenses:
                lens_path = os.path.join(class_path, lens_name)
                frames = sorted([f for f in os.listdir(lens_path)
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if len(frames) < 10:
                    continue
                
                # Skip uncertain_cell
                if class_name == 'uncertain_cell':
                    skipped_uncertain += 1
                    continue
                
                all_lenses.append({
                    'video': video,
                    'lens': lens_name,
                    'class': class_name,
                    'class_idx': CLASS_TO_IDX[class_name],
                    'num_frames': len(frames),
                    'path': lens_path
                })
                
                class_counts[class_name] += 1
    
    # Print summary
    print(f"  Skipped uncertain_cell: {skipped_uncertain} lenses", flush=True)
    print(f"  Total lenses kept: {len(all_lenses)}", flush=True)
    print(f"\n  Class distribution:", flush=True)
    for cls in CLASS_NAMES:
        count = class_counts[cls]
        pct = count / len(all_lenses) * 100
        print(f"    {cls:20s}: {count:5d} ({pct:5.1f}%)", flush=True)
    
    return all_lenses, class_counts


# =============================================================================
# STEP 2: STRATIFIED SPLIT
# =============================================================================

def stratified_split(all_lenses, train_ratio, val_ratio, test_ratio, seed):
    """Split lenses into train/val/test with stratification."""
    
    print("\n" + "=" * 70, flush=True)
    print("  STEP 2: STRATIFIED SPLIT", flush=True)
    print("=" * 70, flush=True)
    
    random.seed(seed)
    
    class_groups = defaultdict(list)
    for lens in all_lenses:
        class_groups[lens['class']].append(lens)
    
    train_lenses = []
    val_lenses = []
    test_lenses = []
    
    print(f"\n  Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}", flush=True)
    print(f"  Random seed: {seed}\n", flush=True)
    
    for class_name in CLASS_NAMES:
        lenses = class_groups[class_name]
        random.shuffle(lenses)
        
        n = len(lenses)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        
        train_lenses.extend(lenses[:n_train])
        val_lenses.extend(lenses[n_train:n_train + n_val])
        test_lenses.extend(lenses[n_train + n_val:])
        
        print(f"  {class_name:20s}: Train={n_train:4d}, Val={n_val:4d}, Test={n_test:4d}", flush=True)
    
    random.shuffle(train_lenses)
    random.shuffle(val_lenses)
    random.shuffle(test_lenses)
    
    print(f"\n  Total: Train={len(train_lenses)}, Val={len(val_lenses)}, Test={len(test_lenses)}", flush=True)
    
    return train_lenses, val_lenses, test_lenses


# =============================================================================
# STEP 3: RESIZE AND COPY
# =============================================================================

def resize_and_copy(lenses, split_name, output_dir, target_size):
    """Resize all frames and copy to output directory."""
    
    print(f"\n  Processing {split_name}...", flush=True)
    
    split_dir = os.path.join(output_dir, split_name)
    
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    total_frames = 0
    
    for idx, lens in enumerate(lenses):
        video_short = lens['video'][:20].replace(' ', '_').replace('-', '_')
        lens_folder_name = f"{video_short}_{lens['lens']}"
        
        out_lens_path = os.path.join(split_dir, lens['class'], lens_folder_name)
        os.makedirs(out_lens_path, exist_ok=True)
        
        frames = sorted([f for f in os.listdir(lens['path'])
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for frame_name in frames:
            src_path = os.path.join(lens['path'], frame_name)
            dst_path = os.path.join(out_lens_path, frame_name)
            
            try:
                with Image.open(src_path) as img:
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    img_resized.save(dst_path)
                    total_frames += 1
            except Exception as e:
                print(f"    Error: {e}", flush=True)
        
        if (idx + 1) % 200 == 0:
            print(f"    Processed {idx + 1}/{len(lenses)} lenses...", flush=True)
    
    print(f"    ✓ {split_name}: {len(lenses)} lenses, {total_frames} frames", flush=True)
    
    return total_frames


# =============================================================================
# STEP 4: COMPUTE CLASS WEIGHTS
# =============================================================================

def compute_class_weights(class_counts, class_names):
    """Compute class weights for handling imbalanced data."""
    
    print("\n" + "=" * 70, flush=True)
    print("  STEP 4: COMPUTING CLASS WEIGHTS", flush=True)
    print("=" * 70, flush=True)
    
    total = sum(class_counts.values())
    num_classes = len(class_names)
    
    weights = []
    print(f"\n  Formula: weight = total / (num_classes × class_count)\n", flush=True)
    
    for class_name in class_names:
        count = class_counts[class_name]
        weight = total / (num_classes * count)
        weights.append(weight)
        print(f"  {class_name:20s}: {count:5d} samples → weight = {weight:.3f}", flush=True)
    
    return weights


# =============================================================================
# STEP 5: SAVE METADATA
# =============================================================================

def save_metadata(output_dir, all_lenses, train_lenses, val_lenses, test_lenses, 
                  class_counts, class_weights, target_size):
    """Save dataset information."""
    
    print("\n" + "=" * 70, flush=True)
    print("  STEP 5: SAVING METADATA", flush=True)
    print("=" * 70, flush=True)
    
    def count_per_class(lenses):
        counts = defaultdict(int)
        for lens in lenses:
            counts[lens['class']] += 1
        return dict(counts)
    
    train_counts = count_per_class(train_lenses)
    val_counts = count_per_class(val_lenses)
    test_counts = count_per_class(test_lenses)
    
    metadata = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'num_classes': 2,
        'total_lenses': len(all_lenses),
        'image_size': list(target_size),
        'frames_per_lens': 30,
        'random_seed': RANDOM_SEED,
        'classes': CLASS_NAMES,
        'class_to_idx': CLASS_TO_IDX,
        'splits': {
            'train': {
                'total': len(train_lenses),
                **{cls: train_counts.get(cls, 0) for cls in CLASS_NAMES}
            },
            'val': {
                'total': len(val_lenses),
                **{cls: val_counts.get(cls, 0) for cls in CLASS_NAMES}
            },
            'test': {
                'total': len(test_lenses),
                **{cls: test_counts.get(cls, 0) for cls in CLASS_NAMES}
            }
        },
        'class_weights': class_weights,
        'class_distribution': {cls: class_counts[cls] for cls in CLASS_NAMES}
    }
    
    json_path = os.path.join(output_dir, 'dataset_info.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Saved: {json_path}", flush=True)
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    weights_path = os.path.join(output_dir, 'class_weights.pt')
    torch.save(weights_tensor, weights_path)
    print(f"  ✓ Saved: {weights_path}", flush=True)
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70, flush=True)
    print("  DATASET PREPARATION - 2 CLASS (no uncertain_cell)", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"  Classes: {CLASS_NAMES}", flush=True)
    print(f"  Target size: {TARGET_SIZE}", flush=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"\n  ERROR: Input directory not found!", flush=True)
        sys.exit(1)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"\n  ⚠️  Output directory exists. Removing old data...", flush=True)
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Scan
    all_lenses, class_counts = scan_dataset(INPUT_DIR)
    
    if len(all_lenses) == 0:
        print("\n  ERROR: No lenses found!", flush=True)
        sys.exit(1)
    
    # Step 2: Split
    train_lenses, val_lenses, test_lenses = stratified_split(
        all_lenses, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    # Step 3: Resize and copy
    print("\n" + "=" * 70, flush=True)
    print("  STEP 3: RESIZING AND COPYING IMAGES", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}", flush=True)
    
    train_frames = resize_and_copy(train_lenses, 'train', OUTPUT_DIR, TARGET_SIZE)
    val_frames = resize_and_copy(val_lenses, 'val', OUTPUT_DIR, TARGET_SIZE)
    test_frames = resize_and_copy(test_lenses, 'test', OUTPUT_DIR, TARGET_SIZE)
    
    total_frames = train_frames + val_frames + test_frames
    print(f"\n  Total frames processed: {total_frames:,}", flush=True)
    
    # Step 4: Class weights
    class_weights = compute_class_weights(class_counts, CLASS_NAMES)
    
    # Step 5: Save metadata
    metadata = save_metadata(
        OUTPUT_DIR, all_lenses, train_lenses, val_lenses, test_lenses,
        class_counts, class_weights, TARGET_SIZE
    )
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  ✅ 2-CLASS DATASET PREPARATION COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  📁 Output: {OUTPUT_DIR}", flush=True)
    print(f"\n  📊 Summary:", flush=True)
    print(f"      Classes: {CLASS_NAMES}", flush=True)
    print(f"      Total lenses: {len(all_lenses):,}", flush=True)
    print(f"      Total frames: {total_frames:,}", flush=True)
    
    print(f"\n  📂 Splits:", flush=True)
    print(f"      Train: {len(train_lenses):,} lenses", flush=True)
    print(f"      Val:   {len(val_lenses):,} lenses", flush=True)
    print(f"      Test:  {len(test_lenses):,} lenses", flush=True)
    
    print(f"\n  ⚖️  Class weights:", flush=True)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"      {cls}: {class_weights[i]:.3f}", flush=True)
    
    print("\n" + "=" * 70, flush=True)
    print("  Next: Run 5_2_train_2class.py", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()


"""
(base) PS D:\Research\Cancer_Cell_Analysis> (C:\Usersl\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate slowfast)
(slowfast) PS D:\Research\Cancer_Cell_Analysis> python -u scripts/5_1_prepare_dataset_2class.py

======================================================================
  DATASET PREPARATION - 2 CLASS (no uncertain_c python -u scripts/5_1_prepare_dataset_2class.py

======================================================================
  DATASET PREPARATION - 2 CLASS (no uncertain_cell)
  2026-03-13 04:18:29
======================================================================

  Input:  D:/Research/Cancer_Cell_Analysis/cell_classification
  Output: D:/Research/Cancer_Cell_Analysis/dataset_2class
  Classes: ['no_cell', 'contain_cell']
  Target size: (96, 96)

  ⚠️  Output directory exists. Removing old data...
======================================================================
  STEP 1: SCANNING DATASET (2-CLASS)
======================================================================

  Found 23 videos
  Scanning lenses (skipping uncertain_cell)... 

  Skipped uncertain_cell: 2127 lenses
  Total lenses kept: 3881

  Class distribution:
    no_cell             :   994 ( 25.6%)       
    contain_cell        :  2887 ( 74.4%)       

======================================================================
  STEP 2: STRATIFIED SPLIT
======================================================================

  Split ratios: Train=70%, Val=15%, Test=15%   
  Random seed: 42

  no_cell             : Train= 695, Val= 149, Test= 150
  contain_cell        : Train=2020, Val= 433, Test= 434

  Total: Train=2715, Val=582, Test=584

======================================================================
  STEP 3: RESIZING AND COPYING IMAGES
======================================================================

  Target size: 96×96

  Processing train...
    Processed 200/2715 lenses...
    Processed 400/2715 lenses...
    Processed 600/2715 lenses...
    Processed 800/2715 lenses...
    Processed 1000/2715 lenses...
    Processed 1200/2715 lenses...
    Processed 1400/2715 lenses...
    Processed 1600/2715 lenses...
    Processed 1800/2715 lenses...
    Processed 2000/2715 lenses...
    Processed 2200/2715 lenses...
    Processed 2400/2715 lenses...
    Processed 2600/2715 lenses...
    ✓ train: 2715 lenses, 81450 frames

  Processing val...
    Processed 200/582 lenses...
    Processed 400/582 lenses...
    ✓ val: 582 lenses, 17460 frames

  Processing test...
    Processed 200/584 lenses...
    Processed 400/584 lenses...
    ✓ test: 584 lenses, 17520 frames

  Total frames processed: 116,430

======================================================================
  STEP 4: COMPUTING CLASS WEIGHTS
======================================================================

  Formula: weight = total / (num_classes × class_count)

  no_cell             :   994 samples → weight = 1.952
  contain_cell        :  2887 samples → weight = 0.672

======================================================================
  STEP 5: SAVING METADATA
======================================================================

  ✓ Saved: D:/Research/Cancer_Cell_Analysis/dataset_2class\dataset_info.json
  ✓ Saved: D:/Research/Cancer_Cell_Analysis/dataset_2class\class_weights.pt

======================================================================
  ✅ 2-CLASS DATASET PREPARATION COMPLETE!     
======================================================================

  📊 Summary:
      Classes: ['no_cell', 'contain_cell']
      Total lenses: 3,881
      Total frames: 116,430

  📂 Splits:
      Train: 2,715 lenses
      Val:   582 lenses
      Test:  584 lenses

  ⚖️  Class weights:
      no_cell: 1.952
      contain_cell: 0.672

======================================================================
  Next: Run 5_2_train_2class.py
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis>
"""