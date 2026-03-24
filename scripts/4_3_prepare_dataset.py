# -*- coding: utf-8 -*-
"""
4_3_prepare_dataset.py

DATASET PREPARATION FOR DEEP LEARNING
=====================================

This script prepares the cell classification dataset for training:
1. Scans all lenses from cell_classification folder
2. Splits by LENS (stratified by class) into train/val/test (70/15/15)
3. Resizes all images to 96×96
4. Computes class weights for imbalanced data
5. Saves metadata (JSON) and class weights (PyTorch tensor)

INPUT:  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\
OUTPUT: D:\\Research\\Cancer_Cell_Analysis\\dataset\\

Run: python -u scripts\\4_3_prepare_dataset.py

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

INPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image size (resize all to this)
TARGET_SIZE = (96, 96)

# Random seed for reproducibility
RANDOM_SEED = 42

# Class names and mapping
CLASS_NAMES = ['no_cell', 'contain_cell', 'uncertain_cell']
CLASS_TO_IDX = {'no_cell': 0, 'contain_cell': 1, 'uncertain_cell': 2}


# =============================================================================
# STEP 1: SCAN DATA
# =============================================================================

def scan_dataset(input_dir):
    """Scan all lenses and collect metadata."""
    
    print("=" * 70, flush=True)
    print("  STEP 1: SCANNING DATASET", flush=True)
    print("=" * 70, flush=True)
    
    all_lenses = []
    class_counts = defaultdict(int)
    
    videos = sorted([v for v in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, v))])
    
    print(f"\n  Found {len(videos)} videos", flush=True)
    print("  Scanning lenses...\n", flush=True)
    
    for video in videos:
        video_path = os.path.join(input_dir, video)
        
        for class_name in CLASS_NAMES:
            class_path = os.path.join(video_path, class_name)
            
            if not os.path.exists(class_path):
                continue
            
            lenses = sorted([l for l in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, l))])
            
            for lens_name in lenses:
                lens_path = os.path.join(class_path, lens_name)
                frames = sorted([f for f in os.listdir(lens_path)
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if len(frames) < 10:  # Skip lenses with too few frames
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
    print(f"  Total lenses found: {len(all_lenses)}", flush=True)
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
    """Split lenses into train/val/test with stratification by class."""
    
    print("\n" + "=" * 70, flush=True)
    print("  STEP 2: STRATIFIED SPLIT", flush=True)
    print("=" * 70, flush=True)
    
    random.seed(seed)
    
    # Group lenses by class
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
    
    # Shuffle each split
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
    
    # Create class subdirectories
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    total_frames = 0
    
    for idx, lens in enumerate(lenses):
        # Create unique lens folder name (video + lens to avoid duplicates)
        video_short = lens['video'][:20].replace(' ', '_').replace('-', '_')
        lens_folder_name = f"{video_short}_{lens['lens']}"
        
        # Output path
        out_lens_path = os.path.join(split_dir, lens['class'], lens_folder_name)
        os.makedirs(out_lens_path, exist_ok=True)
        
        # Get all frames
        frames = sorted([f for f in os.listdir(lens['path'])
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Resize and save each frame
        for frame_name in frames:
            src_path = os.path.join(lens['path'], frame_name)
            dst_path = os.path.join(out_lens_path, frame_name)
            
            try:
                with Image.open(src_path) as img:
                    # Resize with high-quality resampling
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    img_resized.save(dst_path)
                    total_frames += 1
            except Exception as e:
                print(f"    Error processing {src_path}: {e}", flush=True)
        
        # Progress
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
    """Save dataset information to JSON and class weights to PyTorch tensor."""
    
    print("\n" + "=" * 70, flush=True)
    print("  STEP 5: SAVING METADATA", flush=True)
    print("=" * 70, flush=True)
    
    # Count per split
    def count_per_class(lenses):
        counts = defaultdict(int)
        for lens in lenses:
            counts[lens['class']] += 1
        return dict(counts)
    
    train_counts = count_per_class(train_lenses)
    val_counts = count_per_class(val_lenses)
    test_counts = count_per_class(test_lenses)
    
    # Create metadata
    metadata = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
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
    
    # Save JSON
    json_path = os.path.join(output_dir, 'dataset_info.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Saved: {json_path}", flush=True)
    
    # Save class weights as PyTorch tensor
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    weights_path = os.path.join(output_dir, 'class_weights.pt')
    torch.save(weights_tensor, weights_path)
    print(f"  ✓ Saved: {weights_path}", flush=True)
    
    # Save lens lists for each split (for reference)
    for split_name, lenses in [('train', train_lenses), ('val', val_lenses), ('test', test_lenses)]:
        lens_list = [{'video': l['video'], 'lens': l['lens'], 'class': l['class']} for l in lenses]
        list_path = os.path.join(output_dir, f'{split_name}_lenses.json')
        with open(list_path, 'w') as f:
            json.dump(lens_list, f, indent=2)
        print(f"  ✓ Saved: {list_path}", flush=True)
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70, flush=True)
    print("  DATASET PREPARATION FOR DEEP LEARNING", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"  Target size: {TARGET_SIZE}", flush=True)
    
    # Check input exists
    if not os.path.exists(INPUT_DIR):
        print(f"\n  ERROR: Input directory not found: {INPUT_DIR}", flush=True)
        sys.exit(1)
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"\n  ⚠️  Output directory exists. Removing old data...", flush=True)
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Scan data
    all_lenses, class_counts = scan_dataset(INPUT_DIR)
    
    if len(all_lenses) == 0:
        print("\n  ERROR: No lenses found!", flush=True)
        sys.exit(1)
    
    # Step 2: Stratified split
    train_lenses, val_lenses, test_lenses = stratified_split(
        all_lenses, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    # Step 3: Resize and copy
    print("\n" + "=" * 70, flush=True)
    print("  STEP 3: RESIZING AND COPYING IMAGES", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}", flush=True)
    print("  This may take 10-15 minutes...\n", flush=True)
    
    train_frames = resize_and_copy(train_lenses, 'train', OUTPUT_DIR, TARGET_SIZE)
    val_frames = resize_and_copy(val_lenses, 'val', OUTPUT_DIR, TARGET_SIZE)
    test_frames = resize_and_copy(test_lenses, 'test', OUTPUT_DIR, TARGET_SIZE)
    
    total_frames = train_frames + val_frames + test_frames
    print(f"\n  Total frames processed: {total_frames:,}", flush=True)
    
    # Step 4: Compute class weights
    class_weights = compute_class_weights(class_counts, CLASS_NAMES)
    
    # Step 5: Save metadata
    metadata = save_metadata(
        OUTPUT_DIR, all_lenses, train_lenses, val_lenses, test_lenses,
        class_counts, class_weights, TARGET_SIZE
    )
    
    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("  ✅ DATASET PREPARATION COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  📁 Output: {OUTPUT_DIR}", flush=True)
    print(f"\n  📊 Summary:", flush=True)
    print(f"      Total lenses: {len(all_lenses):,}", flush=True)
    print(f"      Total frames: {total_frames:,}", flush=True)
    print(f"      Image size:   {TARGET_SIZE[0]}×{TARGET_SIZE[1]}", flush=True)
    
    print(f"\n  📂 Splits:", flush=True)
    print(f"      Train: {len(train_lenses):,} lenses ({len(train_lenses)/len(all_lenses)*100:.1f}%)", flush=True)
    print(f"      Val:   {len(val_lenses):,} lenses ({len(val_lenses)/len(all_lenses)*100:.1f}%)", flush=True)
    print(f"      Test:  {len(test_lenses):,} lenses ({len(test_lenses)/len(all_lenses)*100:.1f}%)", flush=True)
    
    print(f"\n  ⚖️  Class weights (for imbalance):", flush=True)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"      {cls}: {class_weights[i]:.3f}", flush=True)
    
    print("\n" + "=" * 70, flush=True)
    print("  Ready for training! Next: 4_4_dataset.py", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> python -u scripts\4_3_prepare_dataset.py

======================================================================
  DATASET PREPARATION FOR DEEP LEARNING
  2026-03-12 02:42:29
======================================================================

  Input:  D:\Research\Cancer_Cell_Analysis\cell_classification
  Output: D:\Research\Cancer_Cell_Analysis\dataset
  Target size: (96, 96)
======================================================================
  STEP 1: SCANNING DATASET
======================================================================

  Found 23 videos
  Scanning lenses...

  Total lenses found: 6008

  Class distribution:
    no_cell             :   994 ( 16.5%)
    contain_cell        :  2887 ( 48.1%)
    uncertain_cell      :  2127 ( 35.4%)

======================================================================
  STEP 2: STRATIFIED SPLIT
======================================================================

  Split ratios: Train=70%, Val=15%, Test=15%
  Random seed: 42

  no_cell             : Train= 695, Val= 149, Test= 150
  contain_cell        : Train=2020, Val= 433, Test= 434
  uncertain_cell      : Train=1488, Val= 319, Test= 320

  Total: Train=4203, Val=901, Test=904

======================================================================
  STEP 3: RESIZING AND COPYING IMAGES
======================================================================

  Target size: 96×96
  This may take 10-15 minutes...


  Processing train...
    Processed 200/4203 lenses...
    Processed 400/4203 lenses...
    Processed 600/4203 lenses...
    Processed 800/4203 lenses...
    Processed 1000/4203 lenses...
    Processed 1200/4203 lenses...
    Processed 1400/4203 lenses...
    Processed 1600/4203 lenses...
    Processed 1800/4203 lenses...
    Processed 2000/4203 lenses...
    Processed 2200/4203 lenses...
    Processed 2400/4203 lenses...
    Processed 2600/4203 lenses...
    Processed 2800/4203 lenses...
    Processed 3000/4203 lenses...
    Processed 3200/4203 lenses...
    Processed 3400/4203 lenses...
    Processed 3600/4203 lenses...
    Processed 3800/4203 lenses...
    Processed 4000/4203 lenses...
    Processed 4200/4203 lenses...
    ✓ train: 4203 lenses, 126090 frames

  Processing val...
    Processed 200/901 lenses...
    Processed 400/901 lenses...
    Processed 600/901 lenses...
    Processed 800/901 lenses...
    ✓ val: 901 lenses, 27030 frames

  Processing test...
    Processed 200/904 lenses...
    Processed 400/904 lenses...
    Processed 600/904 lenses...
    Processed 800/904 lenses...
    ✓ test: 904 lenses, 27120 frames

  Total frames processed: 180,240

======================================================================
  STEP 4: COMPUTING CLASS WEIGHTS
======================================================================

  Formula: weight = total / (num_classes × class_count)

  no_cell             :   994 samples → weight = 2.015
  contain_cell        :  2887 samples → weight = 0.694
  uncertain_cell      :  2127 samples → weight = 0.942

======================================================================
  STEP 5: SAVING METADATA
======================================================================

  ✓ Saved: D:\Research\Cancer_Cell_Analysis\dataset\dataset_info.json
  ✓ Saved: D:\Research\Cancer_Cell_Analysis\dataset\class_weights.pt
  ✓ Saved: D:\Research\Cancer_Cell_Analysis\dataset\train_lenses.json
  ✓ Saved: D:\Research\Cancer_Cell_Analysis\dataset\val_lenses.json
  ✓ Saved: D:\Research\Cancer_Cell_Analysis\dataset\test_lenses.json

======================================================================
  ✅ DATASET PREPARATION COMPLETE!
======================================================================

  📁 Output: D:\Research\Cancer_Cell_Analysis\dataset

  📊 Summary:
      Total lenses: 6,008
      Total frames: 180,240
      Image size:   96×96

  📂 Splits:
      Train: 4,203 lenses (70.0%)
      Val:   901 lenses (15.0%)
      Test:  904 lenses (15.0%)

  ⚖️  Class weights (for imbalance):
      no_cell: 2.015
      contain_cell: 0.694
      uncertain_cell: 0.942

======================================================================
  Ready for training! Next: 4_4_dataset.py
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""