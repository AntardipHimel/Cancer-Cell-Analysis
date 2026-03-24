# -*- coding: utf-8 -*-
"""
7_1_prepare_dataset_good.py

DATASET PREPARATION: GOOD vs NOT_GOOD
======================================

New 2-class classification:
  - good (1):     contain_cell
  - not_good (0): no_cell + uncertain_cell (merged)

INPUT:  D:/Research/Cancer_Cell_Analysis/cell_classification/
OUTPUT: D:/Research/Cancer_Cell_Analysis/dataset_good_vs_notgood/

Run: python -u scripts/7_1_prepare_dataset_good.py

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
OUTPUT_DIR = "D:/Research/Cancer_Cell_Analysis/dataset_good_vs_notgood"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image size
TARGET_SIZE = (96, 96)

# Random seed
RANDOM_SEED = 42

# NEW 2 Classes
CLASS_NAMES = ['not_good', 'good']
CLASS_TO_IDX = {'not_good': 0, 'good': 1}

# Mapping from original classes to new classes
ORIGINAL_TO_NEW = {
    'contain_cell': 'good',        # contain_cell → good
    'no_cell': 'not_good',         # no_cell → not_good
    'uncertain_cell': 'not_good'   # uncertain_cell → not_good
}


# =============================================================================
# STEP 1: SCAN DATA
# =============================================================================

def scan_dataset(input_dir):
    """Scan all lenses and map to new classes."""
    
    print("=" * 70, flush=True)
    print("  STEP 1: SCANNING DATASET", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  Class mapping:", flush=True)
    print(f"    contain_cell    → good", flush=True)
    print(f"    no_cell         → not_good", flush=True)
    print(f"    uncertain_cell  → not_good", flush=True)
    
    all_lenses = []
    original_counts = defaultdict(int)
    new_counts = defaultdict(int)
    
    videos = sorted([v for v in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, v))])
    
    print(f"\n  Found {len(videos)} videos", flush=True)
    print("  Scanning lenses...\n", flush=True)
    
    for video in videos:
        video_path = os.path.join(input_dir, video)
        
        for original_class in ['contain_cell', 'no_cell', 'uncertain_cell']:
            class_path = os.path.join(video_path, original_class)
            
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
                
                # Map to new class
                new_class = ORIGINAL_TO_NEW[original_class]
                
                all_lenses.append({
                    'video': video,
                    'lens': lens_name,
                    'original_class': original_class,
                    'class': new_class,
                    'class_idx': CLASS_TO_IDX[new_class],
                    'num_frames': len(frames),
                    'path': lens_path
                })
                
                original_counts[original_class] += 1
                new_counts[new_class] += 1
    
    # Print summary
    print(f"  Original class distribution:", flush=True)
    for cls in ['contain_cell', 'no_cell', 'uncertain_cell']:
        count = original_counts[cls]
        print(f"    {cls:20s}: {count:5d}", flush=True)
    
    print(f"\n  NEW class distribution:", flush=True)
    total = len(all_lenses)
    for cls in CLASS_NAMES:
        count = new_counts[cls]
        pct = count / total * 100
        print(f"    {cls:20s}: {count:5d} ({pct:5.1f}%)", flush=True)
    
    return all_lenses, new_counts


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
        
        if (idx + 1) % 300 == 0:
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
    
    metadata = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'good vs not_good (contain_cell vs no_cell+uncertain_cell)',
        'source_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'num_classes': 2,
        'total_lenses': len(all_lenses),
        'image_size': list(target_size),
        'frames_per_lens': 30,
        'random_seed': RANDOM_SEED,
        'classes': CLASS_NAMES,
        'class_to_idx': CLASS_TO_IDX,
        'class_mapping': ORIGINAL_TO_NEW,
        'splits': {
            'train': len(train_lenses),
            'val': len(val_lenses),
            'test': len(test_lenses)
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
    print("  DATASET PREPARATION: GOOD vs NOT_GOOD", flush=True)
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n  Input:  {INPUT_DIR}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"\n  NEW Classes:", flush=True)
    print(f"    good     = contain_cell", flush=True)
    print(f"    not_good = no_cell + uncertain_cell", flush=True)
    print(f"\n  Target size: {TARGET_SIZE}", flush=True)
    
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
    print("  ✅ DATASET PREPARATION COMPLETE!", flush=True)
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
    
    print(f"\n  ⚖️  Class distribution:", flush=True)
    for cls in CLASS_NAMES:
        print(f"      {cls}: {class_counts[cls]} (weight: {class_weights[CLASS_NAMES.index(cls)]:.3f})", flush=True)
    
    print("\n" + "=" * 70, flush=True)
    print("  Next: Run 7_2_train_good.py", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()