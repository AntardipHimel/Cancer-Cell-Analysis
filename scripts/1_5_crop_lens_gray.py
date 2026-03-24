"""
1_5_crop_lenses_gray.py
With CLAHE
Grayscale Version of Cropped Lenses
Reads the already-cropped color images from cropped_lens/
Converts to grayscale with enhanced contrast for cell structure visualization

Why grayscale?
  - Bright colorful lensless images can be confusing
  - Grayscale reveals actual cell structure (present/absent, morphology)
  - CLAHE enhancement brings out subtle intensity differences

Input:  D:\Research\Cancer_Cell_Analysis\cropped_lens\lens\<video>\lens_NNN\
Output: D:\Research\Cancer_Cell_Analysis\cropped_lens_gray\lens\<video>\lens_NNN\
        D:\Research\Cancer_Cell_Analysis\cropped_lens_gray\logs\<video>\

Author: Based on Antardip Himel's pipeline
Date: February 2026
"""

import os
import cv2
import csv
import json
import numpy as np
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR  = r"D:\Research\Cancer_Cell_Analysis\cropped_lens\lens"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cropped_lens_gray"

# CLAHE for contrast enhancement — brings out cell structure
CLAHE_CLIP  = 3.0       # Higher clip = stronger local contrast
CLAHE_GRID  = (4, 4)    # Smaller grid for small crop regions


def process_video(video_name):
    """Convert all cropped lenses for one video to enhanced grayscale."""

    input_dir  = os.path.join(INPUT_DIR, video_name)
    out_lens   = os.path.join(OUTPUT_DIR, "lens", video_name)
    out_logs   = os.path.join(OUTPUT_DIR, "logs", video_name)

    os.makedirs(out_lens, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    # Skip if already processed
    if os.path.exists(os.path.join(out_logs, "gray_metadata.json")):
        print(f"   ⏭ Already processed — skipping. Delete logs to reprocess.\n")
        return

    # ─── Find lens folders ───────────────────────────────────────────
    if not os.path.exists(input_dir):
        print(f"   ⚠ Input not found: {input_dir} — skipping.\n")
        return

    lens_folders = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f)) and f.startswith('lens_')
    ])

    if not lens_folders:
        print(f"   ⚠ No lens folders found — skipping.\n")
        return

    num_lenses = len(lens_folders)

    # Count frames from first lens
    first_lens_path = os.path.join(input_dir, lens_folders[0])
    frame_files = sorted([
        f for f in os.listdir(first_lens_path)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    num_frames = len(frame_files)

    print(f"   Found {num_lenses} lenses × {num_frames} frames")

    # CLAHE enhancer
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

    total_crops = 0

    # ─── Process each lens ───────────────────────────────────────────
    for li, lens_name in enumerate(lens_folders):
        lens_in  = os.path.join(input_dir, lens_name)
        lens_out = os.path.join(out_lens, lens_name)
        os.makedirs(lens_out, exist_ok=True)

        frames = sorted([
            f for f in os.listdir(lens_in)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        for fname in frames:
            # Read color crop
            img = cv2.imread(os.path.join(lens_in, fname))
            if img is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE enhancement — brings out cell structure
            enhanced = clahe.apply(gray)

            # Save
            cv2.imwrite(os.path.join(lens_out, fname), enhanced)
            total_crops += 1

        # Progress every 10 lenses
        if (li + 1) % 10 == 0 or li == 0 or li == num_lenses - 1:
            print(f"   Lens {li+1}/{num_lenses} done")

    # ─── Save logs ───────────────────────────────────────────────────
    print(f"   Saving logs...")

    meta = {
        'video_name': video_name,
        'source_dir': input_dir,
        'output_dir': os.path.join(out_lens),
        'num_lenses': num_lenses,
        'num_frames_per_lens': num_frames,
        'total_crops': total_crops,
        'grayscale_method': 'cv2.cvtColor BGR2GRAY + CLAHE enhancement',
        'clahe_parameters': {
            'clip_limit': CLAHE_CLIP,
            'tile_grid_size': list(CLAHE_GRID),
            'note': 'CLAHE enhances local contrast to reveal cell structure'
        },
        'lens_folders': lens_folders,
        'numbering_note': 'Lens numbering matches cropped_lens and circle detection exactly',
        'processed_at': datetime.now().isoformat()
    }

    json_path = os.path.join(out_logs, "gray_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"   ✅ Done! {num_lenses} lenses × {num_frames} frames = {total_crops} grayscale crops\n")


def main():
    print("=" * 80)
    print("  GRAYSCALE LENS CONVERSION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\n  Method: Grayscale + CLAHE (clip={CLAHE_CLIP}, grid={CLAHE_GRID})")
    print(f"  Purpose: Enhanced contrast for cell structure visualization")
    print()

    # Check input
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    # Get video folders
    video_folders = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, f))
    ])

    if not video_folders:
        print("ERROR: No video folders found!")
        return

    print(f"  Found {len(video_folders)} videos to process\n")
    print("-" * 80)

    for i, vfolder in enumerate(video_folders, 1):
        print(f"\n▶ [{i}/{len(video_folders)}] {vfolder}")
        try:
            process_video(vfolder)
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("  🎉 Grayscale conversion complete!")
    print(f"  Gray lenses → {os.path.join(OUTPUT_DIR, 'lens')}")
    print(f"  Logs        → {os.path.join(OUTPUT_DIR, 'logs')}")
    print("=" * 80)


if __name__ == "__main__":
    main()