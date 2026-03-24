# -*- coding: utf-8 -*-
"""
1_5b_crop_lenses_gray_pure.py

Pure Grayscale Version of Cropped Lenses (NO CLAHE)
Reads the already-cropped color images from cropped_lens/
Converts to grayscale WITHOUT any enhancement - preserves original texture

Why pure grayscale (no CLAHE)?
  - CLAHE can make images look softer/blurrier
  - Pure grayscale preserves exact texture and cell detail
  - Better for manual labeling where you need to see true structure

Input:  D:\\Research\\Cancer_Cell_Analysis\\cropped_lens\\lens\\<video>\\lens_NNN\\
Output: D:\\Research\\Cancer_Cell_Analysis\\cropped_lens_gray_pure\\lens\\<video>\\lens_NNN\\
        D:\\Research\\Cancer_Cell_Analysis\\cropped_lens_gray_pure\\logs\\<video>\\

Author: Based on Antardip Himel's pipeline
Date: February 2026
"""

import os
import cv2
import json
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR  = r"D:\Research\Cancer_Cell_Analysis\cropped_lens\lens"
OUTPUT_DIR = r"D:\Research\Cancer_Cell_Analysis\cropped_lens_gray_pure"


def process_video(video_name):
    """Convert all cropped lenses for one video to pure grayscale (no enhancement)."""

    input_dir  = os.path.join(INPUT_DIR, video_name)
    out_lens   = os.path.join(OUTPUT_DIR, "lens", video_name)
    out_logs   = os.path.join(OUTPUT_DIR, "logs", video_name)

    os.makedirs(out_lens, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    # Skip if already processed
    if os.path.exists(os.path.join(out_logs, "gray_pure_metadata.json")):
        print("   Already processed - skipping. Delete logs to reprocess.\n")
        return

    # Find lens folders
    if not os.path.exists(input_dir):
        print("   Input not found: " + input_dir + " - skipping.\n")
        return

    lens_folders = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f)) and f.startswith('lens_')
    ])

    if not lens_folders:
        print("   No lens folders found - skipping.\n")
        return

    num_lenses = len(lens_folders)

    # Count frames from first lens
    first_lens_path = os.path.join(input_dir, lens_folders[0])
    frame_files = sorted([
        f for f in os.listdir(first_lens_path)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    num_frames = len(frame_files)

    print("   Found " + str(num_lenses) + " lenses x " + str(num_frames) + " frames")

    total_crops = 0

    # Process each lens
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

            # Convert to grayscale - NO CLAHE, NO ENHANCEMENT
            # Just pure BGR to Grayscale conversion
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save as PNG (lossless)
            cv2.imwrite(os.path.join(lens_out, fname), gray)
            total_crops += 1

        # Progress every 10 lenses
        if (li + 1) % 10 == 0 or li == 0 or li == num_lenses - 1:
            print("   Lens " + str(li+1) + "/" + str(num_lenses) + " done")

    # Save logs
    print("   Saving logs...")

    meta = {
        'video_name': video_name,
        'source_dir': input_dir,
        'output_dir': out_lens,
        'num_lenses': num_lenses,
        'num_frames_per_lens': num_frames,
        'total_crops': total_crops,
        'grayscale_method': 'cv2.cvtColor BGR2GRAY - NO ENHANCEMENT',
        'enhancement': 'NONE - pure grayscale preserves original texture',
        'note': 'This version has NO CLAHE - better for seeing true cell texture and detail',
        'lens_folders': lens_folders,
        'numbering_note': 'Lens numbering matches cropped_lens and circle detection exactly',
        'processed_at': datetime.now().isoformat()
    }

    json_path = os.path.join(out_logs, "gray_pure_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("   Done! " + str(num_lenses) + " lenses x " + str(num_frames) + " frames = " + str(total_crops) + " pure grayscale crops\n")


def main():
    print("=" * 80)
    print("  PURE GRAYSCALE LENS CONVERSION (NO CLAHE)")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)
    print("\n  Input:  " + INPUT_DIR)
    print("  Output: " + OUTPUT_DIR)
    print("\n  Method: Pure BGR2GRAY conversion - NO enhancement")
    print("  Purpose: Preserve original texture for manual labeling")
    print()

    # Check input
    if not os.path.exists(INPUT_DIR):
        print("ERROR: Input directory not found: " + INPUT_DIR)
        return

    # Get video folders
    video_folders = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, f))
    ])

    if not video_folders:
        print("ERROR: No video folders found!")
        return

    print("  Found " + str(len(video_folders)) + " videos to process\n")
    print("-" * 80)

    for i, vfolder in enumerate(video_folders, 1):
        print("\n[" + str(i) + "/" + str(len(video_folders)) + "] " + vfolder)
        try:
            process_video(vfolder)
        except Exception as e:
            print("   ERROR: " + str(e) + "\n")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("  Pure grayscale conversion complete!")
    print("  Output: " + os.path.join(OUTPUT_DIR, 'lens'))
    print("  Logs:   " + os.path.join(OUTPUT_DIR, 'logs'))
    print("=" * 80)


if __name__ == "__main__":
    main()