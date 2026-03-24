"""
1_4_crop_lenses.py

Crop Individual Lenses from Drift-Corrected Frames
Reads circle positions from MATLAB's circle_positions.csv
Crops all usable lenses (green + yellow), skips edge-cut (red)
Keeps numbering identical to MATLAB detection (lens_001 ... lens_N)

Input:
  Frames:  D:\Research\Cancer_Cell_Analysis\extracted_frames\image_frames\<video>\
  Circles: D:\Research\Cancer_Cell_Analysis\image_circle\logs\<video>\circle_positions.csv

Output:
  Crops:   D:\Research\Cancer_Cell_Analysis\cropped_lens\lens\<video>\lens_001\frame_001.png ...
  Logs:    D:\Research\Cancer_Cell_Analysis\cropped_lens\logs\<video>\

Author: Based on Antardip Himel's MATLAB pipeline (STEP 1_8)
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
FRAMES_DIR  = r"D:\Research\Cancer_Cell_Analysis\extracted_frames\image_frames"
CIRCLES_DIR = r"D:\Research\Cancer_Cell_Analysis\image_circle\logs"
OUTPUT_DIR  = r"D:\Research\Cancer_Cell_Analysis\cropped_lens"


def read_circle_positions(csv_path):
    """
    Read circle_positions.csv from MATLAB output.
    Returns list of dicts with: circle_id, center_x, center_y, radius, status, is_edge, is_shrunk
    """
    circles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            circles.append({
                'circle_id':  int(row['circle_id']),
                'center_x':   float(row['center_x']),
                'center_y':   float(row['center_y']),
                'radius':     float(row['radius']),
                'status':     row['status'].strip(),
                'is_edge':    row['is_edge'].strip().lower() == 'true',
                'is_shrunk':  row['is_shrunk'].strip().lower() == 'true',
            })
    return circles


def get_usable_circles(circles):
    """Filter to only usable circles (green + yellow), skip edge-cut (red)."""
    return [c for c in circles if not c['is_edge']]


def compute_crop_box(center_x, center_y, radius, img_w, img_h):
    """
    Compute square crop bounding box around circle.
    Crop size = 2 * radius (diameter), same as MATLAB imcrop.
    Clamps to image boundaries.

    Returns: x1, y1, x2, y2 (pixel coordinates, integers)
             and the original unclamped values for logging.
    """
    r = radius

    # Unclamped box
    x1_raw = center_x - r
    y1_raw = center_y - r
    x2_raw = center_x + r
    y2_raw = center_y + r

    # Clamp to image boundaries
    x1 = max(0, int(round(x1_raw)))
    y1 = max(0, int(round(y1_raw)))
    x2 = min(img_w, int(round(x2_raw)))
    y2 = min(img_h, int(round(y2_raw)))

    return x1, y1, x2, y2, x1_raw, y1_raw, x2_raw, y2_raw


def save_crop_logs(usable_circles, crop_boxes, log_dir, video_name,
                   num_frames, img_w, img_h, all_circles):
    """
    Save comprehensive logs for reverse engineering:
      - crop_positions.csv: exact pixel coords for each lens
      - crop_metadata.json: full reconstruction info
    """

    # ─── crop_positions.csv ──────────────────────────────────────────
    csv_path = os.path.join(log_dir, "crop_positions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lens_id', 'circle_id',
            'center_x', 'center_y', 'radius',
            'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
            'crop_width', 'crop_height',
            'unclamped_x1', 'unclamped_y1', 'unclamped_x2', 'unclamped_y2',
            'was_clamped', 'status'
        ])
        for i, (circ, box) in enumerate(zip(usable_circles, crop_boxes)):
            x1, y1, x2, y2, x1_raw, y1_raw, x2_raw, y2_raw = box
            was_clamped = (int(round(x1_raw)) != x1 or int(round(y1_raw)) != y1 or
                           int(round(x2_raw)) != x2 or int(round(y2_raw)) != y2)
            writer.writerow([
                i + 1,  # lens_id (1-based folder name)
                circ['circle_id'],
                round(circ['center_x'], 2),
                round(circ['center_y'], 2),
                round(circ['radius'], 2),
                x1, y1, x2, y2,
                x2 - x1, y2 - y1,
                round(x1_raw, 2), round(y1_raw, 2),
                round(x2_raw, 2), round(y2_raw, 2),
                was_clamped,
                circ['status']
            ])

    # ─── crop_metadata.json ──────────────────────────────────────────
    meta = {
        'video_name': video_name,
        'source_frames_dir': os.path.join(FRAMES_DIR, video_name),
        'circle_data_source': os.path.join(CIRCLES_DIR, video_name, 'circle_positions.csv'),
        'total_circles_detected': len(all_circles),
        'usable_circles_cropped': len(usable_circles),
        'edge_circles_skipped': len(all_circles) - len(usable_circles),
        'num_frames_per_lens': num_frames,
        'total_crops_generated': len(usable_circles) * num_frames,
        'source_image_resolution': f"{img_w}x{img_h}",
        'crop_method': 'square bounding box, 2 * radius (diameter)',
        'crop_sizes': {},
        'lens_to_circle_mapping': {},
        'reconstruction_info': {
            'description': 'To reconstruct original frame from crops: place each crop at (crop_x1, crop_y1) on a blank canvas of source_image_resolution.',
            'steps': [
                '1. Create blank image of source_image_resolution',
                '2. For each lens in crop_positions.csv:',
                '3.   Read crop from lens_NNN/frame_NNN.png',
                '4.   Place crop at position (crop_x1, crop_y1)',
                '5. Overlapping regions: later lenses overwrite earlier ones'
            ]
        },
        'processed_at': datetime.now().isoformat()
    }

    # Per-lens details
    for i, (circ, box) in enumerate(zip(usable_circles, crop_boxes)):
        x1, y1, x2, y2 = box[:4]
        lens_key = f"lens_{i+1:03d}"
        meta['crop_sizes'][lens_key] = f"{x2-x1}x{y2-y1}"
        meta['lens_to_circle_mapping'][lens_key] = {
            'circle_id': circ['circle_id'],
            'center_x': round(circ['center_x'], 2),
            'center_y': round(circ['center_y'], 2),
            'radius': round(circ['radius'], 2),
            'status': circ['status'],
            'crop_box': [x1, y1, x2, y2]
        }

    json_path = os.path.join(log_dir, "crop_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # ─── crop_summary.txt ────────────────────────────────────────────
    txt_path = os.path.join(log_dir, "crop_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Crop Summary — {video_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Source frames:     {os.path.join(FRAMES_DIR, video_name)}\n")
        f.write(f"Circle data:       {os.path.join(CIRCLES_DIR, video_name, 'circle_positions.csv')}\n")
        f.write(f"Source resolution:  {img_w}x{img_h}\n\n")
        f.write(f"Total circles:     {len(all_circles)}\n")
        f.write(f"Usable (cropped):  {len(usable_circles)}\n")
        f.write(f"Edge (skipped):    {len(all_circles) - len(usable_circles)}\n")
        f.write(f"Frames per lens:   {num_frames}\n")
        f.write(f"Total crops:       {len(usable_circles) * num_frames}\n\n")
        f.write(f"{'Lens':<10} {'Circle#':<10} {'Center':<20} {'Radius':<10} {'Crop Size':<12} {'Status':<10}\n")
        f.write(f"{'-' * 72}\n")
        for i, (circ, box) in enumerate(zip(usable_circles, crop_boxes)):
            x1, y1, x2, y2 = box[:4]
            f.write(f"lens_{i+1:03d}  {circ['circle_id']:<10} "
                    f"({circ['center_x']:.1f}, {circ['center_y']:.1f})  "
                    f"{circ['radius']:<10.1f} "
                    f"{x2-x1}x{y2-y1}{'':>6} {circ['status']:<10}\n")


def process_video(video_name):
    """Full cropping pipeline for one video."""

    frames_dir  = os.path.join(FRAMES_DIR, video_name)
    circles_csv = os.path.join(CIRCLES_DIR, video_name, "circle_positions.csv")
    out_lens    = os.path.join(OUTPUT_DIR, "lens", video_name)
    out_logs    = os.path.join(OUTPUT_DIR, "logs", video_name)

    os.makedirs(out_lens, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    # Skip if already processed
    if os.path.exists(os.path.join(out_logs, "crop_metadata.json")):
        print(f"   ⏭ Already processed — skipping. Delete logs to reprocess.\n")
        return

    # ─── Check inputs exist ──────────────────────────────────────────
    if not os.path.exists(frames_dir):
        print(f"   ⚠ Frames not found: {frames_dir} — skipping.\n")
        return
    if not os.path.exists(circles_csv):
        print(f"   ⚠ Circle CSV not found: {circles_csv} — skipping.\n")
        return

    # ─── Load circle positions ───────────────────────────────────────
    all_circles = read_circle_positions(circles_csv)
    usable_circles = get_usable_circles(all_circles)

    if not usable_circles:
        print(f"   ⚠ No usable circles found — skipping.\n")
        return

    print(f"   Circles: {len(all_circles)} total, {len(usable_circles)} usable, "
          f"{len(all_circles) - len(usable_circles)} edge (skipped)")

    # ─── Load frame list ─────────────────────────────────────────────
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not frame_files:
        print(f"   ⚠ No frames found — skipping.\n")
        return

    num_frames = len(frame_files)

    # Get image dimensions from first frame
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    if first_frame is None:
        print(f"   ⚠ Cannot read first frame — skipping.\n")
        return
    img_h, img_w = first_frame.shape[:2]
    print(f"   Frames: {num_frames} ({img_w}x{img_h})")

    # ─── Compute crop boxes ──────────────────────────────────────────
    crop_boxes = []
    for circ in usable_circles:
        box = compute_crop_box(circ['center_x'], circ['center_y'],
                               circ['radius'], img_w, img_h)
        crop_boxes.append(box)

    # ─── Create lens folders ─────────────────────────────────────────
    for i in range(len(usable_circles)):
        lens_folder = os.path.join(out_lens, f"lens_{i+1:03d}")
        os.makedirs(lens_folder, exist_ok=True)

    # ─── Crop all frames ─────────────────────────────────────────────
    print(f"   Cropping {len(usable_circles)} lenses × {num_frames} frames "
          f"= {len(usable_circles) * num_frames} crops...")

    for fi, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            print(f"   ⚠ Cannot read {fname} — skipping frame.")
            continue

        for i, (circ, box) in enumerate(zip(usable_circles, crop_boxes)):
            x1, y1, x2, y2 = box[:4]

            # Crop (numpy slicing: [y1:y2, x1:x2])
            crop = frame[y1:y2, x1:x2]

            # Save
            lens_folder = os.path.join(out_lens, f"lens_{i+1:03d}")
            cv2.imwrite(os.path.join(lens_folder, fname), crop)

        # Progress
        if (fi + 1) % 5 == 0 or fi == 0 or fi == num_frames - 1:
            print(f"   Frame {fi+1}/{num_frames} done")

    # ─── Save logs ───────────────────────────────────────────────────
    print(f"   Saving logs...")
    save_crop_logs(usable_circles, crop_boxes, out_logs, video_name,
                   num_frames, img_w, img_h, all_circles)

    print(f"   ✅ Done! {len(usable_circles)} lenses × {num_frames} frames "
          f"= {len(usable_circles) * num_frames} crops\n")


def main():
    print("=" * 80)
    print("  LENS CROPPING PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n  Frames:  {FRAMES_DIR}")
    print(f"  Circles: {CIRCLES_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"\n  Method:  Square crop, 2×radius (diameter), usable circles only")
    print()

    # Check inputs
    if not os.path.exists(FRAMES_DIR):
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        return
    if not os.path.exists(CIRCLES_DIR):
        print(f"ERROR: Circles directory not found: {CIRCLES_DIR}")
        return

    # Get video folders (from circles dir since that's what MATLAB processed)
    video_folders = sorted([
        f for f in os.listdir(CIRCLES_DIR)
        if os.path.isdir(os.path.join(CIRCLES_DIR, f))
    ])

    if not video_folders:
        print("ERROR: No video folders found in circles directory!")
        return

    print(f"  Found {len(video_folders)} videos to process\n")
    print("-" * 80)

    total_processed = 0
    total_crops = 0

    for i, vfolder in enumerate(video_folders, 1):
        print(f"\n▶ [{i}/{len(video_folders)}] {vfolder}")
        try:
            process_video(vfolder)
            total_processed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()

    # Final summary
    print("=" * 80)
    print("  🎉 Lens cropping complete!")
    print(f"  Cropped lenses → {os.path.join(OUTPUT_DIR, 'lens')}")
    print(f"  Crop logs      → {os.path.join(OUTPUT_DIR, 'logs')}")
    print(f"  Videos processed: {total_processed}")
    print("=" * 80)


if __name__ == "__main__":
    main()