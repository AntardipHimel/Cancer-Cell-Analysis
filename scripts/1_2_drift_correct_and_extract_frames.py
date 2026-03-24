"""
1_2_drift_correct_and_extract.py

Option C (v2): Direct-to-Reference Drift Estimation + Corrected Frame Extraction
Author: Based on Antardip Himel's MATLAB pipeline, converted & improved in Python
Date: February 2026

─── KEY CHANGE FROM v1 ───────────────────────────────────────────────────
  v1: Frame-to-frame chaining (drift accumulates errors over time)
  v2: Every frame compared DIRECTLY to the first selected frame (reference)
      - First & last video frames are skipped
      - 30 frames evenly spaced from the middle
      - The first of those 30 = reference = frame_001.png
      - All other frames corrected to match this reference exactly
      - Circles detected on frame_001.png stay locked on all 30 frames
───────────────────────────────────────────────────────────────────────────

Pipeline:
  1. Reads each full video from original_videos/videos/
  2. Selects 30 evenly-spaced frames (skipping first & last video frame)
  3. Uses the FIRST selected frame as drift reference
  4. Computes drift for EVERY frame directly against this reference
  5. Applies drift correction to the 30 selected frames
  6. Saves:
     - Stabilized full video   → fixed_videos/videos/<video_name>_stabilized.avi
     - Video drift logs        → fixed_videos/logs/<video_name>/
     - 30 extracted frames     → extracted_frames/image_frames/<video_name>/
     - Extraction logs         → extracted_frames/logs/<video_name>/

Input:  D:\Research\Cancer_Cell_Analysis\original_videos\videos
"""

import os
import cv2
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR       = r"D:\Research\Cancer_Cell_Analysis\original_videos\videos"
FIXED_VIDEO_DIR = r"D:\Research\Cancer_Cell_Analysis\fixed_videos"
EXTRACTED_DIR   = r"D:\Research\Cancer_Cell_Analysis\extracted_frames"
NUM_FRAMES      = 30
VIDEO_EXTS      = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.m4v')


def compute_full_drift(video_path, ref_frame_idx):
    """
    Read entire video and compute drift for EVERY frame directly against
    the specified reference frame using phase correlation.

    ─── v2 CHANGE ────────────────────────────────────────────────────
    v1 (old): Chained frame-to-frame shifts → cumulative error grows
    v2 (new): Each frame vs ref_frame_idx → independent, no accumulation

    The reference frame is the FIRST of the 30 selected frames (not frame 0).
    This means frame_001.png = reference = zero drift = exact match for circles.
    ──────────────────────────────────────────────────────────────────

    Args:
        video_path    : path to video file
        ref_frame_idx : video frame index to use as drift reference
                        (this is the first of the 30 selected frames)

    Returns:
        cumulative_drift : array (total_frames, 2) -> [dx, dy] per frame
                           (shift of each frame relative to reference frame)
        frame_drift      : array (total_frames, 2) -> frame-to-frame shifts
                           (derived from cumulative, for logging only)
        total_frames, fps, width, height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   Total frames: {total_frames} | FPS: {fps} | Resolution: {width}x{height}")
    print(f"   Drift method: DIRECT-TO-REFERENCE (every frame vs frame {ref_frame_idx})")

    # ─── First pass: read the reference frame ────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError(f"Cannot read reference frame {ref_frame_idx}")

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = np.float64(ref_gray)

    # ─── Second pass: compute drift for every frame vs reference ─────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start
    cumulative_drift = np.zeros((total_frames, 2), dtype=np.float64)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            total_frames = i
            cumulative_drift = cumulative_drift[:total_frames]
            break

        if i == ref_frame_idx:
            # Reference frame: zero drift by definition
            cumulative_drift[i] = [0.0, 0.0]
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = np.float64(curr_gray)

        # Phase correlation: shift of current frame relative to REFERENCE
        shift, response = cv2.phaseCorrelate(ref_gray, curr_gray)
        cumulative_drift[i] = [shift[0], shift[1]]

        # Progress indicator every 10%
        if i % max(1, total_frames // 10) == 0:
            pct = (i / total_frames) * 100
            print(f"   Drift estimation: {pct:.0f}% ({i}/{total_frames})")

    cap.release()

    # Compute frame-to-frame shifts from cumulative (for logging only)
    frame_drift = np.zeros_like(cumulative_drift)
    frame_drift[1:] = np.diff(cumulative_drift, axis=0)

    return cumulative_drift, frame_drift, total_frames, fps, width, height


def select_frame_indices(total_frames, num_extract=30):
    """
    Select evenly spaced frame indices, skipping first and last frame.
    Mirrors the MATLAB linspace(2, totalFrames-1, 30) behavior.

    The FIRST selected index becomes the drift reference frame.
    frame_001.png = this reference = zero drift = exact match for circle detection.
    """
    # Skip frame 0 (index 0) and last frame (index total_frames-1)
    indices = np.round(np.linspace(1, total_frames - 2, num_extract)).astype(int)
    # Remove duplicates while preserving order
    indices = list(dict.fromkeys(indices))
    return indices


def extract_corrected_frames(video_path, cumulative_drift, frame_indices, output_folder):
    """
    Re-read the video, extract only the selected frames,
    apply drift correction, and save as PNG.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_indices_set = set(frame_indices)
    extracted = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_indices_set:
            # Get drift for this frame
            dx, dy = cumulative_drift[frame_idx]

            # Build affine transform to correct drift (shift back)
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            h, w = frame.shape[:2]
            corrected = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Save
            out_idx = frame_indices.index(frame_idx) + 1  # 1-based
            fname = f"frame_{out_idx:03d}.png"
            cv2.imwrite(os.path.join(output_folder, fname), corrected)
            extracted[frame_idx] = corrected

        frame_idx += 1

    cap.release()
    return extracted


def write_stabilized_video(video_path, cumulative_drift, total_frames, fps, output_path):
    """
    Write a fully drift-corrected version of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy = cumulative_drift[i]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        corrected = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        out.write(corrected)

        if i % max(1, total_frames // 10) == 0:
            pct = (i / total_frames) * 100
            print(f"   Writing stabilized video: {pct:.0f}%")

    cap.release()
    out.release()


def save_drift_logs(cumulative_drift, frame_drift, frame_indices, log_folder, video_name,
                    total_frames, fps, width, height):
    """
    Save drift CSV, plots, and metadata.
    """
    # --- Full drift CSV ---
    csv_path = os.path.join(log_folder, "full_drift_log.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'frame_shift_x', 'frame_shift_y',
                          'cumulative_shift_x', 'cumulative_shift_y', 'magnitude'])
        for i in range(len(cumulative_drift)):
            mag = np.sqrt(cumulative_drift[i, 0]**2 + cumulative_drift[i, 1]**2)
            writer.writerow([
                i,
                round(frame_drift[i, 0], 4), round(frame_drift[i, 1], 4),
                round(cumulative_drift[i, 0], 4), round(cumulative_drift[i, 1], 4),
                round(mag, 4)
            ])

    # --- Extracted frames drift CSV ---
    ext_csv_path = os.path.join(log_folder, "extracted_frames_drift.csv")
    with open(ext_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['extract_idx', 'video_frame', 'cumulative_dx', 'cumulative_dy', 'magnitude'])
        for idx, fi in enumerate(frame_indices, 1):
            dx, dy = cumulative_drift[fi]
            mag = np.sqrt(dx**2 + dy**2)
            writer.writerow([idx, fi, round(dx, 4), round(dy, 4), round(mag, 4)])

    # --- Metadata JSON ---
    meta = {
        'video_name': video_name,
        'total_frames': int(total_frames),
        'fps': float(fps),
        'resolution': f"{width}x{height}",
        'frames_extracted': int(len(frame_indices)),
        'frame_indices': [int(x) for x in frame_indices],
        'drift_method': 'direct-to-reference',
        'reference_frame': int(frame_indices[0]),
        'reference_note': 'frame_001.png = reference frame (zero drift, zero correction)',
        'max_drift_x': round(float(np.max(np.abs(cumulative_drift[:, 0]))), 4),
        'max_drift_y': round(float(np.max(np.abs(cumulative_drift[:, 1]))), 4),
        'max_drift_magnitude': round(float(np.max(np.sqrt(
            cumulative_drift[:, 0]**2 + cumulative_drift[:, 1]**2))), 4),
        'processed_at': datetime.now().isoformat()
    }
    with open(os.path.join(log_folder, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # --- Plot 1: Cumulative X/Y drift over all frames ---
    fig, ax = plt.subplots(figsize=(12, 5))
    frames_arr = np.arange(len(cumulative_drift))
    ax.plot(frames_arr, cumulative_drift[:, 0], '-', linewidth=1, label='X drift', color='#2196F3')
    ax.plot(frames_arr, cumulative_drift[:, 1], '-', linewidth=1, label='Y drift', color='#FF5722')
    # Mark extracted frames
    for fi in frame_indices:
        ax.axvline(x=fi, color='gray', alpha=0.15, linewidth=0.5)
    ax.scatter(frame_indices, cumulative_drift[frame_indices, 0],
               c='#2196F3', s=30, zorder=5, edgecolors='black', linewidths=0.5)
    ax.scatter(frame_indices, cumulative_drift[frame_indices, 1],
               c='#FF5722', s=30, zorder=5, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Drift from Reference Frame (pixels)')
    ax.set_title(f'Direct-to-Reference Drift — {video_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_folder, 'drift_plot_XY.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Drift magnitude ---
    magnitude = np.sqrt(cumulative_drift[:, 0]**2 + cumulative_drift[:, 1]**2)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames_arr, magnitude, '-', linewidth=1, color='#9C27B0')
    ax.scatter(frame_indices, magnitude[frame_indices],
               c='#FF9800', s=40, zorder=5, edgecolors='black', linewidths=0.5, label='Extracted frames')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Drift Magnitude from Reference (pixels)')
    ax.set_title(f'Total Drift Magnitude — {video_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_folder, 'drift_plot_magnitude.png'), dpi=150)
    plt.close(fig)

    # --- Plot 3: 2D drift path ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(cumulative_drift[:, 0], cumulative_drift[:, 1], '-',
            linewidth=0.8, color='#607D8B', alpha=0.6)
    ax.scatter(cumulative_drift[frame_indices, 0], cumulative_drift[frame_indices, 1],
               c=np.arange(len(frame_indices)), cmap='viridis', s=50,
               zorder=5, edgecolors='black', linewidths=0.5)
    ax.scatter(0, 0, c='red', s=100, marker='x', zorder=6, label='Origin (reference frame)')
    ax.set_xlabel('X Drift (pixels)')
    ax.set_ylabel('Y Drift (pixels)')
    ax.set_title(f'2D Drift Path — {video_name}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_folder, 'drift_plot_2D_path.png'), dpi=150)
    plt.close(fig)


def create_overlay_check(frames_folder, log_folder, num_frames):
    """
    Create overlay QC image: blend first and last extracted frame.
    If correction is good, structures should align perfectly.
    """
    first_path = os.path.join(frames_folder, "frame_001.png")
    last_path = os.path.join(frames_folder, f"frame_{num_frames:03d}.png")

    if os.path.exists(first_path) and os.path.exists(last_path):
        first = cv2.imread(first_path)
        last = cv2.imread(last_path)

        # False-color overlay (first=magenta, last=green)
        overlay = np.zeros_like(first)
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
        overlay[:, :, 0] = first_gray    # Blue channel = first
        overlay[:, :, 1] = last_gray     # Green channel = last
        overlay[:, :, 2] = first_gray    # Red channel = first (magenta)

        cv2.imwrite(os.path.join(log_folder, 'overlay_check_first_last.png'), overlay)


def process_video(video_path, video_name):
    """
    Full pipeline for a single video.
    """
    name_no_ext = os.path.splitext(video_name)[0]

    # Output paths — clean structure
    # fixed_videos/videos/                        → all stabilized .avi files
    # fixed_videos/logs/<video_name>/             → drift logs per video
    # extracted_frames/image_frames/<video_name>/ → 30 corrected PNGs
    # extracted_frames/logs/<video_name>/         → extraction logs per video
    fixed_videos_dir   = os.path.join(FIXED_VIDEO_DIR, "videos")
    fixed_log_dir      = os.path.join(FIXED_VIDEO_DIR, "logs", name_no_ext)
    extract_images_dir = os.path.join(EXTRACTED_DIR, "image_frames", name_no_ext)
    extract_log_dir    = os.path.join(EXTRACTED_DIR, "logs", name_no_ext)

    # Create directories
    for d in [fixed_videos_dir, fixed_log_dir, extract_images_dir, extract_log_dir]:
        os.makedirs(d, exist_ok=True)

    # Skip if already processed
    check_file = os.path.join(extract_log_dir, "metadata.json")
    if os.path.exists(check_file):
        print(f"   ⏭ Already processed — skipping. Delete logs to reprocess.\n")
        return

    # First, get total frame count to determine frame selection
    cap_tmp = cv2.VideoCapture(video_path)
    if not cap_tmp.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames_raw = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()

    # Step 1: Select 30 frame indices (skip first & last)
    frame_indices = select_frame_indices(total_frames_raw, NUM_FRAMES)
    frame_indices = [int(x) for x in frame_indices]
    actual_extracted = len(frame_indices)
    ref_frame_idx = frame_indices[0]  # FIRST selected frame = reference
    print(f"   [1/4] Selected {actual_extracted} frames | Reference = frame {ref_frame_idx}")

    # Step 2: Compute drift — every frame vs reference frame
    print(f"   [2/4] Computing drift (every frame vs frame {ref_frame_idx})...")
    cumulative_drift, frame_drift, total_frames, fps, width, height = compute_full_drift(video_path, ref_frame_idx)
    print(f"   Max drift: X={np.max(np.abs(cumulative_drift[:,0])):.2f}px, "
          f"Y={np.max(np.abs(cumulative_drift[:,1])):.2f}px")

    # Update frame indices if total_frames was truncated during reading
    frame_indices = [fi for fi in frame_indices if fi < total_frames]
    actual_extracted = len(frame_indices)
    print(f"   [2/4] {actual_extracted} frame indices valid from {total_frames} total")

    # Step 3: Extract drift-corrected frames
    print(f"   [3/4] Extracting {actual_extracted} corrected frames...")
    extract_corrected_frames(video_path, cumulative_drift, frame_indices, extract_images_dir)

    # Step 4: Write stabilized full video
    print(f"   [4/4] Writing stabilized video...")
    stabilized_path = os.path.join(fixed_videos_dir, f"{name_no_ext}_stabilized.avi")
    write_stabilized_video(video_path, cumulative_drift, total_frames, fps, stabilized_path)

    # Save logs and plots to BOTH log folders
    print(f"   Saving logs and plots...")
    for log_dir in [fixed_log_dir, extract_log_dir]:
        save_drift_logs(cumulative_drift, frame_drift, frame_indices, log_dir,
                        video_name, total_frames, fps, width, height)

    # Overlay QC
    create_overlay_check(extract_images_dir, extract_log_dir, actual_extracted)

    print(f"   ✅ Done!\n")


def main():
    print("=" * 80)
    print("  DRIFT CORRECTION + FRAME EXTRACTION (v2: Direct-to-Reference)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n  Input:     {INPUT_DIR}")
    print(f"  Fixed:     {FIXED_VIDEO_DIR}")
    print(f"  Extracted: {EXTRACTED_DIR}")
    print(f"\n  Method:    Every frame vs first selected frame (no error accumulation)")
    print()

    # Check input
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    # Get video files
    video_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.lower().endswith(VIDEO_EXTS)
    ])

    if not video_files:
        print("ERROR: No video files found!")
        return

    print(f"  Found {len(video_files)} videos to process\n")
    print("-" * 80)

    # Process each video
    for i, vfile in enumerate(video_files, 1):
        vpath = os.path.join(INPUT_DIR, vfile)
        print(f"\n▶ [{i}/{len(video_files)}] {vfile}")
        try:
            process_video(vpath, vfile)
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}\n")

    # Final summary
    print("=" * 80)
    print("  🎉 Pipeline complete!")
    print(f"  Stabilized videos → {FIXED_VIDEO_DIR}")
    print(f"  Extracted frames  → {EXTRACTED_DIR}")
    print(f"  Method: Direct-to-Reference (v2)")
    print("=" * 80)


if __name__ == "__main__":
    main()