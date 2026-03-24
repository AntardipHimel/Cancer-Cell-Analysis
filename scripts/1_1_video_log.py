"""
1_1_video_log.py

Video Length Logger
Reads all videos from the input folder and logs their duration, 
frame count, FPS, and resolution.

Input:  D:\Research\Cancer_Cell_Analysis\original_videos\videos
Output: D:\Research\Cancer_Cell_Analysis\original_videos\video_log.csv
"""

import os
import cv2
import csv
from datetime import datetime


def get_video_info(video_path):
    """Extract video metadata using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Get properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration
    if fps > 0:
        duration_sec = total_frames / fps
    else:
        duration_sec = 0
    
    # Format duration as mm:ss
    minutes = int(duration_sec // 60)
    seconds = duration_sec % 60
    duration_formatted = f"{minutes}m {seconds:.2f}s"
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'duration_sec': round(duration_sec, 2),
        'duration_formatted': duration_formatted,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}"
    }


def main():
    input_dir = r"D:\Research\Cancer_Cell_Analysis\original_videos\videos"
    output_dir = r"D:\Research\Cancer_Cell_Analysis\original_videos"
    output_csv = os.path.join(output_dir, "video_log.csv")
    output_txt = os.path.join(output_dir, "video_log.txt")
    
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Get all video files
    video_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(video_extensions)
    ])
    
    if not video_files:
        print(f"No video files found in: {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos\n")
    print(f"{'#':<4} {'Video Name':<60} {'Frames':<10} {'FPS':<8} {'Duration':<15} {'Resolution':<12}")
    print("-" * 115)
    
    results = []
    total_frames_all = 0
    total_duration_all = 0
    
    for i, filename in enumerate(video_files, 1):
        filepath = os.path.join(input_dir, filename)
        info = get_video_info(filepath)
        
        if info:
            total_frames_all += info['total_frames']
            total_duration_all += info['duration_sec']
            
            print(f"{i:<4} {filename:<60} {info['total_frames']:<10} {info['fps']:<8} {info['duration_formatted']:<15} {info['resolution']:<12}")
            
            results.append({
                'index': i,
                'filename': filename,
                'total_frames': info['total_frames'],
                'fps': info['fps'],
                'duration_sec': info['duration_sec'],
                'duration_formatted': info['duration_formatted'],
                'width': info['width'],
                'height': info['height'],
                'resolution': info['resolution']
            })
        else:
            print(f"{i:<4} {filename:<60} ** COULD NOT READ **")
            results.append({
                'index': i,
                'filename': filename,
                'total_frames': 'ERROR',
                'fps': 'ERROR',
                'duration_sec': 'ERROR',
                'duration_formatted': 'ERROR',
                'width': 'ERROR',
                'height': 'ERROR',
                'resolution': 'ERROR'
            })
    
    # Summary
    total_min = int(total_duration_all // 60)
    total_sec = total_duration_all % 60
    print("-" * 115)
    print(f"\nSummary:")
    print(f"  Total videos:   {len(video_files)}")
    print(f"  Total frames:   {total_frames_all}")
    print(f"  Total duration: {total_min}m {total_sec:.2f}s ({total_duration_all:.2f} seconds)")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV log
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'filename', 'total_frames', 'fps',
            'duration_sec', 'duration_formatted', 'width', 'height', 'resolution'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    # Save readable TXT log
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"Video Log - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {input_dir}\n")
        f.write(f"{'=' * 115}\n\n")
        f.write(f"{'#':<4} {'Video Name':<60} {'Frames':<10} {'FPS':<8} {'Duration':<15} {'Resolution':<12}\n")
        f.write(f"{'-' * 115}\n")
        
        for r in results:
            f.write(f"{r['index']:<4} {r['filename']:<60} {str(r['total_frames']):<10} {str(r['fps']):<8} {str(r['duration_formatted']):<15} {str(r['resolution']):<12}\n")
        
        f.write(f"\n{'=' * 115}\n")
        f.write(f"Total videos:   {len(video_files)}\n")
        f.write(f"Total frames:   {total_frames_all}\n")
        f.write(f"Total duration: {total_min}m {total_sec:.2f}s ({total_duration_all:.2f} seconds)\n")
    
    print(f"\nLogs saved:")
    print(f"  CSV: {output_csv}")
    print(f"  TXT: {output_txt}")


if __name__ == "__main__":
    main()