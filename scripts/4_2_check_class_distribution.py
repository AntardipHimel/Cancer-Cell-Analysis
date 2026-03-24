# -*- coding: utf-8 -*-
"""
4_2_check_class_distribution.py

Counts lenses per class and shows distribution across videos.

Run: python -u scripts\4_2_check_class_distribution.py
"""

import os
import sys

def main():
    root = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
    
    print("=" * 70, flush=True)
    print("  CLASS DISTRIBUTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    
    # Initialize counters
    classes = {'contain_cell': 0, 'no_cell': 0, 'uncertain_cell': 0}
    videos_with_class = {'contain_cell': [], 'no_cell': [], 'uncertain_cell': []}
    video_details = {}
    
    # Get all videos
    videos = sorted([v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v))])
    
    print(f"\n  Found {len(videos)} videos\n", flush=True)
    print("  Scanning...", flush=True)
    
    for video in videos:
        video_path = os.path.join(root, video)
        video_details[video] = {}
        
        for cls_name in classes.keys():
            cls_path = os.path.join(video_path, cls_name)
            
            if os.path.exists(cls_path):
                lenses = [l for l in os.listdir(cls_path) 
                         if os.path.isdir(os.path.join(cls_path, l))]
                count = len(lenses)
                classes[cls_name] += count
                video_details[video][cls_name] = count
                
                if count > 0:
                    videos_with_class[cls_name].append(video)
            else:
                video_details[video][cls_name] = 0
    
    # Print overall distribution
    print("\n" + "=" * 70, flush=True)
    print("  OVERALL CLASS DISTRIBUTION", flush=True)
    print("=" * 70, flush=True)
    
    total = sum(classes.values())
    for cls, count in classes.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:20s}: {count:5d} lenses  ({pct:5.1f}%)  {bar}", flush=True)
    
    print(f"  {'-'*50}", flush=True)
    print(f"  {'TOTAL':20s}: {total:5d} lenses", flush=True)
    
    # Print videos per class
    print("\n" + "=" * 70, flush=True)
    print("  VIDEOS PER CLASS", flush=True)
    print("=" * 70, flush=True)
    
    for cls, vids in videos_with_class.items():
        print(f"  {cls:20s}: {len(vids):3d} videos", flush=True)
    
    # Print per-video breakdown
    print("\n" + "=" * 70, flush=True)
    print("  PER-VIDEO BREAKDOWN", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Video':<45} {'Cell':>7} {'NoCell':>7} {'Uncertain':>9}", flush=True)
    print(f"  {'-'*70}", flush=True)
    
    for video in videos:
        cell = video_details[video].get('contain_cell', 0)
        no_cell = video_details[video].get('no_cell', 0)
        uncertain = video_details[video].get('uncertain_cell', 0)
        
        # Truncate long video names
        video_short = video[:43] + ".." if len(video) > 45 else video
        print(f"  {video_short:<45} {cell:>7} {no_cell:>7} {uncertain:>9}", flush=True)
    
    print("=" * 70, flush=True)
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total Videos:  {len(videos)}", flush=True)
    print(f"  Total Lenses:  {total}", flush=True)
    print(f"  Frames/Lens:   30 (assumed)", flush=True)
    print(f"  Total Frames:  {total * 30:,}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> python -u scripts\4_2_check_class_distribution.py
======================================================================
  CLASS DISTRIBUTION ANALYSIS
======================================================================

  Found 23 videos

  Scanning...

======================================================================
  OVERALL CLASS DISTRIBUTION
======================================================================
  contain_cell        :  2887 lenses  ( 48.1%)  ████████████████████████
  no_cell             :   994 lenses  ( 16.5%)  ████████
  uncertain_cell      :  2127 lenses  ( 35.4%)  █████████████████
  --------------------------------------------------
  TOTAL               :  6008 lenses

======================================================================
  VIDEOS PER CLASS
======================================================================
  contain_cell        :  23 videos
  no_cell             :  23 videos
  uncertain_cell      :  23 videos

======================================================================
  PER-VIDEO BREAKDOWN
======================================================================
  Video                                            Cell  NoCell Uncertain
  ----------------------------------------------------------------------
  01.16.2025-cell w 100 uM dox-white led             98      65       106
  01.30.25-cell trated w 70uM dox-PI- w ms          191       5        79
  02.04.25-70uM dox-exp2                            220       4        30
  02.04.25-bt-20-70uM dox                            86      46       113
  02.10.25-control                                  182      11        63
  02.11.25 BT-20-70 uM dox                          179       7        71
  02.17.25-BT-20-70uM dox                           186      11        59
  02.18.25- cell-PI-50 uM dox- w ms-2 min ima..     168      33        59
  02.20.25- cell-PI- w 20uM dox_w ms-2 min im..     216      12        21
  05.29.25- MDA cell-NO DRUG-CONTROL_w ms-2 m..     125      79        68
  06.05.25- MDA cell-20uM momipp_w ms-2 min i..      63      99       108
  06.19.25-mda cell -20uM momipp                     43     130        85
  07.09.25- MDA cell-30uM DMC_w ms-2 min imag..      73      88       101
  07.18.24-cell w 50uM dox-w-ms-white led           112      53        95
  07.30.24- cell w 70 uM dox-white led               95      21       146
  08.02.24- cell w 120 uM dox-white led             149      15       103
  08.05.24-cell w 110 uM dox-white led              124      26       119
  08.06.24-cell w 110 uM dox-white led              122      37       106
  08.07.24- cell w 60 mM Ethanol-white led           59      52       160
  08.15.24- cell w 800 uM dox-white led              45     122        93
  12.16.24- bt20-20mg&#x3a;ml etharynic acid        154       3       110
  12.19.24- cell treated w 2mg-ml etharynic a..      84      24       147
  12.20.24- cell treated w 2mg-ml etharynic a..     113      51        85
======================================================================

======================================================================
  SUMMARY
======================================================================
  Total Videos:  23
  Total Lenses:  6008
  Frames/Lens:   30 (assumed)
  Total Frames:  180,240
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""