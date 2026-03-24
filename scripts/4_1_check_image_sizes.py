# -*- coding: utf-8 -*-
"""
check_image_sizes.py

Safely scans all images and reports size distribution.
- Closes images after reading (prevents memory leak)
- Shows progress
- Won't freeze your PC

Run: python check_image_sizes.py
"""

import os
import sys
from PIL import Image
from collections import defaultdict

def main():
    root = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
    
    print("=" * 60, flush=True)
    print("  SCANNING ALL IMAGES", flush=True)
    print("=" * 60, flush=True)
    
    sizes = defaultdict(int)
    total = 0
    errors = 0
    
    # First count total files
    print("\n  Counting files...", flush=True)
    total_files = sum(1 for _, _, files in os.walk(root) for f in files if f.endswith('.png'))
    print(f"  Found {total_files:,} PNG files to scan\n", flush=True)
    
    # Scan images
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.endswith('.png'):
                try:
                    filepath = os.path.join(dirpath, f)
                    with Image.open(filepath) as img:  # 'with' ensures image is closed
                        sizes[img.size] += 1
                    total += 1
                except Exception as e:
                    errors += 1
                
                # Progress every 5000 images
                if total % 5000 == 0:
                    pct = total / total_files * 100
                    print(f"  Processed {total:,} / {total_files:,} ({pct:.1f}%)", flush=True)
    
    # Final results
    print(f"\n" + "=" * 60, flush=True)
    print(f"  IMAGE SIZE DISTRIBUTION", flush=True)
    print("=" * 60, flush=True)
    print(f"\n  Total images scanned: {total:,}", flush=True)
    if errors > 0:
        print(f"  Errors: {errors}", flush=True)
    
    print(f"\n  SIZE BREAKDOWN:", flush=True)
    print(f"  {'-'*40}", flush=True)
    
    for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {size[0]:4d} x {size[1]:4d}  :  {count:>7,} images  ({pct:5.1f}%)", flush=True)
    
    print(f"\n  Unique sizes: {len(sizes)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()


    """
(slowfast) PS D:\Research\Cancer_Cell_Analysis> cd D:\Research\Cancer_Cell_Analysis
(slowfast) PS D:\Research\Cancer_Cell_Analysis> python -u scripts\4_1_check_image_sizes.py
============================================================
  SCANNING ALL IMAGES
============================================================

  Counting files...
  Found 180,240 PNG files to scan

  Processed 5,000 / 180,240 (2.8%)
  Processed 10,000 / 180,240 (5.5%)
  Processed 15,000 / 180,240 (8.3%)
  Processed 20,000 / 180,240 (11.1%)
  Processed 25,000 / 180,240 (13.9%)
  Processed 30,000 / 180,240 (16.6%)
  Processed 35,000 / 180,240 (19.4%)
  Processed 40,000 / 180,240 (22.2%)
  Processed 45,000 / 180,240 (25.0%)
  Processed 50,000 / 180,240 (27.7%)
  Processed 55,000 / 180,240 (30.5%)
  Processed 60,000 / 180,240 (33.3%)
  Processed 65,000 / 180,240 (36.1%)
  Processed 70,000 / 180,240 (38.8%)
  Processed 75,000 / 180,240 (41.6%)
  Processed 80,000 / 180,240 (44.4%)
  Processed 85,000 / 180,240 (47.2%)
  Processed 90,000 / 180,240 (49.9%)
  Processed 95,000 / 180,240 (52.7%)
  Processed 100,000 / 180,240 (55.5%)
  Processed 105,000 / 180,240 (58.3%)
  Processed 110,000 / 180,240 (61.0%)
  Processed 115,000 / 180,240 (63.8%)
  Processed 120,000 / 180,240 (66.6%)
  Processed 125,000 / 180,240 (69.4%)
  Processed 130,000 / 180,240 (72.1%)
  Processed 135,000 / 180,240 (74.9%)
  Processed 140,000 / 180,240 (77.7%)
  Processed 145,000 / 180,240 (80.4%)
  Processed 150,000 / 180,240 (83.2%)
  Processed 155,000 / 180,240 (86.0%)
  Processed 160,000 / 180,240 (88.8%)
  Processed 165,000 / 180,240 (91.5%)
  Processed 170,000 / 180,240 (94.3%)
  Processed 175,000 / 180,240 (97.1%)
  Processed 180,000 / 180,240 (99.9%)

============================================================
  IMAGE SIZE DISTRIBUTION
============================================================

  Total images scanned: 180,240

  SIZE BREAKDOWN:
  ----------------------------------------
    76 x   76  :   23,430 images  ( 13.0%)
    80 x   80  :   21,390 images  ( 11.9%)
    78 x   78  :   19,650 images  ( 10.9%)
    82 x   82  :   16,680 images  (  9.3%)
    74 x   74  :   14,790 images  (  8.2%)
    84 x   84  :   12,390 images  (  6.9%)
    86 x   86  :    8,940 images  (  5.0%)
    72 x   72  :    6,150 images  (  3.4%)
    70 x   70  :    3,960 images  (  2.2%)
    68 x   68  :    2,850 images  (  1.6%)
    88 x   88  :    2,730 images  (  1.5%)
    66 x   66  :    2,160 images  (  1.2%)
    76 x   77  :    1,560 images  (  0.9%)
    77 x   76  :    1,380 images  (  0.8%)
    75 x   76  :    1,320 images  (  0.7%)
    81 x   80  :    1,320 images  (  0.7%)
    64 x   64  :    1,170 images  (  0.6%)
    79 x   79  :    1,140 images  (  0.6%)
    78 x   79  :    1,110 images  (  0.6%)
    74 x   73  :    1,080 images  (  0.6%)
    80 x   79  :    1,080 images  (  0.6%)
    80 x   81  :    1,020 images  (  0.6%)
    72 x   73  :      990 images  (  0.5%)
    83 x   82  :      990 images  (  0.5%)
    79 x   80  :      960 images  (  0.5%)
    78 x   77  :      960 images  (  0.5%)
    90 x   90  :      960 images  (  0.5%)
    79 x   78  :      930 images  (  0.5%)
    75 x   74  :      900 images  (  0.5%)
    77 x   77  :      900 images  (  0.5%)
    73 x   72  :      870 images  (  0.5%)
    75 x   75  :      840 images  (  0.5%)
    81 x   82  :      840 images  (  0.5%)
    73 x   73  :      810 images  (  0.4%)
    76 x   75  :      780 images  (  0.4%)
    74 x   75  :      780 images  (  0.4%)
    60 x   60  :      750 images  (  0.4%)
    81 x   81  :      720 images  (  0.4%)
    83 x   83  :      690 images  (  0.4%)
    84 x   83  :      690 images  (  0.4%)
    82 x   81  :      690 images  (  0.4%)
    72 x   71  :      660 images  (  0.4%)
    62 x   62  :      660 images  (  0.4%)
    77 x   78  :      660 images  (  0.4%)
    82 x   83  :      660 images  (  0.4%)
    85 x   84  :      630 images  (  0.3%)
    73 x   74  :      630 images  (  0.3%)
    83 x   84  :      600 images  (  0.3%)
    67 x   67  :      600 images  (  0.3%)
    70 x   71  :      570 images  (  0.3%)
    71 x   72  :      570 images  (  0.3%)
    84 x   85  :      570 images  (  0.3%)
    71 x   70  :      510 images  (  0.3%)
    69 x   70  :      510 images  (  0.3%)
    68 x   67  :      450 images  (  0.2%)
    69 x   68  :      450 images  (  0.2%)
    87 x   86  :      420 images  (  0.2%)
    69 x   69  :      390 images  (  0.2%)
    58 x   58  :      390 images  (  0.2%)
    67 x   66  :      360 images  (  0.2%)
    67 x   68  :      360 images  (  0.2%)
    71 x   71  :      330 images  (  0.2%)
    86 x   85  :      330 images  (  0.2%)
    85 x   86  :      330 images  (  0.2%)
    70 x   69  :      330 images  (  0.2%)
    87 x   87  :      300 images  (  0.2%)
    85 x   85  :      270 images  (  0.1%)
    86 x   87  :      270 images  (  0.1%)
    65 x   65  :      240 images  (  0.1%)
    65 x   64  :      240 images  (  0.1%)
    64 x   65  :      210 images  (  0.1%)
    61 x   61  :      180 images  (  0.1%)
    68 x   69  :      180 images  (  0.1%)
    63 x   63  :      180 images  (  0.1%)
    66 x   65  :      150 images  (  0.1%)
    57 x   57  :      150 images  (  0.1%)
    65 x   66  :      150 images  (  0.1%)
    66 x   67  :      150 images  (  0.1%)
    62 x   63  :      120 images  (  0.1%)
    64 x   63  :      120 images  (  0.1%)
    87 x   88  :      120 images  (  0.1%)
    88 x   89  :      120 images  (  0.1%)
    63 x   62  :      120 images  (  0.1%)
    52 x   52  :       90 images  (  0.0%)
    61 x   62  :       90 images  (  0.0%)
    57 x   56  :       90 images  (  0.0%)
    76 x   74  :       90 images  (  0.0%)
    82 x   84  :       90 images  (  0.0%)
    59 x   60  :       90 images  (  0.0%)
    62 x   61  :       90 images  (  0.0%)
    63 x   64  :       90 images  (  0.0%)
    54 x   54  :       90 images  (  0.0%)
    56 x   56  :       60 images  (  0.0%)
    88 x   87  :       60 images  (  0.0%)
    72 x   70  :       60 images  (  0.0%)
    55 x   55  :       60 images  (  0.0%)
    50 x   50  :       60 images  (  0.0%)
    58 x   59  :       60 images  (  0.0%)
    89 x   89  :       60 images  (  0.0%)
    57 x   58  :       30 images  (  0.0%)
    61 x   60  :       30 images  (  0.0%)
    59 x   59  :       30 images  (  0.0%)
    70 x   72  :       30 images  (  0.0%)
    60 x   61  :       30 images  (  0.0%)
    86 x   84  :       30 images  (  0.0%)
    56 x   57  :       30 images  (  0.0%)
    80 x   82  :       30 images  (  0.0%)
    59 x   58  :       30 images  (  0.0%)
    74 x   76  :       30 images  (  0.0%)
    51 x   50  :       30 images  (  0.0%)
    56 x   55  :       30 images  (  0.0%)
    66 x   68  :       30 images  (  0.0%)
    74 x   72  :       30 images  (  0.0%)

  Unique sizes: 113
============================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""