# -*- coding: utf-8 -*-
"""
m4_matlab_bridge.py

ONCOLENS - MODULE 4: MATLAB BRIDGE
===================================
Interface to call MATLAB scripts for circle detection.

Features:
    - Run MATLAB scripts from Python
    - Call standalone detect_circles.m script
    - Parse MATLAB output (CSV, JSON)
    - Handle errors and timeouts

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m4_matlab_bridge.py

Author: Antardip Himel
Date: March 2026
"""

import os
import subprocess
import csv
import json

from . import m1_config as config


class MatlabBridge:
    """Bridge to run MATLAB scripts from Python."""
    
    def __init__(self, matlab_executable=None):
        """
        Initialize MATLAB bridge.
        
        Args:
            matlab_executable: Path to MATLAB executable or just "matlab" if in PATH
        """
        self.matlab_exe = matlab_executable or config.MATLAB_EXECUTABLE
        self._is_available = None
    
    @property
    def is_available(self):
        """Check if MATLAB is available (cached)."""
        if self._is_available is None:
            self._is_available = self._check_matlab_available()
        return self._is_available
    
    def _check_matlab_available(self):
        """Check if MATLAB is available."""
        try:
            result = subprocess.run(
                [self.matlab_exe, "-batch", "disp('ok')"],
                capture_output=True,
                timeout=60
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def run_script(self, script_dir, script_name, args=None, timeout=300):
        """
        Run a MATLAB script with arguments.
        
        Args:
            script_dir: Directory containing the script
            script_name: Script name (without .m extension)
            args: List of string arguments to pass to the function
            timeout: Timeout in seconds
            
        Returns:
            dict with success status, output, and error
        """
        # Normalize path for MATLAB
        script_dir = script_dir.replace('\\', '/')
        
        # Build MATLAB command
        if args:
            # Format arguments as MATLAB strings
            args_str = ", ".join([f"'{arg}'" for arg in args])
            matlab_cmd = f"cd('{script_dir}'); {script_name}({args_str}); exit;"
        else:
            matlab_cmd = f"cd('{script_dir}'); {script_name}; exit;"
        
        try:
            result = subprocess.run(
                [self.matlab_exe, "-batch", matlab_cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else '',
                'returncode': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'MATLAB timed out after {timeout} seconds',
                'output': ''
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': f'MATLAB not found at: {self.matlab_exe}',
                'output': ''
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': ''
            }
    
    def run_command(self, matlab_cmd, timeout=300):
        """
        Run a raw MATLAB command.
        
        Args:
            matlab_cmd: MATLAB command string
            timeout: Timeout in seconds
            
        Returns:
            dict with success status, output, and error
        """
        try:
            result = subprocess.run(
                [self.matlab_exe, "-batch", matlab_cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else '',
                'returncode': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'MATLAB timed out after {timeout} seconds',
                'output': ''
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': f'MATLAB not found at: {self.matlab_exe}',
                'output': ''
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': ''
            }


def get_matlab_script_path():
    """
    Get path to the detect_circles.m script.
    
    Returns:
        Path to MATLAB script directory
    """
    return config.MATLAB_SCRIPT_DIR


def check_matlab_script_exists():
    """
    Check if the detect_circles.m script exists.
    
    Returns:
        True if script exists, False otherwise
    """
    script_path = os.path.join(config.MATLAB_SCRIPT_DIR, "detect_circles.m")
    return os.path.exists(script_path)


def run_circle_detection(frames_dir, output_dir, video_name, 
                         matlab_exe=None, save_images=True, progress_callback=None):
    """
    Run circle detection using MATLAB.
    
    Calls the standalone detect_circles.m script with arguments.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_dir: Directory for circle detection output
        video_name: Name of the video
        matlab_exe: Path to MATLAB executable (default: from config)
        save_images: If True, save annotated frames; if False, save only data
        progress_callback: Optional callback(message, progress_pct)
        
    Returns:
        dict with:
            - success: True/False
            - num_circles: number of circles found
            - circles: list of circle dicts
            - csv_path: path to circles.csv
            - json_path: path to circles.json
            - error: error message if failed
    """
    if progress_callback:
        progress_callback("Checking MATLAB...", 5)
    
    # Check if MATLAB script exists
    if not check_matlab_script_exists():
        return {
            'success': False,
            'error': f'MATLAB script not found: {os.path.join(config.MATLAB_SCRIPT_DIR, "detect_circles.m")}',
            'num_circles': 0
        }
    
    # Normalize paths (use forward slashes for MATLAB)
    frames_dir = os.path.abspath(frames_dir).replace('\\', '/')
    output_dir = os.path.abspath(output_dir).replace('\\', '/')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if progress_callback:
        progress_callback("Starting MATLAB...", 10)
    
    # Create bridge
    bridge = MatlabBridge(matlab_exe)
    
    # Check MATLAB availability
    if not bridge.is_available:
        return {
            'success': False,
            'error': f'MATLAB not available at: {bridge.matlab_exe}\n\nPlease check:\n1. MATLAB is installed\n2. MATLAB is in your system PATH\n3. Or set full path in m1_config.py',
            'num_circles': 0
        }
    
    if progress_callback:
        progress_callback("Running circle detection...", 20)
    
    # Run the MATLAB script with save_images option
    save_images_str = 'true' if save_images else 'false'
    result = bridge.run_script(
        script_dir=config.MATLAB_SCRIPT_DIR,
        script_name="detect_circles",
        args=[frames_dir, output_dir, video_name, save_images_str],
        timeout=180
    )
    
    if not result['success']:
        return {
            'success': False,
            'error': result.get('error', 'Unknown MATLAB error'),
            'output': result.get('output', ''),
            'num_circles': 0
        }
    
    if progress_callback:
        progress_callback("Parsing results...", 80)
    
    # Parse results
    csv_path = os.path.join(output_dir, 'logs', 'circle_positions.csv')
    json_path = os.path.join(output_dir, 'logs', 'detection_metadata.json')
    annotated_path = os.path.join(output_dir, 'images')  # folder, not single file
    
    if not os.path.exists(csv_path):
        return {
            'success': False,
            'error': f'Circle detection output not found: {csv_path}\n\nMATLAB output:\n{result.get("output", "")}',
            'num_circles': 0
        }
    
    # Parse CSV
    circles = parse_circles_csv(csv_path)
    
    if progress_callback:
        progress_callback(f"Found {len(circles)} circles!", 100)
    
    return {
        'success': True,
        'num_circles': len(circles),
        'usable_circles': sum(1 for c in circles if not c.get('is_edge', False)),
        'circles': circles,
        'csv_path': csv_path,
        'json_path': json_path,
        'images_dir': annotated_path,
        'output_dir': output_dir,
        'matlab_output': result.get('output', '')
    }


def parse_circles_csv(csv_path):
    """
    Parse circles CSV file.
    
    Args:
        csv_path: Path to circle_positions.csv
        
    Returns:
        List of circle dicts with circle_id, center_x, center_y, radius, status
    """
    circles = []
    
    if not os.path.exists(csv_path):
        return circles
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            circle = {
                'circle_id': int(row.get('circle_id', row.get('lens_id', '0').replace('lens_', ''))),
                'center_x': float(row['center_x']),
                'center_y': float(row['center_y']),
                'radius': float(row['radius']),
            }
            # Handle optional status fields
            # NOTE: MATLAB string() writes booleans as "1"/"0", not "true"/"false"
            if 'status' in row:
                circle['status'] = row['status']
                circle['is_edge'] = row.get('is_edge', 'false').lower() in ('true', '1')
                circle['is_shrunk'] = row.get('is_shrunk', 'false').lower() in ('true', '1')
            else:
                circle['status'] = 'ok'
                circle['is_edge'] = False
                circle['is_shrunk'] = False
            
            circles.append(circle)
    
    return circles


def parse_circles_json(json_path):
    """
    Parse circles JSON file.
    
    Args:
        json_path: Path to circles.json
        
    Returns:
        Full detection data as dictionary
    """
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def get_detection_summary(json_path):
    """
    Get a summary of circle detection results.
    
    Args:
        json_path: Path to detection_metadata.json
        
    Returns:
        Formatted summary string
    """
    data = parse_circles_json(json_path)
    
    if data is None:
        return "No detection results found."
    
    # Keys match MATLAB saveLogs output: total_circles, estimated_mean_radius, image_resolution
    return (
        f"Circles Detected: {data.get('total_circles', 0)}\n"
        f"Mean Radius: {data.get('estimated_mean_radius', 0):.1f} pixels\n"
        f"Resolution: {data.get('image_resolution', 'unknown')}\n"
        f"Video: {data.get('video_name', 'unknown')}"
    )