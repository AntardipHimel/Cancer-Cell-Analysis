# -*- coding: utf-8 -*-
"""
1_6_cell_classification_gui.py

Cell Classification GUI Tool
For lensless microscopy cancer cell analysis

This tool helps you manually classify cropped lens regions as:
- Contains Cell
- No Cell  
- Uncertain

Features:
- THREE-PANEL view: Color, Pure Grayscale, CLAHE Grayscale
- Video-like playback to see all 30 frames of each lens
- Keyboard shortcuts for fast classification
- Progress tracking and logging
- Resume capability
- Undo functionality

Input:
  Color:      D:\\Research\\Cancer_Cell_Analysis\\cropped_lens\\lens\\<video>\\lens_NNN\\
  Pure Gray:  D:\\Research\\Cancer_Cell_Analysis\\cropped_lens_gray_pure\\lens\\<video>\\lens_NNN\\
  CLAHE Gray: D:\\Research\\Cancer_Cell_Analysis\\cropped_lens_gray\\lens\\<video>\\lens_NNN\\

Output:
  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\<video>\\contain_cell\\lens_NNN\\
  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\<video>\\no_cell\\lens_NNN\\
  D:\\Research\\Cancer_Cell_Analysis\\cell_classification\\<video>\\uncertain_cell\\lens_NNN\\

Author: Antardip Himel
Date: February 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import threading
import time


class CellClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell Classification Tool - Lensless Microscopy")
        self.root.geometry("1800x900")
        self.root.state('zoomed')  # Start maximized on Windows
        self.root.configure(bg='#1e1e1e')
        
        # Base paths - THREE input sources
        self.input_color = r"D:\Research\Cancer_Cell_Analysis\cropped_lens\lens"
        self.input_gray_pure = r"D:\Research\Cancer_Cell_Analysis\cropped_lens_gray_pure\lens"
        self.input_gray_clahe = r"D:\Research\Cancer_Cell_Analysis\cropped_lens_gray\lens"
        self.output_base = r"D:\Research\Cancer_Cell_Analysis\cell_classification"
        
        # State variables
        self.video_folders = []
        self.current_video_idx = 0
        self.lens_folders = []
        self.current_lens_idx = 0
        self.frames_color = []
        self.frames_gray_pure = []
        self.frames_gray_clahe = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 100  # ms between frames
        
        # Log file
        self.log_file = os.path.join(self.output_base, "classification_log.json")
        self.classification_log = {}
        
        # Undo history (stack of recent classifications)
        self.undo_history = []
        
        # Zoom level
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 10.0
        
        # Setup UI
        self.setup_ui()
        
        # Load data
        self.load_video_folders()
        self.load_log()
        
        # Keyboard bindings
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('1', lambda e: self.classify('contain_cell'))
        self.root.bind('2', lambda e: self.classify('no_cell'))
        self.root.bind('3', lambda e: self.classify('uncertain_cell'))
        self.root.bind('n', lambda e: self.next_lens())
        self.root.bind('p', lambda e: self.prev_lens())
        self.root.bind('<Return>', lambda e: self.next_lens())
        self.root.bind('<BackSpace>', lambda e: self.undo_last())
        self.root.bind('<Control-z>', lambda e: self.undo_last())
        
        # Mouse wheel zoom bindings
        self.root.bind('<MouseWheel>', self.on_mouse_wheel)  # Windows
        self.root.bind('<Button-4>', lambda e: self.zoom_in())  # Linux scroll up
        self.root.bind('<Button-5>', lambda e: self.zoom_out())  # Linux scroll down
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10), padding=10)
        style.configure('Green.TButton', background='#4CAF50')
        style.configure('Red.TButton', background='#f44336')
        style.configure('Yellow.TButton', background='#ff9800')
        
        # Top info bar
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_label = ttk.Label(info_frame, text="Video: -", style='Header.TLabel')
        self.video_label.pack(side=tk.LEFT)
        
        self.progress_label = ttk.Label(info_frame, text="Progress: -")
        self.progress_label.pack(side=tk.RIGHT)
        
        self.lens_label = ttk.Label(info_frame, text="Lens: -")
        self.lens_label.pack(side=tk.RIGHT, padx=20)
        
        # Image display area - THREE PANELS
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel 1: Color image
        color_panel = ttk.Frame(image_frame)
        color_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        ttk.Label(color_panel, text="Color (Original)", style='Header.TLabel').pack()
        self.color_canvas = tk.Canvas(color_panel, bg='#2d2d2d', highlightthickness=0)
        self.color_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Panel 2: Pure Grayscale image
        gray_pure_panel = ttk.Frame(image_frame)
        gray_pure_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        ttk.Label(gray_pure_panel, text="Pure Grayscale (Sharp)", style='Header.TLabel').pack()
        self.gray_pure_canvas = tk.Canvas(gray_pure_panel, bg='#2d2d2d', highlightthickness=0)
        self.gray_pure_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Panel 3: CLAHE Grayscale image
        gray_clahe_panel = ttk.Frame(image_frame)
        gray_clahe_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        
        ttk.Label(gray_clahe_panel, text="CLAHE Enhanced", style='Header.TLabel').pack()
        self.gray_clahe_canvas = tk.Canvas(gray_clahe_panel, bg='#2d2d2d', highlightthickness=0)
        self.gray_clahe_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame info
        self.frame_label = ttk.Label(main_frame, text="Frame: 1 / 30")
        self.frame_label.pack(pady=5)
        
        # Playback controls
        playback_frame = ttk.Frame(main_frame)
        playback_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(playback_frame, text="<< Prev Lens (P)", command=self.prev_lens).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="< Prev Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        
        self.play_btn = ttk.Button(playback_frame, text="[>] Play (Space)", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(playback_frame, text="Next Frame >", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="Next Lens (N) >>", command=self.next_lens).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Label(playback_frame, text="  |  ").pack(side=tk.LEFT)
        
        # Undo button
        self.undo_btn = ttk.Button(playback_frame, text="Undo (Backspace)", command=self.undo_last)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        ttk.Label(playback_frame, text="  Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.IntVar(value=100)
        speed_scale = ttk.Scale(playback_frame, from_=20, to=500, variable=self.speed_var, 
                                orient=tk.HORIZONTAL, length=150, command=self.update_speed)
        speed_scale.pack(side=tk.LEFT)
        self.speed_label = ttk.Label(playback_frame, text="100ms")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Zoom controls
        ttk.Label(playback_frame, text="  |  Zoom:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(playback_frame, text="-", width=3, command=self.zoom_out).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(playback_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="+", width=3, command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(playback_frame, text="Reset", command=self.zoom_reset).pack(side=tk.LEFT, padx=2)
        
        # Classification buttons
        classify_frame = ttk.Frame(main_frame)
        classify_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(classify_frame, text="Classify this lens:", style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 20))
        
        self.btn_cell = tk.Button(classify_frame, text="[1] Contains Cell", bg='#4CAF50', fg='white',
                                   font=('Segoe UI', 12, 'bold'), width=20, height=2,
                                   command=lambda: self.classify('contain_cell'))
        self.btn_cell.pack(side=tk.LEFT, padx=5)
        
        self.btn_no_cell = tk.Button(classify_frame, text="[2] No Cell", bg='#f44336', fg='white',
                                      font=('Segoe UI', 12, 'bold'), width=20, height=2,
                                      command=lambda: self.classify('no_cell'))
        self.btn_no_cell.pack(side=tk.LEFT, padx=5)
        
        self.btn_uncertain = tk.Button(classify_frame, text="[3] Uncertain", bg='#ff9800', fg='white',
                                        font=('Segoe UI', 12, 'bold'), width=20, height=2,
                                        command=lambda: self.classify('uncertain_cell'))
        self.btn_uncertain.pack(side=tk.LEFT, padx=5)
        
        # Video navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_frame, text="<- Previous Video", command=self.prev_video).pack(side=tk.LEFT, padx=5)
        
        self.video_combo = ttk.Combobox(nav_frame, state='readonly', width=50)
        self.video_combo.pack(side=tk.LEFT, padx=10)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_selected)
        
        ttk.Button(nav_frame, text="Next Video ->", command=self.next_video).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready. Use keyboard shortcuts: 1=Cell, 2=No Cell, 3=Uncertain, Space=Play, N=Next Lens")
        self.status_label.pack(fill=tk.X, pady=(10, 0))
        
    def load_video_folders(self):
        """Load all video folders from input directory"""
        if not os.path.exists(self.input_color):
            messagebox.showerror("Error", "Input folder not found:\n" + self.input_color)
            return
            
        self.video_folders = sorted([
            f for f in os.listdir(self.input_color) 
            if os.path.isdir(os.path.join(self.input_color, f))
        ])
        
        if not self.video_folders:
            messagebox.showerror("Error", "No video folders found!")
            return
            
        self.video_combo['values'] = self.video_folders
        self.video_combo.current(0)
        self.load_current_video()
        
    def load_log(self):
        """Load existing classification log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.classification_log = json.load(f)
            except:
                self.classification_log = {}
                
    def save_log(self):
        """Save classification log"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.classification_log, f, indent=2)
            
    def load_current_video(self):
        """Load lens folders for current video"""
        if not self.video_folders:
            return
            
        video_name = self.video_folders[self.current_video_idx]
        color_video_path = os.path.join(self.input_color, video_name)
        
        # Get all lens folders from color directory
        if not os.path.exists(color_video_path):
            messagebox.showwarning("Warning", "No color folder for " + video_name)
            return
            
        self.lens_folders = sorted([
            f for f in os.listdir(color_video_path)
            if os.path.isdir(os.path.join(color_video_path, f)) and f.startswith('lens_')
        ])
        
        if not self.lens_folders:
            messagebox.showwarning("Warning", "No lens folders in " + video_name)
            return
        
        # Create output structure
        self.create_output_structure(video_name)
        
        # Update UI
        self.video_label.config(text="Video: " + video_name)
        self.video_combo.current(self.current_video_idx)
        
        # Find first unclassified lens
        self.current_lens_idx = self.find_next_unclassified(-1)
        if self.current_lens_idx == -1:
            self.current_lens_idx = 0
            
        self.load_current_lens()
        self.update_progress()
        
    def create_output_structure(self, video_name):
        """Create output folder structure for a video"""
        output_video_path = os.path.join(self.output_base, video_name)
        for category in ['contain_cell', 'no_cell', 'uncertain_cell']:
            os.makedirs(os.path.join(output_video_path, category), exist_ok=True)
            
    def load_current_lens(self):
        """Load frames for current lens from all THREE sources"""
        if not self.lens_folders:
            return
            
        video_name = self.video_folders[self.current_video_idx]
        lens_name = self.lens_folders[self.current_lens_idx]
        
        # Paths for all three sources
        color_path = os.path.join(self.input_color, video_name, lens_name)
        gray_pure_path = os.path.join(self.input_gray_pure, video_name, lens_name)
        gray_clahe_path = os.path.join(self.input_gray_clahe, video_name, lens_name)
        
        # Load color frames
        self.frames_color = []
        if os.path.exists(color_path):
            frame_files = sorted([f for f in os.listdir(color_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in frame_files:
                try:
                    img = Image.open(os.path.join(color_path, f))
                    self.frames_color.append(img)
                except:
                    pass
                    
        # Load pure grayscale frames
        self.frames_gray_pure = []
        if os.path.exists(gray_pure_path):
            frame_files = sorted([f for f in os.listdir(gray_pure_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in frame_files:
                try:
                    img = Image.open(os.path.join(gray_pure_path, f))
                    self.frames_gray_pure.append(img)
                except:
                    pass
                    
        # Load CLAHE grayscale frames
        self.frames_gray_clahe = []
        if os.path.exists(gray_clahe_path):
            frame_files = sorted([f for f in os.listdir(gray_clahe_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in frame_files:
                try:
                    img = Image.open(os.path.join(gray_clahe_path, f))
                    self.frames_gray_clahe.append(img)
                except:
                    pass
                    
        self.current_frame_idx = 0
        self.zoom_level = 1.0  # Reset zoom for new lens
        self.zoom_label.config(text="100%")
        self.lens_label.config(text="Lens: " + lens_name + " (" + str(self.current_lens_idx + 1) + "/" + str(len(self.lens_folders)) + ")")
        self.display_current_frame()
        self.update_progress()
        
    def display_current_frame(self):
        """Display current frame on all THREE canvases - enlarged with sharp pixels"""
        # Get canvas sizes
        self.root.update_idletasks()
        canvas_w = max(self.color_canvas.winfo_width(), 300)
        canvas_h = max(self.color_canvas.winfo_height(), 300)
        
        # Base target display size - make images large for visibility
        base_target_size = min(canvas_w - 20, canvas_h - 20, 400)
        
        # Display color frame
        if self.frames_color and self.current_frame_idx < len(self.frames_color):
            img = self.frames_color[self.current_frame_idx].copy()
            orig_w, orig_h = img.size
            base_scale = base_target_size / max(orig_w, orig_h)
            final_scale = base_scale * self.zoom_level
            
            if final_scale != 1:
                new_w = max(1, int(orig_w * final_scale))
                new_h = max(1, int(orig_h * final_scale))
                img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
            
            self.color_photo = ImageTk.PhotoImage(img)
            self.color_canvas.delete("all")
            self.color_canvas.create_image(canvas_w//2, canvas_h//2, image=self.color_photo)
        else:
            self.color_canvas.delete("all")
            self.color_canvas.create_text(canvas_w//2, canvas_h//2, text="No color frames", fill='white')
            
        # Display pure grayscale frame
        if self.frames_gray_pure and self.current_frame_idx < len(self.frames_gray_pure):
            img = self.frames_gray_pure[self.current_frame_idx].copy()
            orig_w, orig_h = img.size
            base_scale = base_target_size / max(orig_w, orig_h)
            final_scale = base_scale * self.zoom_level
            
            if final_scale != 1:
                new_w = max(1, int(orig_w * final_scale))
                new_h = max(1, int(orig_h * final_scale))
                img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
            
            self.gray_pure_photo = ImageTk.PhotoImage(img)
            self.gray_pure_canvas.delete("all")
            self.gray_pure_canvas.create_image(canvas_w//2, canvas_h//2, image=self.gray_pure_photo)
        else:
            self.gray_pure_canvas.delete("all")
            self.gray_pure_canvas.create_text(canvas_w//2, canvas_h//2, text="No pure gray frames", fill='white')
            
        # Display CLAHE grayscale frame
        if self.frames_gray_clahe and self.current_frame_idx < len(self.frames_gray_clahe):
            img = self.frames_gray_clahe[self.current_frame_idx].copy()
            orig_w, orig_h = img.size
            base_scale = base_target_size / max(orig_w, orig_h)
            final_scale = base_scale * self.zoom_level
            
            if final_scale != 1:
                new_w = max(1, int(orig_w * final_scale))
                new_h = max(1, int(orig_h * final_scale))
                img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
            
            self.gray_clahe_photo = ImageTk.PhotoImage(img)
            self.gray_clahe_canvas.delete("all")
            self.gray_clahe_canvas.create_image(canvas_w//2, canvas_h//2, image=self.gray_clahe_photo)
        else:
            self.gray_clahe_canvas.delete("all")
            self.gray_clahe_canvas.create_text(canvas_w//2, canvas_h//2, text="No CLAHE frames", fill='white')
            
        # Update frame label
        total_frames = max(len(self.frames_color), len(self.frames_gray_pure), len(self.frames_gray_clahe), 1)
        self.frame_label.config(text="Frame: " + str(self.current_frame_idx + 1) + " / " + str(total_frames))
        
    def toggle_play(self):
        """Toggle video playback"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="[||] Pause (Space)")
            self.play_frames()
        else:
            self.play_btn.config(text="[>] Play (Space)")
            
    def play_frames(self):
        """Play frames in sequence"""
        if not self.is_playing:
            return
            
        self.next_frame()
        
        # Loop back to start
        total_frames = max(len(self.frames_color), len(self.frames_gray_pure), len(self.frames_gray_clahe), 1)
        if self.current_frame_idx >= total_frames - 1:
            self.current_frame_idx = -1
            
        self.root.after(self.play_speed, self.play_frames)
        
    def next_frame(self):
        """Go to next frame"""
        total_frames = max(len(self.frames_color), len(self.frames_gray_pure), len(self.frames_gray_clahe), 1)
        self.current_frame_idx = (self.current_frame_idx + 1) % total_frames
        self.display_current_frame()
        
    def prev_frame(self):
        """Go to previous frame"""
        total_frames = max(len(self.frames_color), len(self.frames_gray_pure), len(self.frames_gray_clahe), 1)
        self.current_frame_idx = (self.current_frame_idx - 1) % total_frames
        self.display_current_frame()
        
    def update_speed(self, val):
        """Update playback speed"""
        self.play_speed = int(float(val))
        self.speed_label.config(text=str(self.play_speed) + "ms")
        
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
            
    def zoom_in(self):
        """Zoom in on images"""
        if self.zoom_level < self.max_zoom:
            self.zoom_level = min(self.zoom_level * 1.2, self.max_zoom)
            self.zoom_label.config(text=str(int(self.zoom_level * 100)) + "%")
            self.display_current_frame()
            
    def zoom_out(self):
        """Zoom out on images"""
        if self.zoom_level > self.min_zoom:
            self.zoom_level = max(self.zoom_level / 1.2, self.min_zoom)
            self.zoom_label.config(text=str(int(self.zoom_level * 100)) + "%")
            self.display_current_frame()
            
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.zoom_level = 1.0
        self.zoom_label.config(text="100%")
        self.display_current_frame()
        
    def undo_last(self):
        """Undo the last classification"""
        if not self.undo_history:
            self.status_label.config(text="Nothing to undo")
            return
            
        # Get last action
        last_action = self.undo_history.pop()
        video_name = last_action['video']
        lens_name = last_action['lens']
        category = last_action['category']
        
        # Remove from output folder
        dst_path = os.path.join(self.output_base, video_name, category, lens_name)
        try:
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
                
            # Remove from log
            if video_name in self.classification_log:
                if lens_name in self.classification_log[video_name]:
                    del self.classification_log[video_name][lens_name]
            self.save_log()
            
            # Navigate back to that lens if in same video
            if video_name == self.video_folders[self.current_video_idx]:
                # Find the lens index
                if lens_name in self.lens_folders:
                    self.current_lens_idx = self.lens_folders.index(lens_name)
                    self.load_current_lens()
            
            self.status_label.config(text="Undone: " + lens_name + " removed from " + category)
            self.update_progress()
            
        except Exception as e:
            messagebox.showerror("Error", "Failed to undo:\n" + str(e))

    def classify(self, category):
        """Classify current lens and copy color frames only"""
        if not self.lens_folders:
            return
            
        video_name = self.video_folders[self.current_video_idx]
        lens_name = self.lens_folders[self.current_lens_idx]
        
        # Source path (color only)
        src_color = os.path.join(self.input_color, video_name, lens_name)
        
        # Destination path (keep original lens name)
        dst_base = os.path.join(self.output_base, video_name, category)
        dst_color = os.path.join(dst_base, lens_name)
        
        # Copy color files only (not move, to preserve original data)
        try:
            if os.path.exists(src_color):
                if os.path.exists(dst_color):
                    shutil.rmtree(dst_color)
                shutil.copytree(src_color, dst_color)
                
            # Update log
            if video_name not in self.classification_log:
                self.classification_log[video_name] = {}
            self.classification_log[video_name][lens_name] = {
                'category': category,
                'timestamp': datetime.now().isoformat()
            }
            self.save_log()
            
            # Add to undo history
            self.undo_history.append({
                'video': video_name,
                'lens': lens_name,
                'category': category
            })
            
            self.status_label.config(text="Classified " + lens_name + " as " + category)
            
            # Auto-advance to next lens
            self.next_lens()
            
        except Exception as e:
            messagebox.showerror("Error", "Failed to copy files:\n" + str(e))
            
    def find_next_unclassified(self, start_idx):
        """Find next unclassified lens"""
        video_name = self.video_folders[self.current_video_idx]
        video_log = self.classification_log.get(video_name, {})
        
        for i in range(start_idx + 1, len(self.lens_folders)):
            if self.lens_folders[i] not in video_log:
                return i
        return -1
        
    def next_lens(self):
        """Go to next lens"""
        if self.current_lens_idx < len(self.lens_folders) - 1:
            # Try to find next unclassified
            next_unclassified = self.find_next_unclassified(self.current_lens_idx)
            if next_unclassified != -1:
                self.current_lens_idx = next_unclassified
            else:
                self.current_lens_idx += 1
            self.is_playing = False
            self.play_btn.config(text="[>] Play (Space)")
            self.load_current_lens()
        else:
            if messagebox.askyesno("Video Complete", 
                                   "All lenses in this video have been reviewed.\nMove to next video?"):
                self.next_video()
                
    def prev_lens(self):
        """Go to previous lens"""
        if self.current_lens_idx > 0:
            self.current_lens_idx -= 1
            self.is_playing = False
            self.play_btn.config(text="[>] Play (Space)")
            self.load_current_lens()
            
    def next_video(self):
        """Go to next video folder"""
        if self.current_video_idx < len(self.video_folders) - 1:
            self.current_video_idx += 1
            self.current_lens_idx = 0
            self.is_playing = False
            self.play_btn.config(text="[>] Play (Space)")
            self.load_current_video()
        else:
            messagebox.showinfo("Complete", "You have reviewed all videos!")
            
    def prev_video(self):
        """Go to previous video folder"""
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.current_lens_idx = 0
            self.is_playing = False
            self.play_btn.config(text="[>] Play (Space)")
            self.load_current_video()
            
    def on_video_selected(self, event):
        """Handle video selection from combobox"""
        idx = self.video_combo.current()
        if idx != self.current_video_idx:
            self.current_video_idx = idx
            self.current_lens_idx = 0
            self.is_playing = False
            self.play_btn.config(text="[>] Play (Space)")
            self.load_current_video()
            
    def update_progress(self):
        """Update progress display"""
        if not self.video_folders or not self.lens_folders:
            return
            
        video_name = self.video_folders[self.current_video_idx]
        video_log = self.classification_log.get(video_name, {})
        classified = len(video_log)
        total = len(self.lens_folders)
        
        # Count by category
        cells = sum(1 for v in video_log.values() if v['category'] == 'contain_cell')
        no_cells = sum(1 for v in video_log.values() if v['category'] == 'no_cell')
        uncertain = sum(1 for v in video_log.values() if v['category'] == 'uncertain_cell')
        
        self.progress_label.config(
            text="Progress: " + str(classified) + "/" + str(total) + " | Cells: " + str(cells) + " | No Cell: " + str(no_cells) + " | Uncertain: " + str(uncertain)
        )


def main():
    root = tk.Tk()
    app = CellClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()