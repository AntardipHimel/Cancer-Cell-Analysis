# -*- coding: utf-8 -*-
"""
cell_classifier_app.py

GUI APPLICATION: CANCER CELL CLASSIFIER
========================================

A user-friendly tool to:
1. Select/upload video files
2. Extract lenses (30 frames each)
3. Classify each lens (contain_cell vs no_cell)
4. Save results to organized folders

Uses the trained ResNet+LSTM model.

Run: python cell_classifier_app.py

Author: Antardip Himel
Date: March 2026
"""

import os
import sys
import json
import shutil
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.models as models

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model path - UPDATE THIS IF NEEDED
MODEL_PATH = "D:/Research/Cancer_Cell_Analysis/dl_output_2class/checkpoints/best_model.pt"

# Output directory for predictions
OUTPUT_BASE = "D:/Research/Cancer_Cell_Analysis/predictions"

# Model settings
NUM_FRAMES = 30
NUM_CLASSES = 2
CLASS_NAMES = ['no_cell', 'contain_cell']
IMAGE_SIZE = (96, 96)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class ResNetLSTM(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256, num_layers=2, dropout=0.5):
        super(ResNetLSTM, self).__init__()
        
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnet(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        h_combined = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        return self.classifier(h_combined)


# =============================================================================
# VIDEO PROCESSING FUNCTIONS
# =============================================================================

def extract_lenses_from_video(video_path, progress_callback=None):
    """
    Extract lenses from video.
    
    Returns list of:
    {
        'lens_id': str,
        'frames': list of numpy arrays (30 frames, grayscale, 96x96)
    }
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Read all frames
    all_frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        all_frames.append(gray)
        frame_idx += 1
        
        if progress_callback and frame_idx % 100 == 0:
            progress_callback(f"Reading frames: {frame_idx}/{total_frames}")
    
    cap.release()
    
    # Group into lenses (every 30 frames = 1 lens)
    lenses = []
    num_lenses = len(all_frames) // NUM_FRAMES
    
    for i in range(num_lenses):
        start_idx = i * NUM_FRAMES
        end_idx = start_idx + NUM_FRAMES
        
        lens_frames = []
        for frame in all_frames[start_idx:end_idx]:
            # Resize to 96x96
            resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
            lens_frames.append(resized)
        
        lenses.append({
            'lens_id': f"lens_{i+1:04d}",
            'frames': lens_frames,
            'start_frame': start_idx,
            'end_frame': end_idx
        })
        
        if progress_callback:
            progress_callback(f"Extracted lens {i+1}/{num_lenses}")
    
    return lenses


def preprocess_lens(frames):
    """Convert list of frames to tensor for model input."""
    
    # Stack frames: (30, 96, 96)
    frames_array = np.stack(frames, axis=0).astype(np.float32)
    
    # Normalize to [0, 1]
    frames_array = frames_array / 255.0
    
    # Add channel dimension: (30, 1, 96, 96)
    frames_array = frames_array[:, np.newaxis, :, :]
    
    # Convert to tensor and add batch dimension: (1, 30, 1, 96, 96)
    tensor = torch.from_numpy(frames_array).unsqueeze(0)
    
    return tensor


def classify_lens(model, frames, device):
    """Classify a single lens."""
    
    model.eval()
    
    with torch.no_grad():
        tensor = preprocess_lens(frames).to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()
        confidence = probs[pred_idx].item()
    
    return {
        'prediction': CLASS_NAMES[pred_idx],
        'prediction_idx': pred_idx,
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: probs[i].item() for i in range(NUM_CLASSES)
        }
    }


def save_lens_frames(frames, output_dir, lens_id):
    """Save lens frames to directory."""
    
    lens_dir = os.path.join(output_dir, lens_id)
    os.makedirs(lens_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(lens_dir, f"frame_{i+1:03d}.png")
        cv2.imwrite(frame_path, frame)
    
    return lens_dir


# =============================================================================
# GUI APPLICATION
# =============================================================================

class CellClassifierApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Cell Classifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Variables
        self.video_paths = []
        self.model = None
        self.is_processing = False
        self.results = []
        
        # Create UI
        self.create_widgets()
        
        # Load model on startup
        self.load_model()
    
    def create_widgets(self):
        """Create all UI elements."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔬 Cancer Cell Classifier", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        subtitle = ttk.Label(main_frame, 
                            text="Classify microscopy lenses as contain_cell or no_cell",
                            font=('Helvetica', 10))
        subtitle.grid(row=1, column=0, pady=(0, 20))
        
        # Model status
        self.model_status = ttk.Label(main_frame, text="Model: Loading...", 
                                      font=('Helvetica', 10))
        self.model_status.grid(row=2, column=0, pady=(0, 10))
        
        # Video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding="10")
        video_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        
        # Browse button
        btn_frame = ttk.Frame(video_frame)
        btn_frame.grid(row=0, column=0, sticky="ew")
        
        self.browse_btn = ttk.Button(btn_frame, text="📁 Browse Videos", 
                                     command=self.browse_videos)
        self.browse_btn.pack(side="left", padx=(0, 10))
        
        self.clear_btn = ttk.Button(btn_frame, text="🗑️ Clear", 
                                    command=self.clear_videos)
        self.clear_btn.pack(side="left")
        
        # Video list
        self.video_listbox = tk.Listbox(video_frame, height=4, 
                                        selectmode=tk.EXTENDED)
        self.video_listbox.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        # Output directory
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding="10")
        output_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        
        self.output_var = tk.StringVar(value=OUTPUT_BASE)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        output_btn = ttk.Button(output_frame, text="📂 Change", 
                               command=self.change_output_dir)
        output_btn.grid(row=0, column=1)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="🚀 Process Videos", 
                                      command=self.start_processing,
                                      style='Accent.TButton')
        self.process_btn.grid(row=5, column=0, pady=10)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        
        self.status_label = ttk.Label(progress_frame, text="Ready", 
                                     font=('Helvetica', 9))
        self.status_label.grid(row=1, column=0, pady=(5, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=7, column=0, sticky="nsew", pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=12, wrap=tk.WORD,
                                   font=('Consolas', 10))
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                 command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Open output folder button
        self.open_folder_btn = ttk.Button(main_frame, text="📂 Open Output Folder",
                                         command=self.open_output_folder,
                                         state="disabled")
        self.open_folder_btn.grid(row=8, column=0, pady=(0, 10))
        
        # Device info
        device_text = f"Device: {DEVICE}"
        if DEVICE.type == 'cuda':
            device_text += f" ({torch.cuda.get_device_name(0)})"
        device_label = ttk.Label(main_frame, text=device_text, font=('Helvetica', 8))
        device_label.grid(row=9, column=0)
    
    def load_model(self):
        """Load the trained model."""
        
        try:
            if not os.path.exists(MODEL_PATH):
                self.model_status.config(text=f"❌ Model not found: {MODEL_PATH}")
                messagebox.showerror("Error", f"Model file not found:\n{MODEL_PATH}")
                return
            
            self.model = ResNetLSTM(num_classes=NUM_CLASSES).to(DEVICE)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_status.config(text=f"✅ Model loaded (Epoch {checkpoint['epoch']+1}, "
                                         f"Val Acc: {checkpoint['val_acc']:.2f}%)")
            
        except Exception as e:
            self.model_status.config(text=f"❌ Error loading model")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def browse_videos(self):
        """Open file dialog to select videos."""
        
        filetypes = [
            ("Video files", "*.avi *.mp4 *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
        
        paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=filetypes
        )
        
        if paths:
            for path in paths:
                if path not in self.video_paths:
                    self.video_paths.append(path)
                    self.video_listbox.insert(tk.END, os.path.basename(path))
    
    def clear_videos(self):
        """Clear selected videos."""
        
        self.video_paths = []
        self.video_listbox.delete(0, tk.END)
    
    def change_output_dir(self):
        """Change output directory."""
        
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_var.get()
        )
        
        if directory:
            self.output_var.set(directory)
    
    def update_status(self, message):
        """Update status label (thread-safe)."""
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def update_progress(self, value):
        """Update progress bar (thread-safe)."""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def append_result(self, text):
        """Append text to results (thread-safe)."""
        def _append():
            self.results_text.insert(tk.END, text + "\n")
            self.results_text.see(tk.END)
        self.root.after(0, _append)
    
    def start_processing(self):
        """Start processing videos in a separate thread."""
        
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please select at least one video.")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Cannot process videos.")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress.")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results = []
        
        # Disable buttons
        self.process_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.is_processing = True
        
        # Start processing thread
        thread = threading.Thread(target=self.process_videos)
        thread.daemon = True
        thread.start()
    
    def process_videos(self):
        """Process all selected videos."""
        
        try:
            total_videos = len(self.video_paths)
            
            for video_idx, video_path in enumerate(self.video_paths):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(self.output_var.get(), f"{video_name}_{timestamp}")
                
                self.append_result(f"\n{'='*50}")
                self.append_result(f"Processing: {video_name}")
                self.append_result(f"{'='*50}")
                
                # Create output directories
                os.makedirs(os.path.join(output_dir, "contain_cell"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "no_cell"), exist_ok=True)
                
                # Extract lenses
                self.update_status(f"Extracting lenses from {video_name}...")
                lenses = extract_lenses_from_video(video_path, self.update_status)
                
                self.append_result(f"Extracted {len(lenses)} lenses")
                
                # Classify each lens
                video_results = []
                contain_count = 0
                no_cell_count = 0
                
                for i, lens in enumerate(lenses):
                    # Update progress
                    progress = ((video_idx / total_videos) + 
                               (i / len(lenses) / total_videos)) * 100
                    self.update_progress(progress)
                    self.update_status(f"Classifying lens {i+1}/{len(lenses)}...")
                    
                    # Classify
                    result = classify_lens(self.model, lens['frames'], DEVICE)
                    
                    # Save frames to appropriate folder
                    class_folder = os.path.join(output_dir, result['prediction'])
                    save_lens_frames(lens['frames'], class_folder, lens['lens_id'])
                    
                    # Track results
                    video_results.append({
                        'lens_id': lens['lens_id'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities']
                    })
                    
                    if result['prediction'] == 'contain_cell':
                        contain_count += 1
                    else:
                        no_cell_count += 1
                
                # Save CSV results
                csv_path = os.path.join(output_dir, "results.csv")
                with open(csv_path, 'w') as f:
                    f.write("lens_id,prediction,confidence,prob_no_cell,prob_contain_cell\n")
                    for r in video_results:
                        f.write(f"{r['lens_id']},{r['prediction']},{r['confidence']:.4f},"
                               f"{r['probabilities']['no_cell']:.4f},"
                               f"{r['probabilities']['contain_cell']:.4f}\n")
                
                # Save summary JSON
                summary = {
                    'video': video_name,
                    'video_path': video_path,
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_lenses': len(lenses),
                    'contain_cell_count': contain_count,
                    'no_cell_count': no_cell_count,
                    'model_path': MODEL_PATH,
                    'results': video_results
                }
                
                with open(os.path.join(output_dir, "summary.json"), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Display results
                self.append_result(f"\n📊 Results for {video_name}:")
                self.append_result(f"   Total lenses: {len(lenses)}")
                self.append_result(f"   contain_cell: {contain_count} ({contain_count/len(lenses)*100:.1f}%)")
                self.append_result(f"   no_cell: {no_cell_count} ({no_cell_count/len(lenses)*100:.1f}%)")
                self.append_result(f"   Output: {output_dir}")
                
                self.results.append({
                    'video': video_name,
                    'output_dir': output_dir,
                    'summary': summary
                })
            
            # Complete
            self.update_progress(100)
            self.update_status("✅ Processing complete!")
            self.append_result(f"\n{'='*50}")
            self.append_result("✅ ALL VIDEOS PROCESSED!")
            self.append_result(f"{'='*50}")
            
            # Enable open folder button
            self.root.after(0, lambda: self.open_folder_btn.config(state="normal"))
            
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            self.append_result(f"\n❌ ERROR: {str(e)}")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
        
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.browse_btn.config(state="normal"))
            self.is_processing = False
    
    def open_output_folder(self):
        """Open output folder in file explorer."""
        
        if self.results:
            output_dir = self.results[-1]['output_dir']
            output_dir = os.path.dirname(output_dir)  # Open parent
        else:
            output_dir = self.output_var.get()
        
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            messagebox.showwarning("Warning", f"Directory not found:\n{output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Modern look
    
    # Create app
    app = CellClassifierApp(root)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()

