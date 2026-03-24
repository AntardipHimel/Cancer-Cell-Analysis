# -*- coding: utf-8 -*-
"""
oncolens_app.py

ONCOLENS - MAIN GUI APPLICATION
================================
PyQt5-based graphical interface for cancer cell classification.

Features:
    - Step-by-step workflow
    - Video preview and selection
    - Drift correction visualization
    - Circle detection preview
    - Classification with model selection
    - Results dashboard and export

Requirements:
    pip install PyQt5

Run:
    python oncolens_app.py

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/oncolens_app.py

Author: Antardip Himel
Date: March 2026
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QMessageBox, QComboBox,
    QStackedWidget, QListWidget, QListWidgetItem, QCheckBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPalette, QColor

import cv2
import numpy as np

# Import OncoLens modules
from modules import m1_config as config
from modules import m2_video_utils as video_utils
from modules import m3_drift_correction as drift_correction
from modules import m4_matlab_bridge as matlab_bridge
from modules import m5_lens_cropping as lens_cropping
from modules import m7_classifier as classifier
from modules import m8_export_utils as export_utils


# =============================================================================
# WORKER THREADS
# =============================================================================

class DriftCorrectionWorker(QThread):
    """Background worker for drift correction."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, output_dir, num_frames=None):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.num_frames = num_frames
    
    def run(self):
        try:
            result = drift_correction.process_video(
                self.video_path,
                self.output_dir,
                num_frames=self.num_frames,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CircleDetectionWorker(QThread):
    """Background worker for MATLAB circle detection."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, frames_dir, output_dir, video_name, save_images=True):
        super().__init__()
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.video_name = video_name
        self.save_images = save_images
    
    def run(self):
        try:
            result = matlab_bridge.run_circle_detection(
                self.frames_dir,
                self.output_dir,
                self.video_name,
                save_images=self.save_images,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LensCroppingWorker(QThread):
    """Background worker for lens cropping."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, frames_dir, circles, output_dir):
        super().__init__()
        self.frames_dir = frames_dir
        self.circles = circles
        self.output_dir = output_dir
    
    def run(self):
        try:
            result = lens_cropping.crop_all_lenses(
                self.frames_dir,
                self.circles,
                self.output_dir,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ClassificationWorker(QThread):
    """Background worker for lens classification."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, lenses_dir, model_type):
        super().__init__()
        self.lenses_dir = lenses_dir
        self.model_type = model_type
    
    def run(self):
        try:
            # Create classifier
            clf = classifier.LensClassifier(self.model_type)
            
            self.progress.emit("Loading model...", 5)
            if not clf.load():
                self.error.emit("Failed to load model")
                return
            
            # Classify all lenses
            results = classifier.classify_all_lenses(
                self.lenses_dir,
                clf,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ExportWorker(QThread):
    """Background worker for exporting results."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, results, paths, video_name, model_type, export_videos=True):
        super().__init__()
        self.results = results
        self.paths = paths
        self.video_name = video_name
        self.model_type = model_type
        self.export_videos = export_videos
    
    def run(self):
        try:
            # Save classification results
            self.progress.emit("Saving classification results...", 10)
            export_utils.save_classification_results(
                self.results,
                self.paths['classification'],
                self.video_name
            )
            
            # Organize lenses by class
            self.progress.emit("Organizing lenses by class...", 30)
            export_utils.organize_lenses_by_class(
                self.results,
                self.paths['lenses'],
                self.paths['classified'],
                progress_callback=lambda msg, pct: self.progress.emit(msg, 30 + int(pct * 0.2))
            )
            
            # Export videos if requested
            if self.export_videos:
                self.progress.emit("Exporting lens videos...", 50)
                export_utils.export_all_lenses_as_videos(
                    self.results,
                    self.paths['lenses'],
                    self.paths['videos'],
                    progress_callback=lambda msg, pct: self.progress.emit(msg, 50 + int(pct * 0.3))
                )
            
            # Generate report
            self.progress.emit("Generating report...", 85)
            report_path = os.path.join(self.paths['main'], 'report.txt')
            export_utils.generate_report(
                self.results,
                report_path,
                self.video_name,
                self.model_type
            )
            
            # Create summary image
            self.progress.emit("Creating summary visualization...", 95)
            summary_path = os.path.join(self.paths['classification'], 'summary.png')
            export_utils.create_summary_image(
                self.results,
                summary_path,
                f"Classification Results - {self.video_name}"
            )
            
            self.progress.emit("Export complete!", 100)
            self.finished.emit({
                'report_path': report_path,
                'summary_path': summary_path
            })
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# STEP WIDGETS
# =============================================================================

class StepWidget(QWidget):
    """Base class for step widgets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Override in subclasses."""
        pass
    
    def reset(self):
        """Reset the step to initial state."""
        pass


class Step1VideoSelection(StepWidget):
    """Step 1: Video Selection."""
    
    video_selected = pyqtSignal(str, dict)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 1: Select Video")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Select a microscopy video file to analyze. You can browse anywhere on your PC "
            "or use the default input folder."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Browse section
        browse_group = QGroupBox("Video File")
        browse_layout = QVBoxLayout(browse_group)
        
        # File path display
        path_layout = QHBoxLayout()
        self.path_label = QLabel("No video selected")
        self.path_label.setStyleSheet("color: #666; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        path_layout.addWidget(self.path_label, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setMinimumWidth(100)
        self.browse_btn.clicked.connect(self.browse_video)
        path_layout.addWidget(self.browse_btn)
        
        browse_layout.addLayout(path_layout)
        
        # Default folder button
        default_btn = QPushButton(f"Open Default Input Folder: {config.DEFAULT_INPUT_DIR}")
        default_btn.clicked.connect(self.open_default_folder)
        default_btn.setStyleSheet("color: #1976D2;")
        browse_layout.addWidget(default_btn)
        
        layout.addWidget(browse_group)
        
        # Video info section
        self.info_group = QGroupBox("Video Information")
        info_layout = QVBoxLayout(self.info_group)
        
        self.info_label = QLabel("Select a video to see its information")
        self.info_label.setStyleSheet("color: #666;")
        info_layout.addWidget(self.info_label)
        
        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("background: #222; border-radius: 5px;")
        info_layout.addWidget(self.preview_label)
        
        layout.addWidget(self.info_group)
        layout.addStretch()
        
        # State
        self.video_path = None
        self.video_info = None
    
    def browse_video(self):
        """Open file dialog to select video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Microscopy Video",
            config.DEFAULT_INPUT_DIR,
            f"Video Files ({' '.join(['*' + ext for ext in config.VIDEO_EXTENSIONS])});;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def open_default_folder(self):
        """Open the default input folder in file explorer."""
        os.makedirs(config.DEFAULT_INPUT_DIR, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(config.DEFAULT_INPUT_DIR)
        elif sys.platform == 'darwin':
            os.system(f'open "{config.DEFAULT_INPUT_DIR}"')
        else:
            os.system(f'xdg-open "{config.DEFAULT_INPUT_DIR}"')
    
    def load_video(self, video_path):
        """Load and preview a video."""
        # Validate
        is_valid, error = video_utils.validate_video(video_path)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Video", f"Cannot load video:\n{error}")
            return
        
        # Get info
        info = video_utils.get_video_info(video_path)
        if info is None:
            QMessageBox.warning(self, "Error", "Failed to read video information")
            return
        
        self.video_path = video_path
        self.video_info = info
        
        # Update UI
        self.path_label.setText(video_path)
        self.path_label.setStyleSheet("color: #333; padding: 10px; background: #e8f5e9; border-radius: 5px;")
        
        self.info_label.setText(video_utils.format_video_info(info))
        self.info_label.setStyleSheet("color: #333;")
        
        # Show preview (first frame)
        frame = video_utils.read_frame(video_path, 0)
        if frame is not None:
            self.show_frame_preview(frame)
        
        # Emit signal
        self.video_selected.emit(video_path, info)
    
    def show_frame_preview(self, frame):
        """Display a frame in the preview label."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale to fit preview area
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 500, 350
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to QPixmap (.copy() prevents segfault if numpy array is GC'd)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        
        self.preview_label.setPixmap(pixmap)
    
    def reset(self):
        self.video_path = None
        self.video_info = None
        self.path_label.setText("No video selected")
        self.path_label.setStyleSheet("color: #666; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        self.info_label.setText("Select a video to see its information")
        self.info_label.setStyleSheet("color: #666;")
        self.preview_label.clear()


class Step2DriftCorrection(StepWidget):
    """Step 2: Drift Correction."""
    
    drift_completed = pyqtSignal(dict)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 2: Drift Correction")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Stabilize the video and extract frames. This corrects for microscope drift "
            "using phase correlation and extracts 30 evenly spaced frames."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Frames to extract:"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 100)
        self.frames_spin.setValue(config.NUM_FRAMES)
        settings_layout.addWidget(self.frames_spin)
        settings_layout.addStretch()
        
        layout.addWidget(settings_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready to start")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        
        self.run_btn = QPushButton("Run Drift Correction")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_drift_correction)
        progress_layout.addWidget(self.run_btn)
        
        layout.addWidget(progress_group)
        
        # Results section
        self.results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(self.results_group)
        
        self.results_label = QLabel("Run drift correction to see results")
        self.results_label.setStyleSheet("color: #666;")
        results_layout.addWidget(self.results_label)
        
        # Frame preview (scrollable)
        self.frame_preview = QLabel()
        self.frame_preview.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.frame_preview)
        
        layout.addWidget(self.results_group)
        layout.addStretch()
        
        # State
        self.video_path = None
        self.output_paths = None
        self.result = None
        self.worker = None
    
    def set_video(self, video_path, output_paths):
        """Set the video to process."""
        self.video_path = video_path
        self.output_paths = output_paths
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Ready to process: {os.path.basename(video_path)}")
    
    def run_drift_correction(self):
        """Start drift correction in background thread."""
        if not self.video_path or not self.output_paths:
            return
        if self.worker and self.worker.isRunning():
            return
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = DriftCorrectionWorker(
            self.video_path,
            self.output_paths['frames'],
            num_frames=self.frames_spin.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message, percent):
        """Handle progress updates."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, result):
        """Handle completion."""
        self.result = result
        self.run_btn.setEnabled(True)
        self.status_label.setText("Drift correction complete!")
        self.progress_bar.setValue(100)
        
        # Update results display
        self.results_label.setText(
            f"✓ Extracted {result['num_frames']} frames\n"
            f"✓ Saved to: {result['output_dir']}\n"
            f"✓ {drift_correction.get_drift_summary(result['drift_data'])}"
        )
        self.results_label.setStyleSheet("color: #2e7d32;")
        
        # Show first frame preview
        if result['frames']:
            self.show_frame_montage(result['frames'][:6])
        
        self.drift_completed.emit(result)
    
    def on_error(self, error_msg):
        """Handle errors."""
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #c62828;")
        QMessageBox.critical(self, "Error", f"Drift correction failed:\n{error_msg}")
    
    def show_frame_montage(self, frames):
        """Show a montage of frames."""
        if not frames:
            return
        
        # Create montage (2 rows x 3 cols)
        thumb_size = 120
        cols = min(3, len(frames))
        rows = (len(frames) + cols - 1) // cols
        
        montage = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames[:6]):
            row, col = i // cols, i % cols
            thumb = cv2.resize(frame, (thumb_size, thumb_size))
            if len(thumb.shape) == 2:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            montage[row*thumb_size:(row+1)*thumb_size, col*thumb_size:(col+1)*thumb_size] = thumb
        
        # Convert to QPixmap (.copy() prevents segfault if numpy array is GC'd)
        montage_rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        h, w, ch = montage_rgb.shape
        q_img = QImage(montage_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.frame_preview.setPixmap(QPixmap.fromImage(q_img))
    
    def reset(self):
        self.video_path = None
        self.output_paths = None
        self.result = None
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to start")
        self.results_label.setText("Run drift correction to see results")
        self.results_label.setStyleSheet("color: #666;")
        self.frame_preview.clear()


class Step3CircleDetection(StepWidget):
    """Step 3: Circle Detection (MATLAB)."""
    
    detection_completed = pyqtSignal(dict)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 3: Circle Detection")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Detect circular lenses in the frames using MATLAB's Hough Transform. "
            "Make sure MATLAB is installed and accessible from command line."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # MATLAB status
        status_group = QGroupBox("MATLAB Status")
        status_layout = QVBoxLayout(status_group)
        
        self.matlab_status = QLabel("Checking MATLAB...")
        status_layout.addWidget(self.matlab_status)
        
        layout.addWidget(status_group)
        
        # Options
        options_group = QGroupBox("Detection Options")
        options_layout = QVBoxLayout(options_group)
        
        self.save_images_cb = QCheckBox("Save annotated frames (green=OK, yellow=shrunk, red=edge-cut)")
        self.save_images_cb.setChecked(True)
        options_layout.addWidget(self.save_images_cb)
        
        self.save_qc_cb = QCheckBox("Save QC images (enhanced, smoothed, binary mask, plots)")
        self.save_qc_cb.setChecked(True)
        self.save_qc_cb.setEnabled(False)  # Always saved with data
        options_layout.addWidget(self.save_qc_cb)
        
        layout.addWidget(options_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready to start")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.run_btn = QPushButton("Run Circle Detection")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_detection)
        self.run_btn.setEnabled(False)
        progress_layout.addWidget(self.run_btn)
        
        layout.addWidget(progress_group)
        
        # Results
        self.results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(self.results_group)
        
        self.results_label = QLabel("Run detection to see results")
        self.results_label.setStyleSheet("color: #666;")
        results_layout.addWidget(self.results_label)
        
        self.detection_preview = QLabel()
        self.detection_preview.setAlignment(Qt.AlignCenter)
        self.detection_preview.setMinimumHeight(300)
        results_layout.addWidget(self.detection_preview)
        
        layout.addWidget(self.results_group)
        layout.addStretch()
        
        # State
        self.frames_dir = None
        self.output_dir = None
        self.video_name = None
        self.result = None
        self.worker = None
        
        # Check MATLAB (deferred so it doesn't block GUI startup)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self.check_matlab)
    
    def check_matlab(self):
        """Check if MATLAB is available (runs deferred to avoid blocking UI)."""
        self.matlab_status.setText("Checking MATLAB availability...")
        self.matlab_status.setStyleSheet("color: #666;")
        QApplication.processEvents()  # Paint the "checking" message first
        
        bridge = matlab_bridge.MatlabBridge()
        if bridge.is_available:
            self.matlab_status.setText("✓ MATLAB is available")
            self.matlab_status.setStyleSheet("color: #2e7d32;")
        else:
            self.matlab_status.setText(
                f"✗ MATLAB not found at: {bridge.matlab_exe}\n"
                "Please install MATLAB or update the path in m1_config.py"
            )
            self.matlab_status.setStyleSheet("color: #c62828;")
    
    def set_data(self, frames_dir, output_dir, video_name):
        """Set the data for detection."""
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.video_name = video_name
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Ready to detect circles in: {video_name}")
    
    def run_detection(self):
        """Run circle detection."""
        if not self.frames_dir:
            return
        if self.worker and self.worker.isRunning():
            return
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = CircleDetectionWorker(
            self.frames_dir,
            self.output_dir,
            self.video_name,
            save_images=self.save_images_cb.isChecked()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, result):
        self.result = result
        self.run_btn.setEnabled(True)
        
        if result['success']:
            self.status_label.setText("Circle detection complete!")
            self.progress_bar.setValue(100)
            
            # Count by status
            circles = result.get('circles', [])
            num_ok = sum(1 for c in circles if c.get('status') == 'ok')
            num_shrunk = sum(1 for c in circles if c.get('status') == 'shrunk')
            num_edge = sum(1 for c in circles if c.get('status') == 'edge')
            usable = result.get('usable_circles', len(circles) - num_edge)
            
            self.results_label.setText(
                f"✓ Detected {result['num_circles']} circles\n"
                f"   🟢 OK: {num_ok}  |  🟡 Shrunk: {num_shrunk}  |  🔴 Edge-cut: {num_edge}\n"
                f"   Usable for cropping: 1 to {usable}\n"
                f"✓ Results saved to: {result['output_dir']}"
            )
            self.results_label.setStyleSheet("color: #2e7d32;")
            
            # Show circle map image from logs
            circle_map = os.path.join(result['output_dir'], 'logs', 'circle_map.png')
            if os.path.exists(circle_map):
                self.show_annotated_image(circle_map)
            else:
                # Try first annotated frame
                images_dir = result.get('images_dir', '')
                if images_dir and os.path.exists(images_dir):
                    frames = sorted(os.listdir(images_dir))
                    if frames:
                        self.show_annotated_image(os.path.join(images_dir, frames[0]))
            
            self.detection_completed.emit(result)
        else:
            self.on_error(result.get('error', 'Unknown error'))
    
    def on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Detection failed")
        self.status_label.setStyleSheet("color: #c62828;")
        QMessageBox.critical(self, "Error", f"Circle detection failed:\n{error_msg}")
    
    def show_annotated_image(self, image_path):
        """Display the annotated image."""
        pixmap = QPixmap(image_path)
        scaled = pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.detection_preview.setPixmap(scaled)
    
    def reset(self):
        self.frames_dir = None
        self.output_dir = None
        self.video_name = None
        self.result = None
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to start")
        self.results_label.setText("Run detection to see results")
        self.results_label.setStyleSheet("color: #666;")
        self.detection_preview.clear()


class Step4LensCropping(StepWidget):
    """Step 4: Lens Cropping."""
    
    cropping_completed = pyqtSignal(dict)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 4: Lens Cropping")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Crop each detected lens from all 30 frames. The cropped images will be "
            "resized to 96x96 pixels and converted to grayscale for the classifier."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready to start")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.run_btn = QPushButton("Crop Lenses")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_cropping)
        self.run_btn.setEnabled(False)
        progress_layout.addWidget(self.run_btn)
        
        layout.addWidget(progress_group)
        
        # Results
        self.results_group = QGroupBox("Cropped Lenses")
        results_layout = QVBoxLayout(self.results_group)
        
        self.results_label = QLabel("Run cropping to see results")
        self.results_label.setStyleSheet("color: #666;")
        results_layout.addWidget(self.results_label)
        
        self.lens_preview = QLabel()
        self.lens_preview.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.lens_preview)
        
        layout.addWidget(self.results_group)
        layout.addStretch()
        
        # State
        self.frames_dir = None
        self.circles = None
        self.output_dir = None
        self.result = None
        self.worker = None
    
    def set_data(self, frames_dir, circles, output_dir):
        """Set data for cropping."""
        self.frames_dir = frames_dir
        self.circles = circles
        self.output_dir = output_dir
        self.run_btn.setEnabled(True)
        usable = sum(1 for c in circles if not c.get('is_edge', False))
        edge_cut = len(circles) - usable
        if edge_cut > 0:
            self.status_label.setText(f"Ready to crop {usable} usable lenses (skipping {edge_cut} edge-cut)")
        else:
            self.status_label.setText(f"Ready to crop {usable} lenses")
    
    def run_cropping(self):
        """Run lens cropping."""
        if not self.frames_dir or not self.circles:
            return
        if self.worker and self.worker.isRunning():
            return
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = LensCroppingWorker(
            self.frames_dir,
            self.circles,
            self.output_dir
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, result):
        self.result = result
        self.run_btn.setEnabled(True)
        
        if result['success']:
            self.status_label.setText("Cropping complete!")
            self.progress_bar.setValue(100)
            
            self.results_label.setText(
                f"✓ Cropped {result['num_lenses']} lenses\n"
                f"✓ Each lens has 30 frames (96x96 grayscale)\n"
                f"✓ Saved to: {result['output_dir']}"
            )
            self.results_label.setStyleSheet("color: #2e7d32;")
            
            # Show lens montage
            self.show_lens_montage(result['lenses'][:12])
            
            self.cropping_completed.emit(result)
        else:
            self.on_error(result.get('error', 'Unknown error'))
    
    def on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Cropping failed")
        QMessageBox.critical(self, "Error", f"Lens cropping failed:\n{error_msg}")
    
    def show_lens_montage(self, lenses):
        """Show montage of cropped lenses (in original color)."""
        if not lenses:
            return
        
        thumb_size = 64
        cols = min(6, len(lenses))
        rows = (len(lenses) + cols - 1) // cols
        
        montage = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
        
        for i, lens_data in enumerate(lenses):
            if lens_data['frames']:
                row, col = i // cols, i % cols
                frame = lens_data['frames'][0]
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                thumb = cv2.resize(frame, (thumb_size, thumb_size))
                montage[row*thumb_size:(row+1)*thumb_size, col*thumb_size:(col+1)*thumb_size] = thumb
        
        # Convert to QPixmap (.copy() prevents segfault if numpy array is GC'd)
        montage_rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        h, w, ch = montage_rgb.shape
        q_img = QImage(montage_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.lens_preview.setPixmap(QPixmap.fromImage(q_img))
    
    def reset(self):
        self.frames_dir = None
        self.circles = None
        self.output_dir = None
        self.result = None
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to start")
        self.results_label.setText("Run cropping to see results")
        self.results_label.setStyleSheet("color: #666;")
        self.lens_preview.clear()


class Step5Classification(StepWidget):
    """Step 5: Classification."""
    
    classification_completed = pyqtSignal(list)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 5: Classification")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Classify each lens as 'contain_cell' or 'no_cell' using the trained deep learning model."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("ResNet+LSTM (96.58% accuracy) - Recommended", "resnet_lstm")
        self.model_combo.addItem("3D CNN (94.01% accuracy)", "3dcnn")
        model_layout.addWidget(self.model_combo)
        
        # Model status
        self.model_status = QLabel()
        self.update_model_status()
        model_layout.addWidget(self.model_status)
        
        layout.addWidget(model_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready to start")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.run_btn = QPushButton("Run Classification")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_classification)
        self.run_btn.setEnabled(False)
        progress_layout.addWidget(self.run_btn)
        
        layout.addWidget(progress_group)
        
        # Results summary
        self.results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout(self.results_group)
        
        self.results_label = QLabel("Run classification to see results")
        self.results_label.setStyleSheet("color: #666;")
        results_layout.addWidget(self.results_label)
        
        layout.addWidget(self.results_group)
        layout.addStretch()
        
        # State
        self.lenses_dir = None
        self.results = None
        self.worker = None
    
    def update_model_status(self):
        """Update model availability status."""
        resnet_exists = config.check_model_exists('resnet_lstm')
        cnn3d_exists = config.check_model_exists('3dcnn')
        
        status = []
        if resnet_exists:
            status.append("✓ ResNet+LSTM model found")
        else:
            status.append(f"✗ ResNet+LSTM not found: {config.RESNET_LSTM_MODEL_PATH}")
        
        if cnn3d_exists:
            status.append("✓ 3D CNN model found")
        else:
            status.append(f"✗ 3D CNN not found: {config.CNN3D_MODEL_PATH}")
        
        self.model_status.setText("\n".join(status))
        color = "#2e7d32" if (resnet_exists or cnn3d_exists) else "#c62828"
        self.model_status.setStyleSheet(f"color: {color};")
    
    def set_data(self, lenses_dir):
        """Set lenses directory."""
        self.lenses_dir = lenses_dir
        self.run_btn.setEnabled(True)
        
        num_lenses = lens_cropping.count_lenses_in_directory(lenses_dir)
        self.status_label.setText(f"Ready to classify {num_lenses} lenses")
    
    def run_classification(self):
        """Run classification."""
        if not self.lenses_dir:
            return
        if self.worker and self.worker.isRunning():
            return
        
        model_type = self.model_combo.currentData()
        
        if not config.check_model_exists(model_type):
            QMessageBox.warning(
                self, "Model Not Found",
                f"The selected model was not found:\n{config.get_model_path(model_type)}"
            )
            return
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = ClassificationWorker(self.lenses_dir, model_type)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, results):
        self.results = results
        self.run_btn.setEnabled(True)
        self.status_label.setText("Classification complete!")
        self.progress_bar.setValue(100)
        
        # Get summary
        summary = classifier.get_classification_summary(results)
        
        self.results_label.setText(
            f"Total Lenses: {summary['total']}\n\n"
            f"✓ Contains Cell: {summary['contain_cell']} ({summary['contain_cell_pct']:.1f}%)\n"
            f"✗ No Cell: {summary['no_cell']} ({summary['no_cell_pct']:.1f}%)\n\n"
            f"Average Confidence: {summary['avg_confidence_pct']:.1f}%\n"
            f"Low Confidence (<{config.CONFIDENCE_THRESHOLD*100:.0f}%): {summary['low_confidence_count']}"
        )
        self.results_label.setStyleSheet("color: #333; font-size: 14px;")
        
        self.classification_completed.emit(results)
    
    def on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Classification failed")
        QMessageBox.critical(self, "Error", f"Classification failed:\n{error_msg}")
    
    def get_model_type(self):
        """Get selected model type."""
        return self.model_combo.currentData()
    
    def reset(self):
        self.lenses_dir = None
        self.results = None
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to start")
        self.results_label.setText("Run classification to see results")
        self.results_label.setStyleSheet("color: #666;")


class Step6Export(StepWidget):
    """Step 6: Export Results."""
    
    export_completed = pyqtSignal(dict)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Step 6: Export Results")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Export classification results, organize lenses by class, and generate reports."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)
        
        self.export_videos_cb = QCheckBox("Export lens videos (organized by class)")
        self.export_videos_cb.setChecked(True)
        options_layout.addWidget(self.export_videos_cb)
        
        layout.addWidget(options_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready to export")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.run_btn = QPushButton("Export All")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_export)
        self.run_btn.setEnabled(False)
        progress_layout.addWidget(self.run_btn)
        
        layout.addWidget(progress_group)
        
        # Output info
        self.output_group = QGroupBox("Output Location")
        output_layout = QVBoxLayout(self.output_group)
        
        self.output_label = QLabel("Export to see output location")
        self.output_label.setStyleSheet("color: #666;")
        output_layout.addWidget(self.output_label)
        
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        output_layout.addWidget(self.open_folder_btn)
        
        layout.addWidget(self.output_group)
        layout.addStretch()
        
        # State
        self.results = None
        self.paths = None
        self.video_name = None
        self.model_type = None
        self.worker = None
    
    def set_data(self, results, paths, video_name, model_type):
        """Set data for export."""
        self.results = results
        self.paths = paths
        self.video_name = video_name
        self.model_type = model_type
        self.run_btn.setEnabled(True)
        
        summary = classifier.get_classification_summary(results)
        self.status_label.setText(
            f"Ready to export {summary['total']} classifications"
        )
    
    def run_export(self):
        """Run export."""
        if not self.results or not self.paths:
            return
        if self.worker and self.worker.isRunning():
            return
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = ExportWorker(
            self.results,
            self.paths,
            self.video_name,
            self.model_type,
            self.export_videos_cb.isChecked()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.status_label.setText("Export complete!")
        self.progress_bar.setValue(100)
        
        self.output_label.setText(
            f"✓ All results exported to:\n{self.paths['main']}\n\n"
            f"Contents:\n"
            f"  • 1_extracted_frames/ - Drift-corrected frames\n"
            f"  • 2_circle_detection/ - MATLAB results\n"
            f"  • 3_cropped_lenses/ - Individual lens frames\n"
            f"  • 4_classification/ - CSV & JSON results\n"
            f"  • 5_classified_lenses/ - Organized by class\n"
            f"  • 6_videos/ - Lens videos by class\n"
            f"  • report.txt - Summary report"
        )
        self.output_label.setStyleSheet("color: #2e7d32;")
        
        self.export_completed.emit(result)
        
        # Ask to open folder
        reply = QMessageBox.question(
            self, "Export Complete",
            "Export completed successfully!\n\nWould you like to open the output folder?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.open_output_folder()
    
    def on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Export failed")
        QMessageBox.critical(self, "Error", f"Export failed:\n{error_msg}")
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        if self.paths and 'main' in self.paths:
            path = self.paths['main']
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
    
    def reset(self):
        self.results = None
        self.paths = None
        self.video_name = None
        self.model_type = None
        self.run_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to export")
        self.output_label.setText("Export to see output location")
        self.output_label.setStyleSheet("color: #666;")


# =============================================================================
# MAIN WINDOW
# =============================================================================

class OncoLensApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setMinimumSize(1200, 800)
        
        # State
        self.video_path = None
        self.video_info = None
        self.output_paths = None
        self.circles = None
        self.classification_results = None
        self.current_step = 0
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left sidebar - Steps
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #1976D2;
            }
            QLabel {
                color: white;
            }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(5)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        
        # Logo/Title
        title_label = QLabel("🔬 OncoLens")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)
        
        version_label = QLabel(f"v{config.APP_VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #90CAF9;")
        sidebar_layout.addWidget(version_label)
        
        sidebar_layout.addSpacing(30)
        
        # Step buttons
        self.step_buttons = []
        step_names = [
            "1. Select Video",
            "2. Drift Correction",
            "3. Circle Detection",
            "4. Lens Cropping",
            "5. Classification",
            "6. Export Results"
        ]
        
        for i, name in enumerate(step_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #90CAF9;
                    text-align: left;
                    padding-left: 15px;
                    border: none;
                    border-radius: 5px;
                    font-size: 13px;
                }
                QPushButton:checked {
                    background-color: #0D47A1;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
                QPushButton:disabled {
                    color: #5C6BC0;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self.go_to_step(idx))
            self.step_buttons.append(btn)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        
        # Device info
        device_info = config.get_device_info()
        device_label = QLabel(f"Device: {device_info['device']}")
        device_label.setStyleSheet("color: #90CAF9; font-size: 11px;")
        sidebar_layout.addWidget(device_label)
        
        main_layout.addWidget(sidebar)
        
        # Right content area
        content_area = QFrame()
        content_area.setStyleSheet("background-color: #FAFAFA;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Stacked widget for steps
        self.stack = QStackedWidget()
        
        self.step1 = Step1VideoSelection()
        self.step2 = Step2DriftCorrection()
        self.step3 = Step3CircleDetection()
        self.step4 = Step4LensCropping()
        self.step5 = Step5Classification()
        self.step6 = Step6Export()
        
        self.stack.addWidget(self.step1)
        self.stack.addWidget(self.step2)
        self.stack.addWidget(self.step3)
        self.stack.addWidget(self.step4)
        self.stack.addWidget(self.step5)
        self.stack.addWidget(self.step6)
        
        content_layout.addWidget(self.stack)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        
        self.back_btn = QPushButton("◄ Back")
        self.back_btn.setMinimumWidth(100)
        self.back_btn.clicked.connect(self.go_back)
        nav_layout.addWidget(self.back_btn)
        
        self.next_btn = QPushButton("Next ►")
        self.next_btn.setMinimumWidth(100)
        self.next_btn.setStyleSheet("background-color: #1976D2; color: white;")
        self.next_btn.clicked.connect(self.go_next)
        nav_layout.addWidget(self.next_btn)
        
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.setMinimumWidth(100)
        self.reset_btn.clicked.connect(self.reset_all)
        nav_layout.addWidget(self.reset_btn)
        
        content_layout.addLayout(nav_layout)
        
        main_layout.addWidget(content_area)
        
        # Initialize
        self.go_to_step(0)
        self.update_step_states()
    
    def setup_connections(self):
        """Setup signal connections."""
        self.step1.video_selected.connect(self.on_video_selected)
        self.step2.drift_completed.connect(self.on_drift_completed)
        self.step3.detection_completed.connect(self.on_detection_completed)
        self.step4.cropping_completed.connect(self.on_cropping_completed)
        self.step5.classification_completed.connect(self.on_classification_completed)
        self.step6.export_completed.connect(self.on_export_completed)
    
    def go_to_step(self, step_idx):
        """Navigate to a specific step."""
        self.current_step = step_idx
        self.stack.setCurrentIndex(step_idx)
        
        # Update button states
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == step_idx)
        
        self.back_btn.setEnabled(step_idx > 0)
        self.next_btn.setEnabled(step_idx < 5)
    
    def go_back(self):
        """Go to previous step."""
        if self.current_step > 0:
            self.go_to_step(self.current_step - 1)
    
    def go_next(self):
        """Go to next step."""
        if self.current_step < 5:
            self.go_to_step(self.current_step + 1)
    
    def update_step_states(self):
        """Update which steps are enabled based on progress."""
        # Step 1 always enabled
        self.step_buttons[0].setEnabled(True)
        
        # Other steps depend on previous completion
        self.step_buttons[1].setEnabled(self.video_path is not None)
        self.step_buttons[2].setEnabled(self.step2.result is not None)
        self.step_buttons[3].setEnabled(self.step3.result is not None and self.step3.result.get('success'))
        self.step_buttons[4].setEnabled(self.step4.result is not None and self.step4.result.get('success'))
        self.step_buttons[5].setEnabled(self.step5.results is not None)
    
    def on_video_selected(self, video_path, video_info):
        """Handle video selection."""
        self.video_path = video_path
        self.video_info = video_info
        
        # Create output structure
        video_name = video_info['name']
        self.output_paths = config.create_output_structure(video_name)
        
        # Setup step 2
        self.step2.set_video(video_path, self.output_paths)
        
        self.update_step_states()
    
    def on_drift_completed(self, result):
        """Handle drift correction completion."""
        # Setup step 3
        self.step3.set_data(
            result['output_dir'],
            self.output_paths['circles'],
            self.video_info['name']
        )
        
        self.update_step_states()
        
        # Auto-advance
        self.go_to_step(2)
    
    def on_detection_completed(self, result):
        """Handle circle detection completion."""
        self.circles = result['circles']
        
        # Setup step 4
        self.step4.set_data(
            self.output_paths['frames'],
            self.circles,
            self.output_paths['lenses']
        )
        
        self.update_step_states()
        
        # Auto-advance
        self.go_to_step(3)
    
    def on_cropping_completed(self, result):
        """Handle cropping completion."""
        # Setup step 5
        self.step5.set_data(self.output_paths['lenses'])
        
        self.update_step_states()
        
        # Auto-advance
        self.go_to_step(4)
    
    def on_classification_completed(self, results):
        """Handle classification completion."""
        self.classification_results = results
        
        # Setup step 6
        self.step6.set_data(
            results,
            self.output_paths,
            self.video_info['name'],
            self.step5.get_model_type()
        )
        
        self.update_step_states()
        
        # Auto-advance
        self.go_to_step(5)
    
    def on_export_completed(self, result):
        """Handle export completion."""
        self.update_step_states()
    
    def reset_all(self):
        """Reset all steps."""
        reply = QMessageBox.question(
            self, "Reset All",
            "Are you sure you want to reset all steps?\nThis will clear all progress.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.video_path = None
            self.video_info = None
            self.output_paths = None
            self.circles = None
            self.classification_results = None
            
            self.step1.reset()
            self.step2.reset()
            self.step3.reset()
            self.step4.reset()
            self.step5.reset()
            self.step6.reset()
            
            self.update_step_states()
            self.go_to_step(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set palette for modern look
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.Highlight, QColor(25, 118, 210))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = OncoLensApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()