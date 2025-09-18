# src/sirf_viewer/gui.py
"""
GUI application for SIRF ImageData and AcquisitionData viewer.

This module provides a complete GUI application using PyQt5 for viewing
SIRF data objects with file selection, interactive controls, and export capabilities.
"""

import sys
import os
from typing import Optional, Any
import numpy as np

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                               QSlider, QComboBox, QSpinBox, QGroupBox, 
                               QMessageBox, QSplitter, QFrame)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

try:
    import sirf.STIR as sirf
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False

from .viewers import SIRFViewer


class SIRFViewerGUI(QMainWindow):
    """
    Main GUI application for SIRF data viewer.
    
    This class provides a complete GUI with file selection, viewing controls,
    and export options for SIRF ImageData and AcquisitionData objects.
    """
    
    def __init__(self):
        super().__init__()
        
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for GUI functionality")
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF package is required. Install with: pip install sirf")
            
        self.current_data = None
        self.viewer = None
        self.animation_timer = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('SIRF Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Display
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 900])
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel with file selection and viewing controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Load ImageData button
        self.load_image_btn = QPushButton('Load ImageData (.hv)')
        self.load_image_btn.clicked.connect(self.load_image_data)
        file_layout.addWidget(self.load_image_btn)
        
        # Load AcquisitionData button
        self.load_acq_btn = QPushButton('Load AcquisitionData (.hs)')
        self.load_acq_btn.clicked.connect(self.load_acquisition_data)
        file_layout.addWidget(self.load_acq_btn)
        
        # File info label
        self.file_info_label = QLabel('No file loaded')
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Viewing controls group
        self.view_group = QGroupBox("Viewing Controls")
        view_layout = QVBoxLayout()
        self.view_group.setLayout(view_layout)
        layout.addWidget(self.view_group)
        self.view_group.setEnabled(False)
        
        # Dimension controls will be added dynamically
        self.dimension_widgets = []
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel('Colormap:'))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        display_layout.addLayout(colormap_layout)
        
        # Window/Level controls
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel('Window:'))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10000)
        self.window_spin.setValue(1000)
        self.window_spin.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(self.window_spin)
        
        window_layout.addWidget(QLabel('Level:'))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 10000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(self.level_spin)
        
        display_layout.addLayout(window_layout)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Animation controls
        anim_group = QGroupBox("Animation")
        anim_layout = QVBoxLayout()
        
        # Animation controls
        anim_controls_layout = QHBoxLayout()
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_animation)
        anim_controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_animation)
        anim_controls_layout.addWidget(self.stop_btn)
        
        anim_layout.addLayout(anim_controls_layout)
        
        # FPS control
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel('FPS:'))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        fps_layout.addWidget(self.fps_spin)
        anim_layout.addLayout(fps_layout)
        
        # Create GIF button
        self.create_gif_btn = QPushButton('Create GIF')
        self.create_gif_btn.clicked.connect(self.create_gif)
        anim_layout.addWidget(self.create_gif_btn)
        
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)
        anim_group.setEnabled(False)
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.save_image_btn = QPushButton('Save Current View')
        self.save_image_btn.clicked.connect(self.save_current_view)
        export_layout.addWidget(self.save_image_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        export_group.setEnabled(False)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        return panel
        
    def create_display_panel(self) -> QWidget:
        """Create the display panel with matplotlib canvas."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        return panel
        
    def load_image_data(self):
        """Load ImageData from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load ImageData', '', 'ImageData files (*.hv);;All files (*)'
        )
        
        if filename:
            try:
                self.current_data = sirf.ImageData(filename)
                self.file_info_label.setText(f'Loaded: {os.path.basename(filename)}\nType: ImageData')
                self.setup_viewer()
                self.statusBar().showMessage(f'Loaded ImageData: {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load ImageData: {str(e)}')
                
    def load_acquisition_data(self):
        """Load AcquisitionData from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load AcquisitionData', '', 'AcquisitionData files (*.hs);;All files (*)'
        )
        
        if filename:
            try:
                self.current_data = sirf.AcquisitionData(filename)
                self.file_info_label.setText(f'Loaded: {os.path.basename(filename)}\nType: AcquisitionData')
                self.setup_viewer()
                self.statusBar().showMessage(f'Loaded AcquisitionData: {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load AcquisitionData: {str(e)}')
                
    def setup_viewer(self):
        """Set up the viewer for the current data."""
        if self.current_data is None:
            return
            
        # Create viewer
        self.viewer = SIRFViewer(self.current_data, "SIRF Viewer")
        
        # Set up dimension controls
        self.setup_dimension_controls()
        
        # Enable controls
        self.view_group.setEnabled(True)
        self.findChild(QGroupBox, "Animation").setEnabled(True)
        self.findChild(QGroupBox, "Export").setEnabled(True)
        
        # Initial display
        self.update_display()
        
    def setup_dimension_controls(self):
        """Set up dimension control sliders based on data dimensions."""
        # Clear existing dimension widgets
        for widget in self.dimension_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.dimension_widgets = []
        
        if self.viewer is None:
            return
            
        # Create sliders for scrollable dimensions
        num_sliders = min(2, len(self.viewer.dimensions) - 2)
        
        for i in range(num_sliders):
            # Create slider
            slider_layout = QHBoxLayout()
            
            label = QLabel(f'{self.viewer.dimension_names[i]}:')
            slider_layout.addWidget(label)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, self.viewer.dimensions[i] - 1)
            slider.setValue(self.viewer.current_indices[i])
            slider.valueChanged.connect(lambda val, idx=i: self.update_dimension(idx, val))
            slider_layout.addWidget(slider)
            
            # Value label
            value_label = QLabel(str(self.viewer.current_indices[i]))
            value_label.setMinimumWidth(30)
            slider_layout.addWidget(value_label)
            
            # Add to layout
            self.view_group.layout().insertLayout(
                len(self.dimension_widgets), slider_layout
            )
            
            self.dimension_widgets.extend([label, slider, value_label])
            
    def update_dimension(self, dim_idx: int, value: int):
        """Update a specific dimension."""
        if self.viewer is None:
            return
            
        self.viewer.current_indices[dim_idx] = value
        
        # Update value label
        value_label_idx = 2 + dim_idx * 3  # Find the value label
        if value_label_idx < len(self.dimension_widgets):
            self.dimension_widgets[value_label_idx].setText(str(value))
            
        self.update_display()
        
    def update_display(self):
        """Update the display."""
        if self.viewer is None:
            return
            
        # Clear the figure
        self.figure.clear()
        
        # Get current slice
        slice_data = self.viewer._get_current_slice()
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        im = ax.imshow(slice_data, cmap=self.viewer.colormap, origin='lower')
        self.figure.colorbar(im, ax=ax)
        
        # Update title
        index_str = ', '.join([f'{name}: {idx}' 
                             for name, idx in zip(self.viewer.dimension_names, self.viewer.current_indices)])
        ax.set_title(f'Slice - {index_str}')
        
        # Redraw canvas
        self.canvas.draw()
        
    def update_colormap(self, colormap: str):
        """Update the colormap."""
        if self.viewer is not None:
            self.viewer.set_colormap(colormap)
            self.update_display()
            
    def update_window_level(self):
        """Update window/level settings."""
        if self.viewer is not None:
            window = self.window_spin.value()
            level = self.level_spin.value()
            self.viewer.set_window(level, window)
            self.update_display()
            
    def toggle_animation(self):
        """Toggle animation playback."""
        if self.animation_timer is None:
            # Start animation
            fps = self.fps_spin.value()
            interval = 1000 // fps
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.animation_step)
            self.animation_timer.start(interval)
            self.play_btn.setText('Pause')
        else:
            # Pause animation
            self.animation_timer.stop()
            self.animation_timer = None
            self.play_btn.setText('Play')
            
    def stop_animation(self):
        """Stop animation and reset to first frame."""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
            self.play_btn.setText('Play')
            
        # Reset to first frame
        if self.viewer is not None and len(self.dimension_widgets) > 0:
            slider = self.dimension_widgets[1]  # First slider
            slider.setValue(0)
            
    def animation_step(self):
        """Advance animation by one frame."""
        if self.viewer is None or len(self.dimension_widgets) == 0:
            return
            
        # Get first slider
        slider = self.dimension_widgets[1]  # First slider
        current_val = slider.value()
        max_val = slider.maximum()
        
        # Advance to next frame or loop back to start
        next_val = current_val + 1
        if next_val > max_val:
            next_val = 0
            
        slider.setValue(next_val)
        
    def create_gif(self):
        """Create animated GIF."""
        if self.viewer is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save GIF', '', 'GIF files (*.gif);;All files (*)'
        )
        
        if filename:
            try:
                fps = self.fps_spin.value()
                self.viewer.create_gif(filename, fps)
                self.statusBar().showMessage(f'Created GIF: {filename}')
                QMessageBox.information(self, 'Success', f'GIF saved to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to create GIF: {str(e)}')
                
    def save_current_view(self):
        """Save current view as image."""
        if self.viewer is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 'PNG files (*.png);;All files (*)'
        )
        
        if filename:
            try:
                self.viewer.save_current_view(filename)
                self.statusBar().showMessage(f'Saved image: {filename}')
                QMessageBox.information(self, 'Success', f'Image saved to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image: {str(e)}')


def launch_gui():
    """Launch the SIRF Viewer GUI application."""
    app = QApplication(sys.argv)
    viewer = SIRFViewerGUI()
    viewer.show()
    sys.exit(app.exec_())


def main():
    """Main entry point for the GUI application."""
    launch_gui()


if __name__ == '__main__':
    main()