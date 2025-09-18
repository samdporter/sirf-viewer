# src/sirf_viewer/widgets/controls.py
"""
Custom control widgets for SIRF viewer.

This module provides specialized control widgets for dimension navigation,
animation control, and other interactive elements.
"""

from typing import Optional, List, Any
import warnings

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QPushButton, QSpinBox, QGroupBox, 
                               QComboBox, QCheckBox)
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class DimensionControlWidget(QWidget):
    """
    Widget for controlling data dimensions.
    
    This widget provides sliders and labels for controlling multiple
    dimensions of SIRF data objects.
    
    Signals
    -------
    dimension_changed(int, int)
        Emitted when a dimension value changes (dimension_index, value)
    """
    
    dimension_changed = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for widget functionality")
            
        self.dimensions = []
        self.dimension_names = []
        self.sliders = []
        self.labels = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        
    def set_dimensions(self, dimensions: List[int], dimension_names: Optional[List[str]] = None):
        """
        Set the dimensions to control.
        
        Parameters
        ----------
        dimensions : list of int
            List of dimension sizes
        dimension_names : list of str, optional
            Names for each dimension
        """
        # Clear existing widgets
        self.clear_widgets()
        
        self.dimensions = dimensions
        self.dimension_names = dimension_names or [f'Dim {i}' for i in range(len(dimensions))]
        
        # Create controls for each dimension
        for i, (size, name) in enumerate(zip(self.dimensions, self.dimension_names)):
            # Create horizontal layout for this dimension
            dim_layout = QHBoxLayout()
            
            # Label
            label = QLabel(f"{name}:")
            dim_layout.addWidget(label)
            self.labels.append(label)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, size - 1)
            slider.setValue(size // 2)
            slider.valueChanged.connect(lambda val, idx=i: self.on_dimension_changed(idx, val))
            dim_layout.addWidget(slider)
            self.sliders.append(slider)
            
            # Value label
            value_label = QLabel(str(size // 2))
            value_label.setMinimumWidth(30)
            dim_layout.addWidget(value_label)
            self.labels.append(value_label)
            
            # Add to main layout
            self.layout.addLayout(dim_layout)
            
        # Add stretch at the end
        self.layout.addStretch()
        
    def clear_widgets(self):
        """Clear all existing widgets."""
        for slider in self.sliders:
            slider.setParent(None)
            slider.deleteLater()
        for label in self.labels:
            label.setParent(None)
            label.deleteLater()
            
        # Clear layouts
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                        
        self.sliders = []
        self.labels = []
        
    def on_dimension_changed(self, dimension_index: int, value: int):
        """Handle dimension value change."""
        # Update value label
        label_idx = dimension_index * 2 + 1  # Value label is second for each dimension
        if label_idx < len(self.labels):
            self.labels[label_idx].setText(str(value))
            
        self.dimension_changed.emit(dimension_index, value)
        
    def set_dimension_value(self, dimension_index: int, value: int):
        """Set the value for a specific dimension."""
        if dimension_index < len(self.sliders):
            self.sliders[dimension_index].setValue(value)
            
    def get_dimension_values(self) -> List[int]:
        """Get current values for all dimensions."""
        return [slider.value() for slider in self.sliders]


class AnimationControlWidget(QWidget):
    """
    Widget for controlling animation playback.
    
    This widget provides controls for playing, pausing, and configuring
    animations of SIRF data objects.
    
    Signals
    -------
    play_clicked()
        Emitted when play button is clicked
    pause_clicked()
        Emitted when pause button is clicked
    stop_clicked()
        Emitted when stop button is clicked
    fps_changed(int)
        Emitted when FPS value changes
    create_gif_clicked()
        Emitted when create GIF button is clicked
    """
    
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    fps_changed = pyqtSignal(int)
    create_gif_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for widget functionality")
            
        self.is_playing = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        playback_layout.addWidget(self.play_pause_btn)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        playback_layout.addWidget(self.stop_btn)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # Animation settings
        settings_group = QGroupBox("Animation Settings")
        settings_layout = QVBoxLayout()
        
        # FPS control
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(self.fps_spin)
        
        settings_layout.addLayout(fps_layout)
        
        # Loop option
        self.loop_checkbox = QCheckBox("Loop animation")
        self.loop_checkbox.setChecked(True)
        settings_layout.addWidget(self.loop_checkbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        # Create GIF button
        self.create_gif_btn = QPushButton("Create GIF")
        self.create_gif_btn.clicked.connect(self.on_create_gif_clicked)
        export_layout.addWidget(self.create_gif_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
    def on_play_pause_clicked(self):
        """Handle play/pause button click."""
        if self.is_playing:
            self.pause_clicked.emit()
            self.play_pause_btn.setText("Play")
            self.is_playing = False
        else:
            self.play_clicked.emit()
            self.play_pause_btn.setText("Pause")
            self.is_playing = True
            
    def on_stop_clicked(self):
        """Handle stop button click."""
        self.stop_clicked.emit()
        self.play_pause_btn.setText("Play")
        self.is_playing = False
        
    def on_fps_changed(self, value):
        """Handle FPS value change."""
        self.fps_changed.emit(value)
        
    def on_create_gif_clicked(self):
        """Handle create GIF button click."""
        self.create_gif_clicked.emit()
        
    def set_playing_state(self, is_playing: bool):
        """Set the playing state of the widget."""
        self.is_playing = is_playing
        self.play_pause_btn.setText("Pause" if is_playing else "Play")
        
    def get_fps(self) -> int:
        """Get the current FPS setting."""
        return self.fps_spin.value()
        
    def get_loop_enabled(self) -> bool:
        """Get whether looping is enabled."""
        return self.loop_checkbox.isChecked()


class DisplayOptionsWidget(QWidget):
    """
    Widget for controlling display options.
    
    This widget provides controls for colormap selection, window/level settings,
    and other display options.
    
    Signals
    -------
    colormap_changed(str)
        Emitted when colormap changes
    window_changed(int)
        Emitted when window value changes
    level_changed(int)
        Emitted when level value changes
    auto_window_level_clicked()
        Emitted when auto window/level button is clicked
    """
    
    colormap_changed = pyqtSignal(str)
    window_changed = pyqtSignal(int)
    level_changed = pyqtSignal(int)
    auto_window_level_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for widget functionality")
            
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Colormap selection
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QHBoxLayout()
        
        colormap_layout.addWidget(QLabel("Colormap:"))
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        colormap_layout.addWidget(self.colormap_combo)
        
        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)
        
        # Window/Level controls
        window_level_group = QGroupBox("Window/Level")
        window_level_layout = QVBoxLayout()
        
        # Window control
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10000)
        self.window_spin.setValue(1000)
        self.window_spin.valueChanged.connect(self.on_window_changed)
        window_layout.addWidget(self.window_spin)
        
        window_level_layout.addLayout(window_layout)
        
        # Level control
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Level:"))
        
        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 10000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self.on_level_changed)
        level_layout.addWidget(self.level_spin)
        
        window_level_layout.addLayout(level_layout)
        
        # Auto window/level button
        self.auto_btn = QPushButton("Auto Window/Level")
        self.auto_btn.clicked.connect(self.on_auto_window_level_clicked)
        window_level_layout.addWidget(self.auto_btn)
        
        window_level_group.setLayout(window_level_layout)
        layout.addWidget(window_level_group)
        
    def on_colormap_changed(self, colormap):
        """Handle colormap change."""
        self.colormap_changed.emit(colormap)
        
    def on_window_changed(self, value):
        """Handle window value change."""
        self.window_changed.emit(value)
        
    def on_level_changed(self, value):
        """Handle level value change."""
        self.level_changed.emit(value)
        
    def on_auto_window_level_clicked(self):
        """Handle auto window/level button click."""
        self.auto_window_level_clicked.emit()
        
    def set_colormap(self, colormap: str):
        """Set the current colormap."""
        index = self.colormap_combo.findText(colormap)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
            
    def set_window_level(self, window: int, level: int):
        """Set window and level values."""
        self.window_spin.setValue(window)
        self.level_spin.setValue(level)
        
    def get_colormap(self) -> str:
        """Get the current colormap."""
        return self.colormap_combo.currentText()
        
    def get_window(self) -> int:
        """Get the current window value."""
        return self.window_spin.value()
        
    def get_level(self) -> int:
        """Get the current level value."""
        return self.level_spin.value()