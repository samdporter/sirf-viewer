# src/sirf_viewer/widgets/acquisition_viewer.py
"""
Custom widget for viewing SIRF AcquisitionData objects.

This module provides a specialized widget for viewing 4D AcquisitionData objects
with interactive controls for multiple dimensions and display options.
"""

import numpy as np
from typing import Tuple, Any

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QSlider,
        QComboBox,
        QSpinBox,
        QGroupBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import sirf.STIR as sirf

    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False


class SIRFAcquisitionViewerWidget(QWidget):
    """
    Custom widget for viewing SIRF AcquisitionData objects.

    This widget provides a complete interface for viewing 4D AcquisitionData objects
    with scrolling through ToF Bin and View dimensions, colormap selection,
    and export options.

    Signals
    -------
    tof_bin_changed(int)
        Emitted when the ToF bin changes
    view_changed(int)
        Emitted when the view changes
    colormap_changed(str)
        Emitted when the colormap changes
    window_level_changed(float, float)
        Emitted when window/level settings change
    """

    tof_bin_changed = pyqtSignal(int)
    view_changed = pyqtSignal(int)
    colormap_changed = pyqtSignal(str)
    window_level_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for widget functionality")
        if not SIRF_AVAILABLE:
            raise ImportError(
                "SIRF package is required. Install with: pip install sirf"
            )

        self.data = None
        self.current_tof_bin = 0
        self.current_view = 0
        self.colormap = "gray"
        self.window_level = 0.0
        self.window_width = 1000.0

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        # ToF Bin control
        tof_layout = QHBoxLayout()
        tof_layout.addWidget(QLabel("ToF Bin:"))

        self.tof_slider = QSlider(Qt.Horizontal)
        self.tof_slider.setRange(0, 0)
        self.tof_slider.setValue(0)
        self.tof_slider.valueChanged.connect(self.on_tof_bin_changed)
        tof_layout.addWidget(self.tof_slider)

        self.tof_label = QLabel("0 / 0")
        tof_layout.addWidget(self.tof_label)

        controls_layout.addLayout(tof_layout)

        # View control
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))

        self.view_slider = QSlider(Qt.Horizontal)
        self.view_slider.setRange(0, 0)
        self.view_slider.setValue(0)
        self.view_slider.valueChanged.connect(self.on_view_changed)
        view_layout.addWidget(self.view_slider)

        self.view_label = QLabel("0 / 0")
        view_layout.addWidget(self.view_label)

        controls_layout.addLayout(view_layout)

        # Display options
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Colormap:"))

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]
        )
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        display_layout.addWidget(self.colormap_combo)

        controls_layout.addLayout(display_layout)

        # Window/Level controls
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))

        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10000)
        self.window_spin.setValue(1000)
        self.window_spin.valueChanged.connect(self.on_window_level_changed)
        window_layout.addWidget(self.window_spin)

        window_layout.addWidget(QLabel("Level:"))

        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 10000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self.on_window_level_changed)
        window_layout.addWidget(self.level_spin)

        controls_layout.addLayout(window_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Info label
        self.info_label = QLabel("No data loaded")
        layout.addWidget(self.info_label)

    def set_data(self, data: Any):
        """
        Set the AcquisitionData to view.

        Parameters
        ----------
        data : sirf.AcquisitionData
            The AcquisitionData object to view

        Raises
        ------
        TypeError
            If data is not an AcquisitionData object
        AttributeError
            If data doesn't have required methods
        """
        if (
            not hasattr(data, "__class__")
            or data.__class__.__name__ != "AcquisitionData"
        ):
            raise TypeError("Data must be an AcquisitionData object")

        try:
            data_array = data.asarray()
        except AttributeError:
            raise AttributeError("Data object must have asarray() method")

        self.data = data
        self.data_array = data_array

        # Update controls
        if len(data_array.shape) == 4:
            # ToF Bin (dimension 0)
            self.tof_slider.setRange(0, data_array.shape[0] - 1)
            self.tof_slider.setValue(data_array.shape[0] // 2)
            self.current_tof_bin = data_array.shape[0] // 2
            self.update_tof_label()

            # View (dimension 1)
            self.view_slider.setRange(0, data_array.shape[1] - 1)
            self.view_slider.setValue(data_array.shape[1] // 2)
            self.current_view = data_array.shape[1] // 2
            self.update_view_label()

        # Calculate optimal window/level
        level, width = self.calculate_optimal_window_level()
        self.window_level = level
        self.window_width = width
        self.window_spin.setValue(int(width))
        self.level_spin.setValue(int(level))

        # Update info
        self.info_label.setText(f"Shape: {data_array.shape}")

        # Update display
        self.update_display()

    def on_tof_bin_changed(self, value):
        """Handle ToF bin slider change."""
        self.current_tof_bin = value
        self.update_tof_label()
        self.update_display()
        self.tof_bin_changed.emit(value)

    def on_view_changed(self, value):
        """Handle view slider change."""
        self.current_view = value
        self.update_view_label()
        self.update_display()
        self.view_changed.emit(value)

    def on_colormap_changed(self, colormap):
        """Handle colormap change."""
        self.colormap = colormap
        self.update_display()
        self.colormap_changed.emit(colormap)

    def on_window_level_changed(self):
        """Handle window/level change."""
        self.window_width = self.window_spin.value()
        self.window_level = self.level_spin.value()
        self.update_display()
        self.window_level_changed.emit(self.window_level, self.window_width)

    def update_tof_label(self):
        """Update the ToF bin label."""
        if self.data is not None:
            max_tof = self.data_array.shape[0] - 1
            self.tof_label.setText(f"{self.current_tof_bin} / {max_tof}")

    def update_view_label(self):
        """Update the view label."""
        if self.data is not None:
            max_view = self.data_array.shape[1] - 1
            self.view_label.setText(f"{self.current_view} / {max_view}")

    def update_display(self):
        """Update the display."""
        if self.data is None:
            return

        # Clear the figure
        self.figure.clear()

        # Get current slice (Radial vs Axial)
        slice_data = self.data_array[self.current_tof_bin, self.current_view, :, :]

        # Apply window/level
        slice_data = self.apply_window_level(slice_data)

        # Create subplot
        ax = self.figure.add_subplot(111)
        im = ax.imshow(slice_data, cmap=self.colormap, origin="lower")
        self.figure.colorbar(im, ax=ax)
        ax.set_title(f"ToF Bin: {self.current_tof_bin}, View: {self.current_view}")
        ax.set_xlabel("Radial")
        ax.set_ylabel("Axial")

        # Redraw canvas
        self.canvas.draw()

    def apply_window_level(self, data: np.ndarray) -> np.ndarray:
        """Apply window/level to the data."""
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        return np.clip(data, min_val, max_val)

    def calculate_optimal_window_level(
        self, percentile: float = 98.0
    ) -> Tuple[float, float]:
        """Calculate optimal window/level settings."""
        low_percentile = (100 - percentile) / 2
        high_percentile = 100 - low_percentile

        min_val = np.percentile(self.data_array, low_percentile)
        max_val = np.percentile(self.data_array, high_percentile)

        level = (min_val + max_val) / 2
        width = max_val - min_val

        return level, width

    def get_current_slice_data(self) -> np.ndarray:
        """Get the current slice data with window/level applied."""
        if self.data is None:
            return np.array([])

        slice_data = self.data_array[self.current_tof_bin, self.current_view, :, :]
        return self.apply_window_level(slice_data)

    def save_current_view(self, filename: str, dpi: int = 150):
        """Save the current view as an image."""
        self.figure.savefig(filename, dpi=dpi, bbox_inches="tight")
