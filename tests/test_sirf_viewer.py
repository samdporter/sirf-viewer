# tests/test_sirf_viewer.py
"""
Tests for SIRF viewer package.

This module contains unit tests for the SIRF viewer functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
from typing import Any

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sirf_viewer.viewers import SIRFViewer, NotebookViewer
from sirf_viewer.utils import (get_data_info, validate_sirf_data, 
                              get_optimal_window_level, create_thumbnail)
from sirf_viewer.widgets import (SIRFImageViewerWidget, SIRFAcquisitionViewerWidget,
                                DimensionControlWidget, AnimationControlWidget)


class MockSIRFData:
    """Mock SIRF data object for testing."""
    
    def __init__(self, shape, class_name='ImageData'):
        self.shape = shape
        self._class_name = class_name
        self._data = np.random.rand(*shape)
        
    def __class__(self):
        return type(self._class_name, (), {})
        
    @property
    def __class__(self):
        return type(self._class_name, (), {'__name__': self._class_name})
        
    def asarray(self):
        return self._data


class TestSIRFViewer(unittest.TestCase):
    """Test cases for SIRFViewer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock ImageData (3D)
        self.mock_image_data = MockSIRFData((10, 128, 128), 'ImageData')
        
        # Create mock AcquisitionData (4D)
        self.mock_acq_data = MockSIRFData((5, 32, 64, 64), 'AcquisitionData')
        
    def test_init_with_image_data(self):
        """Test initialization with ImageData."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_image_data)
            
            self.assertEqual(viewer.data, self.mock_image_data)
            self.assertEqual(len(viewer.dimensions), 3)
            self.assertEqual(viewer.dimension_names, ['z', 'y', 'x'])
            self.assertEqual(viewer.current_indices, [5, 64, 64])  # Middle indices
            
    def test_init_with_acquisition_data(self):
        """Test initialization with AcquisitionData."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_acq_data)
            
            self.assertEqual(viewer.data, self.mock_acq_data)
            self.assertEqual(len(viewer.dimensions), 4)
            self.assertEqual(viewer.dimension_names, ['ToF Bin', 'View', 'Radial', 'Axial'])
            self.assertEqual(viewer.current_indices, [2, 16, 32, 32])  # Middle indices
            
    def test_get_current_slice_3d(self):
        """Test getting current slice for 3D data."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_image_data)
            viewer.current_indices = [2, 0, 0]
            
            slice_data = viewer._get_current_slice()
            
            self.assertEqual(slice_data.shape, (128, 128))
            np.testing.assert_array_equal(slice_data, viewer.data_array[2, :, :])
            
    def test_get_current_slice_4d(self):
        """Test getting current slice for 4D data."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_acq_data)
            viewer.current_indices = [1, 2, 0, 0]
            
            slice_data = viewer._get_current_slice()
            
            self.assertEqual(slice_data.shape, (64, 64))
            np.testing.assert_array_equal(slice_data, viewer.data_array[1, 2, :, :])
            
    def test_apply_window_level(self):
        """Test applying window/level to data."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_image_data)
            viewer.window_level = 0.5
            viewer.window_width = 1.0
            
            test_data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            result = viewer._apply_window_level(test_data)
            
            expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            np.testing.assert_array_equal(result, expected)
            
    def test_set_colormap(self):
        """Test setting colormap."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_image_data)
            
            viewer.set_colormap('viridis')
            self.assertEqual(viewer.colormap, 'viridis')
            
    def test_set_window_level(self):
        """Test setting window/level."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = SIRFViewer(self.mock_image_data)
            
            viewer.set_window(100, 200)
            self.assertEqual(viewer.window_level, 100)
            self.assertEqual(viewer.window_width, 200)


class TestNotebookViewer(unittest.TestCase):
    """Test cases for NotebookViewer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_image_data = MockSIRFData((10, 128, 128), 'ImageData')
        
    def test_init(self):
        """Test initialization."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = NotebookViewer(self.mock_image_data)
            
            self.assertEqual(viewer.data, self.mock_image_data)
            self.assertEqual(len(viewer.dimensions), 3)
            self.assertEqual(viewer.dimension_names, ['z', 'y', 'x'])
            
    def test_get_current_slice(self):
        """Test getting current slice."""
        with patch('sirf_viewer.viewers.SIRF_AVAILABLE', True):
            viewer = NotebookViewer(self.mock_image_data)
            viewer.current_indices = [3, 0, 0]
            
            slice_data = viewer._get_current_slice()
            
            self.assertEqual(slice_data.shape, (128, 128))
            np.testing.assert_array_equal(slice_data, viewer.data_array[3, :, :])


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_image_data = MockSIRFData((10, 128, 128), 'ImageData')
        self.mock_acq_data = MockSIRFData((5, 32, 64, 64), 'AcquisitionData')
        
    def test_get_data_info_image_data(self):
        """Test getting data info for ImageData."""
        info = get_data_info(self.mock_image_data)
        
        self.assertEqual(info['type'], 'ImageData')
        self.assertEqual(info['shape'], (10, 128, 128))
        self.assertEqual(info['dimensions'], 3)
        self.assertEqual(info['dimension_names'], ['z', 'y', 'x'])
        
    def test_get_data_info_acquisition_data(self):
        """Test getting data info for AcquisitionData."""
        info = get_data_info(self.mock_acq_data)
        
        self.assertEqual(info['type'], 'AcquisitionData')
        self.assertEqual(info['shape'], (5, 32, 64, 64))
        self.assertEqual(info['dimensions'], 4)
        self.assertEqual(info['dimension_names'], ['ToF Bin', 'View', 'Radial', 'Axial'])
        
    def test_validate_sirf_data_valid(self):
        """Test validating valid SIRF data."""
        with patch('sirf_viewer.utils.SIRF_AVAILABLE', True):
            self.assertTrue(validate_sirf_data(self.mock_image_data))
            self.assertTrue(validate_sirf_data(self.mock_acq_data))
            
    def test_validate_sirf_data_invalid(self):
        """Test validating invalid SIRF data."""
        with patch('sirf_viewer.utils.SIRF_AVAILABLE', True):
            # Invalid class name
            invalid_data = MockSIRFData((10, 128, 128), 'InvalidData')
            self.assertFalse(validate_sirf_data(invalid_data))
            
            # Missing asarray method
            class InvalidData:
                __class__.__name__ = 'ImageData'
                
            self.assertFalse(validate_sirf_data(InvalidData()))
            
    def test_get_optimal_window_level(self):
        """Test calculating optimal window/level."""
        level, width = get_optimal_window_level(self.mock_image_data)
        
        self.assertIsInstance(level, float)
        self.assertIsInstance(width, float)
        self.assertGreater(width, 0)
        
    def test_create_thumbnail(self):
        """Test creating thumbnail."""
        with patch('sirf_viewer.utils.SIRF_AVAILABLE', True):
            thumbnail = create_thumbnail(self.mock_image_data, size=(64, 64))
            
            self.assertEqual(thumbnail.shape, (64, 64))
            self.assertEqual(thumbnail.dtype, np.uint8)


class TestWidgets(unittest.TestCase):
    """Test cases for widget classes."""
    
    def test_dimension_control_widget(self):
        """Test DimensionControlWidget."""
        with patch('sirf_viewer.widgets.PYQT_AVAILABLE', True):
            from PyQt5.QtWidgets import QApplication
            import sys
            
            # Create QApplication if it doesn't exist
            if not QApplication.instance():
                app = QApplication(sys.argv)
            else:
                app = QApplication.instance()
                
            widget = DimensionControlWidget()
            widget.set_dimensions([10, 20, 30], ['z', 'y', 'x'])
            
            self.assertEqual(len(widget.sliders), 3)
            self.assertEqual(widget.get_dimension_values(), [5, 10, 15])  # Middle values
            
    def test_animation_control_widget(self):
        """Test AnimationControlWidget."""
        with patch('sirf_viewer.widgets.PYQT_AVAILABLE', True):
            from PyQt5.QtWidgets import QApplication
            import sys
            
            if not QApplication.instance():
                app = QApplication(sys.argv)
            else:
                app = QApplication.instance()
                
            widget = AnimationControlWidget()
            
            self.assertEqual(widget.get_fps(), 10)
            self.assertTrue(widget.get_loop_enabled())
            
            # Test FPS change
            widget.fps_spin.setValue(20)
            self.assertEqual(widget.get_fps(), 20)
            
    def test_display_options_widget(self):
        """Test DisplayOptionsWidget."""
        with patch('sirf_viewer.widgets.PYQT_AVAILABLE', True):
            from PyQt5.QtWidgets import QApplication
            import sys
            
            if not QApplication.instance():
                app = QApplication(sys.argv)
            else:
                app = QApplication.instance()
                
            widget = DisplayOptionsWidget()
            
            self.assertEqual(widget.get_colormap(), 'gray')
            self.assertEqual(widget.get_window(), 1000)
            self.assertEqual(widget.get_level(), 0)
            
            # Test colormap change
            widget.set_colormap('viridis')
            self.assertEqual(widget.get_colormap(), 'viridis')
            
            # Test window/level change
            widget.set_window_level(500, 100)
            self.assertEqual(widget.get_window(), 500)
            self.assertEqual(widget.get_level(), 100)


if __name__ == '__main__':
    unittest.main()