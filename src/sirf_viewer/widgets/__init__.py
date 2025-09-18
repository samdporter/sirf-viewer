# src/sirf_viewer/widgets/__init__.py
"""
Custom widgets for SIRF viewer.

This package contains custom Qt widgets used in the SIRF viewer GUI.
"""

from .image_viewer import SIRFImageViewerWidget
from .acquisition_viewer import SIRFAcquisitionViewerWidget
from .controls import DimensionControlWidget, AnimationControlWidget

__all__ = [
    "SIRFImageViewerWidget",
    "SIRFAcquisitionViewerWidget", 
    "DimensionControlWidget",
    "AnimationControlWidget",
]