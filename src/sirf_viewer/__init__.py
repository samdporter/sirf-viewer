# src/sirf_viewer/__init__.py

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .viewers import SIRFViewer, NotebookViewer
from .utils import create_gif_from_data, save_view_as_image
from .gui import SIRFViewerGUI, SIRF_AVAILABLE

__all__ = [
    "SIRFViewer",
    "NotebookViewer",
    "create_gif_from_data",
    "save_view_as_image",
    "SIRFViewerGUI",
    "SIRF_AVAILABLE",
]