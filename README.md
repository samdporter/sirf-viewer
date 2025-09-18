# SIRF Viewer

A comprehensive GUI and Python interface for viewing SIRF ImageData and AcquisitionData objects.

## Features

- **GUI Application**: Full-featured desktop application for viewing SIRF data
- **Python Interface**: Programmatic access for use in Jupyter notebooks and scripts
- **Multi-dimensional Support**: Handle both 3D ImageData (z,y,x) and 4D AcquisitionData (ToF Bin, 1, 2, 3)
- **Interactive Controls**: Sliders for scrolling through dimensions
- **Animation Support**: Create GIFs with user-chosen refresh rates
- **File Format Support**: Load ImageData (.hv) and AcquisitionData (.hs) files
- **Cross-platform**: Works on Windows, macOS, and Linux

## Installation

### From PyPI

```bash
pip install sirf-viewer