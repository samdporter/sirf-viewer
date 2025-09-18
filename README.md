# SIRF Viewer

A comprehensive GUI and Python interface for viewing SIRF ImageData and AcquisitionData objects.

## Features

- **GUI Application**: Full-featured desktop application for viewing SIRF data
- **Python Interface**: Programmatic access for use in Jupyter notebooks and scripts
- **Multi-dimensional Support**: Handle both 3D ImageData (z,y,x) and 4D AcquisitionData (ToF Bin, View, Radial, Axial)
- **Interactive Controls**: Sliders for scrolling through dimensions
- **Animation Support**: Create GIFs with user-chosen refresh rates
- **File Format Support**: Load ImageData (.hv) and AcquisitionData (.hs) files
- **Cross-platform**: Works on Windows, macOS, and Linux

## Installation

### From PyPI

```bash
pip install sirf-viewer
```

### From Source

```bash
git clone https://github.com/sirf-viewer/sirf-viewer.git
cd sirf-viewer
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/sirf-viewer/sirf-viewer.git
cd sirf-viewer
pip install -e .[dev]
```

### Notebook Support

For Jupyter notebook widgets support:

```bash
pip install sirf-viewer[notebook]
```

## Requirements

- Python >= 3.8
- SIRF (Software for Synergistic Image Reconstruction Framework)
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- PyQt5 >= 5.15.0 (for GUI)
- Pillow >= 8.0.0
- SciPy >= 1.7.0

Optional dependencies:

- ipywidgets >= 7.0.0 (for Jupyter notebook integration)

## Quick Start

### GUI Application

Launch the standalone GUI application:

```bash
sirf-viewer
```

Or from Python:

```python
from sirf_viewer import SIRFViewerGUI
gui = SIRFViewerGUI()
gui.show()
```

### Python Interface

#### Basic Usage

```python
import sirf.STIR as sirf
from sirf_viewer import SIRFViewer

# Load SIRF data
image_data = sirf.ImageData('path/to/your/image.hv')
# or
acq_data = sirf.AcquisitionData('path/to/your/data.hs')

# Create viewer
viewer = SIRFViewer(image_data, "My Image Data")

# Display the viewer
viewer.show()
```

#### Jupyter Notebook

```python
from sirf_viewer import NotebookViewer

# Create interactive notebook viewer
nb_viewer = NotebookViewer(image_data, width=800, height=600)
nb_viewer.show()  # Shows interactive widgets
```

#### Creating Animations

```python
from sirf_viewer import create_gif_from_data

# Create animated GIF
create_gif_from_data(
    image_data, 
    'animation.gif', 
    fps=10, 
    dimensions=[0],  # Animate through z-dimension
    colormap='viridis'
)
```

#### Saving Images

```python
from sirf_viewer import save_view_as_image

# Save a specific slice
save_view_as_image(
    image_data, 
    'slice.png',
    indices=[10, 64, 64],  # z=10, middle of y,x
    colormap='gray'
)
```

## API Reference

### SIRFViewer

Main viewer class for interactive matplotlib-based viewing.

```python
viewer = SIRFViewer(data, title="SIRF Viewer")
viewer.set_colormap('viridis')
viewer.set_window(level=500, width=1000)
viewer.save_current_view('output.png')
viewer.create_gif('animation.gif', fps=10)
viewer.show()
```

### NotebookViewer

Jupyter notebook-compatible viewer with interactive widgets.

```python
nb_viewer = NotebookViewer(data, width=800, height=600)
nb_viewer.show()
nb_viewer.create_gif('notebook_animation.gif', fps=5)
```

### Utility Functions

```python
from sirf_viewer.utils import (
    get_data_info, 
    print_data_info,
    get_optimal_window_level,
    create_thumbnail,
    batch_process_files
)

# Get data information
info = get_data_info(data)
print_data_info(data)

# Calculate optimal display settings
level, width = get_optimal_window_level(data)

# Create thumbnail
thumbnail = create_thumbnail(data, size=(128, 128))

# Batch process multiple files
output_files = batch_process_files('*.hv', 'output_dir', operation='thumbnail')
```

## Supported Data Types

### ImageData (3D)

- Format: .hv files
- Dimensions: z, y, x (typically axial slices)
- Viewing: Interactive slice scrolling through z-dimension

### AcquisitionData (4D)

- Format: .hs files  
- Dimensions: ToF Bin, View, Radial, Axial
- Viewing: Interactive navigation through ToF bins and views

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Command-line examples
- `notebook_example.ipynb`: Jupyter notebook tutorial

## GUI Features

- **File Loading**: Open .hv and .hs files through file dialogs
- **Multi-dimensional Navigation**: Sliders for each scrollable dimension
- **Display Controls**:
  - Colormap selection (gray, viridis, plasma, inferno, magma, cividis)
  - Window/Level adjustment
  - Auto window/level calculation
- **Animation**:
  - Play/pause/stop controls
  - Adjustable frame rate
  - GIF export
- **Export Options**:
  - Save current view as PNG
  - Create animated GIFs
  - Batch processing capabilities

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses Black for code formatting:

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Format code (`black .`)
7. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SIRF Viewer in your research, please cite:

```bibtex
@software{sirf_viewer,
    title={SIRF Viewer: A GUI and Python interface for viewing SIRF data},
    author={Sam Porter},
    url={https://github.com/sirf-viewer/sirf-viewer},
    year={2024}
}
```

## Acknowledgments

- Built for the SIRF (Software for Synergistic Image Reconstruction Framework) community
- Uses matplotlib, PyQt5, and other excellent Python libraries

## Support

- **Documentation**: [https://sirf-viewer.readthedocs.io](https://sirf-viewer.readthedocs.io) (When ready)
- **Issues**: [GitHub Issues](https://github.com/samdporter/sirf-viewer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/samdporter/sirf-viewer/discussions)
