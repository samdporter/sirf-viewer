# src/sirf_viewer/utils.py
"""
Utility functions for SIRF viewer.

This module provides utility functions for data processing, file operations,
and other helper functions used throughout the package.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, List, Union
import warnings
import os
from pathlib import Path

try:
    import sirf.STIR as sirf
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False
    warnings.warn("SIRF not available. Install sirf package to use SIRF data objects.")


def create_gif_from_data(data: Any, filename: str, fps: int = 10, 
                        dimensions: Optional[List[int]] = None,
                        colormap: str = 'gray', **kwargs) -> None:
    """
    Create an animated GIF from SIRF data.
    
    This is a convenience function that creates a GIF animation from
    SIRF ImageData or AcquisitionData objects.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to animate
    filename : str
        Output filename for the GIF
    fps : int, optional
        Frames per second (default: 10)
    dimensions : list of int, optional
        Which dimensions to animate (default: first scrollable dimension)
    colormap : str, optional
        Colormap to use for the animation (default: 'gray')
    **kwargs
        Additional keyword arguments passed to the animation creation
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    from .viewers import SIRFViewer
    
    # Create temporary viewer
    viewer = SIRFViewer(data, "GIF Creator")
    viewer.set_colormap(colormap)
    
    # Create GIF
    viewer.create_gif(filename, fps, dimensions, **kwargs)


def save_view_as_image(data: Any, filename: str, indices: Optional[List[int]] = None,
                      colormap: str = 'gray', dpi: int = 150, **kwargs) -> None:
    """
    Save a specific view of SIRF data as an image.
    
    This is a convenience function that saves a specific slice/view of
    SIRF ImageData or AcquisitionData objects as an image file.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to save
    filename : str
        Output filename for the image
    indices : list of int, optional
        Specific indices for each dimension (default: middle slices)
    colormap : str, optional
        Colormap to use for the image (default: 'gray')
    dpi : int, optional
        Resolution in dots per inch (default: 150)
    **kwargs
        Additional keyword arguments passed to matplotlib savefig
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    from .viewers import SIRFViewer
    
    # Create temporary viewer
    viewer = SIRFViewer(data, "Image Saver")
    viewer.set_colormap(colormap)
    
    # Set specific indices if provided
    if indices is not None:
        for i, idx in enumerate(indices):
            if i < len(viewer.current_indices):
                viewer.current_indices[i] = idx
                
    # Save the current view
    viewer.save_current_view(filename)


def get_data_info(data: Any) -> dict:
    """
    Get information about SIRF data object.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to analyze
        
    Returns
    -------
    dict
        Dictionary containing data information including:
        - type: Data type (ImageData or AcquisitionData)
        - shape: Data shape as tuple
        - dimensions: Number of dimensions
        - dtype: Data type of the underlying array
        - min_value: Minimum value in the data
        - max_value: Maximum value in the data
        - mean_value: Mean value of the data
        - dimension_names: Names of the dimensions
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    try:
        data_array = data.asarray()
    except AttributeError:
        raise AttributeError("Data object must have asarray() method")
        
    # Get basic information
    info = {
        'type': data.__class__.__name__,
        'shape': data_array.shape,
        'dimensions': len(data_array.shape),
        'dtype': str(data_array.dtype),
        'min_value': float(np.min(data_array)),
        'max_value': float(np.max(data_array)),
        'mean_value': float(np.mean(data_array)),
    }
    
    # Get dimension names based on data type
    if info['type'] == 'ImageData':
        info['dimension_names'] = ['z', 'y', 'x']
    elif info['type'] == 'AcquisitionData':
        info['dimension_names'] = ['ToF Bin', 'View', 'Radial', 'Axial']
    else:
        info['dimension_names'] = [f'Dim {i}' for i in range(info['dimensions'])]
        
    return info


def print_data_info(data: Any) -> None:
    """
    Print information about SIRF data object in a readable format.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to analyze
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    info = get_data_info(data)

    print("SIRF Data Information:")
    print(f"  Type: {info['type']}")
    print(f"  Shape: {info['shape']}")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Data Type: {info['dtype']}")
    print(f"  Value Range: [{info['min_value']:.6f}, {info['max_value']:.6f}]")
    print(f"  Mean Value: {info['mean_value']:.6f}")
    print(f"  Dimension Names: {', '.join(info['dimension_names'])}")


def validate_sirf_data(data: Any) -> bool:
    """
    Validate if an object is a valid SIRF data object.
    
    Parameters
    ----------
    data : Any
        Object to validate
        
    Returns
    -------
    bool
        True if the object is a valid SIRF ImageData or AcquisitionData object
        
    Notes
    -----
    This function checks if the object:
    1. Has the correct class name (ImageData or AcquisitionData)
    2. Has an asarray() method
    3. The asarray() method returns a numpy array
    """
    if not SIRF_AVAILABLE:
        return False
        
    # Check class name
    if not hasattr(data, '__class__'):
        return False
        
    class_name = data.__class__.__name__
    if class_name not in ['ImageData', 'AcquisitionData']:
        return False
        
    # Check asarray method
    if not hasattr(data, 'asarray'):
        return False
        
    try:
        # Try to get array
        data_array = data.asarray()
        if not isinstance(data_array, np.ndarray):
            return False
    except:
        return False
        
    return True


def get_optimal_window_level(data: Any, percentile: float = 98.0) -> Tuple[float, float]:
    """
    Calculate optimal window/level settings for data display.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to analyze
    percentile : float, optional
        Percentile to use for window calculation (default: 98.0)
        
    Returns
    -------
    tuple of (float, float)
        Optimal (level, width) for window/level display
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    try:
        data_array = data.asarray()
    except AttributeError:
        raise AttributeError("Data object must have asarray() method")
        
    # Calculate percentiles
    low_percentile = (100 - percentile) / 2
    high_percentile = 100 - low_percentile
    
    min_val = np.percentile(data_array, low_percentile)
    max_val = np.percentile(data_array, high_percentile)
    
    level = (min_val + max_val) / 2
    width = max_val - min_val
    
    return level, width


def create_thumbnail(data: Any, size: Tuple[int, int] = (128, 128), 
                   indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Create a thumbnail image from SIRF data.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to create thumbnail from
    size : tuple of int, optional
        Size of the thumbnail as (width, height) (default: (128, 128))
    indices : list of int, optional
        Specific indices for each dimension (default: middle slices)
        
    Returns
    -------
    numpy.ndarray
        Thumbnail image as numpy array
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    AttributeError
        If data object doesn't have required methods
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    from .viewers import SIRFViewer
    
    # Create temporary viewer
    viewer = SIRFViewer(data, "Thumbnail Creator")
    
    # Set specific indices if provided
    if indices is not None:
        for i, idx in enumerate(indices):
            if i < len(viewer.current_indices):
                viewer.current_indices[i] = idx
                
    # Get current slice
    slice_data = viewer._get_current_slice()
    
    # Resize to thumbnail size
    from PIL import Image
    
    # Normalize to 0-255
    slice_data_norm = ((slice_data - slice_data.min()) / 
                      (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    
    # Convert to PIL Image and resize
    pil_image = Image.fromarray(slice_data_norm)
    pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
    
    return np.array(pil_image)


def batch_process_files(file_pattern: str, output_dir: str, 
                       operation: str = 'thumbnail', **kwargs) -> List[str]:
    """
    Batch process multiple SIRF files.
    
    Parameters
    ----------
    file_pattern : str
        File pattern (e.g., '/path/to/files/*.hv')
    output_dir : str
        Output directory for processed files
    operation : str, optional
        Operation to perform: 'thumbnail', 'gif', or 'info' (default: 'thumbnail')
    **kwargs
        Additional keyword arguments for the operation
        
    Returns
    -------
    list of str
        List of output file paths
        
    Raises
    ------
    ImportError
        If SIRF package is not available
    ValueError
        If operation is not supported
    """
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
        
    import glob
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    files = glob.glob(file_pattern)
    output_files = []
    
    for file_path in files:
        try:
            # Load data
            if file_path.endswith('.hv'):
                data = sirf.ImageData(file_path)
            elif file_path.endswith('.hs'):
                data = sirf.AcquisitionData(file_path)
            else:
                continue
                
            # Perform operation
            if operation == 'thumbnail':
                thumbnail = create_thumbnail(data, **kwargs)
                output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_thumb.png")
                plt.imsave(output_path, thumbnail, cmap='gray')
                output_files.append(output_path)
                
            elif operation == 'gif':
                output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}.gif")
                create_gif_from_data(data, output_path, **kwargs)
                output_files.append(output_path)
                
            elif operation == 'info':
                info = get_data_info(data)
                output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_info.txt")
                with open(output_path, 'w') as f:
                    for key, value in info.items():
                        f.write(f"{key}: {value}\n")
                output_files.append(output_path)
                
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
            
    return output_files