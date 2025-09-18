# examples/basic_usage.py
"""
Basic usage examples for SIRF viewer.

This script demonstrates how to use the SIRF viewer package with both
ImageData and AcquisitionData objects.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import sirf_viewer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import sirf.STIR as sirf
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False
    print("Warning: SIRF not available. Using mock data for demonstration.")

from sirf_viewer import SIRFViewer, NotebookViewer, create_gif_from_data, save_view_as_image


def create_mock_image_data():
    """Create mock ImageData for demonstration."""
    class MockImageData:
        def __init__(self, shape):
            self.shape = shape
            self._data = np.random.rand(*shape) * 1000
            
        def asarray(self):
            return self._data
            
        @property
        def __class__(self):
            return type('ImageData', (), {'__name__': 'ImageData'})
    
    return MockImageData((20, 128, 128))


def create_mock_acquisition_data():
    """Create mock AcquisitionData for demonstration."""
    class MockAcquisitionData:
        def __init__(self, shape):
            self.shape = shape
            self._data = np.random.rand(*shape) * 500
            
        def asarray(self):
            return self._data
            
        @property
        def __class__(self):
            return type('AcquisitionData', (), {'__name__': 'AcquisitionData'})
    
    return MockAcquisitionData((8, 16, 64, 64))


def example_image_data_viewer():
    """Example of using SIRFViewer with ImageData."""
    print("=== ImageData Viewer Example ===")
    
    # Create or load ImageData
    if SIRF_AVAILABLE:
        # In real usage, you would load from file:
        # image_data = sirf.ImageData('path/to/your/image.hv')
        print("SIRF available - in real usage, load ImageData from .hv file")
        image_data = create_mock_image_data()
    else:
        print("Using mock ImageData for demonstration")
        image_data = create_mock_image_data()
    
    # Create viewer
    viewer = SIRFViewer(image_data, "My Image Data")
    
    # Print data info
    from sirf_viewer.utils import get_data_info, print_data_info
    print_data_info(image_data)
    
    # Set different colormaps
    print("\nTrying different colormaps...")
    for colormap in ['gray', 'viridis', 'plasma']:
        viewer.set_colormap(colormap)
        print(f"  Set colormap to: {colormap}")
    
    # Set window/level
    print("\nSetting window/level...")
    level, width = 500, 1000
    viewer.set_window(level, width)
    print(f"  Set window: {width}, level: {level}")
    
    # Save current view
    print("\nSaving current view...")
    save_view_as_image(image_data, 'example_image_slice.png', 
                      indices=[10, 64, 64], colormap='viridis')
    print("  Saved to: example_image_slice.png")
    
    # Create GIF animation
    print("\nCreating GIF animation...")
    create_gif_from_data(image_data, 'example_image_animation.gif', 
                       fps=5, dimensions=[0])
    print("  Created: example_image_animation.gif")
    
    print("\nTo display the viewer interactively, call:")
    print("  viewer.show()")
    print("(This would open a matplotlib window)")
    
    return viewer


def example_acquisition_data_viewer():
    """Example of using SIRFViewer with AcquisitionData."""
    print("\n=== AcquisitionData Viewer Example ===")
    
    # Create or load AcquisitionData
    if SIRF_AVAILABLE:
        # In real usage, you would load from file:
        # acq_data = sirf.AcquisitionData('path/to/your/data.hs')
        print("SIRF available - in real usage, load AcquisitionData from .hs file")
        acq_data = create_mock_acquisition_data()
    else:
        print("Using mock AcquisitionData for demonstration")
        acq_data = create_mock_acquisition_data()
    
    # Create viewer
    viewer = SIRFViewer(acq_data, "My Acquisition Data")
    
    # Print data info
    from sirf_viewer.utils import get_data_info, print_data_info
    print_data_info(acq_data)
    
    # Navigate through different dimensions
    print("\nNavigating through dimensions...")
    
    # Change ToF bin
    viewer.current_indices[0] = 2
    print(f"  Set ToF bin to: {viewer.current_indices[0]}")
    
    # Change view
    viewer.current_indices[1] = 8
    print(f"  Set view to: {viewer.current_indices[1]}")
    
    # Save current view
    print("\nSaving current view...")
    save_view_as_image(acq_data, 'example_acquisition_slice.png', 
                      indices=[2, 8, 32, 32], colormap='plasma')
    print("  Saved to: example_acquisition_slice.png")
    
    # Create GIF animation (animate through ToF bins)
    print("\nCreating GIF animation...")
    create_gif_from_data(acq_data, 'example_acquisition_animation.gif', 
                       fps=3, dimensions=[0])
    print("  Created: example_acquisition_animation.gif")
    
    print("\nTo display the viewer interactively, call:")
    print("  viewer.show()")
    print("(This would open a matplotlib window)")
    
    return viewer


def example_notebook_viewer():
    """Example of using NotebookViewer for Jupyter notebooks."""
    print("\n=== Notebook Viewer Example ===")
    
    # Create mock data
    image_data = create_mock_image_data()
    
    # Create notebook viewer
    print("Creating notebook viewer...")
    viewer = NotebookViewer(image_data, width=600, height=400)
    
    print("Notebook viewer created with:")
    print(f"  - Dimensions: {viewer.dimensions}")
    print(f"  - Dimension names: {viewer.dimension_names}")
    print(f"  - Size: {viewer.width}x{viewer.height}")
    
    print("\nIn a Jupyter notebook, you would use:")
    print("  viewer.show()")
    print("This would display interactive widgets in the notebook.")
    
    return viewer


def example_batch_processing():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    # Create mock files for demonstration
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Create mock data files
    mock_files = []
    for i in range(3):
        # Create mock ImageData
        image_data = create_mock_image_data()
        
        # Save as numpy file (in real usage, these would be .hv files)
        filename = os.path.join(temp_dir, f'mock_image_{i}.npy')
        np.save(filename, image_data.asarray())
        mock_files.append(filename)
    
    print(f"Created {len(mock_files)} mock files")
    
    # Batch process to create thumbnails
    print("\nBatch processing to create thumbnails...")
    from sirf_viewer.utils import batch_process_files
    
    try:
        # This would normally process .hv files, but we'll demonstrate the concept
        output_files = batch_process_files(
            os.path.join(temp_dir, '*.npy'),
            temp_dir,
            operation='info'  # Just create info files for demo
        )
        print(f"Processed {len(output_files)} files")
        
        # Show created files
        for output_file in output_files:
            if os.path.exists(output_file):
                print(f"  Created: {output_file}")
                with open(output_file, 'r') as f:
                    print(f"    Content: {f.read().strip()}")
                    
    except Exception as e:
        print(f"Batch processing failed (expected for mock files): {e}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary directory: {temp_dir}")


def main():
    """Run all examples."""
    print("SIRF Viewer - Basic Usage Examples")
    print("=" * 40)
    
    try:
        # Run examples
        image_viewer = example_image_data_viewer()
        acq_viewer = example_acquisition_data_viewer()
        notebook_viewer = example_notebook_viewer()
        example_batch_processing()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("  - example_image_slice.png")
        print("  - example_image_animation.gif")
        print("  - example_acquisition_slice.png")
        print("  - example_acquisition_animation.gif")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()