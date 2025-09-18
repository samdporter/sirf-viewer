#!/usr/bin/env python3
"""
Example demonstrating view switching functionality.

This shows how to use the view switching features in both SIRFViewer and NotebookViewer.
"""

import numpy as np
import sys
import os

from sirf_viewer import SIRFViewer, NotebookViewer

# create output directory in current folder
output_dir =  os.path.join(os.path.dirname(__file__),'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.chdir(output_dir)

# Create mock data
class MockSIRFData:
    """Mock SIRF data object for testing view switching."""
    
    def __init__(self, shape, class_name='ImageData'):
        self.shape = shape
        self._class_name = class_name
        if class_name == 'ImageData':
            self._data = self._create_anatomical_image_data(shape)
        else:
            self._data = self._create_structured_acquisition_data(shape)
        
    @property
    def __class__(self):
        return type(self._class_name, (), {'__name__': self._class_name})
        
    def asarray(self):
        return self._data
        
    def _create_anatomical_image_data(self, shape):
        """Create anatomical-like test data with clear directional features."""
        z, y, x = shape
        data = np.zeros((z, y, x))
        
        center_z, center_y, center_x = z//2, y//2, x//2
        
        # Create different features that will look different in each view
        for iz in range(z):
            for iy in range(y):
                for ix in range(x):
                    # Background noise
                    noise = np.random.normal(0, 20)
                    
                    # Axial feature (circular in axial view)
                    axial_dist = np.sqrt((iy - center_y)**2 + (ix - center_x)**2)
                    axial_feature = 500 * np.exp(-axial_dist**2 / 800) if axial_dist < 40 else 0
                    
                    # Sagittal feature (vertical stripe in sagittal view)
                    sagittal_feature = 300 * np.exp(-((iz - center_z)**2 + (iy - center_y + 20)**2) / 200)
                    
                    # Coronal feature (horizontal stripe in coronal view)
                    coronal_feature = 400 * np.exp(-((iz - center_z + 15)**2 + (ix - center_x)**2) / 300)
                    
                    # Z-variation
                    z_factor = 1.0 + 0.3 * np.sin(2 * np.pi * iz / z)
                    
                    data[iz, iy, ix] = noise + z_factor * (axial_feature + sagittal_feature + coronal_feature)
                    
        return np.maximum(data, 0)  # Ensure non-negative
        
    def _create_structured_acquisition_data(self, shape):
        """Create structured acquisition data with clear features."""
        tof, views, radial, axial = shape
        data = np.zeros((tof, views, radial, axial))
        
        for t in range(tof):
            for v in range(views):
                for r in range(radial):
                    for a in range(axial):
                        # Background noise
                        noise = np.random.normal(0, 5)
                        
                        # Sinogram-like pattern
                        angle = v * np.pi / views
                        center_r, center_a = radial//2, axial//2
                        
                        # Create projection features
                        sino_feature = 100 * np.exp(-((r - center_r)**2 + (a - center_a)**2) / 100)
                        
                        # View-dependent variation
                        view_variation = 50 * np.sin(2 * angle) * np.exp(-(r - center_r)**2 / 200)
                        
                        # ToF variation
                        tof_factor = np.exp(-t / 3) if t > 0 else 1.0
                        
                        data[t, v, r, a] = noise + tof_factor * (sino_feature + view_variation)
                        
        return np.maximum(data, 0)


def demo_imagedata_views():
    """Demonstrate view switching with ImageData."""
    print("=== ImageData View Switching Demo ===")
    
    # Create anatomical-like test data
    image_data = MockSIRFData((30, 100, 100), 'ImageData')
    print(f"Created ImageData with shape: {image_data.shape}")
    
    # Create viewer
    viewer = SIRFViewer(image_data, "ImageData View Switching Demo")
    
    print("\nAvailable views:")
    for view in viewer.get_available_views():
        print(f"  - {view}")
    
    print("\nThe interactive viewer will have:")
    print("  - Buttons to switch between Axial, Coronal, and Sagittal views")
    print("  - Sliders that adapt to each view")
    print("  - Different slice orientations for each view")
    
    # Demonstrate programmatic view switching
    print("\nProgrammatic view switching:")
    
    # Set up the plot first
    viewer.setup_plot()
    
    for view in viewer.get_available_views():
        viewer.set_view(view)
        print(f"  - Switched to {view} view")
        print(f"    Scrolling through {viewer.dimension_names[viewer.available_views[view]['scroll_dim']]}")
        
        # Save a sample from each view
        viewer.save_current_view(f'sample_{view.lower()}_view.png')
        print(f"    Saved sample to sample_{view.lower()}_view.png")
    
    print("\nCall viewer.show() to see the interactive version")
    return viewer


def demo_acquisition_views():
    """Demonstrate view switching with AcquisitionData."""
    print("\n=== AcquisitionData View Switching Demo ===")

    # Create acquisition test data
    acq_data = MockSIRFData((5, 20, 48, 48), 'AcquisitionData')
    print(f"Created AcquisitionData with shape: {acq_data.shape}")

    # Create viewer
    viewer = SIRFViewer(acq_data, "AcquisitionData View Switching Demo")

    print("\nAvailable views:")
    for view in viewer.get_available_views():
        print(f"  - {view}")

    print("\nThe interactive viewer will have:")
    print("  - Buttons to switch between different 2D projections")
    print("  - Multiple sliders for each view (e.g., ToF + View for sinogram)")
    print("  - Different combinations of the 4D data dimensions")

    # Demonstrate programmatic view switching
    print("\nProgrammatic view switching:")

    viewer.setup_plot()

    for view in viewer.get_available_views():
        viewer.set_view(view)
        view_config = viewer.available_views[view]
        print(f"  - Switched to {view}")
        print(f"    Primary scroll: {viewer.dimension_names[view_config['scroll_dim']]}")

        if controllable_dims := view_config.get('controllable_dims', []):
            ctrl_names = [viewer.dimension_names[i] for i in controllable_dims]
            print(f"    Also controls: {', '.join(ctrl_names)}")

        # Save a sample
        viewer.save_current_view(f'sample_acq_{view.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png')

    print("\nCall viewer.show() to see the interactive version")
    return viewer


def demo_notebook_viewer():
    """Demonstrate NotebookViewer with view switching."""
    print("\n=== NotebookViewer View Switching Demo ===")
    
    # Create test data
    image_data = MockSIRFData((20, 80, 80), 'ImageData')
    print(f"Created ImageData for notebook viewer: {image_data.shape}")
    
    # Create notebook viewer
    nb_viewer = NotebookViewer(image_data, width=600, height=500)
    
    print("\nNotebookViewer features:")
    print("  - Dropdown to select view (Axial, Coronal, Sagittal)")
    print("  - Sliders that automatically rebuild when view changes")
    print("  - Colormap dropdown")
    print("  - All in interactive Jupyter widgets")
    
    print(f"\nAvailable views: {list(nb_viewer.available_views.keys())}")
    
    # Demonstrate programmatic view changes
    print("\nProgrammatic view switching (for notebook):")
    for view in nb_viewer.available_views.keys():
        print(f"  - {view}: {nb_viewer.available_views[view]['labels']}")
    
    print("\nIn a Jupyter notebook:")
    print("  1. Call nb_viewer.show()")
    print("  2. Use the dropdown to switch views")
    print("  3. Watch the sliders rebuild automatically")
    print("  4. See different slice orientations")
    
    return nb_viewer


def demo_gif_creation():
    """Demonstrate GIF creation with different views."""
    print("\n=== GIF Creation with Views ===")
    
    image_data = MockSIRFData((15, 64, 64), 'ImageData')
    viewer = SIRFViewer(image_data, "GIF Demo")
    
    # Create GIFs for each view
    for view in viewer.get_available_views():
        viewer.set_view(view)
        filename = f'animation_{view.lower()}.gif'
        
        print(f"Creating {view} view animation...")
        viewer.create_gif(filename, fps=5)
        print(f"  Saved: {filename}")
    
    print("\nGIF animations created for all views!")


def main():
    """Run all view switching demos."""
    print("SIRF Viewer - View Switching Demonstration")
    print("=" * 50)
    
    # Run demos
    image_viewer = demo_imagedata_views()
    acq_viewer = demo_acquisition_views()
    nb_viewer = demo_notebook_viewer()
    
    print("\n" + "=" * 50)
    print("View switching demos completed!")
    
    print("\nTo try the interactive viewers:")
    print("  image_viewer.show()  # Interactive ImageData viewer")
    print("  acq_viewer.show()    # Interactive AcquisitionData viewer")
    print("  nb_viewer.show()     # Interactive notebook viewer")
    
    print("\nInteractive features:")
    print("  - Click view buttons to switch orientations")
    print("  - Use sliders to navigate through slices")  
    print("  - Try different colormaps")
    print("  - Save views and create animations")
    
    # Create sample GIFs
    demo_gif_creation()
    
    return {
        'image_viewer': image_viewer,
        'acq_viewer': acq_viewer,
        'nb_viewer': nb_viewer
    }


if __name__ == '__main__':
    viewers = main()
    
    viewers['image_viewer'].show()
    viewers['nb_viewer'].show()