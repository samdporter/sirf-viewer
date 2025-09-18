#!/usr/bin/env python3
"""
Example demonstrating view switching functionality (updated for refactored viewers).

This shows how to use the view switching features in both SIRFViewer and NotebookViewer.
"""

import numpy as np
import os

from sirf_viewer.viewers import SIRFViewer

# create output directory in current folder
output_dir = os.path.join(os.path.dirname(__file__), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.chdir(output_dir)


# Create mock data
class MockSIRFData:
    """Mock SIRF data object for testing view switching."""

    def __init__(self, shape, class_name="ImageData"):
        self.shape = shape
        self._class_name = class_name
        if class_name == "ImageData":
            self._data = self._create_anatomical_image_data(shape)
        else:
            self._data = self._create_structured_acquisition_data(shape)

    def asarray(self):
        return self._data

    def _create_anatomical_image_data(self, shape):
        """Create anatomical-like test data with clear directional features."""
        z, y, x = shape
        data = np.zeros((z, y, x))

        center_z, center_y, center_x = z // 2, y // 2, x // 2

        rng = np.random.default_rng(42)
        for iz in range(z):
            for iy in range(y):
                for ix in range(x):
                    # Background noise
                    noise = rng.normal(0, 20)

                    # Axial feature (circular in axial view)
                    axial_dist = np.sqrt((iy - center_y) ** 2 + (ix - center_x) ** 2)
                    axial_feature = (
                        500 * np.exp(-(axial_dist**2) / 800) if axial_dist < 40 else 0
                    )

                    # Sagittal feature (vertical stripe in sagittal view)
                    sagittal_feature = 300 * np.exp(
                        -((iz - center_z) ** 2 + (iy - center_y + 20) ** 2) / 200
                    )

                    # Coronal feature (horizontal stripe in coronal view)
                    coronal_feature = 400 * np.exp(
                        -((iz - center_z + 15) ** 2 + (ix - center_x) ** 2) / 300
                    )

                    # Z-variation
                    z_factor = 1.0 + 0.3 * np.sin(2 * np.pi * iz / max(1, z))

                    data[iz, iy, ix] = noise + z_factor * (
                        axial_feature + sagittal_feature + coronal_feature
                    )

        return np.maximum(data, 0)  # Ensure non-negative

    def _create_structured_acquisition_data(self, shape):
        """Create structured acquisition data with clear features."""
        tof, views, radial, axial = shape
        data = np.zeros((tof, views, radial, axial))
        rng = np.random.default_rng(123)

        for t in range(tof):
            for v in range(views):
                angle = v * np.pi / max(1, views)
                center_r, center_a = radial // 2, axial // 2
                tof_factor = np.exp(-t / 3) if t > 0 else 1.0
                for r in range(radial):
                    for a in range(axial):
                        # Background noise
                        noise = rng.normal(0, 5)

                        # Sinogram-like pattern
                        sino_feature = 100 * np.exp(
                            -((r - center_r) ** 2 + (a - center_a) ** 2) / 100
                        )

                        # View-dependent variation
                        view_variation = (
                            50
                            * np.sin(2 * angle)
                            * np.exp(-((r - center_r) ** 2) / 200)
                        )

                        data[t, v, r, a] = noise + tof_factor * (
                            sino_feature + view_variation
                        )

        return np.maximum(data, 0)


def demo_imagedata_views():
    """Demonstrate view switching with ImageData."""
    print("=== ImageData View Switching Demo ===")

    # Create anatomical-like test data
    image_data = MockSIRFData((30, 100, 100), "ImageData")
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

    for view in viewer.get_available_views():
        viewer.set_view(view)
        vd = viewer.state.views[view]
        scroll_dim = vd["scroll_dim"]
        dim_name = viewer.state.dim_names[scroll_dim]
        print(f"  - Switched to {view} view")
        print(f"    Scrolling through {dim_name}")

        # Save a sample from each view
        viewer.save_current_view(f"sample_{view.lower()}_view.png")
        print(f"    Saved sample to sample_{view.lower()}_view.png")

    print("\nCall viewer.show() to see the interactive version")
    return viewer


def demo_acquisition_views():
    """Demonstrate view switching with AcquisitionData."""
    print("\n=== AcquisitionData View Switching Demo ===")

    # Create acquisition test data
    acq_data = MockSIRFData((5, 20, 48, 48), "AcquisitionData")
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

    for view in viewer.get_available_views():
        viewer.set_view(view)
        view_config = viewer.state.views[view]
        print(f"  - Switched to {view}")
        primary_name = viewer.state.dim_names[view_config["scroll_dim"]]
        print(f"    Primary scroll: {primary_name}")

        if controllable_dims := view_config.get("controllable_dims", []):
            ctrl_names = [viewer.state.dim_names[i] for i in controllable_dims]
            print(f"    Also controls: {', '.join(ctrl_names)}")

        # Save a sample
        out_name = f"sample_acq_{view.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        viewer.save_current_view(out_name)
        print(f"    Saved sample to {out_name}")

    print("\nCall viewer.show() to see the interactive version")
    return viewer


def demo_gif_creation():
    """Demonstrate GIF creation with different views."""
    print("\n=== GIF Creation with Views ===")

    image_data = MockSIRFData((15, 64, 64), "ImageData")
    viewer = SIRFViewer(image_data, "GIF Demo")

    # Create GIFs for each view
    for view in viewer.get_available_views():
        viewer.set_view(view)
        filename = f"animation_{view.lower()}.gif"

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

    print("\n" + "=" * 50)
    print("View switching demos completed!")

    print("\nTo try the interactive viewers:")
    print("  image_viewer.show()  # Interactive ImageData viewer")
    print("  acq_viewer.show()    # Interactive AcquisitionData viewer")

    print("\nInteractive features:")
    print("  - Click view buttons to switch orientations")
    print("  - Use sliders to navigate through slices")
    print("  - Try different colormaps")
    print("  - Save views and create animations")

    # Create sample GIFs
    demo_gif_creation()

    return {
        "image_viewer": image_viewer,
        "acq_viewer": acq_viewer,
    }


if __name__ == "__main__":
    viewers = main()

    viewers["image_viewer"].show()
    viewers["acq_viewer"].show()
