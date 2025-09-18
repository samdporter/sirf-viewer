# src/sirf_viewer/viewers.py
"""
Viewer classes for SIRF ImageData and AcquisitionData objects.

This module provides the main viewer classes for both GUI and notebook usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from PIL import Image
import io
from typing import Optional, Tuple, List, Union, Any
import warnings

try:
    import sirf.STIR as sirf
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False
    warnings.warn("SIRF not available. Install sirf package to use SIRF data objects.")


class SIRFViewer:
    """
    Main viewer class for SIRF ImageData and AcquisitionData objects.
    
    This class provides a comprehensive viewer for both 3D ImageData and 4D 
    AcquisitionData objects with interactive controls and animation capabilities.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to view
    title : str, optional
        Title for the viewer window (default: "SIRF Viewer")
    """
    
    def __init__(self, data: Any, title: str = "SIRF Viewer"):
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF package is required. Install with: pip install sirf")
            
        self.data = data
        self.title = title
        self.colormap = 'gray'
        self.window_level = None
        self.window_width = None
        
        # Get data array and dimensions FIRST
        self.data_array = self._get_data_array()
        self.dimensions = self._get_dimensions()
        self.dimension_names = self._get_dimension_names()
        
        # Set initial indices to middle of each dimension
        self.current_indices = [dim // 2 for dim in self.dimensions]
        
        # NEW: View selection for different slice orientations
        self.available_views = self._get_available_views()
        self.current_view = list(self.available_views.keys())[0]  # Default to first available view
        
        # Set up the figure and plot
        self.fig = None
        self.ax = None
        self.im = None
        self.sliders = []
        self.view_buttons = []
        
        self.setup_plot()
        
    def _get_available_views(self) -> dict:
        """Get available view orientations based on data type and dimensions."""
        views = {}
        
        if len(self.dimensions) == 3:  # ImageData
            views = {
                'Axial (Z-Y-X)': {
                    'scroll_dim': 0,  # Scroll through Z
                    'display_dims': (1, 2),  # Show Y-X plane
                    'labels': ('Y', 'X')
                },
                'Coronal (Y-Z-X)': {
                    'scroll_dim': 1,  # Scroll through Y
                    'display_dims': (0, 2),  # Show Z-X plane
                    'labels': ('Z', 'X')
                },
                'Sagittal (X-Z-Y)': {
                    'scroll_dim': 2,  # Scroll through X
                    'display_dims': (0, 1),  # Show Z-Y plane
                    'labels': ('Z', 'Y')
                }
            }
        elif len(self.dimensions) == 4:  # AcquisitionData: (ToF bin, view, axial, tangential)
            views = {
                'Axial-Tangential': {
                    'scroll_dim': 0,  # Primary: Scroll through ToF bins
                    'display_dims': (1, 3),  # Show Axial vs Tangential
                    'labels': ('Axial', 'Tangential'),
                    'fixed_dims': [2]  # Secondary scroller: Axial position
                },
                'View-Tangential (Sinogram)': {
                    'scroll_dim': 0,  # Primary: Scroll through ToF bins
                    'display_dims': (2, 3),  # Show View vs Tangential  
                    'labels': ('View', 'Tangential'),
                    'fixed_dims': [1]  # Secondary scroller: View angle
                },
                'View-Axial': {
                    'scroll_dim': 0,  # Primary: Scroll through ToF bins
                    'display_dims': (1, 2),  # Show View vs Axial
                    'labels': ('View', 'Axial'), 
                    'fixed_dims': [3]  # Secondary scroller: Tangential position
                }
            }
        
        return views
        
    def get_available_views(self) -> list:
        """Get list of available view names."""
        return list(self.available_views.keys())
    
    def set_view(self, view_name: str):
        """Set the current view orientation."""
        if view_name in self.available_views:
            self.current_view = view_name
            self._update_display()
            self._update_sliders()
        else:
            available = list(self.available_views.keys())
            raise ValueError(f"View '{view_name}' not available. Available views: {available}")
        
    def _add_view_controls(self):
        """Add view selection controls to the plot."""
        # Add view selection buttons
        view_names = list(self.available_views.keys())
        button_width = 0.15
        button_height = 0.04
        start_x = 0.1
        
        for i, view_name in enumerate(view_names):
            ax_button = plt.axes([start_x + i * (button_width + 0.01), 0.02, button_width, button_height])
            button = Button(ax_button, view_name.split('(')[0])  # Short name
            button.on_clicked(lambda event, view=view_name: self._on_view_change(view))
            self.view_buttons.append(button)

    def _on_view_change(self, view_name: str):
        """Handle view change button click."""
        self.set_view(view_name)
        print(f"Switched to {view_name} view")
        
    def _get_data_array(self) -> np.ndarray:
        """Get numpy array from SIRF data object."""
        try:
            return self.data.asarray()
        except AttributeError:
            raise AttributeError("Data object must have asarray() method")
            
    def _get_num_dimensions(self) -> int:
        """Get number of dimensions in the data."""
        return len(self.data_array.shape)
        
    def _get_dimensions(self) -> Tuple[int, ...]:
        """Get dimensions of the data."""
        return self.data_array.shape
        
    def _get_dimension_names(self) -> List[str]:
        """Get dimension names based on data type and shape."""
        if hasattr(self.data, '__class__'):
            class_name = self.data.__class__.__name__
            
            if class_name == 'ImageData':
                # ImageData is 3D: (z, y, x)
                return ['z', 'y', 'x']
            elif class_name == 'AcquisitionData':
                # AcquisitionData is 4D: (ToF Bin, 1, 2, 3)
                # ToF bin is always dim 1 if SPECT data
                return ['ToF Bin', 'View', 'Radial', 'Axial']
                
        # Default dimension names
        return [f'Dim {i}' for i in range(len(self.dimensions))]
        
    def setup_plot(self):
        """Set up the matplotlib figure and initial plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle(self.title)
        
        # Display initial slice
        self._update_display()
        
        # Add sliders for current view
        self._add_sliders()
        
        # Add view selection controls
        self._add_view_controls()
        
        # Add other controls
        self._add_controls()
        
    def _update_display(self):
        """Update the display based on current indices and view."""
        # Get the current slice
        slice_data = self._get_current_slice()

        # Apply window/level if set
        if self.window_level is not None and self.window_width is not None:
            slice_data = self._apply_window_level(slice_data)

        # Clear and redraw
        self.ax.clear()

        if self.im is None:
            self.im = self.ax.imshow(slice_data, cmap=self.colormap, origin='lower')
            self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_array(slice_data)
            self.im.set_cmap(self.colormap)

        # Update title with current view and indices
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        labels = view_config['labels']

        title_parts = [
            f"{self.current_view}",
            f"{self.dimension_names[scroll_dim]}: {self.current_indices[scroll_dim]}",
        ]
        self.ax.set_title(' - '.join(title_parts))
        self.ax.set_xlabel(labels[1])  # X-axis label
        self.ax.set_ylabel(labels[0])  # Y-axis label

        self.fig.canvas.draw()
        
    def _update_sliders(self):
        """Update slider ranges and values when view changes."""
        # Clear existing sliders
        for slider in self.sliders:
            slider.ax.remove()
        self.sliders = []
        
        # Recreate sliders for new view
        self._add_sliders()
        
    def _get_current_slice(self) -> np.ndarray:
        """Get the current 2D slice based on selected view."""
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        display_dims = view_config['display_dims']

        # Build indexing tuple
        indices = []
        for i in range(len(self.dimensions)):
            if i == scroll_dim or i not in display_dims:
                indices.append(self.current_indices[i])
            else:
                indices.append(slice(None))
        slice_data = self.data_array[tuple(indices)]

        # Ensure we have a 2D array
        if slice_data.ndim > 2:
            # Squeeze out singleton dimensions
            slice_data = np.squeeze(slice_data)

        return slice_data
                
    def _apply_window_level(self, data: np.ndarray) -> np.ndarray:
        """Apply window/level to the data."""
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        return np.clip(data, min_val, max_val)
        
    def _add_sliders(self):
        """Add sliders for the scrollable dimension based on current view."""
        # Clear existing sliders
        for slider in self.sliders:
            slider.ax.remove()
        self.sliders = []
        
        # Get current view configuration
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        
        # Adjust subplot to make room for sliders and buttons
        plt.subplots_adjust(bottom=0.25)
        
        # Add slider for the scrollable dimension
        ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
        slider = Slider(
            ax_slider, 
            self.dimension_names[scroll_dim], 
            0, 
            self.dimensions[scroll_dim] - 1, 
            valinit=self.current_indices[scroll_dim], 
            valfmt='%d'
        )
        slider.on_changed(lambda val: self._on_slider_change(scroll_dim, val))
        self.sliders.append(slider)
        
        # Add sliders for other controllable dimensions if they exist
        other_dims = view_config.get('fixed_dims', [])
        for i, dim_idx in enumerate(other_dims):
            ax_slider = plt.axes([0.2, 0.10 - i * 0.04, 0.6, 0.03])
            slider = Slider(
                ax_slider, 
                self.dimension_names[dim_idx], 
                0, 
                self.dimensions[dim_idx] - 1, 
                valinit=self.current_indices[dim_idx], 
                valfmt='%d'
            )
            slider.on_changed(lambda val, idx=dim_idx: self._on_slider_change(idx, val))
            self.sliders.append(slider)
                
    def _add_controls(self):
        """Add control buttons."""
        # Add colormap button
        ax_colormap = plt.axes([0.85, 0.9, 0.1, 0.04])
        btn_colormap = Button(ax_colormap, 'Colormap')
        btn_colormap.on_clicked(self._on_colormap_click)
        
        # Add save button
        ax_save = plt.axes([0.85, 0.85, 0.1, 0.04])
        btn_save = Button(ax_save, 'Save')
        btn_save.on_clicked(self._on_save_click)
        
        # Add GIF button
        ax_gif = plt.axes([0.85, 0.8, 0.1, 0.04])
        btn_gif = Button(ax_gif, 'Create GIF')
        btn_gif.on_clicked(self._on_gif_click)
        
    def _on_slider_change(self, dim_idx: int, val: float):
        """Handle slider value changes for a specific dimension."""
        self.current_indices[dim_idx] = int(val)
        self._update_display()
        
    def _on_colormap_click(self, event):
        """Handle colormap button click."""
        colormaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        current_idx = colormaps.index(self.colormap) if self.colormap in colormaps else 0
        self.colormap = colormaps[(current_idx + 1) % len(colormaps)]
        self._update_display()
        
    def _on_save_click(self, event):
        """Handle save button click."""
        filename = f"sirf_view_slice_{'_'.join(map(str, self.current_indices))}.png"
        self.save_current_view(filename)
        print(f"Saved current view to {filename}")
        
    def _on_gif_click(self, event):
        """Handle GIF button click."""
        filename = "sirf_animation.gif"
        self.create_gif(filename, fps=10)
        print(f"Created GIF animation: {filename}")
        
    def show(self):
        """Display the viewer."""
        plt.show()
        
    def set_colormap(self, colormap: str):
        """Set the colormap for the display."""
        self.colormap = colormap
        self._update_display()
        
    def set_window(self, level: float, width: float):
        """Set window/level for the display."""
        self.window_level = level
        self.window_width = width
        self._update_display()
        
    def save_current_view(self, filename: str):
        """Save the current view as an image."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        
    def create_gif(self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None):
        """
        Create an animated GIF from the data.
        
        Parameters
        ----------
        filename : str
            Output filename for the GIF
        fps : int, optional
            Frames per second (default: 10)
        dimensions : list of int, optional
            Which dimensions to animate (default: first scrollable dimension)
        """
        if dimensions is None:
            dimensions = [0]  # Animate first dimension by default
            
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine animation frames
        dim_to_animate = dimensions[0]
        num_frames = self.dimensions[dim_to_animate]
        
        def animate(frame):
            ax.clear()
            
            # Set current indices for this frame
            temp_indices = self.current_indices.copy()
            temp_indices[dim_to_animate] = frame
            
            # Get slice data
            if len(self.dimensions) == 3:
                slice_data = self.data_array[temp_indices[0], :, :]
            elif len(self.dimensions) == 4:
                slice_data = self.data_array[temp_indices[0], temp_indices[1], :, :]
            else:
                indices = temp_indices[:-2] + [slice(None), slice(None)]
                slice_data = self.data_array[tuple(indices)]
                
            # Apply window/level if set
            if self.window_level is not None and self.window_width is not None:
                slice_data = self._apply_window_level(slice_data)
                
            im = ax.imshow(slice_data, cmap=self.colormap, origin='lower')
            ax.set_title(f'Frame {frame + 1}/{num_frames}')
            return [im]
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=1000//fps, blit=True
        )
        
        # Save as GIF
        anim.save(filename, writer='pillow', fps=fps)
        plt.close(fig)


class NotebookViewer:
    """
    Jupyter notebook-compatible viewer with interactive widgets.
    
    This class provides a viewer that works well in Jupyter notebooks
    with interactive widgets for controlling the display.
    
    Parameters
    ----------
    data : sirf.ImageData or sirf.AcquisitionData
        The SIRF data object to view
    width : int, optional
        Widget width in pixels (default: 800)
    height : int, optional
        Widget height in pixels (default: 600)
    """
    
    def __init__(self, data: Any, width: int = 800, height: int = 600):
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF package is required. Install with: pip install sirf")
            
        self.data = data
        self.width = width
        self.height = height
        
        # Get data array and dimensions
        self.data_array = self._get_data_array()
        self.dimensions = self._get_dimensions()
        self.dimension_names = self._get_dimension_names()
        
        # Current state
        self.current_indices = [dim // 2 for dim in self.dimensions]
        self.colormap = 'gray'
        
        # Try to import ipywidgets
        try:
            import ipywidgets as widgets
            from IPython.display import display
            self.widgets_available = True
        except ImportError:
            self.widgets_available = False
            warnings.warn("ipywidgets not available. Install with: pip install ipywidgets")
            
    def _get_data_array(self) -> np.ndarray:
        """Get numpy array from SIRF data object."""
        try:
            return self.data.asarray()
        except AttributeError:
            raise AttributeError("Data object must have asarray() method")
            
    def _get_dimensions(self) -> Tuple[int, ...]:
        """Get dimensions of the data."""
        return self.data_array.shape
        
    def _get_dimension_names(self) -> List[str]:
        """Get dimension names based on data type and shape."""
        if hasattr(self.data, '__class__'):
            class_name = self.data.__class__.__name__
            
            if class_name == 'ImageData':
                return ['z', 'y', 'x']
            elif class_name == 'AcquisitionData':
                return ['ToF Bin', 'View', 'Radial', 'Axial']
                
        return [f'Dim {i}' for i in range(len(self.dimensions))]
        
    def _get_current_slice(self) -> np.ndarray:
        """Get the current 2D slice based on indices."""
        if len(self.dimensions) == 3:
            return self.data_array[self.current_indices[0], :, :]
        elif len(self.dimensions) == 4:
            return self.data_array[self.current_indices[0], self.current_indices[1], :, :]
        else:
            if len(self.dimensions) < 2:
                return self.data_array
            indices = self.current_indices[:-2] + [slice(None), slice(None)]
            return self.data_array[tuple(indices)]
                
    def _update_plot(self, *args):
        """Update the plot when controls change."""
        # Get current slice
        slice_data = self._get_current_slice()
        
        # Clear and redraw
        plt.figure(figsize=(self.width/100, self.height/100))
        plt.imshow(slice_data, cmap=self.colormap, origin='lower')
        plt.colorbar()
        
        # Update title
        index_str = ', '.join([f'{name}: {idx}' 
                             for name, idx in zip(self.dimension_names, self.current_indices)])
        plt.title(f'Slice - {index_str}')
        
        plt.show()
        
    def show(self):
        """Display the interactive viewer."""
        if not self.widgets_available:
            print("ipywidgets not available. Using basic matplotlib display.")
            self._update_plot()
            return
            
        import ipywidgets as widgets
        from IPython.display import display
        
        # Create sliders for dimensions
        sliders = []
        for i, (name, size) in enumerate(zip(self.dimension_names, self.dimensions)):
            if i < 2:  # Only create sliders for first 2 dimensions
                slider = widgets.IntSlider(
                    value=self.current_indices[i],
                    min=0,
                    max=size - 1,
                    step=1,
                    description=name,
                    continuous_update=False
                )
                slider.observe(lambda change, idx=i: self._update_index(idx, change.new), names='value')
                sliders.append(slider)
                
        # Create colormap dropdown
        colormap_dropdown = widgets.Dropdown(
            options=['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            value=self.colormap,
            description='Colormap:'
        )
        colormap_dropdown.observe(lambda change: self._update_colormap(change.new), names='value')
        
        # Create output widget
        output = widgets.Output()
        
        # Layout
        controls = widgets.VBox(sliders + [colormap_dropdown])
        viewer = widgets.VBox([controls, output])
        
        # Initial display
        with output:
            self._update_plot()
            
        display(viewer)
        
    def _update_index(self, dim_idx: int, value: int):
        """Update index for a specific dimension."""
        self.current_indices[dim_idx] = value
        self._update_plot()
        
    def _update_colormap(self, colormap: str):
        """Update the colormap."""
        self.colormap = colormap
        self._update_plot()
        
    def create_gif(self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None):
        """
        Create an animated GIF from the data.
        
        Parameters
        ----------
        filename : str
            Output filename for the GIF
        fps : int, optional
            Frames per second (default: 10)
        dimensions : list of int, optional
            Which dimensions to animate (default: first scrollable dimension)
        """
        # Use the same implementation as SIRFViewer
        temp_viewer = SIRFViewer(self.data, "Temporary GIF Creator")
        temp_viewer.create_gif(filename, fps, dimensions)