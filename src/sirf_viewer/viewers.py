# src/sirf_viewer/viewers.py
"""
Viewer classes for SIRF ImageData and AcquisitionData objects.

This module provides the main viewer classes for both GUI and notebook usage.
"""


import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from PIL import Image
import io
from typing import Optional, Tuple, List, Union, Any, Dict
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
    AcquisitionData objects with interactive controls and view switching.
    """
    
    def __init__(self, data: Any, title: str = "SIRF Viewer"):
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF package is required. Install with: pip install sirf")

        self.data = data
        self.title = title
        self.colormap = 'gray'
        self.window_level = None
        self.window_width = None

        # Get data array and dimensions
        self.data_array = self._get_data_array()
        self.dimensions = self._get_dimensions()
        self.dimension_names = self._get_dimension_names()

        self.voxel_sizes = None
        if hasattr(self.data, 'voxel_sizes'):
            with contextlib.suppress(Exception):
                self.voxel_sizes = self.data.voxel_sizes()  # (z_size, y_size, x_size)
        # Set initial indices to middle of each dimension
        self.current_indices = [dim // 2 for dim in self.dimensions]

        # View system
        self.available_views = self._get_available_views()
        self.current_view = list(self.available_views.keys())[0]

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.im = None
        self.colorbar = None
        self.sliders = []
        self.buttons = []
        
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
        
    def _get_available_views(self) -> Dict[str, Dict]:
        """Get available view orientations based on data type."""
        views = {}
        
        if len(self.dimensions) == 3:  # ImageData (z, y, x)
            views = {
                'Axial': {
                    'scroll_dim': 0,  # Scroll through z
                    'display_dims': (1, 2),  # Show y-x plane
                    'labels': ('Y', 'X'),
                    'controllable_dims': []
                },
                'Coronal': {
                    'scroll_dim': 1,  # Scroll through y
                    'display_dims': (0, 2),  # Show z-x plane
                    'labels': ('Z', 'X'),
                    'controllable_dims': []
                },
                'Sagittal': {
                    'scroll_dim': 2,  # Scroll through x
                    'display_dims': (0, 1),  # Show z-y plane
                    'labels': ('Z', 'Y'),
                    'controllable_dims': []
                }
            }
        elif len(self.dimensions) == 4:  # AcquisitionData (tof, view, radial, axial)
            views = {
                'Radial-Axial': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (2, 3),  # Show radial-axial plane
                    'labels': ('Radial', 'Axial'),
                    'controllable_dims': [1]  # Also control view
                },
                'View-Axial (Sinogram)': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (1, 3),  # Show view-axial plane
                    'labels': ('View', 'Axial'),
                    'controllable_dims': [2]  # Also control radial
                },
                'View-Radial': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (1, 2),  # Show view-radial plane
                    'labels': ('View', 'Radial'),
                    'controllable_dims': [3]  # Also control axial
                }
            }
        
        return views
        
    def get_available_views(self) -> List[str]:
        """Get list of available view names."""
        return list(self.available_views.keys())
    
    def set_view(self, view_name: str):
        """Set the current view orientation."""
        if view_name not in self.available_views:
            available = list(self.available_views.keys())
            raise ValueError(f"View '{view_name}' not available. Available views: {available}")
            
        self.current_view = view_name
        if self.fig is not None:
            self._clear_sliders()
            self._add_sliders()
            self._update_display()
        
    def setup_plot(self):
        """Set up the matplotlib figure and initial plot."""
        # Create figure with extra space for controls
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle(self.title)
        
        # Adjust layout to make room for sliders and buttons
        plt.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.9)
        
        # Display initial slice
        self._update_display()
        
        # Add interactive controls
        self._add_sliders()
        self._add_control_buttons()
        
    def _update_display(self):
        """Update the display based on current indices and view."""
        # Get the current slice for the current view
        slice_data = self._get_current_slice()

        # Apply window/level if set
        if self.window_level is not None and self.window_width is not None:
            slice_data = self._apply_window_level(slice_data)

        # Clear the axes
        self.ax.clear()
        
        aspect = 'equal'  # default
        if self.voxel_sizes is not None and len(self.dimensions) == 3:
            vz, vy, vx = self.voxel_sizes
            if self.current_view == 'Axial':        # y-x plane
                aspect = vy / vx
            elif self.current_view == 'Coronal':    # z-x plane
                aspect = vz / vx
            elif self.current_view == 'Sagittal':   # z-y plane
                aspect = vz / vy

        # Display the image
        self.im = self.ax.imshow(slice_data, cmap=self.colormap, origin='lower', aspect=aspect)

        # Add colorbar if it doesn't exist or update it
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.colorbar.update_normal(self.im)

        # Update title and labels
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        labels = view_config['labels']

        title_parts = [
            f"{self.current_view} View",
            f"{self.dimension_names[scroll_dim]}: {self.current_indices[scroll_dim]}"
        ]

        # Add controllable dimensions to title
        title_parts.extend(
            f"{self.dimension_names[dim_idx]}: {self.current_indices[dim_idx]}"
            for dim_idx in view_config.get('controllable_dims', [])
        )
        self.ax.set_title(' - '.join(title_parts))
        self.ax.set_xlabel(labels[1])  # X-axis label
        self.ax.set_ylabel(labels[0])  # Y-axis label

        # Redraw
        self.fig.canvas.draw()
        
    def _get_current_slice(self) -> np.ndarray:
        """Get the current 2D slice based on current view."""
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        display_dims = view_config['display_dims']
        
        # Build the indexing tuple
        indices = list(self.current_indices)
        
        if len(self.dimensions) == 3:  # ImageData
            if self.current_view == 'Axial':
                return self.data_array[indices[0], :, :]
            elif self.current_view == 'Coronal':
                return self.data_array[:, indices[1], :]
            elif self.current_view == 'Sagittal':
                return self.data_array[:, :, indices[2]]

        elif len(self.dimensions) == 4:  # AcquisitionData
            # For 4D data, we need to specify all non-display dimensions
            if display_dims == (2, 3):  # Radial-Axial
                return self.data_array[indices[0], indices[1], :, :]
            elif display_dims == (1, 3):  # View-Axial
                return self.data_array[indices[0], :, indices[2], :]
            elif display_dims == (1, 2):  # View-Radial
                return self.data_array[indices[0], :, :, indices[3]]
        
        # Fallback - shouldn't reach here
        return self.data_array[tuple(indices[:-2]) + (slice(None), slice(None))]
                
    def _apply_window_level(self, data: np.ndarray) -> np.ndarray:
        """Apply window/level to the data."""
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        return np.clip(data, min_val, max_val)
        
    def _clear_sliders(self):
        """Clear existing sliders."""
        for slider in self.sliders:
            if hasattr(slider, 'ax'):
                slider.ax.remove()
        self.sliders = []
        
    def _add_sliders(self):
        """Add sliders based on current view."""
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        controllable_dims = view_config.get('controllable_dims', [])
        
        # List of dimensions that need sliders
        slider_dims = [scroll_dim] + controllable_dims
        
        slider_height = 0.03
        slider_spacing = 0.04
        
        for i, dim_idx in enumerate(slider_dims):
            # Create slider axis
            slider_bottom = 0.30 - i * slider_spacing
            ax_slider = plt.axes([0.15, slider_bottom, 0.5, slider_height])
            
            # Create slider
            slider = Slider(
                ax_slider, 
                self.dimension_names[dim_idx], 
                0, 
                self.dimensions[dim_idx] - 1, 
                valinit=self.current_indices[dim_idx], 
                valfmt='%d',
                valstep=1
            )
            
            # Connect callback
            def make_slider_callback(dim_idx):
                def callback(val):
                    self.current_indices[dim_idx] = int(val)
                    self._update_display()
                return callback
            
            slider.on_changed(make_slider_callback(dim_idx))
            self.sliders.append(slider)
            
    def _add_control_buttons(self):
        """Add control buttons."""
        button_width = 0.12
        button_height = 0.04
        button_spacing = 0.01
        
        button_row = 0.30
        button_col = 0.75
        
        # View switching buttons
        view_names = list(self.available_views.keys())
        for i, view_name in enumerate(view_names):
            ax_view = plt.axes([button_col, button_row - i * (button_height + button_spacing), 
                              button_width, button_height])
            btn_view = Button(ax_view, view_name[:8])  # Truncate long names
            btn_view.on_clicked(lambda event, view=view_name: self._on_view_click(view))
            self.buttons.append(btn_view)
        
        # Other control buttons
        other_buttons_start = button_row - len(view_names) * (button_height + button_spacing) - 0.02
        
        # Colormap button
        ax_colormap = plt.axes([button_col, other_buttons_start, button_width, button_height])
        btn_colormap = Button(ax_colormap, 'Colormap')
        btn_colormap.on_clicked(self._on_colormap_click)
        self.buttons.append(btn_colormap)
        
        # Save button
        ax_save = plt.axes([button_col, other_buttons_start - (button_height + button_spacing), 
                          button_width, button_height])
        btn_save = Button(ax_save, 'Save View')
        btn_save.on_clicked(self._on_save_click)
        self.buttons.append(btn_save)
        
        # Auto W/L button
        ax_auto = plt.axes([button_col, other_buttons_start - 2 * (button_height + button_spacing), 
                          button_width, button_height])
        btn_auto = Button(ax_auto, 'Auto W/L')
        btn_auto.on_clicked(self._on_auto_window_level)
        self.buttons.append(btn_auto)
        
    def _on_view_click(self, view_name: str):
        """Handle view button click."""
        print(f"Switching to {view_name} view")
        self.set_view(view_name)
        
    def _on_colormap_click(self, event):
        """Handle colormap button click."""
        colormaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        current_idx = colormaps.index(self.colormap) if self.colormap in colormaps else 0
        self.colormap = colormaps[(current_idx + 1) % len(colormaps)]
        print(f"Changed colormap to: {self.colormap}")
        self._update_display()
        
    def _on_save_click(self, event):
        """Handle save button click."""
        filename = f"sirf_{self.current_view.lower()}_{'_'.join(map(str, self.current_indices))}.png"
        self.save_current_view(filename)
        print(f"Saved current view to {filename}")
        
    def _on_auto_window_level(self, event):
        """Handle auto window/level button click."""
        data = self._get_current_slice()
        level = float(np.percentile(data, 50))  # Median
        width = float(np.percentile(data, 98) - np.percentile(data, 2))
        self.set_window(level, width)
        print(f"Auto W/L: Level={level:.1f}, Width={width:.1f}")
        
    def show(self):
        """Set up and display the interactive viewer."""
        self.setup_plot()
        plt.show()
        
    def set_colormap(self, colormap: str):
        """Set the colormap for the display."""
        self.colormap = colormap
        if self.fig is not None:
            self._update_display()
        
    def set_window(self, level: float, width: float):
        """Set window/level for the display."""
        self.window_level = level
        self.window_width = width
        if self.fig is not None:
            self._update_display()
        
    def save_current_view(self, filename: str):
        """Save the current view as an image."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        
    def create_gif(self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None):
        """Create an animated GIF from the data."""
        if dimensions is None:
            # Use the scroll dimension of current view
            view_config = self.available_views[self.current_view]
            dimensions = [view_config['scroll_dim']]
            
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
            
            # Temporarily set indices to get slice
            old_indices = self.current_indices.copy()
            self.current_indices = temp_indices
            slice_data = self._get_current_slice()
            self.current_indices = old_indices
                
            # Apply window/level if set
            if self.window_level is not None and self.window_width is not None:
                slice_data = self._apply_window_level(slice_data)

            im = ax.imshow(slice_data, cmap=self.colormap, origin='upper')
            ax.set_title(f'{self.current_view} - {self.dimension_names[dim_to_animate]}: {frame}')
            return [im]
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=1000//fps, blit=True
        )
        
        # Save as GIF
        anim.save(filename, writer='pillow', fps=fps)
        plt.close(fig)
        print(f"Created GIF: {filename}")


class NotebookViewer:
    """Jupyter notebook-compatible viewer with interactive widgets."""
    
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
        
        # Voxel sizes if available
        self.voxel_sizes = None
        if hasattr(self.data, 'voxel_sizes'):
            with contextlib.suppress(Exception):
                self.voxel_sizes = self.data.voxel_sizes()  # (z_size, y_size, x_size)

        # Current state
        self.current_indices = [dim // 2 for dim in self.dimensions]
        self.colormap = 'gray'

        # View system
        self.available_views = self._get_available_views()
        self.current_view = list(self.available_views.keys())[0]

        # Check for ipywidgets
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            self.widgets_available = True
            self._widgets = widgets
            self._display = display
            self._clear_output = clear_output
        except ImportError:
            self.widgets_available = False
            warnings.warn("ipywidgets not available. Install with: pip install ipywidgets")

    def _get_data_array(self) -> np.ndarray:
        """Get numpy array from SIRF data object."""
        try:
            return self.data.asarray()
        except AttributeError as e:
            raise AttributeError("Data object must have asarray() method") from e
            
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
        
    def _get_available_views(self) -> Dict[str, Dict]:
        """Get available view orientations based on data type."""
        views = {}
        
        if len(self.dimensions) == 3:  # ImageData (z, y, x)
            views = {
                'Axial': {
                    'scroll_dim': 0,  # Scroll through z
                    'display_dims': (1, 2),  # Show y-x plane
                    'labels': ('Y', 'X'),
                    'controllable_dims': []
                },
                'Coronal': {
                    'scroll_dim': 1,  # Scroll through y
                    'display_dims': (0, 2),  # Show z-x plane
                    'labels': ('Z', 'X'),
                    'controllable_dims': []
                },
                'Sagittal': {
                    'scroll_dim': 2,  # Scroll through x
                    'display_dims': (0, 1),  # Show z-y plane
                    'labels': ('Z', 'Y'),
                    'controllable_dims': []
                }
            }
        elif len(self.dimensions) == 4:  # AcquisitionData (tof, view, radial, axial)
            views = {
                'Rad-Ax': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (2, 3),  # Show radial-axial plane
                    'labels': ('Radial', 'Axial'),
                    'controllable_dims': [1]  # Also control view
                },
                'View-Ax (Sinogram)': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (1, 3),  # Show view-axial plane
                    'labels': ('View', 'Axial'),
                    'controllable_dims': [2]  # Also control radial
                },
                'View-Rad': {
                    'scroll_dim': 0,  # Primary: ToF bins
                    'display_dims': (1, 2),  # Show view-radial plane
                    'labels': ('View', 'Radial'),
                    'controllable_dims': [3]  # Also control axial
                }
            }
        
        return views
        
    def _get_current_slice(self) -> np.ndarray:
        """Get the current 2D slice based on current view."""
        view_config = self.available_views[self.current_view]
        indices = list(self.current_indices)
        
        if len(self.dimensions) == 3:  # ImageData
            if self.current_view == 'Axial':
                return self.data_array[indices[0], :, :]
            elif self.current_view == 'Coronal':
                return self.data_array[:, indices[1], :]
            elif self.current_view == 'Sagittal':
                return self.data_array[:, :, indices[2]]

        elif len(self.dimensions) == 4:  # AcquisitionData
            display_dims = view_config['display_dims']
            if display_dims == (2, 3):  # Radial-Axial
                return self.data_array[indices[0], indices[1], :, :]
            elif display_dims == (1, 3):  # View-Axial
                return self.data_array[indices[0], :, indices[2], :]
            elif display_dims == (1, 2):  # View-Radial
                return self.data_array[indices[0], :, :, indices[3]]
        
        # Fallback
        return self.data_array[tuple(indices[:-2]) + (slice(None), slice(None))]
        
    def show(self):
        """Display the interactive viewer."""
        if not self.widgets_available:
            print("ipywidgets not available. Using basic matplotlib display.")
            self._show_static()
            return
            
        # Create interactive widgets
        self._create_interactive_viewer()
        
    def _show_static(self):
        """Show static matplotlib display as fallback."""
        self._plot_slice_with_colorbar()
        plt.title(self._get_title())
        plt.show()
        
    def _create_interactive_viewer(self):
        """Create interactive viewer with widgets."""
        widgets = self._widgets
        
        # View selector
        view_dropdown = widgets.Dropdown(
            options=list(self.available_views.keys()),
            value=self.current_view,
            description='View:'
        )
        
        def on_view_change(change):
            self.current_view = change.new
            self._rebuild_sliders()
            
        view_dropdown.observe(on_view_change, names='value')
        
        # Create initial sliders
        self._create_sliders()
                
        # Colormap dropdown
        colormap_dropdown = widgets.Dropdown(
            options=['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            value=self.colormap,
            description='Colormap:'
        )
        
        def on_colormap_change(change):
            self.colormap = change.new
            self._update_interactive_plot()
            
        colormap_dropdown.observe(on_colormap_change, names='value')
        
        # Create output widget for the plot
        self.output = widgets.Output()
        
        # Store references for rebuilding
        self.view_dropdown = view_dropdown
        self.colormap_dropdown = colormap_dropdown
        
        # Initial layout and display
        self._rebuild_layout()
        self._update_interactive_plot()
        
    def _create_sliders(self):
        """Create sliders for current view."""
        widgets = self._widgets
        
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        controllable_dims = view_config.get('controllable_dims', [])
        
        # List of dimensions that need sliders
        slider_dims = [scroll_dim] + controllable_dims
        
        self.sliders = []
        for dim_idx in slider_dims:
            slider = widgets.IntSlider(
                value=self.current_indices[dim_idx],
                min=0,
                max=self.dimensions[dim_idx] - 1,
                step=1,
                description=f'{self.dimension_names[dim_idx]}:',
                continuous_update=False
            )
            
            def make_callback(idx):
                def callback(change):
                    self.current_indices[idx] = change.new
                    self._update_interactive_plot()
                return callback
            
            slider.observe(make_callback(dim_idx), names='value')
            self.sliders.append(slider)
            
    def _rebuild_sliders(self):
        """Rebuild sliders when view changes."""
        self._create_sliders()
        self._rebuild_layout()
        self._update_interactive_plot()
        
    def _rebuild_layout(self):
        """Rebuild the widget layout."""
        widgets = self._widgets
        
        # Create layout
        controls = widgets.VBox([
            self.view_dropdown,
            *self.sliders,
            self.colormap_dropdown
        ])
        viewer = widgets.VBox([controls, self.output])
        
        # Clear and display the updated widget
        self._clear_output(wait=True)
        self._display(viewer)
        
    def _update_interactive_plot(self):
        """Update the interactive plot."""
        with self.output:
            self.output.clear_output(wait=True)

            self._plot_slice_with_colorbar()
            view_config = self.available_views[self.current_view]
            labels = view_config['labels']
            plt.xlabel(labels[1])
            plt.ylabel(labels[0])
            plt.title(self._get_title())
            plt.show()

    def _plot_slice_with_colorbar(self):
        plt.figure(figsize=(self.width / 100, self.height / 100))
        slice_data = self._get_current_slice()
        
        aspect = 'equal'  # default
        if self.voxel_sizes is not None and len(self.dimensions) == 3:
            vz, vy, vx = self.voxel_sizes
            if self.current_view == 'Axial':        # y-x plane
                aspect = vy / vx
            elif self.current_view == 'Coronal':    # z-x plane  
                aspect = vz / vx
            elif self.current_view == 'Sagittal':   # z-y plane
                aspect = vz / vy

        plt.imshow(slice_data, cmap=self.colormap, origin='upper', aspect=aspect)
        plt.colorbar()
            
    def _get_title(self):
        """Get title string with current view and indices."""
        view_config = self.available_views[self.current_view]
        scroll_dim = view_config['scroll_dim']
        controllable_dims = view_config.get('controllable_dims', [])

        title_parts = [
            f"{self.current_view} View",
            f"{self.dimension_names[scroll_dim]}: {self.current_indices[scroll_dim]}",
        ]

        # Add controllable dimensions
        title_parts.extend(
            f"{self.dimension_names[dim_idx]}: {self.current_indices[dim_idx]}"
            for dim_idx in controllable_dims
        )
        return ' - '.join(title_parts)
        
    def create_gif(self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None):
        """Create an animated GIF from the data."""
        # Use the same implementation as SIRFViewer
        temp_viewer = SIRFViewer(self.data, "Temporary GIF Creator")
        temp_viewer.current_view = self.current_view
        temp_viewer.colormap = self.colormap
        temp_viewer.create_gif(filename, fps, dimensions)