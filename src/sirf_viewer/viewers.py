# src/sirf_viewer/viewers.py
"""
Unified SIRF viewers with shared core and optional background overlay for 3D ImageData.

- Core (UI-agnostic): ViewerState + pure helpers
- SIRFViewer (matplotlib desktop)
- NotebookViewer (ipywidgets notebook)

All displays use origin='upper' (standardised).
Background overlay:
  - Supported only for 3D ImageData (z, y, x).
  - Shape must match foreground exactly.
  - If both voxel sizes are available, they must match (within tolerance).
  - The foreground (emission) is drawn on top with configurable alpha.

Intensity control (single, consistent):
  - Display range (vmin, vmax) controls the mapping for imshow.
  - If not set (None), each slice auto-scales on load (matplotlib default).
  - "Auto Range" buttons compute vmin/vmax from percentiles of the current slice.
  - set_window(level, width) maps to display range: vmin=level-width/2, vmax=level+width/2.
"""

from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
import warnings

try:
    import sirf.STIR as sirf  # noqa: F401
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False
    warnings.warn(
        "SIRF not available. Install sirf to use SIRF data objects.", RuntimeWarning
    )

# ----------------------------- Core (UI-agnostic) -----------------------------

COLORMAPS = ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]


def infer_dimension_names(data_obj: Any, shape: Tuple[int, ...]) -> List[str]:
    if hasattr(data_obj, "__class__"):
        name = data_obj.__class__.__name__
        if name == "ImageData" and len(shape) == 3:
            return ["z", "y", "x"]
        if name == "AcquisitionData" and len(shape) == 4:
            return ["ToF Bin", "View", "Radial", "Axial"]
    return [f"Dim {i}" for i in range(len(shape))]


def build_available_views(shape: Tuple[int, ...]) -> Dict[str, Dict[str, Any]]:
    if len(shape) == 3:  # (z, y, x)
        return {
            "Axial": dict(
                scroll_dim=0,
                display_dims=(1, 2),
                labels=("Y", "X"),
                controllable_dims=[],
            ),
            "Coronal": dict(
                scroll_dim=1,
                display_dims=(0, 2),
                labels=("Z", "X"),
                controllable_dims=[],
            ),
            "Sagittal": dict(
                scroll_dim=2,
                display_dims=(0, 1),
                labels=("Z", "Y"),
                controllable_dims=[],
            ),
        }
    if len(shape) == 4:  # (ToF, View, Radial, Axial)
        return {
            "Radial-Axial": {
                "scroll_dim": 0,
                "display_dims": (2, 3),
                "labels": ("Radial", "Axial"),
                "controllable_dims": [1],
            },
            "View-Axial (Sinogram)": {
                "scroll_dim": 0,
                "display_dims": (1, 3),
                "labels": ("View", "Axial"),
                "controllable_dims": [2],
            },
            "View-Radial": {
                "scroll_dim": 0,
                "display_dims": (1, 2),
                "labels": ("View", "Radial"),
                "controllable_dims": [3],
            },
        }
    # Fallback generic
    last = len(shape) - 1
    return {
        "Last-2D": dict(
            scroll_dim=max(0, len(shape) - 3),
            display_dims=(last - 1, last),
            labels=(f"Dim {last - 1}", f"Dim {last}"),
            controllable_dims=[],
        )
    }


def get_slice(
    arr: np.ndarray,
    indices: List[int],
    view_name: str,
    views: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    v = views[view_name]
    if arr.ndim == 3:
        z, y, x = indices
        if view_name == "Axial":
            return arr[z, :, :]
        if view_name == "Coronal":
            return arr[:, y, :]
        if view_name == "Sagittal":
            return arr[:, :, x]
    elif arr.ndim == 4:
        t, v_i, r, a = indices
        dd = v["display_dims"]
        if dd == (2, 3):  # Radial-Axial
            return arr[t, v_i, :, :]
        if dd == (1, 3):  # View-Axial
            return arr[t, :, r, :]
        if dd == (1, 2):  # View-Radial
            return arr[t, :, :, a]
    # Generic fallback: fix all but last two dims using indices
    fixed = list(indices[: max(0, arr.ndim - 2)])
    return arr[tuple(fixed + [slice(None), slice(None)])]


def compute_aspect(
    voxel_sizes: Optional[Tuple[float, float, float]], view_name: str
) -> float | str:
    if voxel_sizes is None:
        return "equal"
    vz, vy, vx = voxel_sizes
    if view_name == "Axial":  # y-x
        return vy / vx
    if view_name == "Coronal":  # z-x
        return vz / vx
    if view_name == "Sagittal":  # z-y
        return vz / vy
    return "equal"


def title_for(state: "ViewerState") -> str:
    v = state.views[state.view]
    parts = [
        f"{state.view} View",
        f"{state.dim_names[v['scroll_dim']]}: {state.indices[v['scroll_dim']]}",
    ]
    parts.extend(
        f"{state.dim_names[d]}: {state.indices[d]}"
        for d in v.get("controllable_dims", [])
    )
    return " - ".join(parts)


def plot_slice(ax: plt.Axes, state: "ViewerState"):
    """Render current foreground slice on provided Axes. Returns (AxesImage, labels)."""
    arr = get_slice(state.data, state.indices, state.view, state.views)
    aspect = compute_aspect(
        state.voxel_sizes if state.data.ndim == 3 else None, state.view
    )
    imshow_kwargs = dict(cmap=state.colormap, origin="upper", aspect=aspect)
    if state.vmin is not None and state.vmax is not None:
        imshow_kwargs["vmin"] = state.vmin
        imshow_kwargs["vmax"] = state.vmax
    im = ax.imshow(arr, **imshow_kwargs)
    v = state.views[state.view]
    ax.set_xlabel(v["labels"][1])
    ax.set_ylabel(v["labels"][0])
    ax.set_title(title_for(state))
    return im, v["labels"]


def create_gif_core(
    state: "ViewerState",
    filename: str,
    fps: int = 10,
    animate_dim: Optional[int] = None,
):
    """Create an animated GIF by sweeping one dimension (default: view scroll_dim)."""
    view_def = state.views[state.view]
    dim = view_def["scroll_dim"] if animate_dim is None else animate_dim
    nframes = state.shape[dim]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Using a local copy of indices to avoid mutating external state during animation
    base_idx = list(state.indices)

    def _frame(i):
        ax.clear()
        idx = base_idx.copy()
        idx[dim] = i
        temp = state.copy_with(indices=idx)
        im, _ = plot_slice(ax, temp)
        return [im]

    anim = animation.FuncAnimation(
        fig, _frame, frames=nframes, interval=1000 // max(1, fps), blit=True
    )
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)


@dataclass
class ViewerState:
    """Holds data and viewing parameters; pure and UI-agnostic."""
    data: np.ndarray
    dim_names: List[str]
    views: Dict[str, Dict[str, Any]]
    view: str
    indices: List[int]
    colormap: str = "gray"
    voxel_sizes: Optional[Tuple[float, float, float]] = None  # (z, y, x) for 3D only
    # Single consistent intensity control:
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def copy_with(self, **updates) -> "ViewerState":
        fields = dict(
            data=self.data,
            dim_names=self.dim_names,
            views=self.views,
            view=self.view,
            indices=list(self.indices),
            colormap=self.colormap,
            voxel_sizes=self.voxel_sizes,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        fields.update(updates)
        return ViewerState(**fields)


def to_state(data_obj: Any, colormap: str = "gray") -> ViewerState:
    if not SIRF_AVAILABLE:
        raise ImportError("SIRF package is required. Install with: pip install sirf")
    try:
        arr = data_obj.asarray()
    except AttributeError as e:
        raise AttributeError("Data object must have asarray() method") from e

    shape = arr.shape
    names = infer_dimension_names(data_obj, shape)
    views = build_available_views(shape)
    view = next(iter(views.keys()))
    indices = [s // 2 for s in shape]

    vxs = None
    if hasattr(data_obj, "voxel_sizes"):
        with contextlib.suppress(Exception):
            vxs = data_obj.voxel_sizes()  # (z,y,x)

    return ViewerState(
        data=arr,
        dim_names=names,
        views=views,
        view=view,
        indices=indices,
        colormap=colormap,
        voxel_sizes=vxs,
    )

# ----------------------------- Matplotlib desktop -----------------------------

class SIRFViewer:
    """
    Matplotlib-based interactive viewer for SIRF ImageData / AcquisitionData.
    Uses the shared ViewerState and core helpers.

    Background overlay (3D ImageData only) via set_background(...).
    """

    def __init__(
        self,
        data: Any,
        title: str = "SIRF Viewer",
        background_image: Optional[Union[Any, np.ndarray]] = None,
        alpha: float = 0.5,
    ):
        self._data_obj = data
        self.title = title

        self.state = to_state(data)
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im = None
        self.colorbar = None
        self.sliders: List[Slider] = []
        self.buttons: List[Button] = []
        self._tb_min: Optional[TextBox] = None
        self._tb_max: Optional[TextBox] = None

        # Background overlay storage
        self._bg_arr: Optional[np.ndarray] = None
        self._bg_alpha: float = alpha  # alpha applied to FOREGROUND (top) image
        self._bg_cmap: str = "gray"
        self._bg_voxel_sizes: Optional[Tuple[float, float, float]] = None

        # Backward-compatible init: allow passing background_image directly
        if background_image is not None:
            try:
                self.set_background(
                    background_image, alpha=self._bg_alpha, cmap=self._bg_cmap
                )
            except Exception as e:  # keep constructor resilient
                warnings.warn(f"Failed to set background at init: {e}", RuntimeWarning)

    # ---- Public API ----

    def show(self):
        if self.fig is None or self.ax is None:
            self._setup_plot()
        plt.show()

    def get_available_views(self) -> List[str]:
        return list(self.state.views)

    def set_view(self, view_name: str):
        if view_name not in self.state.views:
            raise ValueError(
                f"View '{view_name}' not available. Options: {list(self.state.views)}"
            )
        self.state.view = view_name
        if self.fig is not None:
            self._clear_sliders()
            self._add_sliders()
            self._redraw()

    def set_colormap(self, cmap: str):
        self.state.colormap = cmap
        if self.fig is not None:
            self._redraw()

    # Backwards-compatible API: map window/level to display range
    def set_window(self, level: float, width: float):
        vmin = float(level - width / 2.0)
        vmax = float(level + width / 2.0)
        self.set_display_range(vmin, vmax)

    def set_display_range(self, vmin: Optional[float], vmax: Optional[float]):
        self.state.vmin = vmin
        self.state.vmax = vmax
        if self.fig is not None:
            self._redraw()

    def save_current_view(self, filename: str):
        if self.fig is None or self.ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
            # background first (if any)
            if (self._bg_arr is not None) and (self.state.data.ndim == 3):
                bg_slice = get_slice(
                    self._bg_arr, self.state.indices, self.state.view, self.state.views
                )
                aspect = compute_aspect(self.state.voxel_sizes, self.state.view)
                ax.imshow(bg_slice, cmap=self._bg_cmap, origin="upper", aspect=aspect)
            # foreground
            im, _ = plot_slice(ax, self.state)
            im.set_alpha(self._bg_alpha if self._bg_arr is not None else 1.0)
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            self.fig.savefig(filename, dpi=150, bbox_inches="tight")

    def create_gif(
        self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None
    ):
        dim = dimensions[0] if dimensions else None
        create_gif_core(self.state, filename, fps=fps, animate_dim=dim)

    # ---- Background API ----

    def set_background(
        self,
        bg_obj: Union[Any, np.ndarray],
        alpha: float = 0.5,
        cmap: str = "gray",
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ):
        """
        Set a 3D background image (e.g. µ-map) for ImageData overlays.
        """
        if self.state.data.ndim != 3:
            raise ValueError("Background overlay is only supported for 3D ImageData.")

        bg_vxs = None
        # Extract background array
        if isinstance(bg_obj, np.ndarray):
            bg_arr = bg_obj
        else:
            try:
                bg_arr = bg_obj.asarray()
            except AttributeError as e:
                raise TypeError(
                    "Background object must implement asarray() or be a numpy ndarray."
                ) from e
            if hasattr(bg_obj, "voxel_sizes"):
                with contextlib.suppress(Exception):
                    bg_vxs = bg_obj.voxel_sizes()

        if bg_arr.ndim != 3:
            raise ValueError(f"Background must be 3D; got {bg_arr.ndim}D.")
        if tuple(bg_arr.shape) != tuple(self.state.shape):
            raise ValueError(
                f"Shape mismatch: foreground {self.state.shape} vs background {bg_arr.shape}."
            )

        # Compare voxel sizes if available on both
        fg_vxs = self.state.voxel_sizes
        if (fg_vxs is not None) and (bg_vxs is not None):
            if not np.allclose(
                np.asarray(fg_vxs, float),
                np.asarray(bg_vxs, float),
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    f"Voxel size mismatch: foreground {fg_vxs} vs background {bg_vxs}."
                )
        elif (fg_vxs is not None) ^ (bg_vxs is not None):
            warnings.warn(
                "Voxel sizes available for only one image; cannot verify equality.",
                RuntimeWarning,
            )

        self._bg_arr = bg_arr
        self._bg_alpha = alpha
        self._bg_cmap = cmap
        self._bg_voxel_sizes = bg_vxs

        if self.fig is not None:
            self._redraw()

    def clear_background(self):
        """Remove the background overlay."""
        self._bg_arr = None
        self._bg_voxel_sizes = None
        if self.fig is not None:
            self._redraw()

    # ---- Internals ----

    def _set_window_icon(self):
        """Qt-only: set figure window + app icon; no-ops elsewhere."""
        try:
            from sirf_viewer.icons import app_icon  # returns PyQt5.QtGui.QIcon
            from PyQt5.QtWidgets import QApplication

            if "qt" not in str(plt.get_backend()).lower():
                return

            icon = app_icon()
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)

            mgr = self.fig.canvas.manager
            if hasattr(mgr, "window") and hasattr(mgr.window, "setWindowIcon"):
                mgr.window.setWindowIcon(icon)
        except Exception as e:
            warnings.warn(f"Could not set window icon: {e}", RuntimeWarning)

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self._set_window_icon()
        self.fig.suptitle(self.title)
        plt.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.9)
        self._redraw()
        self._add_sliders()
        self._add_buttons()

    def _redraw(self):
        assert self.ax is not None and self.fig is not None
        self.ax.clear()

        # 1) Background (behind), if present and 3D
        if (self._bg_arr is not None) and (self.state.data.ndim == 3):
            bg_slice = get_slice(
                self._bg_arr, self.state.indices, self.state.view, self.state.views
            )
            aspect = compute_aspect(self.state.voxel_sizes, self.state.view)
            self.ax.imshow(bg_slice, cmap=self._bg_cmap, origin="upper", aspect=aspect)

        # 2) Foreground (emission) on top
        self.im, _ = plot_slice(self.ax, self.state)
        self.im.set_alpha(self._bg_alpha if self._bg_arr is not None else 1.0)

        # Colourbar tied to foreground
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.colorbar.update_normal(self.im)

        self.fig.canvas.draw_idle()

    def _clear_sliders(self):
        for s in self.sliders:
            if hasattr(s, "ax"):
                s.ax.remove()
        self.sliders = []

    def _add_sliders(self):
        v = self.state.views[self.state.view]
        slider_dims = [v["scroll_dim"]] + v.get("controllable_dims", [])

        slider_height = 0.03
        slider_spacing = 0.04

        for i, d in enumerate(slider_dims):
            y = 0.30 - i * slider_spacing
            ax_s = plt.axes([0.15, y, 0.5, slider_height])
            sl = Slider(
                ax_s,
                self.state.dim_names[d],
                0,
                self.state.shape[d] - 1,
                valinit=self.state.indices[d],
                valstep=1,
                valfmt="%d",
            )

            def _cb(val, idx=d):
                self.state.indices[idx] = int(val)
                self._redraw()

            sl.on_changed(_cb)
            self.sliders.append(sl)

    def _add_buttons(self):
        button_width = 0.12
        button_height = 0.04
        button_spacing = 0.01
        button_row = 0.30
        button_col = 0.75

        # View buttons
        for i, vn in enumerate(self.get_available_views()):
            ax_b = plt.axes(
                [
                    button_col,
                    button_row - i * (button_height + button_spacing),
                    button_width,
                    button_height,
                ]
            )
            b = Button(ax_b, vn[:8])
            b.on_clicked(lambda _e, vname=vn: self.set_view(vname))
            self.buttons.append(b)

        start = (
            button_row
            - len(self.get_available_views()) * (button_height + button_spacing)
            - 0.02
        )

        # Colormap
        ax_cmap = plt.axes([button_col, start, button_width, button_height])
        btn_cmap = Button(ax_cmap, "Colormap")

        def _cycle_cmap(_e):
            try:
                i = COLORMAPS.index(self.state.colormap)
            except ValueError:
                i = 0
            self.set_colormap(COLORMAPS[(i + 1) % len(COLORMAPS)])

        btn_cmap.on_clicked(_cycle_cmap)
        self.buttons.append(btn_cmap)

        # Save View
        ax_save = plt.axes(
            [
                button_col,
                start - (button_height + button_spacing),
                button_width,
                button_height,
            ]
        )
        btn_save = Button(ax_save, "Save View")

        def _save(_e):
            fname = f"sirf_{self.state.view.lower()}_{'_'.join(map(str, self.state.indices))}.png"
            self.save_current_view(fname)
            print(f"Saved current view to {fname}")

        btn_save.on_clicked(_save)
        self.buttons.append(btn_save)

        # Auto Range (percentiles on current slice)
        ax_auto = plt.axes(
            [
                button_col,
                start - 2 * (button_height + button_spacing),
                button_width,
                button_height,
            ]
        )
        btn_auto = Button(ax_auto, "Auto Range")

        def _auto(_e):
            data = get_slice(
                self.state.data, self.state.indices, self.state.view, self.state.views
            )
            vmin = float(np.percentile(data, 2.0))
            vmax = float(np.percentile(data, 99.0))
            self.set_display_range(vmin, vmax)
            # update text boxes too
            if self._tb_min is not None:
                self._tb_min.set_val(f"{vmin:.6g}")
            if self._tb_max is not None:
                self._tb_max.set_val(f"{vmax:.6g}")
            print(f"Auto Range: [{vmin:.3g}, {vmax:.3g}]")

        btn_auto.on_clicked(_auto)
        self.buttons.append(btn_auto)

        # Manual Range TextBoxes
        # Min
        ax_min = plt.axes(
            [button_col, start - 3 * (button_height + button_spacing), button_width, button_height]
        )
        self._tb_min = TextBox(ax_min, "Min", initial="")
        # Max
        ax_max = plt.axes(
            [button_col, start - 4 * (button_height + button_spacing), button_width, button_height]
        )
        self._tb_max = TextBox(ax_max, "Max", initial="")

        def _parse_textbox_value(val: str) -> Optional[float]:
            s = val.strip()
            if s == "":
                return None
            return float(s)

        def _on_min_submit(text):
            try:
                vmin = _parse_textbox_value(text)
                vmax = self.state.vmax
                # If both None -> auto; if only vmin set, keep vmax as-is
                self.set_display_range(vmin, vmax)
            except Exception as e:
                print(f"Invalid Min: {e}")

        def _on_max_submit(text):
            try:
                vmax = _parse_textbox_value(text)
                vmin = self.state.vmin
                self.set_display_range(vmin, vmax)
            except Exception as e:
                print(f"Invalid Max: {e}")

        self._tb_min.on_submit(_on_min_submit)
        self._tb_max.on_submit(_on_max_submit)

        # Reset to auto
        ax_reset = plt.axes(
            [button_col, start - 5 * (button_height + button_spacing), button_width, button_height]
        )
        btn_reset = Button(ax_reset, "Reset Auto")

        def _reset(_e):
            self.set_display_range(None, None)
            if self._tb_min is not None:
                self._tb_min.set_val("")
            if self._tb_max is not None:
                self._tb_max.set_val("")
            print("Display range reset to automatic")

        btn_reset.on_clicked(_reset)
        self.buttons.append(btn_reset)


# ----------------------------- Jupyter notebook ------------------------------

class NotebookViewer:
    """ipywidgets-based viewer reusing the shared core and identical behaviour."""

    def __init__(self, data: Any, width: int = 800, height: int = 600):
        self._data_obj = data
        self.state = to_state(data)
        self.width = width
        self.height = height

        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output

            self.widgets_available = True
            self._widgets = widgets
            self._display = display
            self._clear_output = clear_output
        except ImportError:
            self.widgets_available = False
            warnings.warn(
                "ipywidgets not available. Install with: pip install ipywidgets",
                RuntimeWarning,
            )

        self.sliders: List[Any] = []
        self.output = None
        self.view_dropdown = None
        self.colormap_dropdown = None

        # Background overlay storage
        self._bg_arr: Optional[np.ndarray] = None
        self._bg_alpha: float = 0.5
        self._bg_cmap: str = "gray"
        self._bg_voxel_sizes: Optional[Tuple[float, float, float]] = None

        # Range controls
        self.range_min = None
        self.range_max = None
        self.reset_button = None

    def show(self):
        if not self.widgets_available:
            print("ipywidgets not available. Falling back to static matplotlib.")
            self._show_static()
            return
        self._create_interactive()

    def create_gif(
        self, filename: str, fps: int = 10, dimensions: Optional[List[int]] = None
    ):
        dim = dimensions[0] if dimensions else None
        create_gif_core(self.state, filename, fps=fps, animate_dim=dim)

    # ---- Background API ----

    def set_background(
        self,
        bg_obj: Union[Any, np.ndarray],
        alpha: float = 0.5,
        cmap: str = "gray",
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ):
        """
        Set a 3D background image (e.g. µ-map) for ImageData overlays. See SIRFViewer.set_background.
        """
        if self.state.data.ndim != 3:
            raise ValueError("Background overlay is only supported for 3D ImageData.")

        bg_vxs = None
        # Extract background array
        if isinstance(bg_obj, np.ndarray):
            bg_arr = bg_obj
        else:
            try:
                bg_arr = bg_obj.asarray()
            except AttributeError as e:
                raise TypeError(
                    "Background object must implement asarray() or be a numpy ndarray."
                ) from e
            if hasattr(bg_obj, "voxel_sizes"):
                with contextlib.suppress(Exception):
                    bg_vxs = bg_obj.voxel_sizes()

        if bg_arr.ndim != 3:
            raise ValueError(f"Background must be 3D; got {bg_arr.ndim}D.")
        if tuple(bg_arr.shape) != tuple(self.state.shape):
            raise ValueError(
                f"Shape mismatch: foreground {self.state.shape} vs background {bg_arr.shape}."
            )

        # Compare voxel sizes if available on both
        fg_vxs = self.state.voxel_sizes
        if (fg_vxs is not None) and (bg_vxs is not None):
            if not np.allclose(
                np.asarray(fg_vxs, float),
                np.asarray(bg_vxs, float),
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    f"Voxel size mismatch: foreground {fg_vxs} vs background {bg_vxs}."
                )
        elif (fg_vxs is not None) ^ (bg_vxs is not None):
            warnings.warn(
                "Voxel sizes available for only one image; cannot verify equality.",
                RuntimeWarning,
            )

        self._bg_arr = bg_arr
        self._bg_alpha = alpha
        self._bg_cmap = cmap
        self._bg_voxel_sizes = bg_vxs

        if getattr(self, "output", None) is not None:
            self._update_plot()

    def clear_background(self):
        """Remove the background overlay."""
        self._bg_arr = None
        self._bg_voxel_sizes = None
        if getattr(self, "output", None) is not None:
            self._update_plot()

    # Single consistent intensity control (public)
    def set_window(self, level: float, width: float):
        """Compatibility: maps window/level to display range."""
        vmin = float(level - width / 2.0)
        vmax = float(level + width / 2.0)
        self.set_display_range(vmin, vmax)

    def set_display_range(self, vmin: Optional[float], vmax: Optional[float]):
        self.state.vmin = vmin
        self.state.vmax = vmax
        if getattr(self, "output", None) is not None:
            # keep widget boxes in sync if present
            if self.range_min is not None:
                self.range_min.value = vmin
            if self.range_max is not None:
                self.range_max.value = vmax
            self._update_plot()

    # ---- internals ----

    def _show_static(self):
        plt.figure(figsize=(self.width / 100, self.height / 100))
        ax = plt.gca()

        # Background first (if any)
        if (self._bg_arr is not None) and (self.state.data.ndim == 3):
            bg_slice = get_slice(
                self._bg_arr, self.state.indices, self.state.view, self.state.views
            )
            aspect = compute_aspect(self.state.voxel_sizes, self.state.view)
            ax.imshow(bg_slice, cmap=self._bg_cmap, origin="upper", aspect=aspect)

        # Foreground
        im, _ = plot_slice(ax, self.state)
        im.set_alpha(self._bg_alpha if self._bg_arr is not None else 1.0)
        plt.colorbar(im, ax=ax)
        plt.show()

    def _create_interactive(self):
        w = self._widgets

        # View selector
        self.view_dropdown = w.Dropdown(
            options=list(self.state.views.keys()),
            value=self.state.view,
            description="View:",
        )

        def _on_view(change):
            self.state.view = change.new
            self._build_sliders()  # rebuild for new view dims
            self._rebuild_layout()
            self._update_plot()

        self.view_dropdown.observe(_on_view, names="value")

        # Initial sliders and layout
        self._build_sliders()

        # Colormap selector
        self.colormap_dropdown = w.Dropdown(
            options=COLORMAPS, value=self.state.colormap, description="Colormap:"
        )

        def _on_cmap(change):
            self.state.colormap = change.new
            self._update_plot()

        self.colormap_dropdown.observe(_on_cmap, names="value")

        # Range controls
        self.range_min = w.FloatText(
            value=self.state.vmin, description="Min:", placeholder="auto"
        )
        self.range_max = w.FloatText(
            value=self.state.vmax, description="Max:", placeholder="auto"
        )

        def _on_range_change(_):
            self.state.vmin = self.range_min.value
            self.state.vmax = self.range_max.value
            self._update_plot()

        self.range_min.observe(_on_range_change, names="value")
        self.range_max.observe(_on_range_change, names="value")

        self.reset_button = w.Button(description="Auto Range")
        def _on_reset(_):
            # compute from current slice percentiles
            data = get_slice(
                self.state.data, self.state.indices, self.state.view, self.state.views
            )
            vmin = float(np.percentile(data, 2.0))
            vmax = float(np.percentile(data, 98.0))
            self.range_min.value = vmin
            self.range_max.value = vmax
            # _on_range_change will trigger update

        self.reset_button.on_click(_on_reset)

        self.output = w.Output()
        self._rebuild_layout()
        self._update_plot()

    def _build_sliders(self):
        w = self._widgets
        v = self.state.views[self.state.view]
        slider_dims = [v["scroll_dim"]] + v.get("controllable_dims", [])
        self.sliders = []

        for d in slider_dims:
            s = w.IntSlider(
                value=self.state.indices[d],
                min=0,
                max=self.state.shape[d] - 1,
                step=1,
                description=f"{self.state.dim_names[d]}:",
                continuous_update=False,
            )

            def _cb(change, idx=d):
                self.state.indices[idx] = int(change.new)
                self._update_plot()

            s.observe(_cb, names="value")
            self.sliders.append(s)

    def _rebuild_layout(self):
        w = self._widgets
        controls = w.VBox(
            [
                self.view_dropdown,
                *self.sliders,
                self.colormap_dropdown,
                self.range_min,
                self.range_max,
                self.reset_button,
            ]
        )
        viewer = w.VBox([controls, self.output])
        self._clear_output(wait=True)
        self._display(viewer)

    def _update_plot(self):
        with self.output:
            self.output.clear_output(wait=True)
            fig = plt.figure(figsize=(self.width / 100, self.height / 100))
            ax = fig.gca()

            # Background first
            if (self._bg_arr is not None) and (self.state.data.ndim == 3):
                bg_slice = get_slice(
                    self._bg_arr, self.state.indices, self.state.view, self.state.views
                )
                aspect = compute_aspect(self.state.voxel_sizes, self.state.view)
                ax.imshow(bg_slice, cmap=self._bg_cmap, origin="upper", aspect=aspect)

            # Foreground
            im, _ = plot_slice(ax, self.state)
            im.set_alpha(self._bg_alpha if self._bg_arr is not None else 1.0)

            fig.colorbar(im, ax=ax)  # colourbar for foreground
            plt.show()
