"""
Main application window for Image Blur, Sharpening & Edge Detection Studio.

Integrates Header, ControlPanel, ImageArea, StatusBar, KernelEditorWindow,
and AnalysisPanel via a ttk.Notebook tabbed layout.
"""

import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable
import numpy as np

from app.gui.header import Header
from app.gui.status_bar import StatusBar
from app.gui.image_view import ImageView
from app.gui.image_area import ImageArea
from app.gui.filter_panel import FilterPanel
from app.gui.kernel_editor import KernelEditorWindow
from app.gui.analysis_panel import AnalysisPanel
from app.gui.kernel_info import KernelInfoPanel
from app.gui.filter_comparison import FilterComparisonPanel

from app.io.image_io import load_image, save_image
from app.filters.blur import apply_blur
from app.filters.sharpen import apply_sharpen, apply_unsharp_mask
from app.filters.edges import apply_edge
from app.core.convolution import convolve2d
from app.core.image_state import ImageState


def _parse_kernel_size(filter_name: str, default: int = 3) -> int:
    """
    Extract kernel size integer from a filter label string.

    Examples:
        "Box Blur (7x7)"      -> 7
        "Gaussian Blur (3x3)" -> 3
        "Box Blur"            -> default

    Args:
        filter_name (str): GUI filter label.
        default (int): Fallback size if parsing fails.

    Returns:
        int: Parsed kernel size.
    """
    import re
    match = re.search(r'\((\d+)x\d+\)', filter_name)
    if match:
        return int(match.group(1))
    return default


class MainWindow(tk.Tk):
    """
    Main application window implementing structured responsive grid layout
    with tabbed Notebook for image view and signal analysis.
    """

    def __init__(self):
        super().__init__()

        self.title("Image Blur, Sharpening & Edge Detection Studio")
        self.geometry("1280x780")
        self.minsize(1000, 620)

        # Image state manager (float64 arrays)
        self.image_state = ImageState()

        # Kernel Editor Window instance reference
        self.kernel_editor_win: Optional[KernelEditorWindow] = None

        self._configure_styles()
        self._build_layout()

    @property
    def original_image(self) -> Optional[np.ndarray]:
        """Return the original loaded image array."""
        return self.image_state.original_image

    @original_image.setter
    def original_image(self, val: Optional[np.ndarray]) -> None:
        if val is None:
            self.image_state.clear()
        else:
            self.image_state.set_image(val)

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Return the current working/processed image array."""
        return self.image_state.current_image

    @current_image.setter
    def current_image(self, val: Optional[np.ndarray]) -> None:
        if val is None:
            self.image_state.clear()
        else:
            self.image_state.update_current(val)

    @property
    def processed_image(self) -> Optional[np.ndarray]:
        """Alias for current_image for backward compatibility."""
        return self.image_state.current_image

    @processed_image.setter
    def processed_image(self, val: Optional[np.ndarray]) -> None:
        if val is None:
            self.image_state.clear()
        else:
            self.image_state.update_current(val)

        self._configure_styles()
        self._build_layout()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _configure_styles(self) -> None:
        """Configure ttk themes and default styling aesthetics."""
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

    # ------------------------------------------------------------------
    # Layout Construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Construct responsive grid layout with Notebook content area."""
        self.columnconfigure(0, weight=0, minsize=260)  # ControlPanel
        self.columnconfigure(1, weight=1)               # Right content
        self.rowconfigure(0, weight=0)                  # Header
        self.rowconfigure(1, weight=1)                  # Content (Notebook)
        self.rowconfigure(2, weight=0)                  # StatusBar

        # 1. Header
        self.header = Header(self)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")

        # 2. ControlPanel
        self.control_panel = FilterPanel(
            self,
            on_load=self.handle_load_image,
            on_apply=self.handle_apply_filter,
            on_reset=self.handle_reset,
            on_save=self.handle_save_image,
            on_open_editor=self.handle_open_kernel_editor,
        )
        self.control_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)

        # 3. Notebook (right-side content area)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)

        # 3a. Images Tab
        self._build_images_tab()

        # 3b. Signal Analysis Tab
        self._build_analysis_tab()

        # 3c. Educational Kernel Info Tab
        self._build_kernel_info_tab()

        # 3d. Multi-Filter Comparison Tab
        self._build_filter_comparison_tab()

        # 4. StatusBar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_images_tab(self) -> None:
        """Build the dual image view & comparison tab inside the Notebook."""
        images_tab = ttk.Frame(self.notebook)
        self.notebook.add(images_tab, text="  🖼  Images & Comparison  ")

        images_tab.columnconfigure(0, weight=1)
        images_tab.rowconfigure(0, weight=1)

        self.image_area = ImageArea(images_tab)
        self.image_area.grid(row=0, column=0, sticky="nsew")

    @property
    def original_view(self) -> ImageView:
        """Alias for backward compatibility with original_view property."""
        return self.image_area.original_frame

    @property
    def processed_view(self) -> ImageView:
        """Alias for backward compatibility with processed_view property."""
        return self.image_area.processed_frame

    def _build_analysis_tab(self) -> None:
        """Build the Signal Analysis panel tab inside the Notebook."""
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="  📊  Signal Analysis  ")

        analysis_tab.columnconfigure(0, weight=1)
        analysis_tab.rowconfigure(0, weight=1)

        self.analysis_panel = AnalysisPanel(analysis_tab)
        self.analysis_panel.grid(row=0, column=0, sticky="nsew")

    def _build_kernel_info_tab(self) -> None:
        """Build the Educational Kernel Information tab inside the Notebook."""
        info_tab = ttk.Frame(self.notebook)
        self.notebook.add(info_tab, text="  📚  Educational Kernel Info  ")

        info_tab.columnconfigure(0, weight=1)
        info_tab.rowconfigure(0, weight=1)

        self.kernel_info_panel = KernelInfoPanel(info_tab)
        self.kernel_info_panel.grid(row=0, column=0, sticky="nsew")

    def _build_filter_comparison_tab(self) -> None:
        """Build the Multi-Filter Comparison tab inside the Notebook."""
        comp_tab = ttk.Frame(self.notebook)
        self.notebook.add(comp_tab, text="  ⚔  Filter Comparison  ")

        comp_tab.columnconfigure(0, weight=1)
        comp_tab.rowconfigure(0, weight=1)

        self.filter_comp_panel = FilterComparisonPanel(comp_tab)
        self.filter_comp_panel.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------

    # Event Handlers
    # ------------------------------------------------------------------

    def handle_open_kernel_editor(self) -> None:
        """Open or focus interactive Kernel Editor window."""
        if self.kernel_editor_win is not None and self.kernel_editor_win.winfo_exists():
            self.kernel_editor_win.focus()
            return

        self.kernel_editor_win = KernelEditorWindow(
            self,
            image_array=self.current_image,
            on_apply_kernel=self.handle_apply_custom_kernel
        )

    def handle_load_image(self) -> None:
        """Open file dialog, load selected image, and update UI views & status."""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Supported Images", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("BMP Files", "*.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            t0 = time.perf_counter()
            image_array = load_image(file_path)
            elapsed = time.perf_counter() - t0

            self.image_state.set_image(image_array)

            self.image_area.set_images(self.original_image, self.current_image)
            self.filter_comp_panel.set_image(self.original_image)

            # Update analysis with initial (identical) original & processed
            self.analysis_panel.update_analysis(self.original_image, self.current_image)

            filename = Path(file_path).name
            h, w = image_array.shape[:2]
            mode_str = "Grayscale" if image_array.ndim == 2 else "RGB"
            self.status_bar.set_status(f"Loaded '{filename}' in {elapsed:.3f}s")
            self.status_bar.set_info(f"Dimensions: {w} × {h} | Mode: {mode_str}")

        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Failed to load image:\n{e}")
            self.status_bar.set_status("Error loading image file.")

    def handle_apply_filter(self, filter_name: str) -> None:
        """Apply selected filter algorithm to current image array and update status."""
        if self.current_image is None:
            messagebox.showwarning("No Image Loaded", "Please load an image before applying a filter.")
            return

        if filter_name == "Custom Kernel (Editor)":
            self.handle_open_kernel_editor()
            return

        # Update educational kernel info tab
        self.kernel_info_panel.display_kernel_info(filter_name)

        self.status_bar.set_status(f"Applying '{filter_name}'...")
        self.update_idletasks()

        input_image = self.current_image

        def compute() -> tuple[np.ndarray, str]:
            desc_str = filter_name
            iterations = self.control_panel.get_iterations()
            fn_lower = filter_name.lower()

            if fn_lower.startswith("box blur"):
                size = _parse_kernel_size(filter_name, default=3)
                result = apply_blur(
                    input_image,
                    blur_type='box',
                    size=size,
                    padding_mode='edge',
                    iterations=iterations,
                )
            elif fn_lower.startswith("gaussian blur"):
                size = _parse_kernel_size(filter_name, default=3)
                result = apply_blur(
                    input_image,
                    blur_type='gaussian',
                    size=size,
                    padding_mode='edge',
                    iterations=iterations,
                )
            elif filter_name == "Basic Sharpen":
                result = apply_sharpen(
                    input_image,
                    method='basic',
                    padding_mode='edge',
                )
                for _ in range(iterations - 1):
                    result = apply_sharpen(result, method='basic', padding_mode='edge')
            elif filter_name == "Strong Sharpen":
                result = apply_sharpen(
                    input_image,
                    method='strong',
                    padding_mode='edge',
                )
                for _ in range(iterations - 1):
                    result = apply_sharpen(result, method='strong', padding_mode='edge')
            elif filter_name.startswith("Unsharp Mask"):
                import re as _re
                r_match = _re.search(r'r=([0-9]+)', filter_name)
                a_match = _re.search(r'a=([0-9.]+)', filter_name)
                usm_radius = int(r_match.group(1)) if r_match else 5
                usm_amount = float(a_match.group(1)) if a_match else 1.5
                usm_amount_total = usm_amount * iterations
                result = apply_unsharp_mask(
                    input_image,
                    radius=usm_radius,
                    amount=usm_amount_total,
                    padding_mode='edge',
                )
                desc_str = f"Unsharp Mask (r={usm_radius}, a={usm_amount_total:.1f})"
            elif filter_name == "Edge Detection" or "Sobel" in filter_name or "Prewitt" in filter_name \
                    or "Roberts" in filter_name or "Laplacian" in filter_name:
                op = self.control_panel.get_edge_operator()
                direction = self.control_panel.get_edge_direction()
                normalize = self.control_panel.get_edge_normalize()

                result = apply_edge(
                    input_image,
                    operator=op,
                    direction=direction,
                    padding_mode='edge'
                )

                if normalize:
                    if direction in ("horizontal", "vertical") or op == "laplacian":
                        result = np.abs(result)
                    max_val = np.max(result)
                    if max_val > 0:
                        result = (result / max_val) * 255.0

                base_desc = f"{op.capitalize()} Edge ({direction.capitalize()})"
                if input_image.ndim == 3:
                    desc_str = f"{base_desc} [RGB→Grayscale luma]"
                else:
                    desc_str = base_desc
            else:
                raise ValueError(f"Unknown filter option '{filter_name}'.")

            return result, desc_str

        self._execute_filter_computation(compute)

    def handle_apply_custom_kernel(self, kernel_matrix: np.ndarray) -> None:
        """Apply custom user-designed matrix kernel using manual convolve2d engine."""
        if self.current_image is None:
            messagebox.showwarning("No Image Loaded", "Please load an image before applying custom kernel.")
            return

        k_h, k_w = kernel_matrix.shape
        self.kernel_info_panel.display_kernel_info(f"Custom Kernel ({k_h}x{k_w})", custom_matrix=kernel_matrix)

        input_image = self.current_image
        is_rgb = (input_image.ndim == 3 and input_image.shape[2] == 3)
        mode_desc = "RGB (per-channel)" if is_rgb else "Grayscale"
        self.status_bar.set_status(f"Applying custom {k_h}x{k_w} kernel to {mode_desc} image...")
        self.update_idletasks()

        def compute() -> tuple[np.ndarray, str]:
            res = convolve2d(input_image, kernel_matrix, padding_mode='edge')
            desc = f"Custom {k_h}x{k_w} Kernel [RGB per-channel]" if is_rgb else f"Custom {k_h}x{k_w} Kernel"
            return res, desc

        self._execute_filter_computation(compute)

    def _execute_filter_computation(self, compute_fn: Callable[[], tuple[np.ndarray, str]]) -> None:
        """Execute filter computation in worker thread to prevent Tkinter GUI freeze."""
        t0 = time.perf_counter()

        def worker():
            try:
                result, desc_str = compute_fn()
                elapsed = time.perf_counter() - t0
                self.after(0, self._on_filter_success, result, desc_str, elapsed)
            except Exception as err:
                self.after(0, self._on_filter_error, str(err))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_filter_success(self, result: np.ndarray, desc_str: str, elapsed: float) -> None:
        """Main thread callback executed after background convolution completes."""
        self.image_state.update_current(result, description=desc_str)
        self.image_area.set_images(self.original_image, self.current_image)
        self.analysis_panel.update_analysis(self.original_image, self.current_image)

        step_count = len(self.image_state.history)
        if step_count > 1:
            self.status_bar.set_status(f"Applied {desc_str} in {elapsed:.3f}s (Step {step_count} in chain)")
        else:
            self.status_bar.set_status(f"Applied {desc_str} in {elapsed:.3f}s")

        # Update info bar: show original mode + current shape
        orig_mode = "RGB" if (self.original_image is not None and self.original_image.ndim == 3) else "Grayscale"
        curr_mode = "RGB" if result.ndim == 3 else "Grayscale"
        h, w = result.shape[:2]
        if orig_mode == curr_mode:
            if curr_mode == "RGB" and "Custom" in desc_str:
                self.status_bar.set_info(f"Image: RGB | Size: {w} × {h} | Per-Channel (R, G, B) Convolved")
            else:
                self.status_bar.set_info(f"Image: {curr_mode} | Size: {w} × {h}")
        else:
            self.status_bar.set_info(f"Input: {orig_mode} → Current: {curr_mode} | Size: {w} × {h}")

    def _on_filter_error(self, err_msg: str) -> None:
        """Main thread callback executed if background convolution fails."""
        messagebox.showerror("Filter Error", f"An error occurred while applying filter:\n{err_msg}")
        self.status_bar.set_status("Error applying filter.")


    def handle_reset(self) -> None:
        """Reset processed result back to original loaded image."""
        if self.original_image is None:
            return

        self.image_state.reset()
        self.image_area.set_images(self.original_image, self.processed_image)

        # Refresh analysis panel after reset (orig == processed again)
        self.analysis_panel.update_analysis(self.original_image, self.processed_image)

        self.status_bar.set_status("Reset processed image to original.")

    def handle_save_image(self) -> None:
        """Open save dialog and write processed image array to disk."""
        if self.processed_image is None:
            messagebox.showwarning("No Result to Save", "No processed image is available to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp")
            ]
        )

        if not file_path:
            return

        try:
            save_image(file_path, self.processed_image)
            filename = Path(file_path).name
            messagebox.showinfo("Success", f"Image saved successfully as '{filename}'!")
            self.status_bar.set_status(f"Saved result image to '{filename}'")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
            self.status_bar.set_status("Error saving image.")
