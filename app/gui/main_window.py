"""
Main application window for Image Blur, Sharpening & Edge Detection Studio.

Integrates Header, ControlPanel, ImageArea, StatusBar, KernelEditorWindow,
and AnalysisPanel via a ttk.Notebook tabbed layout.
"""

import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import numpy as np

from app.gui.header import Header
from app.gui.status_bar import StatusBar
from app.gui.image_view import ImageView
from app.gui.filter_panel import FilterPanel
from app.gui.kernel_editor import KernelEditorWindow
from app.gui.analysis_panel import AnalysisPanel
from app.io.image_io import load_image, save_image
from app.filters.blur import apply_blur
from app.filters.sharpen import apply_sharpen, apply_unsharp_mask
from app.filters.edges import apply_edge
from src.core.convolution import convolve2d


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

        # Image state arrays (float64)
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None

        # Kernel Editor Window instance reference
        self.kernel_editor_win: Optional[KernelEditorWindow] = None

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

        # 4. StatusBar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_images_tab(self) -> None:
        """Build the dual image view tab inside the Notebook."""
        images_tab = ttk.Frame(self.notebook)
        self.notebook.add(images_tab, text="  🖼  Images  ")

        images_tab.columnconfigure(0, weight=1)  # Original Frame
        images_tab.columnconfigure(1, weight=1)  # Processed Frame
        images_tab.rowconfigure(0, weight=1)

        # Original View
        self.original_view = ImageView(images_tab, title="Original Image")
        self.original_view.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        # Processed View
        self.processed_view = ImageView(images_tab, title="Processed Result")
        self.processed_view.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

    def _build_analysis_tab(self) -> None:
        """Build the Signal Analysis panel tab inside the Notebook."""
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="  📊  Signal Analysis  ")

        analysis_tab.columnconfigure(0, weight=1)
        analysis_tab.rowconfigure(0, weight=1)

        self.analysis_panel = AnalysisPanel(analysis_tab)
        self.analysis_panel.grid(row=0, column=0, sticky="nsew")

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
            image_array=self.original_image,
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
            image_array = load_image(file_path, as_grayscale=True)
            elapsed = time.perf_counter() - t0

            self.original_image = image_array
            self.processed_image = image_array.copy()

            self.original_view.set_image(self.original_image)
            self.processed_view.set_image(self.processed_image)

            # Update analysis with initial (identical) original & processed
            self.analysis_panel.update_analysis(self.original_image, self.processed_image)

            filename = Path(file_path).name
            h, w = image_array.shape[:2]
            self.status_bar.set_status(f"Loaded '{filename}' in {elapsed:.3f}s")
            self.status_bar.set_info(f"Dimensions: {w} × {h} | Mode: Grayscale")

        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Failed to load image:\n{e}")
            self.status_bar.set_status("Error loading image file.")

    def handle_apply_filter(self, filter_name: str) -> None:
        """Apply selected filter algorithm to original image array and update status."""
        if self.original_image is None:
            messagebox.showwarning("No Image Loaded", "Please load an image before applying a filter.")
            return

        try:
            if filter_name == "Custom Kernel (Editor)":
                self.handle_open_kernel_editor()
                return

            self.status_bar.set_status(f"Applying '{filter_name}'...")
            self.update_idletasks()

            t0 = time.perf_counter()
            desc_str = filter_name
            iterations = self.control_panel.get_iterations()

            fn_lower = filter_name.lower()

            if fn_lower.startswith("box blur"):
                # Parse size from label e.g. "Box Blur (7x7)" → 7
                size = _parse_kernel_size(filter_name, default=3)
                result = apply_blur(
                    self.original_image,
                    blur_type='box',
                    size=size,
                    padding_mode='edge',
                    iterations=iterations,
                )
            elif fn_lower.startswith("gaussian blur"):
                size = _parse_kernel_size(filter_name, default=3)
                result = apply_blur(
                    self.original_image,
                    blur_type='gaussian',
                    size=size,
                    padding_mode='edge',
                    iterations=iterations,
                )
            elif filter_name == "Basic Sharpen":
                result = apply_sharpen(
                    self.original_image,
                    method='basic',
                    padding_mode='edge',
                )
                # Apply iterations manually for sharpen
                for _ in range(iterations - 1):
                    result = apply_sharpen(result, method='basic', padding_mode='edge')
            elif filter_name == "Strong Sharpen":
                result = apply_sharpen(
                    self.original_image,
                    method='strong',
                    padding_mode='edge',
                )
                for _ in range(iterations - 1):
                    result = apply_sharpen(result, method='strong', padding_mode='edge')
            elif filter_name.startswith("Unsharp Mask"):
                # Parse preset parameters from label, e.g. "Unsharp Mask — Medium (r=7, a=1.5)"
                import re as _re
                r_match = _re.search(r'r=([0-9]+)', filter_name)
                a_match = _re.search(r'a=([0-9.]+)', filter_name)
                usm_radius = int(r_match.group(1)) if r_match else 5
                usm_amount = float(a_match.group(1)) if a_match else 1.5
                # Scale amount by iterations for progressive strengthening
                usm_amount_total = usm_amount * iterations
                result = apply_unsharp_mask(
                    self.original_image,
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
                    self.original_image,
                    operator=op,
                    direction=direction,
                    padding_mode='edge'
                )

                if normalize:
                    # Normalize signed gradients (Gx, Gy, Laplacian) for visual clarity
                    if direction in ("horizontal", "vertical") or op == "laplacian":
                        result = np.abs(result)
                    max_val = np.max(result)
                    if max_val > 0:
                        result = (result / max_val) * 255.0

                desc_str = f"{op.capitalize()} Edge ({direction.capitalize()})"
            else:
                raise ValueError(f"Unknown filter option '{filter_name}'.")

            elapsed = time.perf_counter() - t0

            self.processed_image = result
            self.processed_view.set_image(self.processed_image)

            # Refresh analysis panel with updated arrays
            self.analysis_panel.update_analysis(self.original_image, self.processed_image)

            self.status_bar.set_status(f"Applied {desc_str} in {elapsed:.3f}s")

        except Exception as e:
            messagebox.showerror("Filter Error", f"An error occurred while applying filter:\n{e}")
            self.status_bar.set_status("Error applying filter.")

    def handle_apply_custom_kernel(self, kernel_matrix: np.ndarray) -> None:
        """Apply custom user-designed matrix kernel using manual convolve2d engine."""
        if self.original_image is None:
            messagebox.showwarning("No Image Loaded", "Please load an image before applying custom kernel.")
            return

        try:
            k_h, k_w = kernel_matrix.shape
            self.status_bar.set_status(f"Applying custom {k_h}x{k_w} kernel...")
            self.update_idletasks()

            t0 = time.perf_counter()
            result = convolve2d(self.original_image, kernel_matrix, padding_mode='edge')
            elapsed = time.perf_counter() - t0

            self.processed_image = result
            self.processed_view.set_image(self.processed_image)

            # Refresh analysis panel
            self.analysis_panel.update_analysis(self.original_image, self.processed_image)

            self.status_bar.set_status(f"Applied custom {k_h}x{k_w} kernel in {elapsed:.3f}s")

        except Exception as e:
            messagebox.showerror("Convolution Error", f"Failed to apply custom kernel:\n{e}")
            self.status_bar.set_status("Error applying custom kernel.")

    def handle_reset(self) -> None:
        """Reset processed result back to original loaded image."""
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()
        self.processed_view.set_image(self.processed_image)

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
