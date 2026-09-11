"""
Interactive Kernel Editor component for custom 2D discrete convolution filter design.

Supports 3x3 and 5x5 odd-sized matrices, preset loading, live metadata calculation,
matrix entry validation, debounced live thumbnail preview, and custom kernel dispatch 
to the manual convolution engine.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Dict
import numpy as np
from PIL import Image

from app.gui.image_view import ImageView
from src.core import kernels
from src.core.convolution import convolve2d


class KernelEditorWindow(tk.Toplevel):
    """
    Toplevel window dialog providing an interactive matrix kernel editor with live preview.
    """

    PRESETS: Dict[str, np.ndarray] = {
        "Identity 3x3": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
        "Box Blur 3x3": kernels.BOX_BLUR_3X3,
        "Gaussian Blur 3x3": kernels.GAUSSIAN_BLUR_3X3,
        "Basic Sharpen": kernels.SHARPEN_BASIC,
        "Strong Sharpen": kernels.SHARPEN_STRONG,
        "Laplacian Sharpen": kernels.SHARPEN_LAPLACIAN,
        "Sobel Horizontal": kernels.SOBEL_HORIZONTAL,
        "Sobel Vertical": kernels.SOBEL_VERTICAL,
        "Prewitt Horizontal": kernels.PREWITT_HORIZONTAL,
        "Prewitt Vertical": kernels.PREWITT_VERTICAL,
        "Roberts X (3x3)": kernels.ROBERTS_X,
        "Roberts Y (3x3)": kernels.ROBERTS_Y,
        "Edge Laplacian": kernels.EDGE_LAPLACIAN,
        "Box Blur 5x5": kernels.BOX_BLUR_5X5,
        "Gaussian Blur 5x5": kernels.GAUSSIAN_BLUR_5X5,
        "Identity 5x5": np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)
    }

    def __init__(
        self, 
        parent: tk.Widget, 
        image_array: Optional[np.ndarray] = None,
        on_apply_kernel: Optional[Callable[[np.ndarray], None]] = None
    ):
        super().__init__(parent)

        self.title("Interactive Kernel Editor (Signals Lab)")
        self.geometry("780x620")
        self.minsize(680, 520)

        self.on_apply_kernel_cb = on_apply_kernel
        self._debounce_timer_id: Optional[str] = None

        # Prepare downsampled thumbnail array for live preview
        self._preview_thumbnail: Optional[np.ndarray] = None
        self._prepare_preview_thumbnail(image_array)

        # Kernel Grid Matrix State
        self.current_size: int = 3
        self.entries: list[list[ttk.Entry]] = []
        self.string_vars: list[list[tk.StringVar]] = []

        # Window Grid Configuration: Left Controls (Col 0) | Right Live Preview (Col 1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_ui()
        self.load_preset("Box Blur 3x3")

    def _prepare_preview_thumbnail(self, image_array: Optional[np.ndarray]) -> None:
        """Create a lightweight downsampled thumbnail copy for responsive live preview."""
        if image_array is None:
            self._preview_thumbnail = None
            return

        arr = np.clip(image_array, 0.0, 255.0).astype(np.uint8)
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode='L')
        else:
            pil_img = Image.fromarray(arr)

        # Thumbnail to max 180x180 pixels
        pil_img.thumbnail((180, 180), Image.Resampling.LANCZOS)
        self._preview_thumbnail = np.asarray(pil_img, dtype=np.float64)

    def _build_ui(self) -> None:
        """Construct Kernel Editor split UI widgets using grid()."""
        # --- LEFT SIDE CONTROLLER FRAME ---
        left_frame = ttk.Frame(self, padding=5)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=0)  # Preset / Size
        left_frame.rowconfigure(1, weight=1)  # Matrix Grid
        left_frame.rowconfigure(2, weight=0)  # Metadata
        left_frame.rowconfigure(3, weight=0)  # Action Buttons

        # 1. Preset & Size Controls
        ctrl_frame = ttk.LabelFrame(left_frame, text="Kernel Preset & Dimensions", padding=8)
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ctrl_frame.columnconfigure(1, weight=1)

        ttk.Label(ctrl_frame, text="Size:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.size_var = tk.StringVar(value="3x3")
        size_combo = ttk.Combobox(
            ctrl_frame, 
            textvariable=self.size_var, 
            values=["3x3", "5x5"], 
            width=6, 
            state="readonly"
        )
        size_combo.grid(row=0, column=1, sticky="w", padx=5)
        size_combo.bind("<<ComboboxSelected>>", self._on_size_change)

        ttk.Label(ctrl_frame, text="Preset:").grid(row=0, column=2, sticky="w", padx=(10, 5))
        self.preset_var = tk.StringVar(value="Box Blur 3x3")
        preset_combo = ttk.Combobox(
            ctrl_frame, 
            textvariable=self.preset_var, 
            values=list(self.PRESETS.keys()), 
            state="readonly"
        )
        preset_combo.grid(row=0, column=3, sticky="ew", padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_select)

        # 2. Matrix Entry Grid Frame
        self.grid_frame = ttk.LabelFrame(left_frame, text="Kernel Coefficients Matrix", padding=10)
        self.grid_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        # 3. Metadata Display Panel
        self.meta_frame = ttk.LabelFrame(left_frame, text="Kernel Signal Statistics", padding=8)
        self.meta_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.meta_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.lbl_meta_size = ttk.Label(self.meta_frame, text="Size: 3 × 3", font=("Segoe UI", 9, "bold"))
        self.lbl_meta_size.grid(row=0, column=0, sticky="w")

        self.lbl_meta_sum = ttk.Label(self.meta_frame, text="Sum: 0.0000", font=("Segoe UI", 9))
        self.lbl_meta_sum.grid(row=0, column=1, sticky="w")

        self.lbl_meta_min = ttk.Label(self.meta_frame, text="Min: 0.0000", font=("Segoe UI", 9))
        self.lbl_meta_min.grid(row=0, column=2, sticky="w")

        self.lbl_meta_max = ttk.Label(self.meta_frame, text="Max: 0.0000", font=("Segoe UI", 9))
        self.lbl_meta_max.grid(row=0, column=3, sticky="w")

        # 4. Action Buttons Bar
        action_frame = ttk.Frame(left_frame, padding=(0, 5, 0, 5))
        action_frame.grid(row=3, column=0, sticky="ew")
        action_frame.columnconfigure((0, 1, 2), weight=1)

        btn_reset = ttk.Button(action_frame, text="↺ Reset Zeros", command=self.reset_to_zeros)
        btn_reset.grid(row=0, column=0, sticky="ew", padx=2)

        btn_norm = ttk.Button(action_frame, text="⚖ Normalize Sum", command=self.normalize_kernel_sum)
        btn_norm.grid(row=0, column=1, sticky="ew", padx=2)

        btn_apply = ttk.Button(action_frame, text="⚡ Apply Custom Kernel", command=self._on_apply_click)
        btn_apply.grid(row=0, column=2, sticky="ew", padx=2)

        # --- RIGHT SIDE LIVE PREVIEW FRAME ---
        right_frame = ttk.Frame(self, padding=5)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.preview_view = ImageView(
            right_frame, 
            title="Live Kernel Preview (Thumbnail)"
        )
        self.preview_view.grid(row=0, column=0, sticky="nsew")

        if self._preview_thumbnail is not None:
            self.preview_view.set_image(self._preview_thumbnail)

    def _rebuild_matrix_grid(self, size: int) -> None:
        """Dynamically construct N x N Entry widgets for matrix values."""
        self.current_size = size

        for child in self.grid_frame.winfo_children():
            child.destroy()

        self.entries = []
        self.string_vars = []

        for r in range(size):
            self.grid_frame.rowconfigure(r, weight=1)
            self.grid_frame.columnconfigure(r, weight=1)

            row_entries = []
            row_vars = []

            for c in range(size):
                var = tk.StringVar(value="0")
                var.trace_add("write", self._on_entry_value_changed)

                entry = ttk.Entry(
                    self.grid_frame, 
                    textvariable=var, 
                    justify="center", 
                    width=7
                )
                entry.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)

                row_entries.append(entry)
                row_vars.append(var)

            self.entries.append(row_entries)
            self.string_vars.append(row_vars)

        self.lbl_meta_size.config(text=f"Size: {size} × {size}")
        self.update_metadata()
        self._schedule_live_preview()

    def set_kernel_array(self, kernel: np.ndarray) -> None:
        """Set matrix entry contents from a NumPy 2D array."""
        h, w = kernel.shape
        if h != w or h not in (3, 5):
            raise ValueError(f"Kernel must be 3x3 or 5x5 square array, got {kernel.shape}")

        size_str = f"{h}x{h}"
        if self.size_var.get() != size_str:
            self.size_var.set(size_str)

        self._rebuild_matrix_grid(h)

        for r in range(h):
            for c in range(w):
                val = kernel[r, c]
                if float(val).is_integer():
                    formatted = str(int(val))
                else:
                    formatted = f"{val:.6g}"
                self.string_vars[r][c].set(formatted)

        self.update_metadata()
        self._schedule_live_preview()

    def get_kernel_array(self) -> np.ndarray:
        """
        Validate entry values and convert matrix to a 2D float64 NumPy array.

        Raises:
            ValueError: If any matrix entry is not a valid number.
        """
        size = self.current_size
        matrix = np.zeros((size, size), dtype=np.float64)

        for r in range(size):
            for c in range(size):
                text = self.string_vars[r][c].get().strip()
                if not text:
                    raise ValueError(f"Matrix cell at Row {r+1}, Col {c+1} is empty.")
                try:
                    val = float(text)
                    matrix[r, c] = val
                except ValueError as e:
                    raise ValueError(
                        f"Invalid numerical value '{text}' at Row {r+1}, Col {c+1}."
                    ) from e

        return matrix

    def update_metadata(self) -> None:
        """Calculate and display live metadata statistics (Sum, Min, Max)."""
        try:
            arr = self.get_kernel_array()
            k_sum = np.sum(arr)
            k_min = np.min(arr)
            k_max = np.max(arr)

            self.lbl_meta_sum.config(text=f"Sum: {k_sum:.4f}")
            self.lbl_meta_min.config(text=f"Min: {k_min:.4f}")
            self.lbl_meta_max.config(text=f"Max: {k_max:.4f}")
        except ValueError:
            self.lbl_meta_sum.config(text="Sum: Error")
            self.lbl_meta_min.config(text="Min: Error")
            self.lbl_meta_max.config(text="Max: Error")

    def _on_entry_value_changed(self, *args) -> None:
        """Debounced listener invoked when user edits a matrix cell."""
        self.update_metadata()
        self._schedule_live_preview()

    def _schedule_live_preview(self) -> None:
        """Schedule a debounced live preview calculation using after()."""
        if self._debounce_timer_id is not None:
            self.after_cancel(self._debounce_timer_id)
        self._debounce_timer_id = self.after(300, self._trigger_live_preview)

    def _trigger_live_preview(self) -> None:
        """Execute manual convolve2d on the preview thumbnail array."""
        self._debounce_timer_id = None
        if self._preview_thumbnail is None:
            return

        try:
            kernel = self.get_kernel_array()
            preview_res = convolve2d(self._preview_thumbnail, kernel, padding_mode='edge')
            self.preview_view.set_image(preview_res)
        except Exception:
            # Paused or invalid mid-keystroke entry - silently skip preview update
            pass

    def load_preset(self, preset_name: str) -> None:
        """Load predefined kernel matrix into grid."""
        if preset_name in self.PRESETS:
            self.preset_var.set(preset_name)
            self.set_kernel_array(self.PRESETS[preset_name])

    def reset_to_zeros(self) -> None:
        """Reset all grid matrix entries to zero."""
        size = self.current_size
        zeros = np.zeros((size, size), dtype=np.float64)
        self.set_kernel_array(zeros)

    def normalize_kernel_sum(self) -> None:
        """Scale kernel matrix elements so they sum to 1.0."""
        try:
            arr = self.get_kernel_array()
            k_sum = np.sum(arr)
            if abs(k_sum) < 1e-9:
                messagebox.showwarning("Normalization Warning", "Kernel sum is zero; cannot normalize.")
                return
            normalized = arr / k_sum
            self.set_kernel_array(normalized)
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))

    def _on_size_change(self, event: tk.Event) -> None:
        """Handle kernel size combobox change."""
        size_str = self.size_var.get()
        size = 3 if size_str == "3x3" else 5
        if size == 3:
            self.load_preset("Box Blur 3x3")
        else:
            self.load_preset("Box Blur 5x5")

    def _on_preset_select(self, event: tk.Event) -> None:
        """Handle preset selection combobox change."""
        preset_name = self.preset_var.get()
        self.load_preset(preset_name)

    def _on_apply_click(self) -> None:
        """Validate matrix values and dispatch custom kernel array to callback."""
        try:
            kernel_matrix = self.get_kernel_array()
            if self.on_apply_kernel_cb:
                self.on_apply_kernel_cb(kernel_matrix)
        except ValueError as e:
            messagebox.showerror("Kernel Validation Error", str(e))
