"""
Multi-Filter Comparison Panel.

Allows users to select multiple signal-processing filters simultaneously,
applies manual discrete convolution in parallel/sequence, and renders comparative
results side-by-side in a responsive card grid with performance timing and statistics.
"""

import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional
import numpy as np

from app.gui.image_view import ImageView
from app.filters.blur import apply_blur
from app.filters.sharpen import apply_sharpen
from app.filters.edges import apply_edge
from app.analysis.statistics import calculate_image_statistics


class FilterComparisonPanel(ttk.Frame):
    """
    Tkinter widget enabling multi-filter batch comparison on the loaded image array.
    """

    AVAILABLE_COMPARISONS = [
        ("Box Blur (3x3)", "box_blur_3x3"),
        ("Gaussian Blur (3x3)", "gauss_blur_3x3"),
        ("Basic Sharpen", "basic_sharpen"),
        ("Strong Sharpen", "strong_sharpen"),
        ("Sobel Edge (Combined)", "sobel_edge"),
        ("Prewitt Edge (Combined)", "prewitt_edge"),
        ("Roberts Edge (Combined)", "roberts_edge"),
        ("Laplacian Edge", "laplacian_edge"),
    ]

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=10, **kwargs)

        self._image_array: Optional[np.ndarray] = None
        self.check_vars: Dict[str, tk.BooleanVar] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Controls
        self.rowconfigure(1, weight=1)  # Grid output area

        self._build_controls()
        self._build_grid_area()

    def _build_controls(self) -> None:
        """Construct top selection bar with filter checkboxes and action button."""
        ctrl_frame = ttk.LabelFrame(self, text="Select Filters to Compare", padding=8)
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Checkboxes grid
        chk_container = ttk.Frame(ctrl_frame)
        chk_container.grid(row=0, column=0, sticky="w")

        # Select all / default defaults
        default_selected = {"box_blur_3x3", "gauss_blur_3x3", "basic_sharpen", "sobel_edge"}

        for i, (label_text, key) in enumerate(self.AVAILABLE_COMPARISONS):
            var = tk.BooleanVar(value=(key in default_selected))
            self.check_vars[key] = var

            r = i // 4
            c = i % 4
            chk = ttk.Checkbutton(chk_container, text=label_text, variable=var)
            chk.grid(row=r, column=c, sticky="w", padx=8, pady=3)

        # Run Button
        self.btn_run = ttk.Button(
            ctrl_frame,
            text="⚡ Run Comparison Matrix",
            command=self.run_comparison
        )
        self.btn_run.grid(row=0, column=1, sticky="e", padx=(15, 0))

    def _build_grid_area(self) -> None:
        """Construct scrollable/responsive output grid container."""
        # Main Canvas for dynamic card layout
        self.output_frame = ttk.Frame(self)
        self.output_frame.grid(row=1, column=0, sticky="nsew")
        self.output_frame.columnconfigure((0, 1, 2), weight=1)
        self.output_frame.rowconfigure((0, 1, 2), weight=1)

        self.lbl_empty = ttk.Label(
            self.output_frame,
            text="Load an image and click 'Run Comparison Matrix' to generate filter comparison.",
            font=("Segoe UI", 10, "italic"),
            anchor="center"
        )
        self.lbl_empty.grid(row=0, column=0, columnspan=3, pady=40)

    def set_image(self, image_array: Optional[np.ndarray]) -> None:
        """Update source image array for comparison execution."""
        self._image_array = image_array.copy() if image_array is not None else None

    def run_comparison(self) -> None:
        """Execute selected filters using manual convolution engine and display card matrix."""
        if self._image_array is None:
            messagebox.showwarning("No Image", "Please load an image before running filter comparison.")
            return

        selected_keys = [key for key, var in self.check_vars.items() if var.get()]
        if not selected_keys:
            messagebox.showwarning("No Selection", "Please check at least one filter option to compare.")
            return

        # Clear existing card widgets
        for child in self.output_frame.winfo_children():
            child.destroy()

        # Add Original Reference Card first
        cards_data = [("Original Image", self._image_array, 0.0)]

        for label_text, key in self.AVAILABLE_COMPARISONS:
            if not self.check_vars[key].get():
                continue

            t0 = time.perf_counter()
            res = self._compute_filter_result(key)
            elapsed = time.perf_counter() - t0

            cards_data.append((label_text, res, elapsed))

        # Render Cards in 3-Column Grid
        cols = 3
        for idx, (title, img_arr, timing) in enumerate(cards_data):
            r = idx // cols
            c = idx % cols

            self.output_frame.rowconfigure(r, weight=1)
            self.output_frame.columnconfigure(c, weight=1)

            card_frame = ttk.LabelFrame(
                self.output_frame, 
                text=f"{title} ({timing:.3f}s)" if timing > 0 else title, 
                padding=5
            )
            card_frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            card_frame.columnconfigure(0, weight=1)
            card_frame.rowconfigure(0, weight=1)

            view = ImageView(card_frame, title="")
            view.grid(row=0, column=0, sticky="nsew")
            view.set_image(img_arr)

            # Mini stats label
            stats = calculate_image_statistics(img_arr)
            stats_str = f"Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f} | Edge Density: {stats['edge_density']:.1f}%"
            lbl_stats = ttk.Label(card_frame, text=stats_str, font=("Segoe UI", 8), anchor="center")
            lbl_stats.grid(row=1, column=0, sticky="ew", pady=(2, 0))

    def _compute_filter_result(self, key: str) -> np.ndarray:
        """Dispatch key to manual convolution filter function."""
        arr = self._image_array
        if key == "box_blur_3x3":
            return apply_blur(arr, blur_type="box", size=3, padding_mode="edge")
        elif key == "gauss_blur_3x3":
            return apply_blur(arr, blur_type="gaussian", size=3, padding_mode="edge")
        elif key == "basic_sharpen":
            return apply_sharpen(arr, method="basic", padding_mode="edge")
        elif key == "strong_sharpen":
            return apply_sharpen(arr, method="strong", padding_mode="edge")
        elif key == "sobel_edge":
            res = apply_edge(arr, operator="sobel", direction="combined", padding_mode="edge")
            max_v = np.max(res)
            return (res / max_v * 255.0) if max_v > 0 else res
        elif key == "prewitt_edge":
            res = apply_edge(arr, operator="prewitt", direction="combined", padding_mode="edge")
            max_v = np.max(res)
            return (res / max_v * 255.0) if max_v > 0 else res
        elif key == "roberts_edge":
            res = apply_edge(arr, operator="roberts", direction="combined", padding_mode="edge")
            max_v = np.max(res)
            return (res / max_v * 255.0) if max_v > 0 else res
        elif key == "laplacian_edge":
            res = np.abs(apply_edge(arr, operator="laplacian", padding_mode="edge"))
            max_v = np.max(res)
            return (res / max_v * 255.0) if max_v > 0 else res
        else:
            return arr.copy()
