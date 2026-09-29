"""
ImageArea container component for Image Blur, Sharpening & Edge Detection Studio.

Holds OriginalImageFrame and ProcessedImageFrame, providing side-by-side view,
single image views, previous-step (pre-filter) image inspection, normalized
difference image visualization, quick view toggle, and optional synchronized
display sizing. Strictly uses grid() layout management.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np

from app.gui.image_view import ImageView


class ImageArea(ttk.Frame):
    """
    Structured ImageArea container holding original and processed image frames
    with interactive comparison options.
    """

    VIEW_MODES = ["Side-by-Side", "Original Only", "Processed Only", "Previous Image", "Difference Image"]

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=5, **kwargs)

        self._orig_image: Optional[np.ndarray] = None
        self._proc_image: Optional[np.ndarray] = None
        self._prev_image: Optional[np.ndarray] = None
        self._diff_image: Optional[np.ndarray] = None

        self._active_single_view: str = "original"  # "original" or "processed"

        # Grid configuration for ImageArea frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Toolbar
        self.rowconfigure(1, weight=1)  # Image display container

        self._build_toolbar()
        self._build_display_grid()

    def _build_toolbar(self) -> None:
        """Build top toolbar containing view mode selection and comparison controls."""
        toolbar = ttk.Frame(self, padding=(5, 2, 5, 5))
        toolbar.grid(row=0, column=0, sticky="ew")

        toolbar.columnconfigure(1, weight=0)
        toolbar.columnconfigure(4, weight=1)

        # 1. View Mode Selection
        ttk.Label(toolbar, text="View Mode:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )

        self.view_mode_var = tk.StringVar(value="Side-by-Side")
        self.mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self.view_mode_var,
            values=self.VIEW_MODES,
            state="readonly",
            width=16
        )
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_view_mode_changed)

        # 2. Quick Toggle View Button
        self.btn_toggle = ttk.Button(
            toolbar,
            text="⇄ Toggle Orig/Proc",
            command=self.toggle_view
        )
        self.btn_toggle.grid(row=0, column=2, sticky="w", padx=5)

        # 3. Synchronized Sizing Checkbox
        self.sync_size_var = tk.BooleanVar(value=True)
        self.chk_sync = ttk.Checkbutton(
            toolbar,
            text="🔒 Synchronize Display Sizing",
            variable=self.sync_size_var,
            command=self._on_sync_toggled
        )
        self.chk_sync.grid(row=0, column=3, sticky="w", padx=(10, 5))

    def _build_display_grid(self) -> None:
        """Build the main dual image view grid container."""
        self.grid_container = ttk.Frame(self)
        self.grid_container.grid(row=1, column=0, sticky="nsew")

        # Configure columns for equal side-by-side expansion
        self.grid_container.columnconfigure(0, weight=1)
        self.grid_container.columnconfigure(1, weight=1)
        self.grid_container.rowconfigure(0, weight=1)

        # Original Image Frame
        self.original_frame = ImageView(self.grid_container, title="Original Image")
        self.original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=2)

        # Processed Image Frame
        self.processed_frame = ImageView(self.grid_container, title="Processed Result")
        self.processed_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=2)

    def set_images(
        self, 
        orig_img: Optional[np.ndarray], 
        proc_img: Optional[np.ndarray],
        prev_img: Optional[np.ndarray] = None
    ) -> None:
        """
        Update original, processed, and previous image arrays and recalculate difference.

        Args:
            orig_img (np.ndarray | None): Original image array.
            proc_img (np.ndarray | None): Processed image array.
            prev_img (np.ndarray | None): Image state before the latest filter step;
                None when no filter has been applied yet (or the chain is empty).
        """
        self._orig_image = orig_img.copy() if orig_img is not None else None
        self._proc_image = proc_img.copy() if proc_img is not None else None
        self._prev_image = prev_img.copy() if prev_img is not None else None

        # Compute absolute difference map
        if self._orig_image is not None and self._proc_image is not None and self._orig_image.shape == self._proc_image.shape:
            self._diff_image = np.abs(
                self._orig_image.astype(np.float64) - self._proc_image.astype(np.float64)
            )
        else:
            self._diff_image = None

        self._apply_view_mode()

    def get_view_mode(self) -> str:
        """Return currently selected view mode string."""
        return self.view_mode_var.get()

    def toggle_view(self) -> None:
        """Toggle active view between Original and Processed images."""
        mode = self.get_view_mode()
        if mode == "Side-by-Side":
            # Switch to single view mode and alternate
            self._active_single_view = "processed" if self._active_single_view == "original" else "original"
            self.view_mode_var.set("Processed Only" if self._active_single_view == "processed" else "Original Only")
        elif mode in ("Original Only", "Processed Only"):
            if mode == "Original Only":
                self.view_mode_var.set("Processed Only")
                self._active_single_view = "processed"
            else:
                self.view_mode_var.set("Original Only")
                self._active_single_view = "original"
        elif mode == "Difference Image":
            self.view_mode_var.set("Side-by-Side")

        elif mode == "Previous Image":
            self.view_mode_var.set("Side-by-Side")

        self._apply_view_mode()

    def _on_view_mode_changed(self, event: tk.Event) -> None:
        """Handle view mode combobox selection change."""
        self._apply_view_mode()

    def _on_sync_toggled(self) -> None:
        """Handle synchronized sizing checkbox toggle."""
        self._apply_view_mode()

    def _apply_view_mode(self) -> None:
        """Update sub-frame visibility and titles according to current view mode."""
        mode = self.get_view_mode()

        if mode == "Side-by-Side":
            self.original_frame.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=(0, 4), pady=2)
            self.processed_frame.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=(4, 0), pady=2)

            self.original_frame.set_image(self._orig_image)
            self.processed_frame.config(text="Processed Result")
            self.processed_frame.set_image(self._proc_image)

        elif mode == "Original Only":
            self.processed_frame.grid_forget()
            self.original_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=2)
            self.original_frame.set_image(self._orig_image)

        elif mode == "Processed Only":
            self.original_frame.grid_forget()
            self.processed_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=2)
            self.processed_frame.config(text="Processed Result")
            self.processed_frame.set_image(self._proc_image)

        elif mode == "Previous Image":
            self.original_frame.grid_forget()
            self.processed_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=2)
            if self._prev_image is not None:
                self.processed_frame.config(text="Previous Image (Before Last Filter)")
                self.processed_frame.set_image(self._prev_image)
            else:
                self.processed_frame.config(text="Previous Image (No Filter Applied Yet)")
                self.processed_frame.set_image(None)

        elif mode == "Difference Image":
            self.original_frame.grid_forget()
            self.processed_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=2)
            self.processed_frame.config(text="Difference Image (|Original - Processed|)")
            self.processed_frame.set_image(self._diff_image)
