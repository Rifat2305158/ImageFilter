"""
ImageView component for displaying original and processed images in Tkinter.

Uses strictly grid() geometry management, converts NumPy arrays to Pillow PhotoImage objects,
maintains persistent references to prevent garbage collection, and supports responsive resizing.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np
from PIL import Image, ImageTk


class ImageView(ttk.LabelFrame):
    """
    Tkinter LabelFrame widget to display an image with dynamic responsive scaling.
    """

    def __init__(
        self, 
        parent: tk.Widget, 
        title: str = "Image", 
        **kwargs
    ):
        super().__init__(parent, text=title, padding=10, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Persistent image references
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._current_array: Optional[np.ndarray] = None
        self._last_draw_size: tuple[int, int] = (0, 0)

        # Image Display Label
        self.image_label = ttk.Label(
            self, 
            text="No Image Loaded", 
            anchor="center"
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Bind resize event for responsive image scaling
        self.image_label.bind("<Configure>", self._on_resize)

    def set_image(self, image_array: Optional[np.ndarray]) -> None:
        """
        Update displayed image from a 2D/3D float64 NumPy array.

        Args:
            image_array (np.ndarray | None): Image array or None to clear.
        """
        if image_array is None:
            self.clear()
            return

        self._current_array = image_array.copy()
        self._render_current_image()

    def clear(self) -> None:
        """Clear current image display and release PhotoImage reference."""
        self._photo_image = None
        self._current_array = None
        self._last_draw_size = (0, 0)
        self.image_label.config(image="", text="No Image Loaded")

    def get_image(self) -> Optional[np.ndarray]:
        """Return the current float64 image array."""
        return self._current_array

    def _on_resize(self, event: tk.Event) -> None:
        """Handle label resize event for responsive image scaling."""
        if self._current_array is None:
            return

        # Avoid unnecessary re-draws for tiny pixel changes
        width, height = event.width, event.height
        if width < 30 or height < 30:
            return

        if abs(width - self._last_draw_size[0]) > 5 or abs(height - self._last_draw_size[1]) > 5:
            self._render_current_image(max_size=(width - 10, height - 10))

    def _render_current_image(self, max_size: Optional[tuple[int, int]] = None) -> None:
        """Internal helper to convert NumPy array to ImageTk and display."""
        if self._current_array is None:
            return

        arr = np.clip(self._current_array, 0.0, 255.0).astype(np.uint8)

        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode='L')
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = Image.fromarray(arr, mode='RGB')
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr, mode='RGBA')
        else:
            self.clear()
            self.image_label.config(text="Unsupported Image Format")
            return

        # Determine target max size
        if max_size is None:
            lbl_w = max(self.image_label.winfo_width() - 10, 350)
            lbl_h = max(self.image_label.winfo_height() - 10, 350)
            target_size = (lbl_w, lbl_h)
        else:
            target_size = max_size

        self._last_draw_size = target_size

        # Resize image maintaining aspect ratio
        pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Retain PhotoImage reference bound to self master
        photo = ImageTk.PhotoImage(pil_img, master=self)
        self._photo_image = photo


        # Update label display
        self.image_label.config(image=photo, text="")
