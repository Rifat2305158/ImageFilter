"""
Header component for Image Blur, Sharpening & Edge Detection Studio.
"""

import tkinter as tk
from tkinter import ttk


class Header(ttk.Frame):
    """
    Header banner widget displayed at the top of the main window.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=(15, 10, 15, 5), **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        # Title Label
        title_label = ttk.Label(
            self,
            text="Image Blur, Sharpening & Edge Detection Studio",
            font=("Segoe UI", 14, "bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        # Subtitle / Academic Context Label
        subtitle_label = ttk.Label(
            self,
            text="Signals and Linear Systems | 2D Discrete Convolution Laboratory",
            font=("Segoe UI", 9, "italic")
        )
        subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 8))

        # Bottom Separator
        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.grid(row=2, column=0, sticky="ew")
