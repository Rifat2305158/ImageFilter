"""
StatusBar component for displaying real-time feedback and image info.
"""

import tkinter as tk
from tkinter import ttk


class StatusBar(ttk.Frame):
    """
    Bottom status bar widget displaying application status messages and image metadata.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=(10, 4, 10, 4), **kwargs)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        # Top Separator
        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # Status Message (Left)
        self.status_label = ttk.Label(
            self,
            text="Ready",
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.status_label.grid(row=1, column=0, sticky="ew")

        # Image Info Metadata (Right)
        self.info_label = ttk.Label(
            self,
            text="No Image Loaded",
            font=("Segoe UI", 9),
            anchor="e"
        )
        self.info_label.grid(row=1, column=1, sticky="e")

    def set_status(self, message: str) -> None:
        """Update status message text."""
        self.status_label.config(text=message)

    def set_info(self, info_str: str) -> None:
        """Update image metadata info text."""
        self.info_label.config(text=info_str)

    def clear(self) -> None:
        """Reset status bar labels."""
        self.status_label.config(text="Ready")
        self.info_label.config(text="No Image Loaded")
