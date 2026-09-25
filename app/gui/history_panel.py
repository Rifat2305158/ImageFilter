"""
Processing Chain History Panel component for visual tracking of applied operations.
Displays step-by-step history from baseline original image to latest filter.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.image_state import HistoryEntry


class HistoryPanel(ttk.LabelFrame):
    """
    Tkinter component displaying a structured table of all operations in the current processing chain.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, text="Processing History Chain", padding=8, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_treeview()

    def _build_treeview(self) -> None:
        """Construct Treeview widget with custom columns and vertical scrollbar."""
        columns = ("step", "operation", "filter_type", "kernel_size", "padding_mode", "details")

        self.tree = ttk.Treeview(
            self,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=10
        )

        # Define column headings and column widths
        self.tree.heading("step", text="#", anchor="center")
        self.tree.heading("operation", text="Operation Name", anchor="w")
        self.tree.heading("filter_type", text="Filter Type", anchor="w")
        self.tree.heading("kernel_size", text="Kernel Size", anchor="center")
        self.tree.heading("padding_mode", text="Padding", anchor="center")
        self.tree.heading("details", text="Parameters & Details", anchor="w")

        self.tree.column("step", width=40, minwidth=30, stretch=False, anchor="center")
        self.tree.column("operation", width=170, minwidth=120, stretch=True, anchor="w")
        self.tree.column("filter_type", width=100, minwidth=80, stretch=False, anchor="w")
        self.tree.column("kernel_size", width=80, minwidth=60, stretch=False, anchor="center")
        self.tree.column("padding_mode", width=70, minwidth=60, stretch=False, anchor="center")
        self.tree.column("details", width=220, minwidth=140, stretch=True, anchor="w")

        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def update_history(self, entries: List) -> None:
        """
        Refresh treeview items with the current processing chain.

        Accepts either a list of ``HistoryEntry`` dataclass objects
        (as returned by ``ImageState.history_entries``) or plain
        ``Dict[str, Any]`` metadata dicts for backwards compatibility.

        The first row is always "Original Image" (step 0) regardless of
        whether *entries* is empty, so the chain baseline is always visible
        when an image is loaded.  Pass an empty list to show only the
        baseline; pass ``None`` (or omit the call) when no image is loaded.

        Args:
            entries: Ordered list of operations since the image was loaded.
                     Each element is a HistoryEntry or a metadata dict.
        """
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Always show step-0 baseline row
        self.tree.insert(
            "",
            "end",
            values=("0", "Original Image", "-", "-", "-", "-"),
            tags=("baseline",),
        )
        self.tree.tag_configure("baseline", foreground="#888888")

        if not entries:
            return

        for step_idx, entry in enumerate(entries, start=1):
            # Support both HistoryEntry dataclass and plain dict
            if hasattr(entry, "metadata"):
                meta = entry.metadata or {}
                op_name = entry.description or meta.get("operation_name", "Unknown")
            else:
                meta = entry
                op_name = meta.get("operation_name", "Unknown")

            ftype = meta.get("filter_type", "-") or "-"

            k_size = meta.get("kernel_size")
            if isinstance(k_size, (tuple, list)) and len(k_size) >= 2:
                k_str = f"{k_size[0]}x{k_size[1]}"
            elif k_size:
                k_str = str(k_size)
            else:
                k_str = "-"

            # padding_mode is not stored per-entry by default; use 'edge' as default
            padding = meta.get("padding_mode", "edge")

            # Format the parameters/details column
            params = meta.get("parameters", {}) or {}
            param_parts: List[str] = []

            if "iterations" in params:
                param_parts.append(f"Passes={params['iterations']}")
            if "operator" in params:
                param_parts.append(f"Op={params['operator']}")
            if "direction" in params:
                param_parts.append(f"Dir={params['direction']}")
            if "amount" in params:
                param_parts.append(f"Amount={params['amount']}")
            if "mode" in params:
                param_parts.append(f"Mode={params['mode']}")

            image_mode = meta.get("image_mode")
            if image_mode:
                param_parts.append(f"Image={image_mode}")

            details_str = ", ".join(param_parts) if param_parts else "-"

            self.tree.insert(
                "",
                "end",
                values=(f"→ {step_idx}", op_name, ftype, k_str, padding, details_str),
            )

    def clear(self) -> None:
        """Remove all rows and show only the baseline row header."""
        for item in self.tree.get_children():
            self.tree.delete(item)
