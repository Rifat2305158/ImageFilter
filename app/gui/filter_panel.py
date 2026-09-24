"""
Filter control panel component for GUI parameter selection, edge detection options, and actions.
Uses grid() exclusively for geometry management.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class FilterPanel(ttk.LabelFrame):
    """
    Tkinter control panel providing filter selection, edge detection parameters,
    kernel editor opening, and action buttons.
    """

    AVAILABLE_FILTERS = [
        "Box Blur (3x3)",
        "Box Blur (5x5)",
        "Box Blur (7x7)",
        "Box Blur (9x9)",
        "Gaussian Blur (3x3)",
        "Gaussian Blur (5x5)",
        "Gaussian Blur (7x7)",
        "Gaussian Blur (9x9)",
        "Basic Sharpen",
        "Strong Sharpen",
        "Unsharp Mask — Mild  (r=5, a=1.0)",
        "Unsharp Mask — Medium (r=7, a=1.5)",
        "Unsharp Mask — Strong (r=9, a=2.5)",
        "Edge Detection",
        "Custom Kernel (Editor)",
    ]

    OPERATORS = ["Sobel", "Prewitt", "Roberts", "Laplacian"]
    DIRECTIONS = ["Combined Magnitude (G)", "Horizontal (Gx)", "Vertical (Gy)"]

    def __init__(
        self,
        parent: tk.Widget,
        on_load: Optional[Callable[[], None]] = None,
        on_apply: Optional[Callable[[str], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
        on_save: Optional[Callable[[], None]] = None,
        on_open_editor: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(parent, text="Controls & Filters", padding=12, **kwargs)

        self.on_load_cb = on_load
        self.on_apply_cb = on_apply
        self.on_reset_cb = on_reset
        self.on_save_cb = on_save
        self.on_open_editor_cb = on_open_editor

        self.columnconfigure(0, weight=1)

        self._build_widgets()

    def _build_widgets(self) -> None:
        """Construct control panel UI widgets using grid()."""
        row = 0

        # Section 1: Image I/O Actions
        io_label = ttk.Label(self, text="1. Image Actions", font=("Segoe UI", 10, "bold"))
        io_label.grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        self.btn_load = ttk.Button(self, text="📁 Load Image", command=self._on_load_click)
        self.btn_load.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        sep1 = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep1.grid(row=row, column=0, sticky="ew", pady=8)
        row += 1

        # Section 2: Filter Selection & Processing
        filter_label = ttk.Label(self, text="2. Select Filter", font=("Segoe UI", 10, "bold"))
        filter_label.grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        self.filter_var = tk.StringVar(value=self.AVAILABLE_FILTERS[0])
        self.filter_combo = ttk.Combobox(
            self, 
            textvariable=self.filter_var, 
            values=self.AVAILABLE_FILTERS, 
            state="readonly"
        )
        self.filter_combo.grid(row=row, column=0, sticky="ew", pady=3)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)
        row += 1

        # Iterations spinner
        iter_frame = ttk.Frame(self)
        iter_frame.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        iter_frame.columnconfigure(1, weight=1)
        ttk.Label(iter_frame, text="Iterations:").grid(row=0, column=0, sticky="w")
        self.iter_var = tk.IntVar(value=1)
        self.iter_spin = ttk.Spinbox(
            iter_frame,
            from_=1,
            to=5,
            textvariable=self.iter_var,
            width=4,
            state="readonly"
        )
        self.iter_spin.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(iter_frame, text="(repeated passes)", foreground="gray").grid(row=0, column=2, sticky="w", padx=(4, 0))
        row += 1

        # Sub-panel for Edge Detection Parameters
        self.edge_frame = ttk.LabelFrame(self, text="Edge Detection Parameters", padding=8)
        self.edge_frame.columnconfigure(1, weight=1)

        ttk.Label(self.edge_frame, text="Operator:").grid(row=0, column=0, sticky="w", pady=2)
        self.edge_op_var = tk.StringVar(value="Sobel")
        self.edge_op_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.edge_op_var,
            values=self.OPERATORS,
            state="readonly",
            width=12
        )
        self.edge_op_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.edge_op_combo.bind("<<ComboboxSelected>>", self._on_edge_op_changed)

        ttk.Label(self.edge_frame, text="Direction:").grid(row=1, column=0, sticky="w", pady=2)
        self.edge_dir_var = tk.StringVar(value="Combined Magnitude (G)")
        self.edge_dir_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.edge_dir_var,
            values=self.DIRECTIONS,
            state="readonly",
            width=18
        )
        self.edge_dir_combo.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)

        self.edge_norm_var = tk.BooleanVar(value=True)
        self.chk_norm = ttk.Checkbutton(
            self.edge_frame,
            text="Normalize Response [0, 255]",
            variable=self.edge_norm_var
        )
        self.chk_norm.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Educational note: RGB images are luma-converted before edge detection
        self.lbl_edge_rgb_note = ttk.Label(
            self.edge_frame,
            text="⚠ RGB input: converted to grayscale\n   Y = 0.299R + 0.587G + 0.114B",
            font=("Segoe UI", 8, "italic"),
            foreground="#666666",
            justify="left"
        )
        self.lbl_edge_rgb_note.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Initially hidden (shown when 'Edge Detection' selected)
        self.edge_frame_row = row
        row += 1

        self.btn_apply = ttk.Button(self, text="⚡ Apply Filter", command=self._on_apply_click)
        self.btn_apply.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1

        self.btn_editor = ttk.Button(self, text="🛠 Kernel Editor", command=self._on_editor_click)
        self.btn_editor.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        sep2 = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep2.grid(row=row, column=0, sticky="ew", pady=8)
        row += 1

        # Section 3: Output Controls
        output_label = ttk.Label(self, text="3. Output Options", font=("Segoe UI", 10, "bold"))
        output_label.grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        self.btn_reset = ttk.Button(self, text="↺ Reset Image", command=self._on_reset_click)
        self.btn_reset.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        self.btn_save = ttk.Button(self, text="💾 Save Result", command=self._on_save_click)
        self.btn_save.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        self._update_edge_subpanel_visibility()

    def get_selected_filter(self) -> str:
        """Return the currently selected main filter string."""
        return self.filter_var.get()

    def get_iterations(self) -> int:
        """Return the number of filter application passes (1–5)."""
        try:
            return max(1, int(self.iter_var.get()))
        except (ValueError, tk.TclError):
            return 1

    def get_edge_operator(self) -> str:
        """Return clean lowercase operator name ('sobel', 'prewitt', 'roberts', 'laplacian')."""
        return self.edge_op_var.get().lower().strip()

    def get_edge_direction(self) -> str:
        """Return clean direction string ('combined', 'horizontal', 'vertical')."""
        val = self.edge_dir_var.get().lower()
        if "horizontal" in val or "gx" in val:
            return "horizontal"
        elif "vertical" in val or "gy" in val:
            return "vertical"
        else:
            return "combined"

    def get_edge_normalize(self) -> bool:
        """Return whether visual display normalization is enabled."""
        return self.edge_norm_var.get()

    def _on_filter_changed(self, event: tk.Event) -> None:
        """Toggle edge options sub-panel visibility when filter selection changes."""
        self._update_edge_subpanel_visibility()

    def _on_edge_op_changed(self, event: tk.Event) -> None:
        """Disable direction choice for Laplacian (isotropic 2nd derivative)."""
        op = self.get_edge_operator()
        if op == "laplacian":
            self.edge_dir_combo.config(state="disabled")
        else:
            self.edge_dir_combo.config(state="readonly")

    def _update_edge_subpanel_visibility(self) -> None:
        """Show edge options subpanel if 'Edge Detection' is selected."""
        if self.get_selected_filter() == "Edge Detection":
            self.edge_frame.grid(row=self.edge_frame_row, column=0, sticky="ew", pady=6)
            self._on_edge_op_changed(None)
        else:
            self.edge_frame.grid_forget()

    def _on_load_click(self) -> None:
        if self.on_load_cb:
            self.on_load_cb()

    def _on_apply_click(self) -> None:
        if self.on_apply_cb:
            self.on_apply_cb(self.get_selected_filter())

    def _on_editor_click(self) -> None:
        if self.on_open_editor_cb:
            self.on_open_editor_cb()

    def _on_reset_click(self) -> None:
        if self.on_reset_cb:
            self.on_reset_cb()

    def _on_save_click(self) -> None:
        if self.on_save_cb:
            self.on_save_cb()
