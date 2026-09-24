"""
Signal Analysis & Histogram Visualization Panel.

Embeds Matplotlib charts inside Tkinter using FigureCanvasTkAgg and displays
side-by-side numerical signal & edge statistics.

Supports both 2D grayscale (H, W) and 3D RGB (H, W, 3) image arrays.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.analysis.statistics import calculate_image_statistics
from app.analysis.histogram import calculate_histogram


class AnalysisPanel(ttk.Frame):
    """
    Embedded Tkinter analysis panel displaying signal statistics and Matplotlib histograms.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=10, **kwargs)

        self._orig_array: Optional[np.ndarray] = None
        self._proc_array: Optional[np.ndarray] = None

        self.columnconfigure(0, weight=0, minsize=320)  # Stats Table
        self.columnconfigure(1, weight=1)               # Matplotlib Canvas
        self.rowconfigure(0, weight=1)

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct UI widgets using grid()."""
        # 1. Left Stats Table Frame
        stats_frame = ttk.LabelFrame(self, text="Signal & Edge Statistics", padding=10)
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        stats_frame.columnconfigure((0, 1, 2), weight=1)

        # Header labels
        ttk.Label(stats_frame, text="Metric", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(stats_frame, text="Original", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="e", pady=3)
        ttk.Label(stats_frame, text="Processed", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="e", pady=3)

        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)

        self.stat_rows = {}
        metrics = [
            ("Min Intensity", "min"),
            ("Max Intensity", "max"),
            ("Mean Intensity", "mean"),
            ("Std Deviation", "std"),
            ("RMS Energy", "rms_energy"),
            ("Edge Density (%)", "edge_density"),
            ("Peak Amplitude", "edge_max"),
        ]

        for i, (label_text, key) in enumerate(metrics, start=2):
            ttk.Label(stats_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=2)
            orig_lbl = ttk.Label(stats_frame, text="-", font=("Consolas", 9), anchor="e")
            orig_lbl.grid(row=i, column=1, sticky="e", pady=2)
            proc_lbl = ttk.Label(stats_frame, text="-", font=("Consolas", 9), anchor="e")
            proc_lbl.grid(row=i, column=2, sticky="e", pady=2)

            self.stat_rows[key] = (orig_lbl, proc_lbl)

        # Mode label row (Grayscale / RGB)
        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(stats_frame, text="Image Mode", font=("Segoe UI", 9, "bold")).grid(row=10, column=0, sticky="w", pady=2)
        self.lbl_orig_mode = ttk.Label(stats_frame, text="-", font=("Consolas", 9), anchor="e")
        self.lbl_orig_mode.grid(row=10, column=1, sticky="e", pady=2)
        self.lbl_proc_mode = ttk.Label(stats_frame, text="-", font=("Consolas", 9), anchor="e")
        self.lbl_proc_mode.grid(row=10, column=2, sticky="e", pady=2)

        # 2. Right Matplotlib Canvas Frame
        plot_frame = ttk.LabelFrame(self, text="Histogram Visualization (256 Bins)", padding=5)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Create Matplotlib Figure
        self.figure = Figure(figsize=(7, 5), dpi=100, facecolor="#F0F0F0")
        self.ax_orig = self.figure.add_subplot(221)
        self.ax_proc = self.figure.add_subplot(222)
        self.ax_comp = self.figure.add_subplot(212)

        self.figure.tight_layout(pad=2.5)

        # Embed Figure in Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self._draw_empty_plots()

    def update_analysis(
        self, 
        orig_img: Optional[np.ndarray], 
        proc_img: Optional[np.ndarray]
    ) -> None:
        """
        Recalculate signal statistics and update embedded Matplotlib histogram plots.

        Supports both 2D grayscale (H, W) and 3D RGB (H, W, 3) arrays.

        Args:
            orig_img (np.ndarray | None): Original image array.
            proc_img (np.ndarray | None): Processed image array.
        """
        self._orig_array = orig_img
        self._proc_array = proc_img

        orig_stats = calculate_image_statistics(orig_img) if orig_img is not None else None
        proc_stats = calculate_image_statistics(proc_img) if proc_img is not None else None

        # 1. Update Numerical Table Labels
        for key, (orig_lbl, proc_lbl) in self.stat_rows.items():
            if orig_stats is not None:
                val = orig_stats[key]
                orig_lbl.config(text=f"{val:.2f}" if key == "edge_density" else f"{val:.4f}")
            else:
                orig_lbl.config(text="-")

            if proc_stats is not None:
                val = proc_stats[key]
                proc_lbl.config(text=f"{val:.2f}" if key == "edge_density" else f"{val:.4f}")
            else:
                proc_lbl.config(text="-")

        # Update mode labels
        def _mode_str(arr):
            if arr is None:
                return "-"
            return "RGB" if arr.ndim == 3 else "Grayscale"

        self.lbl_orig_mode.config(text=_mode_str(orig_img))
        self.lbl_proc_mode.config(text=_mode_str(proc_img))

        # 2. Update Matplotlib Subplots
        self.ax_orig.clear()
        self.ax_proc.clear()
        self.ax_comp.clear()

        if orig_img is None and proc_img is None:
            self._draw_empty_plots()
            return

        centers = np.arange(256)

        # Original Histogram
        if orig_img is not None:
            self._plot_histogram(self.ax_orig, orig_img, "Original Image Histogram")
            self._plot_comparison_lines(self.ax_comp, orig_img, label_prefix="Orig",
                                        colors=("#2B5B84", "#1A7A4A", "#B06020"))

        # Processed Histogram
        if proc_img is not None:
            self._plot_histogram(self.ax_proc, proc_img, "Processed Image Histogram")
            self._plot_comparison_lines(self.ax_comp, proc_img, label_prefix="Proc",
                                        colors=("#D9534F", "#E8A020", "#9B59B6"),
                                        linestyle="--")

        # Comparison Overlay
        self.ax_comp.set_title("Side-by-Side Histogram Comparison", fontsize=9, fontweight="bold")
        self.ax_comp.set_xlim(0, 255)
        self.ax_comp.grid(True, linestyle=":", alpha=0.5)
        if orig_img is not None or proc_img is not None:
            self.ax_comp.legend(loc="upper right", fontsize=7)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _plot_histogram(self, ax, img: np.ndarray, title: str) -> None:
        """Plot a histogram on the given axis. Handles both grayscale and RGB arrays."""
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 255)
        ax.grid(True, linestyle=":", alpha=0.5)

        if img.ndim == 3 and img.shape[2] == 3:
            # Per-channel RGB histograms overlaid
            channel_colors = [("#D9534F", "R"), ("#2DB85A", "G"), ("#3B90E0", "B")]
            for ch_idx, (color, ch_label) in enumerate(channel_colors):
                counts, _ = calculate_histogram(img[:, :, ch_idx])
                centers = np.arange(256)
                ax.plot(centers, counts, color=color, linewidth=1.0, alpha=0.8, label=ch_label)
            ax.legend(loc="upper right", fontsize=7)
        else:
            counts, _ = calculate_histogram(img)
            centers = np.arange(256)
            ax.bar(centers, counts, color="#2B5B84", width=1.0, alpha=0.8)

    def _plot_comparison_lines(
        self, ax, img: np.ndarray,
        label_prefix: str,
        colors: tuple,
        linestyle: str = "-"
    ) -> None:
        """Plot comparison lines on the comparison axis. Handles both grayscale and RGB."""
        centers = np.arange(256)
        if img.ndim == 3 and img.shape[2] == 3:
            ch_labels = ("R", "G", "B")
            for ch_idx, (color, ch_label) in enumerate(zip(colors, ch_labels)):
                counts, _ = calculate_histogram(img[:, :, ch_idx])
                ax.plot(centers, counts, color=color, label=f"{label_prefix}-{ch_label}",
                        linewidth=1.2, linestyle=linestyle, alpha=0.8)
        else:
            counts, _ = calculate_histogram(img)
            ax.plot(centers, counts, color=colors[0], label=label_prefix,
                    linewidth=1.5, linestyle=linestyle)

    def _draw_empty_plots(self) -> None:
        """Render placeholder text on empty subplots."""
        for ax, title in [(self.ax_orig, "Original Histogram"), (self.ax_proc, "Processed Histogram"), (self.ax_comp, "Histogram Comparison")]:
            ax.clear()
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlim(0, 255)
            ax.text(128, 0.5, "No Image Data", ha="center", va="center", color="gray", transform=ax.get_xaxis_transform())
            ax.grid(True, linestyle=":", alpha=0.5)

        self.canvas.draw()

