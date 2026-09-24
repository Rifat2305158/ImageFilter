"""
Educational Kernel Information Panel.

Displays kernel matrix representation, coefficient statistics, dual Gx/Gy matrices 
for edge operators, gradient formulas, and academic signal processing explanations
tailored for university viva demonstrations.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
import numpy as np

from app.core import kernels


class KernelInfoPanel(ttk.Frame):
    """
    Tkinter widget presenting comprehensive educational signal processing breakdown
    and mathematical specifications for convolution kernels.
    """

    EXPLANATIONS: Dict[str, str] = {
        "Box Blur (3x3)": (
            "Type: Low-pass Spatial Averaging Filter (3x3)\n\n"
            "• Spatial Domain: Replaces each pixel with the unweighted average of its 3×3 neighborhood.\n"
            "• Frequency Domain: Attenuates high-frequency spatial variation and noise. Sinc-like frequency response.\n"
            "• Mathematical Property: Sum of coefficients = 1.0, preserving overall image DC energy and mean brightness."
        ),
        "Box Blur (5x5)": (
            "Type: Low-pass Spatial Averaging Filter (5x5)\n\n"
            "• Spatial Domain: Wider 5×5 neighborhood averaging resulting in stronger smoothing and blur.\n"
            "• Frequency Domain: Lower cutoff frequency than 3×3 box blur.\n"
            "• Mathematical Property: 25 elements each equal to 1/25 = 0.04. Sum = 1.0."
        ),
        "Gaussian Blur (3x3)": (
            "Type: Weighted Low-pass Gaussian Filter (3x3)\n\n"
            "• Spatial Domain: Samples 2D Gaussian distribution G(x, y) = exp(-(x²+y²)/(2σ²)). Center pixel receives highest weight.\n"
            "• Frequency Domain: Optimal joint time-frequency localization. Avoids sinc ringing artifacts associated with box filters.\n"
            "• Mathematical Property: Normalized binomial weights [1,2,1; 2,4,2; 1,2,1]/16. Sum = 1.0."
        ),
        "Gaussian Blur (5x5)": (
            "Type: Weighted Low-pass Gaussian Filter (5x5)\n\n"
            "• Spatial Domain: Smooth 5×5 Gaussian decay kernel offering natural isotropic blur.\n"
            "• Mathematical Property: Binomial coefficients scaled by 1/256. Sum = 1.0."
        ),
        "Basic Sharpen": (
            "Type: High-pass Edge Enhancement Filter (4-connected)\n\n"
            "• Spatial Domain: Emphasizes center pixel (+5) relative to immediate orthogonal neighbors (-1).\n"
            "• Frequency Domain: Amplifies high-frequency edge details while maintaining baseline DC luminance.\n"
            "• Mathematical Property: Derivation equivalent to I + (I - I_smooth). Sum = 1.0."
        ),
        "Strong Sharpen": (
            "Type: High-pass Edge Enhancement Filter (8-connected)\n\n"
            "• Spatial Domain: Aggressive sharpening placing +9 at center and -1 at all 8 surrounding neighbors.\n"
            "• Frequency Domain: Strongly boosts fine texture and high spatial frequencies.\n"
            "• Mathematical Property: Sum = 1.0."
        ),
        "Sobel Edge": (
            "Type: 1st Derivative Gradient Operator with Gaussian Smoothing\n\n"
            "• Spatial Domain: Combines 1D Gaussian smoothing along one axis with 1D central differentiation along the orthogonal axis.\n"
            "• Formulas:\n"
            "  Gx = Horizontal gradient response\n"
            "  Gy = Vertical gradient response\n"
            "  Gradient Magnitude G = √(Gx² + Gy²)\n"
            "  Direction θ = arctan(Gy / Gx)\n"
            "• Viva Note: Sum of coefficients = 0.0, yielding zero response in constant uniform regions."
        ),
        "Prewitt Edge": (
            "Type: 1st Derivative Gradient Operator with Box Smoothing\n\n"
            "• Spatial Domain: Combines uniform box averaging along one axis with central difference along the other.\n"
            "• Formulas:\n"
            "  Gx = Horizontal gradient response\n"
            "  Gy = Vertical gradient response\n"
            "  Gradient Magnitude G = √(Gx² + Gy²)\n"
            "• Viva Note: Unweighted 1D box smoothing. Sum of coefficients = 0.0."
        ),
        "Roberts Edge": (
            "Type: 2D Diagonal 1st Derivative Gradient Operator\n\n"
            "• Spatial Domain: Measures 2D diagonal cross differences (padded to 3×3 for odd-kernel convolution).\n"
            "• Formulas:\n"
            "  Gx = Main diagonal difference\n"
            "  Gy = Anti-diagonal difference\n"
            "  Gradient Magnitude G = √(Gx² + Gy²)\n"
            "• Viva Note: Highly sensitive to steep diagonal transitions and high-frequency noise."
        ),
        "Laplacian Edge": (
            "Type: Isotropic 2nd Derivative Operator ∇²I\n\n"
            "• Spatial Domain: Computes 2nd spatial derivative ∇²I = ∂²I/∂x² + ∂²I/∂y².\n"
            "• Mathematical Property: Scalar zero-crossing edge detector. Sum = 0.0."
        )
    }

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, padding=10, **kwargs)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_ui()
        self.display_kernel_info("Box Blur (3x3)")

    def _build_ui(self) -> None:
        """Construct Educational Info split layout using grid()."""
        # Left Panel: Matrix Visualization & Statistics
        left_frame = ttk.LabelFrame(self, text="Kernel Matrix & Coefficients", padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)

        # Header Info Label
        self.lbl_title = ttk.Label(left_frame, text="Kernel Name", font=("Segoe UI", 11, "bold"))
        self.lbl_title.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.lbl_size = ttk.Label(left_frame, text="Dimensions: 3 × 3", font=("Segoe UI", 9))
        self.lbl_size.grid(row=1, column=0, sticky="w", pady=(0, 8))

        # Primary Matrix Display Frame
        self.matrix_frame = ttk.LabelFrame(left_frame, text="Matrix Representation", padding=8)
        self.matrix_frame.grid(row=2, column=0, sticky="nsew", pady=5)

        # Statistics Summary Table Frame
        stats_frame = ttk.LabelFrame(left_frame, text="Mathematical Properties", padding=8)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        stats_frame.columnconfigure((0, 1, 2), weight=1)

        self.lbl_sum = ttk.Label(stats_frame, text="Sum: 1.0000", font=("Consolas", 9, "bold"))
        self.lbl_sum.grid(row=0, column=0, sticky="w")

        self.lbl_min = ttk.Label(stats_frame, text="Min: 0.0000", font=("Consolas", 9))
        self.lbl_min.grid(row=0, column=1, sticky="w")

        self.lbl_max = ttk.Label(stats_frame, text="Max: 0.1111", font=("Consolas", 9))
        self.lbl_max.grid(row=0, column=2, sticky="w")

        # Right Panel: Dual Edge Gx/Gy Matrices, Formulas, & Viva Explanation
        right_frame = ttk.LabelFrame(self, text="Academic Explanation & Formulas", padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Edge Operators Dual Matrix Frame (Gx & Gy)
        self.edge_matrices_frame = ttk.LabelFrame(right_frame, text="Directional Gradient Kernels (Gx & Gy)", padding=8)
        self.edge_matrices_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.edge_matrices_frame.columnconfigure((0, 1), weight=1)

        # Gx Frame
        self.gx_frame = ttk.Frame(self.edge_matrices_frame)
        self.gx_frame.grid(row=0, column=0, sticky="nsew", padx=4)

        # Gy Frame
        self.gy_frame = ttk.Frame(self.edge_matrices_frame)
        self.gy_frame.grid(row=0, column=1, sticky="nsew", padx=4)

        # Formula & Text Box
        self.txt_explanation = tk.Text(
            right_frame,
            wrap="word",
            font=("Segoe UI", 10),
            background="#F8F9FA",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=8
        )


        self.txt_explanation.grid(row=1, column=0, sticky="nsew")

    def display_kernel_info(self, filter_name: str, custom_matrix: Optional[np.ndarray] = None) -> None:
        """
        Update educational display for selected filter name or custom matrix.

        Args:
            filter_name (str): Label of selected filter.
            custom_matrix (np.ndarray | None): Optional custom NumPy matrix array.
        """
        self.lbl_title.config(text=filter_name)

        matrix, gx_matrix, gy_matrix = self._resolve_matrices(filter_name, custom_matrix)

        h, w = matrix.shape
        self.lbl_size.config(text=f"Dimensions: {w} × {h}")

        # Update stats
        k_sum = float(np.sum(matrix))
        k_min = float(np.min(matrix))
        k_max = float(np.max(matrix))

        self.lbl_sum.config(text=f"Sum: {k_sum:.4f}")
        self.lbl_min.config(text=f"Min: {k_min:.4f}")
        self.lbl_max.config(text=f"Max: {k_max:.4f}")

        # Render Main Matrix
        self._render_matrix_grid(self.matrix_frame, matrix)

        # Render Edge Gx / Gy matrices if applicable
        if gx_matrix is not None and gy_matrix is not None:
            self.edge_matrices_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            self._render_matrix_grid(self.gx_frame, gx_matrix, title="Gx (Horizontal)")
            self._render_matrix_grid(self.gy_frame, gy_matrix, title="Gy (Vertical)")
        else:
            self.edge_matrices_frame.grid_forget()

        # Update Text Explanation
        exp_text = self.EXPLANATIONS.get(
            filter_name, 
            f"Filter: {filter_name}\n\nCustom 2D Discrete Convolution Kernel.\nSum = {k_sum:.4f}"
        )

        self.txt_explanation.config(state="normal")
        self.txt_explanation.delete("1.0", tk.END)
        self.txt_explanation.insert(tk.END, exp_text)
        self.txt_explanation.config(state="disabled")

    def _resolve_matrices(
        self, 
        filter_name: str, 
        custom_matrix: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Resolve primary matrix, Gx matrix, and Gy matrix for given filter option."""
        if custom_matrix is not None:
            return custom_matrix, None, None

        fn = filter_name.lower()

        if "box blur (5x5)" in fn:
            return kernels.BOX_BLUR_5X5, None, None
        elif "box blur" in fn:
            return kernels.BOX_BLUR_3X3, None, None
        elif "gaussian blur (5x5)" in fn:
            return kernels.GAUSSIAN_BLUR_5X5, None, None
        elif "gaussian blur" in fn:
            return kernels.GAUSSIAN_BLUR_3X3, None, None
        elif "basic sharpen" in fn:
            return kernels.SHARPEN_BASIC, None, None
        elif "strong sharpen" in fn:
            return kernels.SHARPEN_STRONG, None, None
        elif "unsharp" in fn:
            return kernels.GAUSSIAN_BLUR_5X5, None, None
        elif "sobel" in fn or "edge detection" in fn:
            return kernels.SOBEL_HORIZONTAL, kernels.SOBEL_HORIZONTAL, kernels.SOBEL_VERTICAL
        elif "prewitt" in fn:
            return kernels.PREWITT_HORIZONTAL, kernels.PREWITT_HORIZONTAL, kernels.PREWITT_VERTICAL
        elif "roberts" in fn:
            return kernels.ROBERTS_X, kernels.ROBERTS_X, kernels.ROBERTS_Y
        elif "laplacian" in fn:
            return kernels.EDGE_LAPLACIAN, None, None
        else:
            return kernels.BOX_BLUR_3X3, None, None

    def _render_matrix_grid(self, parent_frame: tk.Widget, matrix: np.ndarray, title: Optional[str] = None) -> None:
        """Render matrix coefficients inside a clean table grid."""
        for child in parent_frame.winfo_children():
            child.destroy()

        r_offset = 0
        if title:
            ttk.Label(parent_frame, text=title, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=matrix.shape[1], pady=(0, 4))
            r_offset = 1

        for r in range(matrix.shape[0]):
            parent_frame.rowconfigure(r + r_offset, weight=1)
            for c in range(matrix.shape[1]):
                parent_frame.columnconfigure(c, weight=1)
                val = matrix[r, c]
                val_str = str(int(val)) if float(val).is_integer() else f"{val:.4f}"

                lbl = ttk.Label(
                    parent_frame,
                    text=val_str,
                    font=("Consolas", 9),
                    anchor="center",
                    relief="solid",
                    borderwidth=1,
                    padding=(4, 2)
                )
                lbl.grid(row=r + r_offset, column=c, sticky="nsew", padx=1, pady=1)
