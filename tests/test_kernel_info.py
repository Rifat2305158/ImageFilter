"""
Automated unit tests for KernelInfoPanel educational breakdown widget.
"""

import pytest
import numpy as np
import tkinter as tk
from app.gui.kernel_info import KernelInfoPanel


@pytest.fixture
def tk_root():
    """Pytest fixture providing headless Tk root window."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in test environment.")


def test_kernel_info_panel_initialization(tk_root):
    """Test KernelInfoPanel initialization and initial Box Blur display."""
    panel = KernelInfoPanel(tk_root)
    assert panel is not None
    assert panel.lbl_title.cget("text") == "Box Blur (3x3)"
    assert "Low-pass" in panel.txt_explanation.get("1.0", tk.END)


def test_kernel_info_panel_sobel_display(tk_root):
    """Test KernelInfoPanel Gx and Gy dual matrix display for Sobel Edge."""
    panel = KernelInfoPanel(tk_root)
    panel.display_kernel_info("Sobel Edge")

    assert panel.lbl_title.cget("text") == "Sobel Edge"
    assert "1st Derivative" in panel.txt_explanation.get("1.0", tk.END)
    assert panel.edge_matrices_frame.winfo_manager() != ""  # Placed in grid


def test_kernel_info_panel_custom_matrix(tk_root):
    """Test KernelInfoPanel custom matrix rendering."""
    panel = KernelInfoPanel(tk_root)
    custom = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    panel.display_kernel_info("Custom Viva Filter", custom_matrix=custom)
    assert panel.lbl_title.cget("text") == "Custom Viva Filter"
    assert panel.lbl_sum.cget("text") == "Sum: 0.0000"
