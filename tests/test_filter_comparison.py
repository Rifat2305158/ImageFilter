"""
Automated unit tests for FilterComparisonPanel multi-filter comparison mode.
"""

import pytest
import numpy as np
import tkinter as tk
from app.gui.filter_comparison import FilterComparisonPanel


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


def test_filter_comparison_initialization(tk_root):
    """Test FilterComparisonPanel widget creation and default checkboxes."""
    panel = FilterComparisonPanel(tk_root)
    assert panel is not None
    assert "box_blur_3x3" in panel.check_vars
    assert panel.check_vars["box_blur_3x3"].get() is True


def test_filter_comparison_run(tk_root):
    """Test running multi-filter comparison on synthetic array."""
    panel = FilterComparisonPanel(tk_root)
    synth_img = np.full((12, 12), 100.0, dtype=np.float64)
    synth_img[3:9, 3:9] = 200.0

    panel.set_image(synth_img)

    # Enable only box blur for fast test execution
    for key, var in panel.check_vars.items():
        var.set(key == "box_blur_3x3")

    panel.run_comparison()

    # Verify output frame contains result cards
    children = panel.output_frame.winfo_children()
    assert len(children) >= 2  # Original card + Box Blur card
