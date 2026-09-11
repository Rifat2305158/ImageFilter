"""
Unit tests for Interactive Kernel Editor (app.gui.kernel_editor).
"""

import pytest
import numpy as np
import tkinter as tk
from app.gui.kernel_editor import KernelEditorWindow
from src.core.convolution import convolve2d


@pytest.fixture
def tk_root():
    """Pytest fixture providing a headless Tk root window."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in test environment.")


def test_kernel_editor_initialization(tk_root):
    """Test KernelEditorWindow widget creation and default 3x3 state."""
    editor = KernelEditorWindow(tk_root)
    editor.withdraw()

    assert editor.current_size == 3
    assert len(editor.entries) == 3
    assert len(editor.entries[0]) == 3

    # Default preset is Box Blur 3x3 -> elements are 1/9
    arr = editor.get_kernel_array()
    assert arr.shape == (3, 3)
    np.testing.assert_array_almost_equal(arr, np.ones((3, 3)) / 9.0)

    editor.destroy()


def test_kernel_editor_size_switch(tk_root):
    """Test switching grid dimensions between 3x3 and 5x5."""
    editor = KernelEditorWindow(tk_root)
    editor.withdraw()

    # Switch to 5x5
    editor.size_var.set("5x5")
    editor._on_size_change(None)

    assert editor.current_size == 5
    assert len(editor.entries) == 5
    assert len(editor.entries[0]) == 5

    arr = editor.get_kernel_array()
    assert arr.shape == (5, 5)

    editor.destroy()


def test_kernel_editor_preset_loading(tk_root):
    """Test loading predefined kernels."""
    editor = KernelEditorWindow(tk_root)
    editor.withdraw()

    editor.load_preset("Sobel Horizontal")
    arr = editor.get_kernel_array()

    expected_sobel_h = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    np.testing.assert_array_equal(arr, expected_sobel_h)

    editor.destroy()


def test_kernel_editor_reset_and_normalize(tk_root):
    """Test reset to zeros and normalization functions."""
    editor = KernelEditorWindow(tk_root)
    editor.withdraw()

    # Reset zeros
    editor.reset_to_zeros()
    zeros_arr = editor.get_kernel_array()
    np.testing.assert_array_equal(zeros_arr, np.zeros((3, 3)))

    # Set arbitrary values and normalize sum
    editor.load_preset("Gaussian Blur 3x3")
    editor.normalize_kernel_sum()
    norm_arr = editor.get_kernel_array()
    assert pytest.approx(np.sum(norm_arr)) == 1.0

    editor.destroy()


def test_kernel_editor_validation_error(tk_root):
    """Test ValueError raising on invalid non-numeric matrix entries."""
    editor = KernelEditorWindow(tk_root)
    editor.withdraw()

    # Set invalid text in matrix cell (0, 0)
    editor.string_vars[0][0].set("invalid_text")

    with pytest.raises(ValueError, match="Invalid numerical value"):
        editor.get_kernel_array()

    editor.destroy()


def test_kernel_editor_live_preview(tk_root):
    """Test thumbnail downsampling and live preview trigger."""
    large_image = np.full((300, 300), 128.0, dtype=np.float64)
    editor = KernelEditorWindow(tk_root, image_array=large_image)
    editor.withdraw()

    assert editor._preview_thumbnail is not None
    assert editor._preview_thumbnail.shape[0] <= 180
    assert editor._preview_thumbnail.shape[1] <= 180

    # Trigger live preview explicitly
    editor._trigger_live_preview()
    assert editor.preview_view.get_image() is not None

    editor.destroy()


def test_custom_kernel_convolution_execution():
    """Verify that a custom kernel output from editor works cleanly in convolve2d."""
    image = np.ones((10, 10), dtype=np.float64) * 100.0
    custom_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    output = convolve2d(image, custom_kernel, padding_mode='edge')
    assert output.shape == (10, 10)
    np.testing.assert_array_almost_equal(output, image)
