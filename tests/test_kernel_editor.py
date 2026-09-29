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
    """Verify that a custom kernel output from editor works cleanly in convolve2d for 2D grayscale."""
    image = np.ones((10, 10), dtype=np.float64) * 100.0
    custom_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    output = convolve2d(image, custom_kernel, padding_mode='edge')
    assert output.shape == (10, 10)
    np.testing.assert_array_almost_equal(output, image)


def test_custom_kernel_rgb_convolution_execution():
    """Verify that a 3x3 custom kernel works correctly on a 3D synthetic RGB image."""
    rgb_image = np.zeros((10, 10, 3), dtype=np.float64)
    rgb_image[:, :, 0] = 50.0   # Red channel
    rgb_image[:, :, 1] = 100.0  # Green channel
    rgb_image[:, :, 2] = 200.0  # Blue channel

    custom_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    output = convolve2d(rgb_image, custom_kernel, padding_mode='edge')

    # Output shape must remain (H, W, 3)
    assert output.shape == (10, 10, 3)

    # R, G, B channels must be processed independently with the same kernel
    for ch in range(3):
        expected_ch = convolve2d(rgb_image[:, :, ch], custom_kernel, padding_mode='edge')
        np.testing.assert_array_almost_equal(output[:, :, ch], expected_ch)

    # For constant color fields, custom sharpen filter output matches original channel intensities
    np.testing.assert_array_almost_equal(output[:, :, 0], 50.0)
    np.testing.assert_array_almost_equal(output[:, :, 1], 100.0)
    np.testing.assert_array_almost_equal(output[:, :, 2], 200.0)


def test_custom_kernel_5x5_rgb_convolution():
    """Verify that a 5x5 custom kernel is applied independently across RGB channels."""
    rgb_image = np.random.uniform(0, 255, (12, 12, 3)).astype(np.float64)

    # Custom 5x5 weighted kernel
    custom_5x5 = np.zeros((5, 5), dtype=np.float64)
    custom_5x5[2, 2] = 1.0  # 5x5 Identity

    output = convolve2d(rgb_image, custom_5x5, padding_mode='edge')

    assert output.shape == (12, 12, 3)
    for ch in range(3):
        expected_ch = convolve2d(rgb_image[:, :, ch], custom_5x5, padding_mode='edge')
        np.testing.assert_array_almost_equal(output[:, :, ch], expected_ch)
        np.testing.assert_array_almost_equal(output[:, :, ch], rgb_image[:, :, ch])


def test_kernel_editor_live_preview_rgb(tk_root):
    """Test thumbnail downsampling and live preview trigger with synthetic RGB image."""
    large_rgb_image = np.full((300, 300, 3), 128.0, dtype=np.float64)
    large_rgb_image[:, :, 0] = 200.0  # Red
    large_rgb_image[:, :, 1] = 100.0  # Green
    large_rgb_image[:, :, 2] = 50.0   # Blue

    editor = KernelEditorWindow(tk_root, image_array=large_rgb_image)
    editor.withdraw()

    assert editor._preview_thumbnail is not None
    assert editor._preview_thumbnail.ndim == 3
    assert editor._preview_thumbnail.shape[2] == 3
    assert editor._preview_thumbnail.shape[0] <= 180
    assert editor._preview_thumbnail.shape[1] <= 180

    # Trigger live preview explicitly
    editor._trigger_live_preview()
    preview_img = editor.preview_view.get_image()
    assert preview_img is not None
    assert preview_img.ndim == 3
    assert preview_img.shape[2] == 3

    editor.destroy()

