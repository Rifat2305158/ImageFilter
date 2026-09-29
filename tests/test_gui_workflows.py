"""
Unit and integration tests for GUI filter chaining workflows (A, B, C, D).
"""

import pytest
import numpy as np

from src.core.image_state import ImageState
from src.core.convolution import convolve2d
from src.filters.blur import apply_blur
from src.filters.sharpening import apply_sharpen
from src.filters.edge import apply_edge, apply_sobel


def test_scenario_a_load_rgb_blur_sharpen_edge_detection():
    """
    Scenario A: Load RGB → Blur → Sharpen → Edge Detection
    Verifies sequential filter chaining on 3D RGB array transitioning to 2D Luma edge response.
    """
    rgb = np.zeros((30, 30, 3), dtype=np.float64)
    rgb[:, 15:, 0] = 200.0  # Vertical Red edge
    rgb[15:, :, 1] = 150.0  # Horizontal Green edge

    state = ImageState(rgb)
    assert state.current_image.shape == (30, 30, 3)

    # 1. Blur
    b_res = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(b_res, description="Box Blur (3x3)")
    assert state.current_image.shape == (30, 30, 3)
    assert len(state.history) == 1

    # 2. Sharpen
    s_res = apply_sharpen(state.current_image, method='basic')
    state.update_current(s_res, description="Basic Sharpen")
    assert state.current_image.shape == (30, 30, 3)
    assert len(state.history) == 2

    # 3. Edge Detection
    e_res = apply_edge(state.current_image, operator='sobel', direction='combined')
    state.update_current(e_res, description="Sobel Edge")
    assert state.current_image.shape == (30, 30)  # 2D gradient magnitude map
    assert len(state.history) == 3

    # Baseline original image must remain untouched
    np.testing.assert_array_equal(state.original_image, rgb)


def test_scenario_b_load_rgb_blur_edge_detection_blur():
    """
    Scenario B: Load RGB → Blur → Edge Detection → Blur
    Verifies RGB input blur -> 2D edge map -> 2D gradient map blur chaining.
    """
    rgb = np.full((30, 30, 3), 100.0, dtype=np.float64)
    rgb[10:20, 10:20, :] = 220.0  # Central square object

    state = ImageState(rgb)

    # 1. Blur RGB
    b1 = apply_blur(state.current_image, blur_type='gaussian', size=3)
    state.update_current(b1, description="Gaussian Blur (3x3)")
    assert state.current_image.shape == (30, 30, 3)

    # 2. Edge Detection (converts RGB to 2D Luma edge map)
    e = apply_edge(state.current_image, operator='sobel', direction='combined')
    state.update_current(e, description="Sobel Edge")
    assert state.current_image.shape == (30, 30)

    # 3. Blur on 2D Edge Map
    b2 = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(b2, description="Box Blur on Edge Map")
    assert state.current_image.shape == (30, 30)
    assert len(state.history) == 3

    np.testing.assert_array_equal(state.original_image, rgb)


def test_scenario_c_load_grayscale_sharpen_custom_kernel_undo_blur():
    """
    Scenario C: Load grayscale → Sharpen → Custom Kernel → Undo → Blur
    Verifies Undo restores exact prior current_image state, enabling branching filter application.
    """
    gray = np.random.uniform(50, 200, (25, 25)).astype(np.float64)
    state = ImageState(gray)

    # 1. Sharpen
    s = apply_sharpen(state.current_image, method='strong')
    state.update_current(s, description="Strong Sharpen")
    assert state.history == ["Strong Sharpen"]

    # 2. Custom Kernel
    custom_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    c = convolve2d(state.current_image, custom_k, padding_mode='edge')
    state.update_current(c, description="Custom 3x3 Kernel")
    assert len(state.history) == 2

    # 3. Undo (reverts custom kernel, restores sharpened image `s`)
    undone = state.undo()
    np.testing.assert_array_almost_equal(undone, s)
    np.testing.assert_array_almost_equal(state.current_image, s)
    assert state.history == ["Strong Sharpen"]

    # 4. Blur applied to restored `s`
    b = apply_blur(state.current_image, blur_type='gaussian', size=3)
    state.update_current(b, description="Gaussian Blur (3x3)")
    assert state.history == ["Strong Sharpen", "Gaussian Blur (3x3)"]

    expected_blur_on_sharpen = apply_blur(s, blur_type='gaussian', size=3)
    np.testing.assert_array_almost_equal(state.current_image, expected_blur_on_sharpen)
    np.testing.assert_array_equal(state.original_image, gray)


def test_scenario_d_load_image_blur_reset_sharpen():
    """
    Scenario D: Load image → Blur → Reset → Sharpen
    Verifies Reset returns current_image to original_image, clears history stack, and starts fresh filter sequence.
    """
    img = np.full((20, 20), 120.0, dtype=np.float64)
    img[5:15, 5:15] = 200.0
    state = ImageState(img)

    # 1. Blur
    b = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(b, description="Box Blur")
    assert len(state.history) == 1

    # 2. Reset
    state.reset()
    assert len(state.history) == 0
    assert not state.can_undo
    np.testing.assert_array_equal(state.current_image, img)
    np.testing.assert_array_equal(state.original_image, img)

    # 3. Sharpen on Reset baseline
    s = apply_sharpen(state.current_image, method='basic')
    state.update_current(s, description="Basic Sharpen")
    assert state.history == ["Basic Sharpen"]

    expected_sharpen_on_original = apply_sharpen(img, method='basic')
    np.testing.assert_array_almost_equal(state.current_image, expected_sharpen_on_original)
    np.testing.assert_array_equal(state.original_image, img)
