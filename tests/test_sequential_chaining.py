"""
Integration unit tests for sequential filter chaining (tests/test_sequential_chaining.py).
"""

import pytest
import numpy as np

from src.core.image_state import ImageState
from src.core.convolution import convolve2d
from src.filters.blur import apply_blur
from src.filters.sharpening import apply_sharpen, apply_unsharp_mask
from src.filters.edge import apply_sobel, apply_edge, apply_prewitt


def test_chaining_blur_followed_by_sharpening_grayscale():
    """Verify grayscale: blur followed by sharpening receives blur output as input."""
    img = np.random.uniform(0, 255, (20, 20)).astype(np.float64)
    state = ImageState(img)

    # Step 1: Blur
    blurred = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(blurred, description="Box Blur (3x3)")

    assert state.current_image.shape == (20, 20)
    assert not np.array_equal(state.current_image, state.original_image)

    # Step 2: Sharpen on current_image (blurred result)
    sharpened = apply_sharpen(state.current_image, method='basic')
    state.update_current(sharpened, description="Basic Sharpen")

    # Verify second step matches direct application on blurred result
    expected = apply_sharpen(blurred, method='basic')
    np.testing.assert_array_almost_equal(state.current_image, expected)
    # Verify original_image is untouched
    np.testing.assert_array_equal(state.original_image, img)
    assert len(state.history) == 2


def test_chaining_sharpening_followed_by_blur_grayscale():
    """Verify grayscale: sharpening followed by blur receives sharpened output as input."""
    img = np.random.uniform(0, 255, (16, 16)).astype(np.float64)
    state = ImageState(img)

    # Step 1: Sharpen
    sharpened = apply_sharpen(state.current_image, method='strong')
    state.update_current(sharpened, description="Strong Sharpen")

    # Step 2: Blur on sharpened result
    blurred = apply_blur(state.current_image, blur_type='gaussian', size=3)
    state.update_current(blurred, description="Gaussian Blur (3x3)")

    expected = apply_blur(sharpened, blur_type='gaussian', size=3)
    np.testing.assert_array_almost_equal(state.current_image, expected)
    np.testing.assert_array_equal(state.original_image, img)


def test_chaining_blur_followed_by_edge_detection_grayscale():
    """Verify grayscale: blur followed by edge detection."""
    img = np.zeros((20, 20), dtype=np.float64)
    img[:, 10:] = 255.0
    state = ImageState(img)

    # Step 1: Blur
    blurred = apply_blur(state.current_image, blur_type='gaussian', size=3)
    state.update_current(blurred, description="Gaussian Blur (3x3)")

    # Step 2: Edge detection on blurred result
    edge = apply_sobel(state.current_image, direction='combined')
    state.update_current(edge, description="Sobel Edge")

    expected = apply_sobel(blurred, direction='combined')
    np.testing.assert_array_almost_equal(state.current_image, expected)
    assert state.current_image.shape == (20, 20)


def test_chaining_edge_detection_followed_by_blur_grayscale():
    """Verify grayscale: edge detection (2D gradient) followed by blur on 2D gradient map."""
    img = np.zeros((20, 20), dtype=np.float64)
    img[10:, :] = 200.0
    state = ImageState(img)

    # Step 1: Sobel Edge Detection -> returns 2D gradient magnitude
    edge = apply_sobel(state.current_image, direction='combined')
    state.update_current(edge, description="Sobel Edge")

    # Step 2: Blur on the 2D gradient magnitude map
    blurred_edge = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(blurred_edge, description="Box Blur on Edge Map")

    expected = apply_blur(edge, blur_type='box', size=3)
    np.testing.assert_array_almost_equal(state.current_image, expected)
    assert state.current_image.shape == (20, 20)


def test_chaining_rgb_blur_followed_by_rgb_sharpening():
    """Verify RGB: RGB blur followed by RGB sharpening preserves (H, W, 3) 3D shape."""
    rgb = np.zeros((15, 15, 3), dtype=np.float64)
    rgb[:, :, 0] = 50.0
    rgb[:, :, 1] = 100.0
    rgb[:, :, 2] = 200.0
    state = ImageState(rgb)

    # Step 1: RGB Blur
    blurred_rgb = apply_blur(state.current_image, blur_type='gaussian', size=3)
    state.update_current(blurred_rgb, description="Gaussian Blur 3x3")

    assert state.current_image.shape == (15, 15, 3)

    # Step 2: RGB Sharpen on blurred result
    sharpened_rgb = apply_sharpen(state.current_image, method='basic')
    state.update_current(sharpened_rgb, description="Basic Sharpen")

    assert state.current_image.shape == (15, 15, 3)
    expected = apply_sharpen(blurred_rgb, method='basic')
    np.testing.assert_array_almost_equal(state.current_image, expected)
    np.testing.assert_array_equal(state.original_image, rgb)


def test_chaining_rgb_blur_followed_by_edge_detection():
    """Verify RGB: RGB blur followed by edge detection handles Luma conversion cleanly."""
    rgb = np.zeros((15, 15, 3), dtype=np.float64)
    rgb[:, 8:, 0] = 200.0  # Red edge
    rgb[:, 8:, 1] = 100.0  # Green edge
    state = ImageState(rgb)

    # Step 1: RGB Blur -> returns (15, 15, 3)
    blurred_rgb = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(blurred_rgb, description="Box Blur 3x3")
    assert state.current_image.shape == (15, 15, 3)

    # Step 2: Edge detection on blurred RGB result -> converts to Luma, returns (15, 15) 2D
    edge_map = apply_edge(state.current_image, operator='sobel', direction='combined')
    state.update_current(edge_map, description="Sobel Edge")

    assert state.current_image.shape == (15, 15)  # 2D gradient map
    expected = apply_edge(blurred_rgb, operator='sobel', direction='combined')
    np.testing.assert_array_almost_equal(state.current_image, expected)


def test_chaining_custom_kernel_followed_by_another_filter():
    """Verify custom kernel convolution output used as input for next filter."""
    rgb = np.full((12, 12, 3), 100.0, dtype=np.float64)
    state = ImageState(rgb)

    # Step 1: Custom 3x3 sharpen kernel on RGB
    custom_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    custom_res = convolve2d(state.current_image, custom_kernel, padding_mode='edge')
    state.update_current(custom_res, description="Custom 3x3 Kernel")

    assert state.current_image.shape == (12, 12, 3)

    # Step 2: Unsharp Mask on custom kernel output
    usm_res = apply_unsharp_mask(state.current_image, radius=3, amount=1.0)
    state.update_current(usm_res, description="Unsharp Mask (r=3, a=1.0)")

    assert state.current_image.shape == (12, 12, 3)
    expected = apply_unsharp_mask(custom_res, radius=3, amount=1.0)
    np.testing.assert_array_almost_equal(state.current_image, expected)


def test_chaining_error_preserves_current_and_original_state():
    """Verify that if a filter operation raises an exception, current_image & original_image remain unchanged."""
    img = np.full((10, 10), 128.0, dtype=np.float64)
    state = ImageState(img)

    # Successful Step 1
    blurred = apply_blur(state.current_image, blur_type='box', size=3)
    state.update_current(blurred, description="Step 1 Blur")
    current_before_error = state.current_image.copy()

    # Step 2 raises ValueError (invalid size)
    with pytest.raises(ValueError):
        apply_blur(state.current_image, blur_type='box', size=4)  # 4 is invalid size

    # State must remain unchanged
    np.testing.assert_array_equal(state.current_image, current_before_error)
    np.testing.assert_array_equal(state.original_image, img)
    assert state.history == ["Step 1 Blur"]
