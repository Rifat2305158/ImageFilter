"""Tests for filter implementations."""
import pytest
import numpy as np
from src.filters.blur import apply_blur

def test_blur_invalid_args():
    """Verify that unsupported parameters raise ValueErrors."""
    image = np.ones((10, 10))
    # size=4 is even — never supported (must be one of 3, 5, 7, 9)
    with pytest.raises(ValueError, match="Unsupported size"):
        apply_blur(image, blur_type='box', size=4)
    with pytest.raises(ValueError, match="Unsupported blur_type"):
        apply_blur(image, blur_type='magic')

def test_blur_preserves_constant_image():
    """A flat, constant image should remain flat and unchanged under blur with reflect/edge padding."""
    image = np.ones((10, 10)) * 128.0
    
    blurred_box = apply_blur(image, blur_type='box', size=3, padding_mode='reflect')
    np.testing.assert_array_almost_equal(blurred_box, image)
    
    blurred_gauss = apply_blur(image, blur_type='gaussian', size=5, padding_mode='edge')
    np.testing.assert_array_almost_equal(blurred_gauss, image)

def test_blur_reduces_high_frequency():
    """Verify that blurring reduces the variance (high-frequency energy) of a noisy image."""
    np.random.seed(42)
    # Synthetic image with random high-frequency noise
    # Base 100, noise +/- 50
    image = 100.0 + np.random.randint(-50, 50, (20, 20)).astype(np.float64)
    
    original_variance = np.var(image)
    
    # Apply 3x3 box blur
    blurred_box = apply_blur(image, blur_type='box', size=3, padding_mode='edge')
    box_variance = np.var(blurred_box)
    
    # Apply 3x3 gaussian blur
    blurred_gauss = apply_blur(image, blur_type='gaussian', size=3, padding_mode='edge')
    gauss_variance = np.var(blurred_gauss)
    
    # Variance (energy of variations) should be strictly less after blurring
    assert box_variance < original_variance
    assert gauss_variance < original_variance
    
    # Generally, a 5x5 blur reduces variance more than a 3x3 blur of the same type
    blurred_box_5 = apply_blur(image, blur_type='box', size=5, padding_mode='edge')
    box_5_variance = np.var(blurred_box_5)
    assert box_5_variance < box_variance


def test_rgb_blur_constant_image():
    """Verify that a constant RGB image remains constant under normalized blur."""
    rgb_image = np.ones((8, 8, 3), dtype=np.float64)
    rgb_image[:, :, 0] *= 50.0
    rgb_image[:, :, 1] *= 100.0
    rgb_image[:, :, 2] *= 150.0

    blurred = apply_blur(rgb_image, blur_type='gaussian', size=5, padding_mode='edge')
    assert blurred.shape == (8, 8, 3)
    np.testing.assert_array_almost_equal(blurred, rgb_image)


def test_rgb_blur_channel_changes():
    """Verify that RGB blur smoothes noise independently on each channel."""
    np.random.seed(42)
    noise = np.random.randint(-30, 30, (16, 16, 3)).astype(np.float64)
    rgb_image = np.ones((16, 16, 3), dtype=np.float64) * 128.0 + noise

    orig_var_r = np.var(rgb_image[:, :, 0])
    orig_var_g = np.var(rgb_image[:, :, 1])
    orig_var_b = np.var(rgb_image[:, :, 2])

    blurred = apply_blur(rgb_image, blur_type='box', size=3, padding_mode='edge')
    assert blurred.shape == (16, 16, 3)

    assert np.var(blurred[:, :, 0]) < orig_var_r
    assert np.var(blurred[:, :, 1]) < orig_var_g
    assert np.var(blurred[:, :, 2]) < orig_var_b

