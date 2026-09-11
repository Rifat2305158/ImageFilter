"""
Unit tests for image sharpening filters.

Tests coverage includes:
- Constant image preservation
- Simple edge image contrast enhancement (overshoot/undershoot)
- Impulse response analysis
- Controlled clipping and normalization behavior
- Parameter and input validation
"""

import pytest
import numpy as np
from src.filters.sharpening import apply_sharpen
from app.filters.sharpen import apply_sharpen as app_apply_sharpen


def test_sharpen_module_import_consistency():
    """Verify that both src.filters.sharpening and app.filters.sharpen export the same function."""
    assert apply_sharpen is app_apply_sharpen


def test_sharpen_constant_image():
    """
    A constant/flat intensity image should remain unchanged under sharpening filters,
    because sharpening kernels sum to 1.0 (zero response to zero spatial gradients).
    """
    image = np.full((12, 12), 128.0, dtype=np.float64)

    for method in ['basic', 'strong', 'laplacian']:
        sharpened = apply_sharpen(image, method=method, padding_mode='edge')
        np.testing.assert_array_almost_equal(
            sharpened, 
            image, 
            err_msg=f"Sharpening with method '{method}' modified a constant image."
        )


def test_sharpen_simple_edge_image():
    """
    Sharpening a step edge should amplify high spatial frequencies, creating 
    characteristic overshoot on the bright side and undershoot on the dark side.
    """
    # Create 10x10 image with a vertical step edge (left: 50.0, right: 150.0)
    image = np.zeros((10, 10), dtype=np.float64)
    image[:, :5] = 50.0
    image[:, 5:] = 150.0

    # Apply sharpening with clip=False to inspect exact overshoot/undershoot values
    sharpened = apply_sharpen(image, method='basic', padding_mode='edge', clip=False)

    # Column 4 (left of edge, originally 50): should be decreased below 50 (undershoot)
    assert np.all(sharpened[:, 4] < 50.0), "Dark side of edge should show undershoot"

    # Column 5 (right of edge, originally 150): should be increased above 150 (overshoot)
    assert np.all(sharpened[:, 5] > 150.0), "Bright side of edge should show overshoot"

    # Far away from edge (e.g. col 1 and col 8), intensity should remain unchanged (50 and 150)
    np.testing.assert_array_almost_equal(sharpened[:, 1], 50.0)
    np.testing.assert_array_almost_equal(sharpened[:, 8], 150.0)


def test_sharpen_impulse_image():
    """
    An isolated impulse delta(x, y) should yield the impulse response (kernel coefficients)
    scaled by the impulse amplitude.
    """
    image = np.zeros((7, 7), dtype=np.float64)
    amplitude = 100.0
    image[3, 3] = amplitude

    # Basic sharpening kernel: center = 5, 4-neighbors = -1, diagonals = 0
    sharpened = apply_sharpen(image, method='basic', padding_mode='zero', clip=False)

    # Center pixel (3, 3) -> 5 * 100 = 500
    assert pytest.approx(sharpened[3, 3]) == 500.0

    # 4-connected neighbors -> -1 * 100 = -100
    assert pytest.approx(sharpened[2, 3]) == -100.0
    assert pytest.approx(sharpened[4, 3]) == -100.0
    assert pytest.approx(sharpened[3, 2]) == -100.0
    assert pytest.approx(sharpened[3, 4]) == -100.0

    # Diagonals -> 0 * 100 = 0
    assert pytest.approx(sharpened[2, 2]) == 0.0
    assert pytest.approx(sharpened[2, 4]) == 0.0


def test_sharpen_clipping_behavior():
    """
    Test controlled clipping behavior.
    When clip=True, pixel intensities must be constrained within clip_range (default [0, 255]).
    When clip=False, values can exceed bounds.
    """
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 3] = 200.0  # High intensity impulse

    # Without clipping, center value becomes 1000 (>255) and neighbors become -200 (<0)
    unclipped = apply_sharpen(image, method='basic', padding_mode='zero', clip=False)
    assert unclipped.max() > 255.0
    assert unclipped.min() < 0.0

    # With clipping (default), values must be clamped to [0.0, 255.0]
    clipped = apply_sharpen(image, method='basic', padding_mode='zero', clip=True, clip_range=(0.0, 255.0))
    assert clipped.max() <= 255.0
    assert clipped.min() >= 0.0
    assert pytest.approx(clipped[3, 3]) == 255.0
    assert pytest.approx(clipped[2, 3]) == 0.0


def test_sharpen_invalid_args():
    """Verify ValueError for invalid methods or input array dimensions."""
    image = np.ones((5, 5))

    with pytest.raises(ValueError, match="Unsupported sharpening method"):
        apply_sharpen(image, method='invalid_method')

    with pytest.raises(ValueError, match="Input image must be a 2D array"):
        apply_sharpen(np.ones((5, 5, 3)), method='basic')
