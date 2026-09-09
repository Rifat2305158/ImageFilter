"""Tests for the kernel library."""
import pytest
import numpy as np
from src.core import kernels

def test_kernel_dimensions():
    """All kernels must be 2D arrays with odd dimensions."""
    for name in dir(kernels):
        if not name.startswith("__") and isinstance(getattr(kernels, name), np.ndarray):
            kernel = getattr(kernels, name)
            assert kernel.ndim == 2, f"{name} should be 2D"
            h, w = kernel.shape
            assert h % 2 != 0 and w % 2 != 0, f"{name} should have odd dimensions, got {kernel.shape}"

def test_blur_sums_to_one():
    """Blur kernels should sum to exactly 1.0 to preserve brightness."""
    blur_kernels = [
        kernels.BOX_BLUR_3X3,
        kernels.BOX_BLUR_5X5,
        kernels.GAUSSIAN_BLUR_3X3,
        kernels.GAUSSIAN_BLUR_5X5
    ]
    for k in blur_kernels:
        assert np.isclose(np.sum(k), 1.0)

def test_sharpen_sums_to_one():
    """Sharpen kernels should sum to exactly 1.0 to preserve brightness."""
    sharpen_kernels = [
        kernels.SHARPEN_BASIC,
        kernels.SHARPEN_STRONG,
        kernels.SHARPEN_LAPLACIAN
    ]
    for k in sharpen_kernels:
        assert np.isclose(np.sum(k), 1.0)

def test_edge_sums_to_zero():
    """Edge detection kernels should sum to 0.0 to black out flat regions."""
    edge_kernels = [
        kernels.SOBEL_HORIZONTAL,
        kernels.SOBEL_VERTICAL,
        kernels.PREWITT_HORIZONTAL,
        kernels.PREWITT_VERTICAL,
        kernels.ROBERTS_X,
        kernels.ROBERTS_Y,
        kernels.EDGE_LAPLACIAN
    ]
    for k in edge_kernels:
        assert np.isclose(np.sum(k), 0.0)

def test_symmetry():
    """Check symmetry and transpose relationships of known kernels."""
    # Gaussian blur should be symmetric
    np.testing.assert_array_equal(kernels.GAUSSIAN_BLUR_3X3, kernels.GAUSSIAN_BLUR_3X3.T)
    np.testing.assert_array_equal(kernels.GAUSSIAN_BLUR_5X5, kernels.GAUSSIAN_BLUR_5X5.T)
    
    # Sobel/Prewitt Horizontal vs Vertical relationships (transpose of each other)
    np.testing.assert_array_equal(kernels.SOBEL_HORIZONTAL, kernels.SOBEL_VERTICAL.T)
    np.testing.assert_array_equal(kernels.PREWITT_HORIZONTAL, kernels.PREWITT_VERTICAL.T)

def test_roberts_cross_structure():
    """Check that the Roberts Cross is properly embedded in a 3x3 matrix with zeros."""
    assert kernels.ROBERTS_X[0, 0] == 1
    assert kernels.ROBERTS_X[1, 1] == -1
    
    assert kernels.ROBERTS_Y[0, 1] == 1
    assert kernels.ROBERTS_Y[1, 0] == -1
    
    # Check that the 3rd row and 3rd column are purely zero padding
    for k in [kernels.ROBERTS_X, kernels.ROBERTS_Y]:
        assert np.all(k[2, :] == 0)
        assert np.all(k[:, 2] == 0)
