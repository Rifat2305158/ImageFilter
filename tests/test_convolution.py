"""Tests for the manual 2D convolution engine."""
import pytest
import numpy as np
from src.core.convolution import convolve2d

def test_invalid_image_dimensions():
    kernel = np.ones((3, 3))
    # 1D image
    with pytest.raises(ValueError, match="Image must be a 2D array"):
        convolve2d(np.array([1, 2, 3]), kernel)
    # 3D image
    with pytest.raises(ValueError, match="Image must be a 2D array"):
        convolve2d(np.ones((3, 3, 3)), kernel)

def test_invalid_kernel_dimensions():
    image = np.ones((5, 5))
    # Even dimensions
    with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
        convolve2d(image, np.ones((2, 2)))
    with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
        convolve2d(image, np.ones((3, 2)))
    # 1D kernel
    with pytest.raises(ValueError, match="Kernel must be a 2D array"):
        convolve2d(image, np.array([1, 2, 3]))

def test_empty_kernel():
    image = np.ones((5, 5))
    with pytest.raises(ValueError, match="Kernel cannot be empty"):
        convolve2d(image, np.array([[]]))

def test_identity_kernel_3x3():
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    # Identity kernel: 1 in the middle, 0 elsewhere
    kernel = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    
    result = convolve2d(image, kernel, padding_mode='zero')
    np.testing.assert_array_equal(result, image)

def test_identity_kernel_5x5():
    image = np.random.rand(10, 10)
    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1.0
    
    result = convolve2d(image, kernel, padding_mode='zero')
    np.testing.assert_array_almost_equal(result, image)

def test_averaging_kernel():
    image = np.ones((4, 4))
    kernel = np.ones((3, 3)) / 9.0
    
    # Inside the image (no padding effects), a 3x3 region of 1s averaged should be 1
    # Reflect padding on a solid image preserves the 1s exactly.
    result = convolve2d(image, kernel, padding_mode='reflect')
    np.testing.assert_array_almost_equal(result, np.ones((4, 4)))

def test_known_small_matrix():
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    # 3x3 all ones kernel
    kernel = np.ones((3, 3))
    
    # For zero padding, the top-left pixel (val 1) will be surrounded by zeros except for:
    # 1, 2
    # 4, 5
    # Sum = 1 + 2 + 4 + 5 = 12
    result = convolve2d(image, kernel, padding_mode='zero')
    assert result[0, 0] == 12.0
    
    # Center pixel (val 5) will sum all 9 pixels: sum(1 to 9) = 45
    assert result[1, 1] == 45.0
    
def test_kernel_flipping():
    image = np.zeros((5, 5))
    image[2, 2] = 1.0  # single impulse in the middle
    
    # Asymmetric kernel
    kernel = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    # Convolution with an impulse produces the original kernel (impulse response).
    result = convolve2d(image, kernel, padding_mode='zero')
    
    expected = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float64)
                         
    np.testing.assert_array_almost_equal(result[1:4, 1:4], expected)

def test_zero_padding():
    image = np.ones((3, 3))
    kernel = np.ones((3, 3))
    result = convolve2d(image, kernel, padding_mode='zero')
    # Top-left corner gets 4 ones
    assert result[0, 0] == 4.0
    # Center gets 9 ones
    assert result[1, 1] == 9.0

def test_reflect_padding():
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    kernel = np.ones((3, 3))
    result = convolve2d(image, kernel, padding_mode='reflect')
    # Reflected array for top-left (1) with 1 pad:
    # 5 4 5
    # 2 1 2
    # 5 4 5
    # Sum = 5+4+5 + 2+1+2 + 5+4+5 = 33
    assert result[0, 0] == 33.0

def test_edge_padding():
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    kernel = np.ones((3, 3))
    result = convolve2d(image, kernel, padding_mode='edge')
    # Edge array for top-left (1) with 1 pad:
    # 1 1 2
    # 1 1 2
    # 4 4 5
    # Sum = 1+1+2 + 1+1+2 + 4+4+5 = 21
    assert result[0, 0] == 21.0
