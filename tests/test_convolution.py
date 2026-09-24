"""Tests for the manual 2D convolution engine."""
import pytest
import numpy as np
from src.core.convolution import convolve2d

def test_invalid_image_dimensions():
    kernel = np.ones((3, 3))
    # 1D image
    with pytest.raises(ValueError, match="Image must be a 2D array or 3D RGB array"):
        convolve2d(np.array([1, 2, 3]), kernel)
    # 4D image
    with pytest.raises(ValueError, match="Image must be a 2D array or 3D RGB array"):
        convolve2d(np.ones((3, 3, 3, 3)), kernel)
    # 3D image with 4 channels (RGBA)
    with pytest.raises(ValueError, match="3D RGB image must have exactly 3 channels"):
        convolve2d(np.ones((3, 3, 4)), kernel)


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
    assert result[0, 0] == 21.0



def test_rgb_identity_kernel():
    """Test that identity kernel leaves 3D RGB array unchanged."""
    rgb_image = np.zeros((4, 4, 3), dtype=np.float64)
    rgb_image[:, :, 0] = 10.0  # R
    rgb_image[:, :, 1] = 20.0  # G
    rgb_image[:, :, 2] = 30.0  # B

    identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    res = convolve2d(rgb_image, identity_kernel, padding_mode='edge')

    assert res.shape == (4, 4, 3)
    np.testing.assert_array_equal(res, rgb_image)


def test_rgb_averaging_kernel():
    """Test 3D RGB array convolved with averaging kernel across all channels."""
    rgb_image = np.ones((5, 5, 3), dtype=np.float64)
    rgb_image[:, :, 0] *= 90.0
    rgb_image[:, :, 1] *= 180.0
    rgb_image[:, :, 2] *= 270.0

    avg_kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    res = convolve2d(rgb_image, avg_kernel, padding_mode='edge')

    assert res.shape == (5, 5, 3)
    np.testing.assert_array_almost_equal(res, rgb_image)


def test_rgb_channel_independence():
    """Verify that distinct RGB channels are processed independently without cross-channel bleeding."""
    rgb_image = np.zeros((3, 3, 3), dtype=np.float64)
    rgb_image[1, 1, 0] = 100.0  # Impulse on Red channel center
    rgb_image[1, 1, 1] = 50.0   # Impulse on Green channel center
    rgb_image[1, 1, 2] = 10.0   # Impulse on Blue channel center

    kernel = np.array([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]], dtype=np.float64)
    res = convolve2d(rgb_image, kernel, padding_mode='zero')

    assert res[1, 1, 0] == 200.0
    assert res[1, 1, 1] == 100.0
    assert res[1, 1, 2] == 20.0


def test_rgb_output_shape_and_invalid_channels():
    """Verify RGB output shape and invalid channel count validation."""
    valid_rgb = np.random.rand(6, 8, 3)
    k = np.ones((3, 3)) / 9.0
    res = convolve2d(valid_rgb, k)
    assert res.shape == (6, 8, 3)

    invalid_4ch = np.random.rand(6, 8, 4)
    with pytest.raises(ValueError, match="3D RGB image must have exactly 3 channels"):
        convolve2d(invalid_4ch, k)

