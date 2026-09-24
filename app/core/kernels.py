"""
Library of 2D discrete convolution kernels for image filtering.

Defines standard matrices used for blurring, sharpening, and edge detection.
Kernels are represented as 2D NumPy float64 arrays.
No actual convolution is performed in this module.
"""

import numpy as np


def make_gaussian_kernel(size: int, sigma: float = 0.0) -> np.ndarray:
    """
    Generate a square Gaussian blur kernel of arbitrary odd size.

    Uses the 2D Gaussian formula:
        G(x, y) = exp(-(x² + y²) / (2σ²))

    If sigma <= 0, it is computed automatically as:
        σ = 0.3 * ((size - 1) / 2 - 1) + 0.8   (OpenCV heuristic)

    Args:
        size (int): Odd integer kernel size (e.g. 3, 5, 7, 9, 11).
        sigma (float): Standard deviation. Auto-computed if <= 0.

    Returns:
        np.ndarray: Normalized 2D float64 Gaussian kernel (sums to 1.0).

    Raises:
        ValueError: If size is even or less than 1.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {size}.")

    if sigma <= 0.0:
        sigma = 0.3 * ((size - 1) / 2 - 1) + 0.8

    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# ==========================================
# BLUR KERNELS
# Purpose: Low-pass filters that average neighboring pixels to reduce high-frequency noise.
# Sum property: Elements should sum to 1.0 to preserve overall image brightness.
# ==========================================

BOX_BLUR_3X3 = np.ones((3, 3), dtype=np.float64) / 9.0
BOX_BLUR_5X5 = np.ones((5, 5), dtype=np.float64) / 25.0
BOX_BLUR_7X7 = np.ones((7, 7), dtype=np.float64) / 49.0
BOX_BLUR_9X9 = np.ones((9, 9), dtype=np.float64) / 81.0

# 3x3 Gaussian-like kernel (approximation using binomial coefficients)
GAUSSIAN_BLUR_3X3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0

# 5x5 Gaussian-like kernel (approximation using binomial coefficients)
GAUSSIAN_BLUR_5X5 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float64) / 256.0

# 7x7 and 9x9 Gaussian kernels
GAUSSIAN_BLUR_7X7 = make_gaussian_kernel(7)
GAUSSIAN_BLUR_9X9 = make_gaussian_kernel(9)


# ==========================================
# SHARPEN KERNELS
# Purpose: High-pass filters that enhance edges by subtracting the blurred version 
# from the original image (or adding the Laplacian).
# Sum property: Elements typically sum to 1.0 to preserve base brightness while 
# amplifying local differences.
# ==========================================

# Basic sharpening (emphasizes center pixel relative to immediate 4 neighbors)
SHARPEN_BASIC = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float64)

# Strong sharpening (emphasizes center pixel relative to all 8 neighbors)
SHARPEN_STRONG = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float64)

# Laplacian-based sharpening 
SHARPEN_LAPLACIAN = np.array([
    [ 1,  1,  1],
    [ 1, -7,  1],
    [ 1,  1,  1]
], dtype=np.float64)


# ==========================================
# EDGE DETECTION KERNELS
# Purpose: Directional or isotropic high-pass filters that approximate derivatives.
# Sum property: Elements must sum to 0.0, so areas of constant intensity become 0.
# ==========================================

# Sobel
SOBEL_HORIZONTAL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

SOBEL_VERTICAL = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

# Prewitt
PREWITT_HORIZONTAL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float64)

PREWITT_VERTICAL = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float64)

# Roberts Cross
ROBERTS_X = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  0]
], dtype=np.float64)

ROBERTS_Y = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  0]
], dtype=np.float64)

# Laplacian
EDGE_LAPLACIAN = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
], dtype=np.float64)
