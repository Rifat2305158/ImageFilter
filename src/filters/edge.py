"""
Edge detection filter implementations.

This module provides spatial domain edge detection using manual 2D discrete convolution.
Edge detection highlights regions in an image where intensity changes rapidly.

Operators provided:
1. Sobel: First-order derivative approximation combined with Gaussian-like smoothing.
2. Prewitt: First-order derivative approximation combined with box-like smoothing.
3. Roberts Cross: First-order derivative approximation along diagonal directions.
4. Laplacian: Second-order isotropic derivative operator.

For gradient-based operators (Sobel, Prewitt, Roberts):
- Gx: Horizontal gradient response (detects vertical edges/intensity transitions along X-axis).
- Gy: Vertical gradient response (detects horizontal edges/intensity transitions along Y-axis).
- G: Combined gradient magnitude calculated as G = sqrt(Gx^2 + Gy^2).
"""

import numpy as np
from src.core.convolution import convolve2d
from src.core import kernels


def _validate_2d_image(image: np.ndarray) -> np.ndarray:
    """Validate that input image is a 2D numpy array and convert to float64."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.ndim != 2:
        raise ValueError(f"Input image must be a 2D array, got shape {image.shape}.")
    return image.astype(np.float64)


def apply_sobel(
    image: np.ndarray, 
    direction: str = 'combined', 
    padding_mode: str = 'edge'
) -> np.ndarray:
    """
    Apply Sobel edge detection filter.

    Args:
        image (np.ndarray): 2D input grayscale image.
        direction (str): Response type. Options:
            - 'horizontal' (or 'x', 'gx'): Horizontal gradient Gx.
            - 'vertical' (or 'y', 'gy'): Vertical gradient Gy.
            - 'combined' (or 'magnitude', 'g'): Gradient magnitude sqrt(Gx^2 + Gy^2).
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.

    Returns:
        np.ndarray: Calculated edge response as a 2D float64 NumPy array.
    """
    img_f = _validate_2d_image(image)
    dir_clean = direction.lower().strip()

    if dir_clean in ('horizontal', 'x', 'gx'):
        return convolve2d(img_f, kernels.SOBEL_HORIZONTAL, padding_mode=padding_mode)
    elif dir_clean in ('vertical', 'y', 'gy'):
        return convolve2d(img_f, kernels.SOBEL_VERTICAL, padding_mode=padding_mode)
    elif dir_clean in ('combined', 'magnitude', 'g'):
        gx = convolve2d(img_f, kernels.SOBEL_HORIZONTAL, padding_mode=padding_mode)
        gy = convolve2d(img_f, kernels.SOBEL_VERTICAL, padding_mode=padding_mode)
        return np.sqrt(gx**2 + gy**2)
    else:
        raise ValueError(
            f"Invalid direction '{direction}' for Sobel filter. "
            "Choose 'horizontal', 'vertical', or 'combined'."
        )


def apply_prewitt(
    image: np.ndarray, 
    direction: str = 'combined', 
    padding_mode: str = 'edge'
) -> np.ndarray:
    """
    Apply Prewitt edge detection filter.

    Args:
        image (np.ndarray): 2D input grayscale image.
        direction (str): Response type ('horizontal', 'vertical', 'combined').
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.

    Returns:
        np.ndarray: Calculated edge response as a 2D float64 NumPy array.
    """
    img_f = _validate_2d_image(image)
    dir_clean = direction.lower().strip()

    if dir_clean in ('horizontal', 'x', 'gx'):
        return convolve2d(img_f, kernels.PREWITT_HORIZONTAL, padding_mode=padding_mode)
    elif dir_clean in ('vertical', 'y', 'gy'):
        return convolve2d(img_f, kernels.PREWITT_VERTICAL, padding_mode=padding_mode)
    elif dir_clean in ('combined', 'magnitude', 'g'):
        gx = convolve2d(img_f, kernels.PREWITT_HORIZONTAL, padding_mode=padding_mode)
        gy = convolve2d(img_f, kernels.PREWITT_VERTICAL, padding_mode=padding_mode)
        return np.sqrt(gx**2 + gy**2)
    else:
        raise ValueError(
            f"Invalid direction '{direction}' for Prewitt filter. "
            "Choose 'horizontal', 'vertical', or 'combined'."
        )


def apply_roberts(
    image: np.ndarray, 
    direction: str = 'combined', 
    padding_mode: str = 'edge'
) -> np.ndarray:
    """
    Apply Roberts Cross edge detection filter.

    Args:
        image (np.ndarray): 2D input grayscale image.
        direction (str): Response type ('x', 'y', 'combined').
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.

    Returns:
        np.ndarray: Calculated edge response as a 2D float64 NumPy array.
    """
    img_f = _validate_2d_image(image)
    dir_clean = direction.lower().strip()

    if dir_clean in ('x', 'horizontal', 'gx'):
        return convolve2d(img_f, kernels.ROBERTS_X, padding_mode=padding_mode)
    elif dir_clean in ('y', 'vertical', 'gy'):
        return convolve2d(img_f, kernels.ROBERTS_Y, padding_mode=padding_mode)
    elif dir_clean in ('combined', 'magnitude', 'g'):
        gx = convolve2d(img_f, kernels.ROBERTS_X, padding_mode=padding_mode)
        gy = convolve2d(img_f, kernels.ROBERTS_Y, padding_mode=padding_mode)
        return np.sqrt(gx**2 + gy**2)
    else:
        raise ValueError(
            f"Invalid direction '{direction}' for Roberts filter. "
            "Choose 'x' ('horizontal'), 'y' ('vertical'), or 'combined'."
        )


def apply_laplacian(
    image: np.ndarray, 
    padding_mode: str = 'edge'
) -> np.ndarray:
    """
    Apply Laplacian edge detection filter (isotropic 2nd order spatial derivative).

    Args:
        image (np.ndarray): 2D input grayscale image.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.

    Returns:
        np.ndarray: Calculated second derivative response as a 2D float64 NumPy array.
    """
    img_f = _validate_2d_image(image)
    return convolve2d(img_f, kernels.EDGE_LAPLACIAN, padding_mode=padding_mode)


def apply_edge(
    image: np.ndarray, 
    operator: str = 'sobel', 
    direction: str = 'combined', 
    padding_mode: str = 'edge'
) -> np.ndarray:
    """
    Unified edge detection API.

    Args:
        image (np.ndarray): 2D input grayscale image.
        operator (str): Edge detection algorithm. Options: 'sobel', 'prewitt', 'roberts', 'laplacian'.
        direction (str): Response type ('horizontal', 'vertical', 'combined'). Ignored for laplacian.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.

    Returns:
        np.ndarray: Calculated edge response as a 2D float64 NumPy array.
    """
    op_clean = operator.lower().strip()

    if op_clean == 'sobel':
        return apply_sobel(image, direction=direction, padding_mode=padding_mode)
    elif op_clean == 'prewitt':
        return apply_prewitt(image, direction=direction, padding_mode=padding_mode)
    elif op_clean == 'roberts':
        return apply_roberts(image, direction=direction, padding_mode=padding_mode)
    elif op_clean == 'laplacian':
        return apply_laplacian(image, padding_mode=padding_mode)
    else:
        raise ValueError(
            f"Unsupported edge operator '{operator}'. "
            "Choose 'sobel', 'prewitt', 'roberts', or 'laplacian'."
        )
