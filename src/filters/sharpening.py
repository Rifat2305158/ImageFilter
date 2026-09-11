"""
Sharpening filter implementations.

This module provides spatial domain image sharpening using manual 2D discrete convolution.
Sharpening works by high-pass filtering (or adding second derivatives / spatial gradients)
to amplify local differences and enhance high-frequency edge details.
"""

import numpy as np
from src.core.convolution import convolve2d
from src.core import kernels


def apply_sharpen(
    image: np.ndarray,
    method: str = "basic",
    padding_mode: str = "edge",
    clip: bool = True,
    clip_range: tuple = (0.0, 255.0),
) -> np.ndarray:
    """
    Apply a sharpening filter to a 2D grayscale image using manual 2D convolution.

    Mathematical Basis:
        Sharpening enhances contrast along edges. In spatial domain filtering, 
        a sharpening operator can be modeled as:
            Output = Image * K_sharpen
        where K_sharpen is a high-pass emphasizing kernel whose elements sum to 1.0.

    Args:
        image (np.ndarray): 2D input grayscale image array.
        method (str): Sharpening algorithm type. Options are:
            - 'basic': 3x3 kernel emphasizing 4-connected neighbors.
            - 'strong': 3x3 kernel emphasizing 8-connected neighbors.
            - 'laplacian': Laplacian-based isotropic sharpening kernel.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.
        clip (bool): Whether to constrain output values within clip_range. Default True.
        clip_range (tuple): Range (min, max) for clipping output pixel intensities.

    Returns:
        np.ndarray: Sharpened image as a 2D float64 NumPy array.

    Raises:
        ValueError: If input image is not 2D or an invalid method is specified.
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be a 2D array, got shape {image.shape}")

    # Ensure input is floating point for precision during intermediate calculations
    image_float = image.astype(np.float64)

    method_clean = method.lower().strip()
    if method_clean == "basic":
        kernel = kernels.SHARPEN_BASIC
    elif method_clean == "strong":
        kernel = kernels.SHARPEN_STRONG
    elif method_clean == "laplacian":
        kernel = kernels.SHARPEN_LAPLACIAN
    else:
        raise ValueError(f"Unsupported sharpening method '{method}'. Choose 'basic', 'strong', or 'laplacian'.")

    # Perform discrete 2D spatial convolution using manual engine
    sharpened = convolve2d(image_float, kernel, padding_mode=padding_mode)

    # Perform controlled output normalization / clipping
    if clip:
        min_val, max_val = clip_range
        sharpened = np.clip(sharpened, min_val, max_val)

    return sharpened
