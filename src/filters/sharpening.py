"""
Sharpening filter implementations.

Provides two sharpening approaches:

1. apply_sharpen() — Kernel-based sharpening via 3x3 high-pass convolution kernels.
   Works by emphasising center pixel against its neighbors.

2. apply_unsharp_mask() — Unsharp Masking (USM): the industry-standard sharpening technique.

   Mathematical Basis:
       mask      = Original - GaussianBlur(Original, radius)
       Output    = Original + amount * mask
                 = (1 + amount) * Original - amount * GaussianBlur(Original)

   This is a high-pass filter response:
       - mask captures high-frequency (edge/detail) information only.
       - Adding a scaled mask back amplifies those details visibly.
       - Using a larger Gaussian radius (7x7, 9x9) makes the effect
         clearly visible even on high-resolution (1536x2048+) images.

   Unlike kernel-based sharpening, USM allows independent control of:
       - radius  : spatial extent of detail extraction (larger = more visible on big images)
       - amount  : strength of amplification (0.5 = subtle, 1.5 = medium, 3.0 = strong)
"""

import numpy as np
from src.core.convolution import convolve2d
from src.core import kernels
from src.core.kernels import make_gaussian_kernel


def apply_sharpen(
    image: np.ndarray,
    method: str = "basic",
    padding_mode: str = "edge",
    clip: bool = True,
    clip_range: tuple = (0.0, 255.0),
) -> np.ndarray:
    """
    Apply a kernel-based sharpening filter to a 2D grayscale image.

    Mathematical Basis:
        Output = Image * K_sharpen
    where K_sharpen is a high-pass emphasizing kernel (elements sum to 1.0).

    Args:
        image (np.ndarray): 2D input grayscale image array.
        method (str): Sharpening type:
            - 'basic'     : 3x3 kernel, 4-connected neighbors (center weight = 5).
            - 'strong'    : 3x3 kernel, 8-connected neighbors (center weight = 9).
            - 'laplacian' : Laplacian-based isotropic sharpening kernel.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge').
        clip (bool): Constrain output to clip_range. Default True.
        clip_range (tuple): (min, max) intensity range for clipping.

    Returns:
        np.ndarray: Sharpened 2D float64 NumPy array.

    Raises:
        ValueError: If image is not 2D or method is unsupported.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must be a 2D or 3D RGB array, got shape {image.shape}")

    image_float = image.astype(np.float64)

    method_clean = method.lower().strip()
    if method_clean == "basic":
        kernel = kernels.SHARPEN_BASIC
    elif method_clean == "strong":
        kernel = kernels.SHARPEN_STRONG
    elif method_clean == "laplacian":
        kernel = kernels.SHARPEN_LAPLACIAN
    else:
        raise ValueError(
            f"Unsupported sharpening method '{method}'. "
            f"Choose 'basic', 'strong', or 'laplacian'."
        )

    sharpened = convolve2d(image_float, kernel, padding_mode=padding_mode)

    if clip:
        min_val, max_val = clip_range
        sharpened = np.clip(sharpened, min_val, max_val)

    return sharpened


def apply_unsharp_mask(
    image: np.ndarray,
    radius: int = 5,
    amount: float = 1.5,
    padding_mode: str = "edge",
    clip: bool = True,
    clip_range: tuple = (0.0, 255.0),
) -> np.ndarray:
    """
Applies Unsharp Masking (USM) to a 2D grayscale or 3D RGB image.

Formula:
    mask = original - blurred
    output = original + amount * mask

Args:
    image: 2D array or 3D RGB array.
    radius: Gaussian kernel size. Must be odd (e.g., 3, 5, 7).
    amount: Strength multiplier.
    padding_mode: Convolution padding scheme ('zero', 'reflect', 'edge').
    clip: If True, clips intensities to `clip_range`.
    clip_range: Min/max values for clipping.

Returns:
    Sharpened float64 array (2D or 3D).

Raises:
    ValueError: If image is not 2D/3D, radius is even, or amount <= 0.
"""

    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must be a 2D or 3D RGB array, got shape {image.shape}")


    if radius < 3 or radius % 2 == 0:
        raise ValueError(f"radius must be an odd integer >= 3, got {radius}.")

    if amount <= 0:
        raise ValueError(f"amount must be a positive number, got {amount}.")

    image_float = image.astype(np.float64)

    # Step 1: Blur using programmatic Gaussian kernel
    gauss_kernel = make_gaussian_kernel(radius)
    blurred = convolve2d(image_float, gauss_kernel, padding_mode=padding_mode)

    # Step 2: Compute high-frequency detail mask
    #   mask = original - blurred   (isolates edges and fine detail)
    mask = image_float - blurred

    # Step 3: Add amplified mask back to original
    #   output = original + amount * mask
    sharpened = image_float + amount * mask

    # Step 4: Clip to valid range
    if clip:
        min_val, max_val = clip_range
        sharpened = np.clip(sharpened, min_val, max_val)

    return sharpened
