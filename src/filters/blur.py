"""Blur filter implementations."""
import numpy as np
from src.core.convolution import convolve2d
from src.core import kernels


# Supported kernel sizes for each blur type
_BOX_KERNEL_MAP = {
    3: kernels.BOX_BLUR_3X3,
    5: kernels.BOX_BLUR_5X5,
    7: kernels.BOX_BLUR_7X7,
    9: kernels.BOX_BLUR_9X9,
}

_GAUSSIAN_KERNEL_MAP = {
    3: kernels.GAUSSIAN_BLUR_3X3,
    5: kernels.GAUSSIAN_BLUR_5X5,
    7: kernels.GAUSSIAN_BLUR_7X7,
    9: kernels.GAUSSIAN_BLUR_9X9,
}


def apply_blur(
    image: np.ndarray,
    blur_type: str = 'box',
    size: int = 3,
    padding_mode: str = 'edge',
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a blur filter to a 2D grayscale image using the manual convolution engine.

    Args:
        image (np.ndarray): The 2D input image.
        blur_type (str): 'box' or 'gaussian'.
        size (int): Kernel size — 3, 5, 7, or 9.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default 'edge'.
        iterations (int): Number of passes to apply the filter (default 1).
                          Repeated application produces progressively stronger blur,
                          equivalent to a larger effective kernel radius.

    Returns:
        np.ndarray: The blurred image as a float64 numpy array.

    Raises:
        ValueError: If blur_type or size is unsupported.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must be a 2D or 3D RGB array, got shape {image.shape}")

    if blur_type == 'box':
        kernel_map = _BOX_KERNEL_MAP
    elif blur_type == 'gaussian':
        kernel_map = _GAUSSIAN_KERNEL_MAP
    else:
        raise ValueError(f"Unsupported blur_type '{blur_type}'. Choose 'box' or 'gaussian'.")

    if size not in kernel_map:
        raise ValueError(
            f"Unsupported size {size} for {blur_type} blur. "
            f"Choose one of: {sorted(kernel_map.keys())}."
        )

    kernel = kernel_map[size]
    result = image.astype(np.float64)

    for _ in range(max(1, iterations)):
        result = convolve2d(result, kernel, padding_mode=padding_mode)

    return result

