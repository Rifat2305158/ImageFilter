"""
Image histogram calculation module.

Calculates 256-bin intensity distribution histogram counts and bin edges.

Independent of Tkinter.
"""

from typing import Tuple
import numpy as np


def calculate_histogram(
    image: np.ndarray, 
    bins: int = 256, 
    range_bounds: Tuple[float, float] = (0.0, 255.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate N-bin intensity distribution histogram for a 2D float64 image array.

    Args:
        image (np.ndarray): 2D grayscale image array.
        bins (int): Number of histogram bins. Default 256.
        range_bounds (tuple): Intensity range (min, max). Default (0.0, 255.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (counts, bin_edges) arrays.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image array cannot be empty.")

    clipped = np.clip(image.astype(np.float64), range_bounds[0], range_bounds[1])
    counts, bin_edges = np.histogram(clipped, bins=bins, range=range_bounds)

    return counts, bin_edges
