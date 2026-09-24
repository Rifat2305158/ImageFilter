"""
Image histogram calculation module.

Calculates 256-bin intensity distribution histogram counts and bin edges.

Supports both 2D grayscale (H, W) and 3D RGB (H, W, 3) NumPy arrays.

Functions
---------
calculate_histogram(image)
    Single combined histogram (flattened across all channels).

calculate_channel_histograms(image)
    Per-channel histograms.
    - Grayscale (H, W)  ->  {'gray': (counts, bin_edges)}
    - RGB (H, W, 3)     ->  {'red': (...), 'green': (...), 'blue': (...)}
"""

from typing import Dict, Tuple
import numpy as np


def calculate_histogram(
    image: np.ndarray,
    bins: int = 256,
    range_bounds: Tuple[float, float] = (0.0, 255.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate N-bin intensity distribution histogram for a 2D or 3D image array.

    For 3D RGB arrays (H, W, 3), the histogram is computed over all pixel-channel
    values combined (flattened), giving an overall intensity distribution.

    Args:
        image (np.ndarray): 2D (H, W) or 3D (H, W, 3) image array.
        bins (int): Number of histogram bins. Default 256.
        range_bounds (tuple): Intensity range (min, max). Default (0.0, 255.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (counts, bin_edges) arrays.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If image is empty or has unsupported dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image array cannot be empty.")

    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(
            f"Unsupported array shape {image.shape}. "
            "Expected 2D (H, W) or 3D (H, W, 3)."
        )

    clipped = np.clip(image.astype(np.float64).ravel(), range_bounds[0], range_bounds[1])
    counts, bin_edges = np.histogram(clipped, bins=bins, range=range_bounds)

    return counts, bin_edges


def calculate_channel_histograms(
    image: np.ndarray,
    bins: int = 256,
    range_bounds: Tuple[float, float] = (0.0, 255.0)
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate per-channel intensity histograms for a 2D grayscale or 3D RGB image.

    For 2D grayscale (H, W) arrays:
        Returns {'gray': (counts, bin_edges)}

    For 3D RGB (H, W, 3) arrays:
        Returns {'red': (counts, bin_edges), 'green': (...), 'blue': (...)}

    Each channel is histogrammed independently, preserving the true per-channel
    intensity distribution without mixing channels.

    Args:
        image (np.ndarray): 2D (H, W) or 3D (H, W, 3) image array.
        bins (int): Number of histogram bins. Default 256.
        range_bounds (tuple): Intensity range (min, max). Default (0.0, 255.0).

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Mapping of channel name to
        (counts, bin_edges) tuple.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If image is empty or has unsupported dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image array cannot be empty.")

    def _hist(channel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(channel_data.astype(np.float64).ravel(), range_bounds[0], range_bounds[1])
        return np.histogram(clipped, bins=bins, range=range_bounds)

    if image.ndim == 2:
        return {"gray": _hist(image)}

    if image.ndim == 3 and image.shape[2] == 3:
        return {
            "red":   _hist(image[:, :, 0]),
            "green": _hist(image[:, :, 1]),
            "blue":  _hist(image[:, :, 2]),
        }

    raise ValueError(
        f"Unsupported array shape {image.shape}. "
        "Expected 2D (H, W) or 3D (H, W, 3)."
    )
