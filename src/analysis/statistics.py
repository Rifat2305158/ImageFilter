"""
Image signal statistics analysis module.

Calculates key spatial domain metrics: minimum, maximum, mean, standard deviation,
RMS signal energy, and edge-strength metrics (edge density & peak amplitude).

Supports both 2D grayscale (H, W) and 3D RGB (H, W, 3) NumPy arrays.

Functions
---------
calculate_image_statistics(image)
    Overall signal statistics (flattened across all channels).

calculate_channel_statistics(image)
    Per-channel statistics.
    - Grayscale (H, W)  ->  {'gray': {'min', 'max', 'mean', 'std'}}
    - RGB (H, W, 3)     ->  {'red': {...}, 'green': {...}, 'blue': {...}}
"""

from typing import Dict, Any
import numpy as np


def calculate_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """
    Calculate spatial domain statistics for a 2D grayscale or 3D RGB float64 image array.

    For 3D RGB arrays (H, W, 3), statistics are computed on all pixel-channel
    values combined (flattened), giving overall signal-level metrics.

    Args:
        image (np.ndarray): 2D (H, W) or 3D (H, W, 3) NumPy image array.

    Returns:
        Dict[str, Any]: Dictionary containing calculated statistical metrics:
            - 'min': float, minimum intensity
            - 'max': float, maximum intensity
            - 'mean': float, mean spatial intensity
            - 'std': float, spatial standard deviation
            - 'rms_energy': float, root mean square signal energy
            - 'edge_density': float, percentage of pixels exceeding 25.0 threshold
            - 'edge_max': float, peak amplitude
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image array cannot be empty.")

    img_f = image.astype(np.float64).ravel()

    min_val = float(np.min(img_f))
    max_val = float(np.max(img_f))
    mean_val = float(np.mean(img_f))
    std_val = float(np.std(img_f))

    # Signal Energy & Edge Strength Metrics
    rms_energy = float(np.sqrt(np.mean(img_f ** 2)))
    # Edge density: percentage of pixel-channel values with response > 25.0
    edge_density = float(np.mean(img_f > 25.0) * 100.0)

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "rms_energy": rms_energy,
        "edge_density": edge_density,
        "edge_max": max_val,
    }


def _channel_stats(channel: np.ndarray) -> Dict[str, float]:
    """Compute min, max, mean, std for a single 2D float64 channel array."""
    ch = channel.astype(np.float64)
    return {
        "min": float(np.min(ch)),
        "max": float(np.max(ch)),
        "mean": float(np.mean(ch)),
        "std": float(np.std(ch)),
    }


def calculate_channel_statistics(image: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-channel statistics for a 2D grayscale or 3D RGB image array.

    For 2D grayscale (H, W) arrays:
        Returns {'gray': {'min', 'max', 'mean', 'std'}}

    For 3D RGB (H, W, 3) arrays:
        Returns {'red': {...}, 'green': {...}, 'blue': {...}}

    Args:
        image (np.ndarray): 2D (H, W) or 3D (H, W, 3) NumPy image array.

    Returns:
        Dict[str, Dict[str, float]]: Per-channel statistics dictionaries.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If image is empty or has unsupported dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image array cannot be empty.")

    if image.ndim == 2:
        return {"gray": _channel_stats(image)}

    if image.ndim == 3 and image.shape[2] == 3:
        return {
            "red":   _channel_stats(image[:, :, 0]),
            "green": _channel_stats(image[:, :, 1]),
            "blue":  _channel_stats(image[:, :, 2]),
        }

    raise ValueError(
        f"Unsupported array shape {image.shape}. "
        "Expected 2D (H, W) or 3D (H, W, 3)."
    )
