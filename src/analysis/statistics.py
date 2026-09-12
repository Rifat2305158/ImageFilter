"""
Image signal statistics analysis module.

Calculates key spatial domain metrics: minimum, maximum, mean, standard deviation,
RMS signal energy, and edge-strength metrics (edge density & peak amplitude).

Independent of Tkinter.
"""

from typing import Dict, Any
import numpy as np


def calculate_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """
    Calculate spatial domain statistics for a 2D grayscale float64 image array.

    Args:
        image (np.ndarray): 2D grayscale image NumPy array.

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

    img_f = image.astype(np.float64)

    min_val = float(np.min(img_f))
    max_val = float(np.max(img_f))
    mean_val = float(np.mean(img_f))
    std_val = float(np.std(img_f))

    # Signal Energy & Edge Strength Metrics
    rms_energy = float(np.sqrt(np.mean(img_f ** 2)))
    # Edge density: percentage of pixels with response > 25.0
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
