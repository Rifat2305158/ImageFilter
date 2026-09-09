"""Blur filter implementations."""
import numpy as np
from src.core.convolution import convolve2d
from src.core import kernels

def apply_blur(image: np.ndarray, blur_type: str = 'box', size: int = 3, padding_mode: str = 'edge') -> np.ndarray:
    """
    Apply a blur filter to a 2D grayscale image using the manual convolution engine.
    
    Args:
        image (np.ndarray): The 2D input image.
        blur_type (str): The type of blur. Options are 'box' or 'gaussian'.
        size (int): The size of the filter kernel. Options are 3 or 5.
        padding_mode (str): Padding strategy ('zero', 'reflect', 'edge'). Default is 'edge'.
        
    Returns:
        np.ndarray: The blurred image as a float64 numpy array.
    """
    if blur_type == 'box':
        if size == 3:
            kernel = kernels.BOX_BLUR_3X3
        elif size == 5:
            kernel = kernels.BOX_BLUR_5X5
        else:
            raise ValueError("Unsupported size for box blur. Choose 3 or 5.")
    elif blur_type == 'gaussian':
        if size == 3:
            kernel = kernels.GAUSSIAN_BLUR_3X3
        elif size == 5:
            kernel = kernels.GAUSSIAN_BLUR_5X5
        else:
            raise ValueError("Unsupported size for gaussian blur. Choose 3 or 5.")
    else:
        raise ValueError("Unsupported blur_type. Choose 'box' or 'gaussian'.")
        
    return convolve2d(image, kernel, padding_mode=padding_mode)
