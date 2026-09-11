"""
Image Input/Output Module.

This module provides functions to load, convert, and save images using Pillow (PIL)
and NumPy. Intermediate arrays are converted to float64 for signal processing,
and safely converted back to uint8 when saved to disk.

The module is strictly independent of Tkinter.
"""

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def load_image(
    filepath: Union[str, Path], 
    as_grayscale: bool = True
) -> np.ndarray:
    """
    Load an image from disk using Pillow and convert to a float64 NumPy array.

    Args:
        filepath (str | Path): Path to the image file (PNG, JPG, BMP, etc.).
        as_grayscale (bool): If True, converts image to 2D grayscale array.
                             Default is True.

    Returns:
        np.ndarray: 2D or 3D float64 NumPy array.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If Pillow cannot open or parse the image file.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    try:
        with Image.open(path) as img:
            img_copy = img.copy()
    except Exception as e:
        raise ValueError(f"Failed to open image file '{filepath}': {e}") from e

    if as_grayscale:
        return convert_to_grayscale(img_copy)
    else:
        if img_copy.mode != 'RGB':
            img_copy = img_copy.convert('RGB')
        return np.asarray(img_copy, dtype=np.float64)


def convert_to_grayscale(
    image: Union[np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Convert an image (Pillow Image or NumPy array) to a 2D float64 grayscale NumPy array.

    Uses ITU-R 601-2 luma formula: Y = 0.299 * R + 0.587 * G + 0.114 * B.

    Args:
        image (np.ndarray | Image.Image): Input image.

    Returns:
        np.ndarray: 2D float64 grayscale NumPy array.

    Raises:
        TypeError: If image is not a Pillow Image or NumPy array.
        ValueError: If NumPy array has invalid dimensions.
    """
    if isinstance(image, Image.Image):
        gray_pil = image.convert('L')
        return np.asarray(gray_pil, dtype=np.float64)

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a PIL Image or NumPy array.")

    arr = image.astype(np.float64)

    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        elif channels == 4:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        elif channels == 1:
            return arr[:, :, 0]
        else:
            raise ValueError(f"Unsupported channel dimension {channels} in 3D array.")
    else:
        raise ValueError(f"Input array must be 2D or 3D, got {arr.ndim}D.")


def save_image(
    filepath: Union[str, Path],
    image: np.ndarray,
    clip: bool = True,
    clip_range: tuple = (0.0, 255.0),
    normalize: bool = False
) -> None:
    """
    Save a NumPy array as an image file on disk using Pillow.

    Converts floating-point values safely back to uint8 prior to saving.

    Args:
        filepath (str | Path): Destination file path.
        image (np.ndarray): 2D or 3D NumPy array to save.
        clip (bool): Whether to clamp values within clip_range before uint8 conversion. Default True.
        clip_range (tuple): Range (min, max) for clipping pixel values. Default (0.0, 255.0).
        normalize (bool): If True, scales min-max of array to [0, 255] range. Default False.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If input array is not a valid 2D or 3D NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image to save must be a NumPy array.")

    if image.ndim not in (2, 3):
        raise ValueError(f"Image array must be 2D or 3D, got {image.ndim}D.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = image.astype(np.float64)

    if normalize:
        min_v = arr.min()
        max_v = arr.max()
        if max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v) * 255.0
        else:
            arr = np.zeros_like(arr)

    if clip:
        arr = np.clip(arr, clip_range[0], clip_range[1])

    uint8_arr = arr.astype(np.uint8)

    mode = None
    if uint8_arr.ndim == 2:
        mode = 'L'
    elif uint8_arr.ndim == 3 and uint8_arr.shape[2] == 3:
        mode = 'RGB'
    elif uint8_arr.ndim == 3 and uint8_arr.shape[2] == 4:
        mode = 'RGBA'

    pil_img = Image.fromarray(uint8_arr, mode=mode)
    pil_img.save(path)
