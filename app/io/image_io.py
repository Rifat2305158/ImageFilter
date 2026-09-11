"""
Image Input/Output module interface (app package).

Re-exports image load, grayscale conversion, and save functions from src.io.image_io.
"""

from src.io.image_io import (
    load_image,
    convert_to_grayscale,
    save_image,
)

__all__ = [
    "load_image",
    "convert_to_grayscale",
    "save_image",
]
