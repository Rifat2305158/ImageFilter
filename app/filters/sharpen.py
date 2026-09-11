"""
Image sharpening module (app package interface).
Re-exports apply_sharpen from src.filters.sharpening for compatibility with app/ path structure.
"""

from src.filters.sharpening import apply_sharpen

__all__ = ["apply_sharpen"]
