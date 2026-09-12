"""
Image sharpening module (app package interface).
Re-exports apply_sharpen and apply_unsharp_mask from src.filters.sharpening.
"""

from src.filters.sharpening import apply_sharpen, apply_unsharp_mask

__all__ = ["apply_sharpen", "apply_unsharp_mask"]
