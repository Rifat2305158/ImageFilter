"""
Edge detection module interface (app package alias).
Re-exports edge detection functions from src.filters.edge.
"""

from src.filters.edge import (
    apply_sobel,
    apply_prewitt,
    apply_roberts,
    apply_laplacian,
    apply_edge,
)

__all__ = [
    "apply_sobel",
    "apply_prewitt",
    "apply_roberts",
    "apply_laplacian",
    "apply_edge",
]
