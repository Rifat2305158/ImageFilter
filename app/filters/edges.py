"""
Edge detection module interface (app package).

Re-exports edge detection functions from src.filters.edge for compatibility 
with the app package path structure.
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
