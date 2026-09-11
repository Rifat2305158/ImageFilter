"""
Blur filter module interface (app package).

Re-exports apply_blur from src.filters.blur for compatibility with app package path structure.
"""

from src.filters.blur import apply_blur

__all__ = ["apply_blur"]
