"""
app.analysis.statistics — re-exports from src.analysis.statistics.

This thin re-export layer keeps the app package decoupled from src internals
while ensuring a single source of truth for all analysis logic.
"""

from src.analysis.statistics import (
    calculate_image_statistics,
    calculate_channel_statistics,
)

__all__ = [
    "calculate_image_statistics",
    "calculate_channel_statistics",
]
