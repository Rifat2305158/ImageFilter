"""
app.analysis.histogram — re-exports from src.analysis.histogram.

This thin re-export layer keeps the app package decoupled from src internals
while ensuring a single source of truth for all histogram calculation logic.
"""

from src.analysis.histogram import (
    calculate_histogram,
    calculate_channel_histograms,
)

__all__ = [
    "calculate_histogram",
    "calculate_channel_histograms",
]


