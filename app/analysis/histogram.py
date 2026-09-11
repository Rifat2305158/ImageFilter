"""
Histogram module interface (app package).

Re-exports calculate_histogram from src.analysis.histogram.
"""

from src.analysis.histogram import calculate_histogram

__all__ = ["calculate_histogram"]
