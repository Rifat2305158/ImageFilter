"""
Statistics module interface (app package).

Re-exports calculate_image_statistics from src.analysis.statistics.
"""

from src.analysis.statistics import calculate_image_statistics

__all__ = ["calculate_image_statistics"]
