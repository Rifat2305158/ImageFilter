"""
Automated unit tests for image analysis modules (histogram and statistics).
"""

import pytest
import numpy as np
from app.analysis.histogram import calculate_histogram
from app.analysis.statistics import calculate_image_statistics


def test_calculate_histogram_basic():
    """Test standard 256-bin histogram calculation on a synthetic image."""
    img = np.linspace(0, 255, 100).reshape((10, 10))
    counts, bin_edges = calculate_histogram(img)

    assert isinstance(counts, np.ndarray)
    assert isinstance(bin_edges, np.ndarray)
    assert len(counts) == 256
    assert len(bin_edges) == 257
    assert np.sum(counts) == 100
    assert bin_edges[0] == 0.0
    assert bin_edges[-1] == 255.0


def test_calculate_histogram_custom_bins():
    """Test custom bin count calculation."""
    img = np.array([[50.0, 100.0], [150.0, 200.0]], dtype=np.float64)
    counts, bin_edges = calculate_histogram(img, bins=10)

    assert len(counts) == 10
    assert len(bin_edges) == 11
    assert np.sum(counts) == 4


def test_calculate_histogram_invalid_input():
    """Test error handling for invalid input types or empty arrays."""
    with pytest.raises(TypeError):
        calculate_histogram("not_an_array")

    with pytest.raises(ValueError):
        calculate_histogram(np.array([]))


def test_calculate_image_statistics_basic():
    """Test spatial statistics calculation on known synthetic matrix."""
    img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    stats = calculate_image_statistics(img)

    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["mean"] == 25.0
    assert np.isclose(stats["std"], np.std(img))
    assert np.isclose(stats["rms_energy"], np.sqrt(np.mean(img ** 2)))


def test_calculate_image_statistics_constant_image():
    """Test statistics on uniform constant intensity image."""
    img = np.full((20, 20), 128.0, dtype=np.float64)
    stats = calculate_image_statistics(img)

    assert stats["min"] == 128.0
    assert stats["max"] == 128.0
    assert stats["mean"] == 128.0
    assert stats["std"] == 0.0
    assert stats["rms_energy"] == 128.0
    assert stats["edge_density"] == 100.0  # 128 > 25.0 threshold


def test_calculate_image_statistics_edge_strength():
    """Test edge density thresholding and edge max calculation."""
    # Image with half zeros and half 100.0 intensity
    img = np.zeros((10, 10), dtype=np.float64)
    img[:, 5:] = 100.0

    stats = calculate_image_statistics(img)
    assert stats["min"] == 0.0
    assert stats["max"] == 100.0
    assert stats["edge_max"] == 100.0
    assert stats["edge_density"] == 50.0  # Exactly 50% > 25.0 threshold


def test_calculate_image_statistics_invalid_input():
    """Test error handling for invalid input in statistics calculation."""
    with pytest.raises(TypeError):
        calculate_image_statistics([1, 2, 3])

    with pytest.raises(ValueError):
        calculate_image_statistics(np.array([[]]))
