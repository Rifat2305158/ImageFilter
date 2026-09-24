import pytest
import numpy as np
from app.analysis.histogram import calculate_histogram, calculate_channel_histograms
from app.analysis.statistics import calculate_image_statistics, calculate_channel_statistics


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
    """Test error handling for invalid input types, empty arrays, or unsupported shapes."""
    with pytest.raises(TypeError):
        calculate_histogram("not_an_array")

    with pytest.raises(ValueError):
        calculate_histogram(np.array([]))

    with pytest.raises(ValueError):
        calculate_histogram(np.zeros((10,)))  # 1D array unsupported

    with pytest.raises(ValueError):
        calculate_histogram(np.zeros((10, 10, 5)))  # 5-channel 3D array unsupported


def test_calculate_histogram_rgb():
    """Test combined histogram calculation on 3D RGB image array."""
    rgb_img = np.zeros((10, 10, 3), dtype=np.float64)
    rgb_img[:, :, 0] = 50.0   # Red
    rgb_img[:, :, 1] = 100.0  # Green
    rgb_img[:, :, 2] = 200.0  # Blue

    counts, bin_edges = calculate_histogram(rgb_img)

    assert len(counts) == 256
    assert len(bin_edges) == 257
    assert np.sum(counts) == 300  # 10 * 10 * 3 values


def test_calculate_channel_histograms_grayscale():
    """Test calculate_channel_histograms for 2D grayscale image array."""
    gray_img = np.full((10, 10), 100.0, dtype=np.float64)
    ch_hists = calculate_channel_histograms(gray_img)

    assert "gray" in ch_hists
    counts, bin_edges = ch_hists["gray"]
    assert len(counts) == 256
    assert len(bin_edges) == 257
    assert np.sum(counts) == 100


def test_calculate_channel_histograms_rgb():
    """Test calculate_channel_histograms for 3D RGB image array."""
    rgb_img = np.zeros((10, 10, 3), dtype=np.float64)
    rgb_img[:, :, 0] = 50.0   # Red
    rgb_img[:, :, 1] = 100.0  # Green
    rgb_img[:, :, 2] = 200.0  # Blue

    ch_hists = calculate_channel_histograms(rgb_img)

    assert set(ch_hists.keys()) == {"red", "green", "blue"}
    for ch_name in ("red", "green", "blue"):
        counts, bin_edges = ch_hists[ch_name]
        assert len(counts) == 256
        assert len(bin_edges) == 257
        assert np.sum(counts) == 100


def test_calculate_image_statistics_basic():
    """Test spatial statistics calculation on known synthetic matrix."""
    img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    stats = calculate_image_statistics(img)

    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["mean"] == 25.0
    assert np.isclose(stats["std"], np.std(img))
    assert np.isclose(stats["rms_energy"], np.sqrt(np.mean(img ** 2)))
    assert "channels" in stats
    assert "gray" in stats["channels"]


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


def test_calculate_image_statistics_rgb():
    """Test image statistics calculation on a 3D RGB array."""
    rgb_img = np.zeros((10, 10, 3), dtype=np.float64)
    rgb_img[:, :, 0] = 10.0  # Red
    rgb_img[:, :, 1] = 20.0  # Green
    rgb_img[:, :, 2] = 30.0  # Blue

    stats = calculate_image_statistics(rgb_img)

    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    assert np.isclose(stats["mean"], 20.0)
    assert np.isclose(stats["std"], np.std(rgb_img))
    assert "channels" in stats
    assert set(stats["channels"].keys()) == {"red", "green", "blue"}
    assert stats["channels"]["red"]["mean"] == 10.0
    assert stats["channels"]["green"]["mean"] == 20.0
    assert stats["channels"]["blue"]["mean"] == 30.0


def test_calculate_channel_statistics_grayscale():
    """Test calculate_channel_statistics for 2D grayscale image array."""
    img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    ch_stats = calculate_channel_statistics(img)

    assert set(ch_stats.keys()) == {"gray"}
    assert ch_stats["gray"]["min"] == 10.0
    assert ch_stats["gray"]["max"] == 40.0
    assert ch_stats["gray"]["mean"] == 25.0
    assert np.isclose(ch_stats["gray"]["std"], np.std(img))


def test_calculate_channel_statistics_rgb():
    """Test calculate_channel_statistics for 3D RGB image array."""
    rgb = np.zeros((4, 4, 3), dtype=np.float64)
    rgb[:, :, 0] = np.full((4, 4), 10.0)
    rgb[:, :, 1] = np.full((4, 4), 50.0)
    rgb[:, :, 2] = np.full((4, 4), 100.0)

    ch_stats = calculate_channel_statistics(rgb)

    assert set(ch_stats.keys()) == {"red", "green", "blue"}
    assert ch_stats["red"] == {"min": 10.0, "max": 10.0, "mean": 10.0, "std": 0.0}
    assert ch_stats["green"] == {"min": 50.0, "max": 50.0, "mean": 50.0, "std": 0.0}
    assert ch_stats["blue"] == {"min": 100.0, "max": 100.0, "mean": 100.0, "std": 0.0}


def test_calculate_image_statistics_invalid_input():
    """Test error handling for invalid input in statistics calculation."""
    with pytest.raises(TypeError):
        calculate_image_statistics([1, 2, 3])

    with pytest.raises(ValueError):
        calculate_image_statistics(np.array([[]]))

    with pytest.raises(ValueError):
        calculate_image_statistics(np.zeros((10,)))  # 1D array unsupported

    with pytest.raises(ValueError):
        calculate_image_statistics(np.zeros((10, 10, 5)))  # Unsupported 3D shape

