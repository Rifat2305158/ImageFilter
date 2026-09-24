"""
Unit tests for ImageState state management module (src.core.image_state).
"""

import pytest
import numpy as np
from src.core.image_state import ImageState


def test_image_state_initialization():
    """Test initializing original_image and current_image from loaded array."""
    synth_gray = np.full((20, 20), 100.0, dtype=np.float64)
    state = ImageState(synth_gray)

    assert state.is_loaded
    assert state.original_image is not None
    assert state.current_image is not None
    assert state.original_image.shape == (20, 20)
    assert state.current_image.shape == (20, 20)
    assert len(state.history) == 0
    np.testing.assert_array_equal(state.original_image, synth_gray)
    np.testing.assert_array_equal(state.current_image, synth_gray)


def test_image_state_independent_copies():
    """Verify that original_image and current_image are completely independent memory copies."""
    synth_gray = np.full((10, 10), 50.0, dtype=np.float64)
    state = ImageState(synth_gray)

    orig_copy = state.original_image
    curr_copy = state.current_image

    # Mutate retrieved array copies externally
    orig_copy[0, 0] = 999.0
    curr_copy[0, 0] = 777.0

    # Internal state must remain unaffected
    assert state.original_image[0, 0] == 50.0
    assert state.current_image[0, 0] == 50.0


def test_image_state_changing_current_without_changing_original():
    """Verify changing current_image via update_current does not alter original_image."""
    synth_gray = np.full((10, 10), 100.0, dtype=np.float64)
    state = ImageState(synth_gray)

    new_current = np.full((10, 10), 200.0, dtype=np.float64)
    state.update_current(new_current, description="Gaussian Blur")

    np.testing.assert_array_equal(state.original_image, np.full((10, 10), 100.0))
    np.testing.assert_array_equal(state.current_image, np.full((10, 10), 200.0))
    assert state.history == ["Gaussian Blur"]


def test_image_state_reset_restores_original():
    """Verify reset restores current_image from original_image and clears history."""
    synth_rgb = np.zeros((10, 10, 3), dtype=np.float64)
    synth_rgb[:, :, 0] = 50.0   # R
    synth_rgb[:, :, 1] = 100.0  # G
    synth_rgb[:, :, 2] = 150.0  # B

    state = ImageState(synth_rgb)

    # Process step 1
    step1 = synth_rgb + 20.0
    state.update_current(step1, description="Add Brightness")

    # Process step 2
    step2 = step1 * 1.5
    state.update_current(step2, description="Contrast Boost")

    assert len(state.history) == 2

    # Reset
    state.reset()

    assert len(state.history) == 0
    np.testing.assert_array_equal(state.current_image, synth_rgb)
    np.testing.assert_array_equal(state.original_image, synth_rgb)


def test_image_state_loading_new_image():
    """Verify loading a new image replaces previous session and resets history."""
    first_img = np.full((5, 5), 10.0, dtype=np.float64)
    state = ImageState(first_img)
    state.update_current(np.full((5, 5), 20.0), description="Blur")

    second_img = np.full((8, 8, 3), 128.0, dtype=np.float64)
    state.set_image(second_img)

    assert state.original_image.shape == (8, 8, 3)
    assert state.current_image.shape == (8, 8, 3)
    assert len(state.history) == 0
    np.testing.assert_array_equal(state.original_image, second_img)


def test_image_state_grayscale_shape_preservation():
    """Verify 2D grayscale shape (H, W) is strictly preserved."""
    gray = np.random.uniform(0, 255, (15, 25)).astype(np.float64)
    state = ImageState(gray)

    assert state.original_image.ndim == 2
    assert state.current_image.ndim == 2
    assert state.original_image.shape == (15, 25)
    assert state.current_image.shape == (15, 25)


def test_image_state_rgb_shape_preservation():
    """Verify 3D RGB shape (H, W, 3) is strictly preserved."""
    rgb = np.random.uniform(0, 255, (12, 18, 3)).astype(np.float64)
    state = ImageState(rgb)

    assert state.original_image.ndim == 3
    assert state.current_image.ndim == 3
    assert state.original_image.shape == (12, 18, 3)
    assert state.current_image.shape == (12, 18, 3)


def test_image_state_validation_errors():
    """Verify type and shape validation error handling."""
    state = ImageState()

    with pytest.raises(TypeError, match="Input image must be a NumPy array."):
        state.set_image("invalid_type")

    with pytest.raises(ValueError, match="cannot be empty"):
        state.set_image(np.array([]))

    with pytest.raises(ValueError, match="Unsupported image shape"):
        state.set_image(np.zeros((10,)))  # 1D array

    with pytest.raises(ValueError, match="Unsupported image shape"):
        state.set_image(np.zeros((10, 10, 4)))  # 4-channel array
