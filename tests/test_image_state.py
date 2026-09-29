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


def test_one_filter_followed_by_undo():
    """Verify applying one filter followed by Undo restores previous current_image state."""
    img = np.full((10, 10), 50.0, dtype=np.float64)
    state = ImageState(img)

    filtered = np.full((10, 10), 100.0, dtype=np.float64)
    state.update_current(filtered, description="Box Blur (3x3)")

    assert state.can_undo
    assert len(state.history) == 1
    np.testing.assert_array_equal(state.current_image, filtered)

    restored = state.undo()
    assert restored is not None
    np.testing.assert_array_equal(restored, img)
    np.testing.assert_array_equal(state.current_image, img)
    np.testing.assert_array_equal(state.original_image, img)
    assert len(state.history) == 0
    assert not state.can_undo


def test_two_filters_followed_by_two_undo_operations():
    """Verify applying two filters followed by two Undo operations unwinds state step by step."""
    img = np.full((10, 10), 10.0, dtype=np.float64)
    state = ImageState(img)

    step1 = np.full((10, 10), 20.0, dtype=np.float64)
    state.update_current(step1, description="Step 1 Blur")

    step2 = np.full((10, 10), 30.0, dtype=np.float64)
    state.update_current(step2, description="Step 2 Sharpen")

    assert len(state.history) == 2
    assert state.history == ["Step 1 Blur", "Step 2 Sharpen"]
    np.testing.assert_array_equal(state.current_image, step2)

    # First Undo
    undone1 = state.undo()
    np.testing.assert_array_equal(undone1, step1)
    np.testing.assert_array_equal(state.current_image, step1)
    assert state.history == ["Step 1 Blur"]
    np.testing.assert_array_equal(state.original_image, img)

    # Second Undo
    undone2 = state.undo()
    np.testing.assert_array_equal(undone2, img)
    np.testing.assert_array_equal(state.current_image, img)
    assert len(state.history) == 0
    np.testing.assert_array_equal(state.original_image, img)


def test_undo_with_no_history():
    """Verify calling Undo with an empty history returns None without altering state."""
    img = np.full((5, 5), 42.0, dtype=np.float64)
    state = ImageState(img)

    assert not state.can_undo
    assert state.undo() is None
    np.testing.assert_array_equal(state.current_image, img)
    np.testing.assert_array_equal(state.original_image, img)
    assert len(state.history) == 0

    # Also test empty state (no image loaded)
    empty_state = ImageState()
    assert not empty_state.can_undo
    assert empty_state.undo() is None


def test_reset_clearing_history():
    """Verify reset restores current_image to original_image and clears processing history."""
    img = np.full((8, 8), 15.0, dtype=np.float64)
    state = ImageState(img)

    state.update_current(np.full((8, 8), 25.0), description="Filter 1")
    state.update_current(np.full((8, 8), 35.0), description="Filter 2")
    assert len(state.history) == 2

    state.reset()
    assert len(state.history) == 0
    assert not state.can_undo
    np.testing.assert_array_equal(state.current_image, img)
    np.testing.assert_array_equal(state.original_image, img)


def test_loading_new_image_clearing_history():
    """Verify loading a new image replaces session state and clears history stack."""
    img1 = np.full((6, 6), 5.0, dtype=np.float64)
    state = ImageState(img1)

    state.update_current(np.full((6, 6), 10.0), description="Filter A")
    assert len(state.history) == 1

    img2 = np.full((12, 12, 3), 200.0, dtype=np.float64)
    state.set_image(img2)

    assert len(state.history) == 0
    assert not state.can_undo
    np.testing.assert_array_equal(state.original_image, img2)
    np.testing.assert_array_equal(state.current_image, img2)


def test_failed_processing_not_changing_history():
    """Verify failed filter application attempts do not alter history stack or image states."""
    img = np.full((10, 10), 100.0, dtype=np.float64)
    state = ImageState(img)

    step1 = np.full((10, 10), 150.0, dtype=np.float64)
    state.update_current(step1, description="Valid Blur")
    assert len(state.history) == 1

    # Attempt 1: Attempt to update with invalid array type
    with pytest.raises(TypeError):
        state.update_current("invalid_array", description="Failed Filter")

    assert len(state.history) == 1
    np.testing.assert_array_equal(state.current_image, step1)
    np.testing.assert_array_equal(state.original_image, img)

    # Attempt 2: Attempt to update with empty array
    with pytest.raises(ValueError):
        state.update_current(np.array([]), description="Failed Filter")

    assert len(state.history) == 1
    np.testing.assert_array_equal(state.current_image, step1)
    np.testing.assert_array_equal(state.original_image, img)


def test_original_image_remaining_unchanged():
    """Verify original_image remains strictly immutable through filters, undos, and resets."""
    img = np.random.uniform(0, 255, (10, 10)).astype(np.float64)
    initial_copy = img.copy()
    state = ImageState(img)

    # Filter 1
    state.update_current(np.full((10, 10), 50.0), description="Filter 1")
    np.testing.assert_array_equal(state.original_image, initial_copy)

    # Filter 2
    state.update_current(np.full((10, 10), 99.0), description="Filter 2")
    np.testing.assert_array_equal(state.original_image, initial_copy)

    # Undo 1
    state.undo()
    np.testing.assert_array_equal(state.original_image, initial_copy)

    # Reset
    state.reset()
    np.testing.assert_array_equal(state.original_image, initial_copy)


def test_applying_filter_after_undo_and_reset():
    """Verify applying filters after Undo or Reset correctly builds history from restored state."""
    img = np.full((5, 5), 10.0, dtype=np.float64)
    state = ImageState(img)

    # Filter 1 then Undo
    state.update_current(np.full((5, 5), 20.0), description="Filter 1")
    state.undo()
    assert len(state.history) == 0

    # Filter 2 after Undo
    f2_res = np.full((5, 5), 30.0, dtype=np.float64)
    state.update_current(f2_res, description="Filter 2")
    assert state.history == ["Filter 2"]
    np.testing.assert_array_equal(state.current_image, f2_res)


    # Undo again
    state.undo()
    np.testing.assert_array_equal(state.current_image, img)

    # Filter 3 then Reset
    state.update_current(np.full((5, 5), 40.0), description="Filter 3")
    state.reset()
    assert len(state.history) == 0

    # Filter 4 after Reset
    f4_res = np.full((5, 5), 50.0, dtype=np.float64)
    state.update_current(f4_res, description="Filter 4")
    assert state.history == ["Filter 4"]
    np.testing.assert_array_equal(state.current_image, f4_res)


def test_previous_image_tracking_through_chain_and_undo():
    """Verify previous_image exposes the image right before the latest filter, including after undo."""
    img = np.full((10, 10), 10.0, dtype=np.float64)
    state = ImageState(img)

    # No filter applied yet -> no previous image
    assert state.previous_image is None

    step1 = np.full((10, 10), 20.0, dtype=np.float64)
    state.update_current(step1, description="Filter 1")
    np.testing.assert_array_equal(state.previous_image, img)

    # Chained second filter: previous image is step1, NOT the original image
    step2 = np.full((10, 10), 30.0, dtype=np.float64)
    state.update_current(step2, description="Filter 2")
    np.testing.assert_array_equal(state.previous_image, step1)
    np.testing.assert_array_equal(state.current_image, step2)

    # Undo: previous image becomes the input of the newest remaining step
    state.undo()
    np.testing.assert_array_equal(state.current_image, step1)
    np.testing.assert_array_equal(state.previous_image, img)

    # Undo everything: chain empty -> no previous image
    state.undo()
    assert state.previous_image is None

    # Reset also clears previous image availability
    state.update_current(np.full((10, 10), 40.0), description="Filter 3")
    state.reset()
    assert state.previous_image is None


def test_previous_image_independent_copy():
    """Verify previous_image returns an independent copy safe from external mutation."""
    img = np.full((8, 8), 60.0, dtype=np.float64)
    state = ImageState(img)
    state.update_current(np.full((8, 8), 90.0), description="Blur")

    prev_copy = state.previous_image
    prev_copy[0, 0] = 999.0

    # Internal history snapshot must remain unaffected
    assert state.previous_image[0, 0] == 60.0
    np.testing.assert_array_equal(state.previous_image, img)


def test_history_operation_metadata():
    """Verify useful operation metadata is captured in HistoryEntry."""
    img = np.zeros((10, 10, 3), dtype=np.float64)
    state = ImageState(img)

    meta_dict = {"filter_type": "blur", "kernel_size": (3, 3), "parameters": {"iterations": 2}}
    state.update_current(img + 10, description="Gaussian Blur (3x3)", metadata=meta_dict)

    entries = state.history_entries
    assert len(entries) == 1
    entry = entries[0]
    assert entry.description == "Gaussian Blur (3x3)"
    assert entry.metadata["filter_type"] == "blur"
    assert entry.metadata["kernel_size"] == (3, 3)
    assert entry.metadata["parameters"]["iterations"] == 2
    assert entry.metadata["image_mode"] == "RGB"

