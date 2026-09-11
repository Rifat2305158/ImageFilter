"""
Unit tests for edge detection filters (Sobel, Prewitt, Roberts, Laplacian).

Coverage includes:
- Verification of horizontal response (Gx) on synthetic vertical edges.
- Verification of vertical response (Gy) on synthetic horizontal edges.
- Verification of cross-axis zero-responses.
- Exact magnitude calculation verification: G = sqrt(Gx^2 + Gy^2).
- Constant image response (should be zero for derivative operators).
- Laplacian 2nd derivative response on step edges.
- Input validation and exception raising.
- Import consistency across app and src package aliases.
"""

import pytest
import numpy as np
from src.filters.edge import (
    apply_sobel,
    apply_prewitt,
    apply_roberts,
    apply_laplacian,
    apply_edge,
)
from app.filters.edges import (
    apply_sobel as app_apply_sobel,
    apply_prewitt as app_apply_prewitt,
    apply_roberts as app_apply_roberts,
    apply_laplacian as app_apply_laplacian,
    apply_edge as app_apply_edge,
)


def test_edge_module_import_consistency():
    """Verify that app.filters.edges re-exports the identical functions from src.filters.edge."""
    assert apply_sobel is app_apply_sobel
    assert apply_prewitt is app_apply_prewitt
    assert apply_roberts is app_apply_roberts
    assert apply_laplacian is app_apply_laplacian
    assert apply_edge is app_apply_edge


def test_constant_image_zero_response():
    """Constant intensity images have zero spatial derivatives, yielding 0 edge responses."""
    constant_img = np.full((10, 10), 150.0, dtype=np.float64)

    for op in ['sobel', 'prewitt', 'roberts']:
        for direction in ['horizontal', 'vertical', 'combined']:
            res = apply_edge(constant_img, operator=op, direction=direction, padding_mode='edge')
            np.testing.assert_array_almost_equal(
                res, 
                0.0, 
                err_msg=f"Operator {op} ({direction}) returned non-zero for constant image."
            )

    lap_res = apply_laplacian(constant_img, padding_mode='edge')
    np.testing.assert_array_almost_equal(lap_res, 0.0, err_msg="Laplacian returned non-zero for constant image.")


def test_sobel_synthetic_edges():
    """
    Test Sobel response on synthetic vertical and horizontal edges:
    - Vertical edge (X step): Gx != 0 at edge, Gy == 0 everywhere.
    - Horizontal edge (Y step): Gy != 0 at edge, Gx == 0 everywhere.
    """
    # 1. Vertical step edge (intensity changes horizontally across columns)
    vert_edge = np.zeros((10, 10), dtype=np.float64)
    vert_edge[:, 5:] = 100.0

    gx_v = apply_sobel(vert_edge, direction='horizontal', padding_mode='edge')
    gy_v = apply_sobel(vert_edge, direction='vertical', padding_mode='edge')

    # Horizontal gradient (Gx) must detect vertical edge at column 4/5
    assert np.any(np.abs(gx_v) > 0.0), "Sobel Gx failed to detect vertical edge."
    # Vertical gradient (Gy) must be 0 everywhere for vertical edge
    np.testing.assert_array_almost_equal(gy_v, 0.0, err_msg="Sobel Gy gave false response on vertical edge.")

    # 2. Horizontal step edge (intensity changes vertically across rows)
    horiz_edge = np.zeros((10, 10), dtype=np.float64)
    horiz_edge[5:, :] = 100.0

    gx_h = apply_sobel(horiz_edge, direction='horizontal', padding_mode='edge')
    gy_h = apply_sobel(horiz_edge, direction='vertical', padding_mode='edge')

    # Vertical gradient (Gy) must detect horizontal edge at row 4/5
    assert np.any(np.abs(gy_h) > 0.0), "Sobel Gy failed to detect horizontal edge."
    # Horizontal gradient (Gx) must be 0 everywhere for horizontal edge
    np.testing.assert_array_almost_equal(gx_h, 0.0, err_msg="Sobel Gx gave false response on horizontal edge.")


def test_prewitt_synthetic_edges():
    """
    Test Prewitt response on synthetic vertical and horizontal edges.
    """
    vert_edge = np.zeros((10, 10), dtype=np.float64)
    vert_edge[:, 5:] = 100.0

    gx_v = apply_prewitt(vert_edge, direction='horizontal', padding_mode='edge')
    gy_v = apply_prewitt(vert_edge, direction='vertical', padding_mode='edge')

    assert np.any(np.abs(gx_v) > 0.0), "Prewitt Gx failed to detect vertical edge."
    np.testing.assert_array_almost_equal(gy_v, 0.0)

    horiz_edge = np.zeros((10, 10), dtype=np.float64)
    horiz_edge[5:, :] = 100.0

    gx_h = apply_prewitt(horiz_edge, direction='horizontal', padding_mode='edge')
    gy_h = apply_prewitt(horiz_edge, direction='vertical', padding_mode='edge')

    assert np.any(np.abs(gy_h) > 0.0), "Prewitt Gy failed to detect horizontal edge."
    np.testing.assert_array_almost_equal(gx_h, 0.0)


def test_roberts_synthetic_edges():
    """
    Test Roberts Cross response on synthetic diagonal step edges.
    """
    vert_edge = np.zeros((10, 10), dtype=np.float64)
    vert_edge[:, 5:] = 100.0

    gx = apply_roberts(vert_edge, direction='x', padding_mode='edge')
    gy = apply_roberts(vert_edge, direction='y', padding_mode='edge')
    g_comb = apply_roberts(vert_edge, direction='combined', padding_mode='edge')

    # Roberts cross detects change across diagonal neighbors
    assert np.any(np.abs(gx) > 0.0)
    assert np.any(np.abs(gy) > 0.0)
    assert np.all(g_comb >= 0.0)


def test_magnitude_calculation():
    """
    Verify that combined magnitude is strictly equal to sqrt(Gx^2 + Gy^2).
    """
    np.random.seed(123)
    rand_img = np.random.uniform(0.0, 255.0, (12, 12)).astype(np.float64)

    # Test for Sobel
    gx_s = apply_sobel(rand_img, direction='horizontal', padding_mode='reflect')
    gy_s = apply_sobel(rand_img, direction='vertical', padding_mode='reflect')
    expected_g_s = np.sqrt(gx_s**2 + gy_s**2)
    actual_g_s = apply_sobel(rand_img, direction='combined', padding_mode='reflect')
    np.testing.assert_array_almost_equal(actual_g_s, expected_g_s, err_msg="Sobel magnitude mismatch.")

    # Test for Prewitt
    gx_p = apply_prewitt(rand_img, direction='horizontal', padding_mode='reflect')
    gy_p = apply_prewitt(rand_img, direction='vertical', padding_mode='reflect')
    expected_g_p = np.sqrt(gx_p**2 + gy_p**2)
    actual_g_p = apply_prewitt(rand_img, direction='combined', padding_mode='reflect')
    np.testing.assert_array_almost_equal(actual_g_p, expected_g_p, err_msg="Prewitt magnitude mismatch.")

    # Test for Roberts
    gx_r = apply_roberts(rand_img, direction='x', padding_mode='reflect')
    gy_r = apply_roberts(rand_img, direction='y', padding_mode='reflect')
    expected_g_r = np.sqrt(gx_r**2 + gy_r**2)
    actual_g_r = apply_roberts(rand_img, direction='combined', padding_mode='reflect')
    np.testing.assert_array_almost_equal(actual_g_r, expected_g_r, err_msg="Roberts magnitude mismatch.")


def test_laplacian_edge():
    """
    Laplacian is a 2nd-order isotropic derivative operator.
    On a step edge, it produces positive response on one side and negative on the other (zero crossing).
    """
    step_edge = np.zeros((8, 8), dtype=np.float64)
    step_edge[:, 4:] = 100.0

    lap_res = apply_laplacian(step_edge, padding_mode='edge')

    # Away from edge, response is 0
    np.testing.assert_array_almost_equal(lap_res[:, 1], 0.0)
    np.testing.assert_array_almost_equal(lap_res[:, 6], 0.0)

    # Near edge, 2nd derivative has opposite signs across the transition boundary
    assert np.any(lap_res[:, 3] < 0.0) or np.any(lap_res[:, 3] > 0.0)
    assert np.any(lap_res[:, 4] < 0.0) or np.any(lap_res[:, 4] > 0.0)


def test_edge_detection_input_validation():
    """Verify ValueError / TypeError on invalid parameter inputs."""
    img = np.ones((5, 5))

    with pytest.raises(ValueError, match="Invalid direction"):
        apply_sobel(img, direction='diagonal')

    with pytest.raises(ValueError, match="Invalid direction"):
        apply_prewitt(img, direction='diagonal')

    with pytest.raises(ValueError, match="Invalid direction"):
        apply_roberts(img, direction='diagonal')

    with pytest.raises(ValueError, match="Unsupported edge operator"):
        apply_edge(img, operator='canny')

    with pytest.raises(ValueError, match="Input image must be a 2D array"):
        apply_sobel(np.ones((5, 5, 3)))

    with pytest.raises(TypeError, match="Input image must be a numpy array"):
        apply_sobel([[1, 2], [3, 4]])
