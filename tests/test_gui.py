"""
Automated unit tests for GUI components and main window state logic.
"""

import pytest
import numpy as np
import tkinter as tk
from app.gui.header import Header
from app.gui.status_bar import StatusBar
from app.gui.image_view import ImageView
from app.gui.image_area import ImageArea
from app.gui.filter_panel import FilterPanel
from app.gui.main_window import MainWindow


@pytest.fixture
def tk_root():
    """Pytest fixture providing a headless Tk root window."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in test environment.")


def test_header_widget(tk_root):
    """Test Header widget creation."""
    header = Header(tk_root)
    assert header is not None


def test_status_bar_widget(tk_root):
    """Test StatusBar set_status, set_info, and clear methods."""
    status_bar = StatusBar(tk_root)
    status_bar.set_status("Processing complete")
    status_bar.set_info("Dimensions: 100x100")

    assert status_bar.status_label.cget("text") == "Processing complete"
    assert status_bar.info_label.cget("text") == "Dimensions: 100x100"

    status_bar.clear()
    assert status_bar.status_label.cget("text") == "Ready"
    assert status_bar.info_label.cget("text") == "No Image Loaded"


def test_image_view_widget(tk_root):
    """Test ImageView set_image, get_image, and clear methods."""
    view = ImageView(tk_root, title="Test View")
    assert view.get_image() is None

    img_data = np.full((20, 20), 100.0, dtype=np.float64)
    view.set_image(img_data)

    assert view.get_image() is not None
    assert view._photo_image is not None
    np.testing.assert_array_equal(view.get_image(), img_data)

    view.clear()
    assert view.get_image() is None
    assert view._photo_image is None


def test_image_area_widget(tk_root):
    """Test ImageArea view mode selection, difference calculation, and view toggle."""
    area = ImageArea(tk_root)
    assert area.get_view_mode() == "Side-by-Side"

    orig = np.full((10, 10), 100.0, dtype=np.float64)
    proc = np.full((10, 10), 150.0, dtype=np.float64)

    area.set_images(orig, proc)

    # Difference should be absolute difference 50.0
    assert area._diff_image is not None
    np.testing.assert_array_almost_equal(area._diff_image, np.full((10, 10), 50.0))

    # Toggle view switching
    area.toggle_view()
    assert area.get_view_mode() in ("Original Only", "Processed Only")

    area.view_mode_var.set("Difference Image")
    area._apply_view_mode()
    assert area.get_view_mode() == "Difference Image"


def test_filter_panel_selection(tk_root):
    """Test FilterPanel selection and button callbacks."""
    selected_filter = []

    def on_apply_cb(filter_name):
        selected_filter.append(filter_name)

    panel = FilterPanel(tk_root, on_apply=on_apply_cb)

    assert panel.get_selected_filter() in FilterPanel.AVAILABLE_FILTERS

    panel._on_apply_click()
    assert len(selected_filter) == 1


def test_edge_detection_gui_controls(tk_root):
    """Test FilterPanel edge detection subpanel controls and getters."""
    panel = FilterPanel(tk_root)
    panel.filter_var.set("Edge Detection")
    panel._on_filter_changed(None)

    assert panel.get_selected_filter() == "Edge Detection"
    assert panel.get_edge_operator() in ["sobel", "prewitt", "roberts", "laplacian"]
    assert panel.get_edge_direction() in ["combined", "horizontal", "vertical"]
    assert isinstance(panel.get_edge_normalize(), bool)


def test_main_window_initialization():
    """Test MainWindow creation, layout components, and initial state."""
    try:
        window = MainWindow()
        window.withdraw()
    except tk.TclError:
        pytest.skip("Tkinter display not available in test environment.")

    assert window.original_image is None
    assert window.processed_image is None
    assert hasattr(window, "header")
    assert hasattr(window, "status_bar")
    assert hasattr(window, "control_panel")
    assert hasattr(window, "image_area")
    assert hasattr(window, "original_view")
    assert hasattr(window, "processed_view")

    # Set synthetic image and verify reset handler
    synth_img = np.zeros((10, 10), dtype=np.float64)
    window.original_image = synth_img
    window.processed_image = synth_img.copy() + 50.0

    window.handle_reset()
    np.testing.assert_array_equal(window.processed_image, synth_img)

    window.destroy()


def test_image_view_displays_rgb_array(tk_root):
    """ImageView must correctly render a 3D RGB array (H, W, 3) without errors."""
    view = ImageView(tk_root, title="RGB Test View")

    rgb_array = np.zeros((20, 20, 3), dtype=np.float64)
    rgb_array[:, :, 0] = 200.0  # red channel
    rgb_array[:, :, 1] = 100.0  # green channel
    rgb_array[:, :, 2] = 50.0   # blue channel

    # Must not raise – should produce a valid PhotoImage
    view.set_image(rgb_array)

    assert view.get_image() is not None
    assert view._photo_image is not None
    assert view.get_image().shape == (20, 20, 3)


def test_image_area_handles_rgb_images(tk_root):
    """ImageArea must compute difference image correctly for matching-shape RGB arrays."""
    area = ImageArea(tk_root)

    orig = np.full((10, 10, 3), 100.0, dtype=np.float64)
    proc = np.full((10, 10, 3), 150.0, dtype=np.float64)

    area.set_images(orig, proc)

    # Difference should be absolute difference 50.0 across all channels
    assert area._diff_image is not None
    assert area._diff_image.shape == (10, 10, 3)
    np.testing.assert_array_almost_equal(area._diff_image, np.full((10, 10, 3), 50.0))


def test_image_area_rgb_grayscale_mismatch(tk_root):
    """ImageArea should handle shape mismatch (RGB original, grayscale processed) gracefully."""
    area = ImageArea(tk_root)

    orig = np.full((10, 10, 3), 100.0, dtype=np.float64)
    proc = np.full((10, 10), 80.0, dtype=np.float64)   # edge detection output

    # Should not raise; diff_image should be None due to shape mismatch
    area.set_images(orig, proc)
    assert area._diff_image is None

