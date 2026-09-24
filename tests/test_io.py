"""
Unit tests for Image Input/Output operations (load_image, convert_to_grayscale, save_image).

Coverage includes:
- Saving and loading PNG, JPG, and BMP image formats.
- Converting RGB/RGBA 3D arrays, 2D arrays, and PIL Images to float64 grayscale.
- Safe uint8 conversion with clipping and normalization.
- Error handling for missing files and malformed arrays.
- Re-export import consistency check.
"""

import pytest
import numpy as np
from PIL import Image
from src.io.image_io import load_image, convert_to_grayscale, save_image
from app.io.image_io import (
    load_image as app_load_image,
    convert_to_grayscale as app_convert_to_grayscale,
    save_image as app_save_image,
)


def test_io_module_import_consistency():
    """Verify consistent re-exports between app.io.image_io and src.io.image_io."""
    assert load_image is app_load_image
    assert convert_to_grayscale is app_convert_to_grayscale
    assert save_image is app_save_image


def test_convert_to_grayscale_rgb_array():
    """Verify Luma formula Y = 0.299*R + 0.587*G + 0.114*B on RGB array."""
    # Red=255, Green=0, Blue=0 -> expected 0.299 * 255 = 76.245
    rgb_arr = np.zeros((4, 4, 3), dtype=np.float64)
    rgb_arr[:, :, 0] = 255.0

    gray = convert_to_grayscale(rgb_arr)
    assert gray.shape == (4, 4)
    assert gray.dtype == np.float64
    np.testing.assert_allclose(gray, 76.245, rtol=1e-5)


def test_convert_to_grayscale_rgba_array():
    """Verify grayscale conversion on 4-channel RGBA array."""
    rgba_arr = np.zeros((4, 4, 4), dtype=np.float64)
    rgba_arr[:, :, 1] = 100.0  # Green channel = 100, alpha = 0
    # Expected 0.587 * 100 = 58.7
    gray = convert_to_grayscale(rgba_arr)
    assert gray.shape == (4, 4)
    np.testing.assert_allclose(gray, 58.7, rtol=1e-5)


def test_convert_to_grayscale_pil_image():
    """Verify grayscale conversion from PIL Image object."""
    pil_img = Image.new('RGB', (10, 10), color=(100, 150, 200))
    gray = convert_to_grayscale(pil_img)

    assert isinstance(gray, np.ndarray)
    assert gray.shape == (10, 10)
    assert gray.dtype == np.float64


def test_convert_to_grayscale_already_2d():
    """Passing an already 2D array should return a float64 array of same values."""
    img_2d = np.array([[10, 20], [30, 40]], dtype=np.float32)
    gray = convert_to_grayscale(img_2d)

    assert gray.shape == (2, 2)
    assert gray.dtype == np.float64
    np.testing.assert_array_equal(gray, [[10.0, 20.0], [30.0, 40.0]])


def test_save_and_load_png(tmp_path):
    """Test saving 2D float64 array as PNG and reloading it."""
    file_path = tmp_path / "test_image.png"
    original_data = np.array([
        [0.0, 64.0, 128.0],
        [192.0, 255.0, 100.0]
    ], dtype=np.float64)

    save_image(file_path, original_data)
    assert file_path.exists()

    loaded = load_image(file_path, as_grayscale=True)
    assert loaded.shape == (2, 3)
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, original_data)


def test_save_and_load_bmp(tmp_path):
    """Test saving and loading BMP format."""
    file_path = tmp_path / "test_image.bmp"
    original_data = np.full((5, 5), 120.0, dtype=np.float64)

    save_image(file_path, original_data)
    assert file_path.exists()

    loaded = load_image(file_path, as_grayscale=True)
    np.testing.assert_array_equal(loaded, original_data)


def test_save_and_load_jpg(tmp_path):
    """Test saving and loading JPEG format (allowing lossy compression variance)."""
    file_path = tmp_path / "test_image.jpg"
    original_data = np.full((10, 10), 128.0, dtype=np.float64)

    save_image(file_path, original_data)
    assert file_path.exists()

    loaded = load_image(file_path, as_grayscale=True)
    assert loaded.shape == (10, 10)
    # JPEG is lossy, so check values are close to 128
    np.testing.assert_allclose(loaded, 128.0, atol=5.0)


def test_save_image_clipping_and_normalization(tmp_path):
    """Test float clipping and min-max normalization when saving."""
    file_path = tmp_path / "clipped.png"
    out_of_bounds = np.array([[-50.0, 300.0], [100.0, 150.0]], dtype=np.float64)

    # Standard clipping clamps -50 -> 0 and 300 -> 255
    save_image(file_path, out_of_bounds, clip=True)
    loaded = load_image(file_path, as_grayscale=True)
    expected_clipped = np.array([[0.0, 255.0], [100.0, 150.0]], dtype=np.float64)
    np.testing.assert_array_equal(loaded, expected_clipped)

    # Test normalization (maps [-50, 300] to [0, 255])
    norm_path = tmp_path / "normalized.png"
    save_image(norm_path, out_of_bounds, normalize=True)
    loaded_norm = load_image(norm_path, as_grayscale=True)
    assert loaded_norm.min() == 0.0
    assert loaded_norm.max() == 255.0


def test_io_error_handling(tmp_path):
    """Test exception raising for invalid file paths or arguments."""
    non_existent = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(non_existent)

    with pytest.raises(TypeError, match="Image to save must be a NumPy array"):
        save_image(tmp_path / "bad.png", "not an array")

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        save_image(tmp_path / "bad.png", np.ones((2, 2, 2, 2)))

    with pytest.raises(TypeError, match="must be a PIL Image or NumPy array"):
        convert_to_grayscale(12345)


def test_load_grayscale_image_preserves_shape(tmp_path):
    """Loading a grayscale file produces a NumPy array with shape (H, W)."""
    file_path = tmp_path / "gray.png"
    img_pil = Image.new("L", (8, 6), color=128)  # W=8, H=6
    img_pil.save(file_path)

    loaded = load_image(file_path)
    assert loaded.ndim == 2
    assert loaded.shape == (6, 8)
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, 128.0)


def test_load_rgb_image_preserves_shape(tmp_path):
    """Loading an RGB file produces a NumPy array with shape (H, W, 3)."""
    file_path = tmp_path / "rgb.png"
    img_pil = Image.new("RGB", (8, 6), color=(255, 128, 64))  # W=8, H=6
    img_pil.save(file_path)

    loaded = load_image(file_path)
    assert loaded.ndim == 3
    assert loaded.shape == (6, 8, 3)
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded[:, :, 0], 255.0)
    np.testing.assert_array_equal(loaded[:, :, 1], 128.0)
    np.testing.assert_array_equal(loaded[:, :, 2], 64.0)


def test_load_rgba_image_converts_to_rgb(tmp_path):
    """Loading an RGBA image converts it to RGB shape (H, W, 3)."""
    file_path = tmp_path / "rgba.png"
    img_pil = Image.new("RGBA", (10, 10), color=(100, 150, 200, 255))
    img_pil.save(file_path)

    loaded = load_image(file_path)
    assert loaded.ndim == 3
    assert loaded.shape == (10, 10, 3)
    assert loaded.dtype == np.float64


def test_save_and_load_rgb_output(tmp_path):
    """Saving 3D RGB float64 array saves valid RGB image and reloads as RGB."""
    file_path = tmp_path / "saved_rgb.png"
    rgb_arr = np.zeros((4, 5, 3), dtype=np.float64)
    rgb_arr[:, :, 0] = 200.0  # R
    rgb_arr[:, :, 1] = 100.0  # G
    rgb_arr[:, :, 2] = 50.0   # B

    save_image(file_path, rgb_arr)
    assert file_path.exists()

    loaded = load_image(file_path)
    assert loaded.shape == (4, 5, 3)
    np.testing.assert_array_equal(loaded[:, :, 0], 200.0)
    np.testing.assert_array_equal(loaded[:, :, 1], 100.0)
    np.testing.assert_array_equal(loaded[:, :, 2], 50.0)

    # Verify PIL image mode on disk is RGB
    with Image.open(file_path) as disk_img:
        assert disk_img.mode == "RGB"


def test_save_and_load_grayscale_output(tmp_path):
    """Saving 2D float64 array saves valid grayscale image ('L') and reloads as grayscale."""
    file_path = tmp_path / "saved_gray.png"
    gray_arr = np.full((4, 5), 180.0, dtype=np.float64)

    save_image(file_path, gray_arr)
    assert file_path.exists()

    loaded = load_image(file_path)
    assert loaded.shape == (4, 5)
    assert loaded.ndim == 2
    np.testing.assert_array_equal(loaded, 180.0)

    # Verify PIL image mode on disk is L
    with Image.open(file_path) as disk_img:
        assert disk_img.mode == "L"


def test_load_image_explicit_as_grayscale_flag(tmp_path):
    """Explicitly passing as_grayscale=True converts RGB file to 2D grayscale."""
    file_path = tmp_path / "rgb_to_gray.png"
    img_pil = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img_pil.save(file_path)

    loaded = load_image(file_path, as_grayscale=True)
    assert loaded.ndim == 2
    assert loaded.shape == (10, 10)
    np.testing.assert_allclose(loaded, 76.0, atol=1.0)

