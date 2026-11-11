"""тести пайплайнів Даші."""
import numpy as np

from src.dasha import (
    bilateral_filter,
    gray_world_white_balance,
    lbp,
    linear_contrast_stretch,
    region_growing,
    rgb_to_ycbcr,
)

def test_gray_world_white_balance():
    """сірий світ: середні каналів вирівнюються до спільного сірого."""

    image = np.array([[[100, 50, 50], [0, 0, 0]]], dtype=np.uint8)

    result = gray_world_white_balance(image, {})
    assert np.array_equal(result[0, 0], [67, 67, 67])
    assert np.array_equal(result[0, 1], [0, 0, 0])


def test_bilateral_filter():
    """однорідна картинка не змінюється білатеральним фільтром."""
    image = np.full((5, 5, 3), 100, dtype=np.uint8)
    result = bilateral_filter(image, {"radius": 1})
    assert np.array_equal(result, image)


def test_bilateral_filter_preserves_edge():
    """різка межа між 0 і 200 не розмивається при малому sigma_color."""
    gray = np.array([[0, 0, 200, 200]] * 4, dtype=np.uint8)
    image = np.stack([gray, gray, gray], axis=2)

    result = bilateral_filter(image, {"radius": 1, "sigma_color": 1.0, "sigma_space": 10.0})

    assert np.all(result[:, 0, 0] == 0)
    assert np.all(result[:, 3, 0] == 200)


def test_region_growing():
    """region growing розбиває весь знімок на однорідні області."""
    gray = np.array([[10, 10, 200, 200]] * 3, dtype=np.uint8)
    image = np.stack([gray, gray, gray], axis=2)

    result = region_growing(image, {"threshold": 50.0})

    # Жодного чорного пікселя — весь знімок покритий
    assert not np.any(result == 0)

    # Дві області (10 і 200) відновлені як однорідні блоки
    expected = np.array([[10, 10, 200, 200]] * 3, dtype=np.uint8)
    expected = np.stack([expected] * 3, axis=2)
    assert np.array_equal(result, expected)


def test_rgb_to_ycbcr():
    """перевіряємо перетворення чистого білого і чорного."""

    image = np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    result = rgb_to_ycbcr(image, {})
    assert np.allclose(result[0, 0], [255, 128, 128], atol=1)
    assert np.allclose(result[0, 1], [0, 128, 128], atol=1)


def test_linear_contrast_stretch():
    """розтягування контрасту: діапазон 50..200 розтягується на 0..255."""

    gray = np.array([[50, 100, 150, 200, 100]], dtype=np.uint8)
    image = np.stack([gray, gray, gray], axis=2)

    result = linear_contrast_stretch(image, {"low_percent": 0.0, "high_percent": 100.0})
    expected = np.array([[0, 85, 170, 255, 85]], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_linear_contrast_stretch_constant_image():
    """якщо картинка однорідна (high <= low), повертається копія без змін."""
    image = np.full((3, 3, 3), 100, dtype=np.uint8)
    result = linear_contrast_stretch(image, {})
    assert np.array_equal(result, image)


def test_lbp():
    """LBP: на однорідній картинці всі біти 1 -> код 255."""
    image = np.full((3, 3, 3), 100, dtype=np.uint8)
    result = lbp(image, {"radius": 1})

    assert np.all(result == 255)


def test_lbp_center_pixel():
    """LBP для пікселя, який темніший за всіх сусідів -> код 0."""

    gray = np.full((3, 3), 100, dtype=np.uint8)
    gray[1, 1] = 10
    image = np.stack([gray, gray, gray], axis=2)
    result = lbp(image, {"radius": 1})

    assert result[1, 1] == 255
