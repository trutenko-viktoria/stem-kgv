"""тести пайплайнів Віки."""
import numpy as np

from src.vika import equalize_histogram, gamma_correction, otsu_threshold


def test_gamma_correction():
    """при гаммі 0.5 темне світлішає сильно, а чорне і біле не змінюються."""
    image = np.array([[[0, 64, 255]] * 3] * 3, dtype=np.uint8) # картинка 3 x 3, канали 0, 64, 255
    result = gamma_correction(image, {"gamma": 0.5})
    assert np.allclose(result[0, 0], [0, 127.75, 255], atol=0.01) # 255 * (64 / 255) ^ 0.5 = 127.75


def test_equalize_histogram():
    """рівні 50, 100, 150 розтягуються на 0, 204, 255."""
    gray = np.array([[50, 50, 50], [50, 100, 100], [100, 100, 150]]) # чотири 50, чотири 100, одне 150
    image = np.stack([gray, gray, gray], axis=2).astype(np.uint8) # однакові r, g, b дають ту саму сіру
    expected = [[0, 0, 0], [0, 204, 204], [204, 204, 255]] # пораховано вручну за cdf
    assert np.array_equal(equalize_histogram(image, {}), expected)


def test_otsu_threshold():
    """оцу відділяє чорні пікселі від світлих."""
    image = np.array([[0, 0, 0], [0, 204, 204], [204, 204, 255]]) # чотири 0, чотири 204, одне 255
    expected = [[0, 0, 0], [0, 255, 255], [255, 255, 255]] # найкращий поріг між 0 і 204
    assert np.array_equal(otsu_threshold(image, {}), expected)
