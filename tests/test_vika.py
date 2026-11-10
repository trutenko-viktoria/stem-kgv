"""тести пайплайнів Віки."""
import numpy as np

from src.vika import (
    change_saturation,
    equalize_histogram,
    gamma_correction,
    kuwahara_filter,
    otsu_threshold,
    scale_nearest,
)


def test_gamma_correction():
    """при гаммі 0.5 темне світлішає сильно, а чорне і біле не змінюються."""
    image = np.array([[[0, 64, 255]] * 3] * 3, dtype=np.uint8) # картинка 3 x 3, канали 0, 64, 255
    result = gamma_correction(image, {"gamma": 0.5})
    assert np.allclose(result[0, 0], [0, 127.75, 255], atol=0.01) # 255 * (64 / 255) ^ 0.5 = 127.75


def test_equalize_histogram():
    """рівні 50, 100, 150 розтягуються на 0, 204, 255."""
    gray = np.array([[50, 50, 50], [50, 100, 100], [100, 100, 150]]) # 4 по 50, 4 по 100, одне 150
    image = np.stack([gray, gray, gray], axis=2).astype(np.uint8) # r, g, b однакові
    expected = [[0, 0, 0], [0, 204, 204], [204, 204, 255]] # пораховано вручну за cdf
    assert np.array_equal(equalize_histogram(image, {}), expected)


def test_otsu_threshold():
    """оцу відділяє чорні пікселі від світлих."""
    image = np.array([[0, 0, 0], [0, 204, 204], [204, 204, 255]]) # чотири 0, чотири 204, одне 255
    expected = [[0, 0, 0], [0, 255, 255], [255, 255, 255]] # найкращий поріг між 0 і 204
    assert np.array_equal(otsu_threshold(image, {}), expected)


def test_scale_nearest():
    """при збільшенні вдвічі кожен піксель стає квадратом 2 x 2."""
    image = np.array([[[10, 10, 10], [20, 20, 20]]], dtype=np.uint8) # картинка 1 x 2
    result = scale_nearest(image, {"scale": 2})
    assert np.array_equal(result[:, :, 0], [[10, 10, 20, 20], [10, 10, 20, 20]]) # стала 2 x 4


def test_kuwahara_filter():
    """різка межа між 0 і 90 не розмивається."""
    gray = np.array([[0, 0, 90, 90]] * 4, dtype=np.uint8) # ліва половина 0, права 90
    image = np.stack([gray, gray, gray], axis=2)
    result = kuwahara_filter(image, {"radius": 1})
    assert np.array_equal(result[:, :, 0], gray) # кожен піксель має рівну область на своєму боці


def test_change_saturation():
    """подвоєння насиченості: s 0.5 стає 1, а сірий піксель не змінюється."""
    image = np.array([[[200, 100, 100], [50, 50, 50]]], dtype=np.uint8) # s = 0.5, і сірий піксель
    result = change_saturation(image, {"saturation": 2})
    assert np.allclose(result, [[[200, 0, 0], [50, 50, 50]]]) # мінімум став 0
