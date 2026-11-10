"""тести пайплайнів Кохан Кароліни"""
import numpy as np
from src.karolina import check_valid_image, hough_lines

def test_valid_image_check():
    """перевірка захисту від пошкоджених файлів"""
    # Тест 1: Правильна картинка
    good_image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert check_valid_image(good_image) is True
    
    # Тест 2: Порожня картинка
    bad_image_empty = np.array([])
    assert check_valid_image(bad_image_empty) is False
    
    # Тест 3: Замала картинка (менше 10x10)
    bad_image_small = np.zeros((5, 5, 3), dtype=np.uint8)
    assert check_valid_image(bad_image_small) is False

def test_hough_lines():
    """перевірка алгоритму Хафа на базовій матриці"""
    # Створюємо маленьку картинку 10x10
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Малюємо горизонтальну лінію посередині
    image[5, :, :] = 255
    
    params = {"hough_threshold": 50}
    result = hough_lines(image, params)
    
    # Результат має бути масивом (простором Хафа)
    assert isinstance(result, np.ndarray)
    # Матриця результату не має бути порожньою (там мають бути голоси за лінію)
    assert result.max() > 0