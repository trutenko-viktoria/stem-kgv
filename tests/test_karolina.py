"""тести пайплайнів Кохан Кароліни."""
import numpy as np

from src.karolina import check_valid_image, compute_region_stats, hough_lines, region_statistics


def test_valid_image_check():
    """перевірка захисту від пошкоджених файлів."""
    good_image = np.zeros((20, 20, 3), dtype=np.uint8) # нормальна картинка
    assert check_valid_image(good_image) is True

    bad_image_empty = np.array([]) # порожня картинка
    assert check_valid_image(bad_image_empty) is False

    bad_image_small = np.zeros((5, 5, 3), dtype=np.uint8) # менше 10 x 10
    assert check_valid_image(bad_image_small) is False


def test_hough_lines():
    """перевірка алгоритму Хафа на базовій матриці."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[5, :, :] = 255 # горизонтальна лінія посередині

    result = hough_lines(image, {"hough_threshold": 50})

    assert isinstance(result, np.ndarray) # результат це простір Хафа
    assert result.max() > 0 # у ньому є голоси за лінію


def test_compute_region_stats():
    """дві області, чорна і біла: середні 0 і 255, ентропія 0."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, 1] = 255 # права половина біла
    stats = compute_region_stats(image)
    assert len(stats) == 2
    assert [round(item[1]) for item in stats] == [0, 255] # середні
    assert all(item[3] == 0 for item in stats) # в області одна яскравість


def test_region_statistics_saves_file(tmp_path):
    """статистика записується в папку знімка, а картинка не змінюється."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result = region_statistics(image, {"out_dir": str(tmp_path)})
    assert np.array_equal(result, image)
    assert (tmp_path / "region_stats.csv").exists()