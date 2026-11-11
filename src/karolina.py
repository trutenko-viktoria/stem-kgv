"""пайплайни Кохан Кароліни."""
import csv
import os

import numpy as np


def check_valid_image(image):
    """перевірка чи файл не пошкоджений, порожній, чорно-білий або замалий."""
    if image is None or image.size == 0:
        return False # порожній або битий файл
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False # чорно-білий або має неправильні канали
    if image.shape[0] < 10 or image.shape[1] < 10:
        return False # надто маленький файл
    return True


def rotate_image(image, params):
    """поворот зображення на заданий кут навколо центру."""
    # pylint: disable=too-many-locals
    if not check_valid_image(image):
        return np.zeros((256, 256, 3), dtype=np.uint8) # чорний квадрат для поганих файлів

    angle = params.get("angle", 15)
    angle_rad = np.radians(angle) # переводимо градуси в радіани
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    y, x = np.indices((h, w)) # координати всіх пікселів

    new_x = ((x - cx) * cos_a - (y - cy) * sin_a + cx).astype(int) # поворот по X
    new_y = ((x - cx) * sin_a + (y - cy) * cos_a + cy).astype(int) # поворот по Y

    valid = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h) # відкидаємо те, що поза межами

    result = np.zeros_like(image)
    result[y[valid], x[valid]] = image[new_y[valid], new_x[valid]] # копіюємо пікселі на нові місця

    return result


def kmeans_segmentation(image, params):
    """сегментація зображення алгоритмом k-means."""
    # pylint: disable=too-many-locals
    k = params.get("k", 3)
    np.random.seed(42) # фіксуємо випадковість для однакових результатів

    img_float = image.astype(np.float64)
    h, w, c = img_float.shape
    pixels = img_float.reshape(-1, c) # картинка як довгий список пікселів

    indices = np.random.choice(pixels.shape[0], k, replace=False) # k випадкових пікселів
    centers = pixels[indices] # робимо їх початковими центрами
    labels = np.zeros(pixels.shape[0], dtype=int)

    for _ in range(5): # 5 ітерацій зазвичай достатньо для сегментації
        distances = np.linalg.norm(pixels[:, None] - centers, axis=2) # евклідова відстань
        labels = np.argmin(distances, axis=1) # піксель іде до найближчого центру

        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = pixels[mask].mean(axis=0) # оновлюємо центр

    segmented = centers[labels].reshape(h, w, c) # збираємо картинку назад
    return segmented.astype(np.uint8)


def compute_region_stats(image):
    """середнє, дисперсія й ентропія кожної області, область це унікальний колір після k-means."""
    pixels = image.astype(np.float64).reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0) # усі унікальні кольори, тобто області
    stats = []
    for color in unique_colors:
        region_pixels = pixels[np.all(pixels == color, axis=1)] # пікселі однієї області
        counts, _ = np.histogram(region_pixels, bins=256, range=(0, 255))
        probs = counts[counts > 0] / region_pixels.size # частка пікселів кожної яскравості
        entropy = max(0.0, float(-np.sum(probs * np.log2(probs)))) # max, щоб не вийшло -0
        stats.append((color, region_pixels.mean(), region_pixels.var(), entropy))
    return stats


def region_statistics(image, params):
    """рахує статистику областей і зберігає її в папку знімка, картинку повертає без змін."""
    out_dir = params.get("out_dir") # папку цього знімка передає main.py
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "region_stats.csv")
        with open(path, "w", newline="", encoding="utf-8") as file: # "w", щоб не дублювалось
            writer = csv.writer(file)
            writer.writerow(["r", "g", "b", "mean", "variance", "entropy"])
            for color, mean_val, var_val, entropy in compute_region_stats(image):
                writer.writerow([*color.astype(int), f"{mean_val:.2f}", f"{var_val:.2f}",
                                 f"{entropy:.2f}"])
    return image # повертаємо картинку без змін, щоб пайплайн міг її зберегти


def log_transform(image, _params):
    """логарифмічне перетворення для покращення контрасту в темних областях."""
    if not check_valid_image(image):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    img_float = image.astype(np.float64)
    c = 255.0 / np.log(1 + 255.0) # нормалізуюча константа

    log_img = c * np.log(1 + img_float) # застосовуємо логарифм
    return np.clip(log_img, 0, 255).astype(np.uint8)


def morphological_gradient(image, _params):
    """морфологічний градієнт: різниця між розширенням (dilation) та звуженням (erosion)."""
    # pylint: disable=too-many-locals
    img = image.astype(np.float64)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2) # переводимо в один канал для морфології

    padded = np.pad(img, pad_width=1, mode='edge') # додаємо рамку 1px
    shifts = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            shifts.append(padded[1 + dy:padded.shape[0] - 1 + dy,
                                 1 + dx:padded.shape[1] - 1 + dx]) # зсуви

    stacked = np.stack(shifts, axis=-1)
    dilation = np.max(stacked, axis=-1) # локальний максимум
    erosion = np.min(stacked, axis=-1) # локальний мінімум

    grad = dilation - erosion # градієнт це різниця
    return grad.astype(np.uint8)


def hough_lines(image, params):
    """перетворення Хафа для пошуку прямих (векторизоване для швидкості)."""
    # pylint: disable=too-many-locals
    img = image.astype(np.float64)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    threshold = params.get("hough_threshold", 50)
    y_idxs, x_idxs = np.nonzero(img > threshold) # координати країв

    thetas = np.deg2rad(np.arange(-90, 90)) # кути від -90 до 90
    diag_len = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))

    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64) # простір Хафа

    if len(x_idxs) > 0:
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        rho_vals = np.outer(x_idxs, cos_t) + np.outer(y_idxs, sin_t) # rho для всіх точок
        rho_idxs = np.round(rho_vals).astype(int) + diag_len
        theta_idxs = np.tile(np.arange(len(thetas)), (len(x_idxs), 1))

        np.add.at(accumulator, (rho_idxs.flatten(), theta_idxs.flatten()), 1) # голосуємо

    acc_max = accumulator.max()
    if acc_max > 0:
        vis = (accumulator / acc_max) * 255 # розтягуємо яскравість для збереження картинки
    else:
        vis = accumulator

    return vis.astype(np.uint8)


PIPELINES = {
    "karolina1": [
        ("rotate", rotate_image),
        ("kmeans", kmeans_segmentation),
        ("stats", region_statistics),
    ],
    "karolina2": [
        ("log", log_transform),
        ("morph_grad", morphological_gradient),
        ("hough", hough_lines),
    ],
}
