"""пайплайни Кохан Кароліни"""
import numpy as np


def check_valid_image(image):
    """перевірка чи файл не пошкоджений, порожній, чорно-білий або замалий"""
    if image is None or image.size == 0:
        return False # порожній або битий файл
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False # чорно-білий або має неправильні канали
    if image.shape[0] < 10 or image.shape[1] < 10:
        return False # надто маленький файл
    return True


def rotate_image(image, params):
    """поворот зображення на заданий кут навколо центру"""
    if not check_valid_image(image):
        return np.zeros((256, 256, 3), dtype=np.uint8) # повертаємо чорний квадрат для поганих файлів
        
    angle = params.get("angle", 15)
    angle_rad = np.radians(angle) # переводимо градуси в радіани
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    
    y, x = np.indices((h, w)) # створюємо координати всіх пікселів
    
    new_x = ((x - cx) * cos_a - (y - cy) * sin_a + cx).astype(int) # математика повороту по X
    new_y = ((x - cx) * sin_a + (y - cy) * cos_a + cy).astype(int) # математика повороту по Y
    
    valid = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h) # відкидаємо те, що вийшло за межі
    
    result = np.zeros_like(image)
    result[y[valid], x[valid]] = image[new_y[valid], new_x[valid]] # копіюємо пікселі на нові місця
    
    return result


def kmeans_segmentation(image, params):
    """сегментація зображення алгоритмом k-means"""
    k = params.get("k", 3)
    np.random.seed(42) # фіксуємо випадковість для однакових результатів
    
    img_float = image.astype(np.float64)
    h, w, c = img_float.shape
    pixels = img_float.reshape(-1, c) # перетворюємо картинку на довгий список пікселів
    
    indices = np.random.choice(pixels.shape[0], k, replace=False) # вибираємо k випадкових пікселів
    centers = pixels[indices] # робимо їх початковими центрами
    labels = np.zeros(pixels.shape[0], dtype=int)
    
    for _ in range(5): # 5 ітерацій зазвичай достатньо для сегментації
        distances = np.linalg.norm(pixels[:, None] - centers, axis=2) # рахуємо евклідову відстань
        labels = np.argmin(distances, axis=1) # прив'язуємо піксель до найближчого центру
        
        for i in range(k): 
            mask = (labels == i)
            if np.any(mask):
                centers[i] = pixels[mask].mean(axis=0) # оновлюємо центр
                
    segmented = centers[labels].reshape(h, w, c) # збираємо картинку назад
    return segmented.astype(np.uint8)


def region_statistics(image, params):
    """обчислення середнього, дисперсії та ентропії кожної області"""
    img_float = image.astype(np.float64)
    pixels = img_float.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0) # знаходимо всі унікальні кольори областей після k-means
    
    for color in unique_colors:
        mask = np.all(pixels == color, axis=1)
        region_pixels = pixels[mask] # виділяємо пікселі конкретної області
        
        mean_val = region_pixels.mean() # рахуємо середнє
        var_val = region_pixels.var() # рахуємо дисперсію
        
        hist, _ = np.histogram(region_pixels, bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0] # залишаємо тільки ненульові значення для логарифма
        entropy = -np.sum(hist * np.log2(hist)) # рахуємо ентропію за формулою
        
    return image # повертаємо картинку без змін, щоб пайплайн міг її зберегти


PIPELINES = {
    "karolina1": [("rotate", rotate_image), ("kmeans", kmeans_segmentation), ("stats", region_statistics)]
}