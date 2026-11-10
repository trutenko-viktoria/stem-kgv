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
    # pylint: disable=too-many-locals
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
    # pylint: disable=too-many-locals
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

def region_statistics(image, _params):
    """обчислення середнього, дисперсії та ентропії кожної області"""
    # pylint: disable=too-many-locals
    img_float = image.astype(np.float64)
    pixels = img_float.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0) # знаходимо всі унікальні кольори областей після k-means
    
    with open("results/karolina_stats.txt", "a", encoding="utf-8") as f:
        for color in unique_colors:
            mask = np.all(pixels == color, axis=1)
            region_pixels = pixels[mask] # виділяємо пікселі конкретної області
            
            mean_val = region_pixels.mean() # рахуємо середнє
            var_val = region_pixels.var() # рахуємо дисперсію
            
            counts, _ = np.histogram(region_pixels, bins=256, range=(0, 255))
            probs = counts[counts > 0] / region_pixels.size
            entropy = -np.sum(probs * np.log2(probs)) # рахуємо ентропію за формулою
            
            f.write(f"Область {color}: середнє={mean_val:.2f}, дисперсія={var_val:.2f}, ентропія={entropy:.2f}\n")
        
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
            shifts.append(padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]) # робимо зсуви
            
    stacked = np.stack(shifts, axis=-1)
    dilation = np.max(stacked, axis=-1) # знаходимо локальний максимум
    erosion = np.min(stacked, axis=-1) # знаходимо локальний мінімум
    
    grad = dilation - erosion # градієнт це різниця
    return grad.astype(np.uint8)

def hough_lines(image, params):
    """перетворення Хафа для пошуку прямих (векторизоване для швидкості)."""
    # pylint: disable=too-many-locals
    img = image.astype(np.float64)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        
    threshold = params.get("hough_threshold", 50)
    y_idxs, x_idxs = np.nonzero(img > threshold) # знаходимо координати країв
    
    thetas = np.deg2rad(np.arange(-90, 90)) # кути від -90 до 90
    diag_len = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
    
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64) # простір Хафа
    
    if len(x_idxs) > 0:
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        rho_vals = np.outer(x_idxs, cos_t) + np.outer(y_idxs, sin_t) # рахуємо rho для всіх точок матричним множенням
        rho_idxs = np.round(rho_vals).astype(int) + diag_len
        theta_idxs = np.tile(np.arange(len(thetas)), (len(x_idxs), 1))
        
        np.add.at(accumulator, (rho_idxs.flatten(), theta_idxs.flatten()), 1) # голосуємо в акумулятор
    
    acc_max = accumulator.max()
    if acc_max > 0:
        vis = (accumulator / acc_max) * 255 # розтягуємо яскравість для збереження картинки
    else:
        vis = accumulator
        
    return vis.astype(np.uint8)

PIPELINES = {
    "karolina1": [("rotate", rotate_image), ("kmeans", kmeans_segmentation), ("stats", region_statistics)],
    "karolina2": [("log", log_transform), ("morph_grad", morphological_gradient), ("hough", hough_lines)]
}