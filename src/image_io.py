"""читання і збереження картинок (skimage тут тільки для цього, методи пишемо вручну)."""
import os

import numpy as np
from skimage.io import imread, imsave

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff") # розширення картинок


def list_images(folder):
    """повертає шляхи до всіх картинок у папці за алфавітом."""
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]


def load_image(path):
    """читає картинку як масив висота x ширина x 3 (r, g, b), або None, якщо файл битий."""
    try:
        image = imread(path)
    except (OSError, ValueError): # не картинка або пошкоджений файл
        return None
    if image.ndim == 2: # чорно-біла картинка -> робимо 3 однакові канали
        image = np.stack([image] * 3, axis=2)
    return image[:, :, :3].astype(np.uint8) # прибираємо альфа-канал, якщо він є


def save_image(path, image):
    """зберігає картинку і сама створює потрібні папки."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8) # обрізаємо все, що вийшло за 0..255
    imsave(path, image, check_contrast=False) # check_contrast=False, щоб не було зайвих попереджень
