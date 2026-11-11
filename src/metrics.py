"""метрики якості зображень: MSE, PSNR, SSIM + псування для перевірки."""
import csv
import os
import numpy as np


def _to_gray_if_needed(img):
    """допоміжна функція: перетворює триканальне зображення у двовимірне, якщо це необхідно."""
    if img.ndim == 3 and img.shape[2] == 3:
        return np.mean(img, axis=2)
    return img

def mse(original, processed):
    """середньоквадратична помилка між двома зображеннями (0 = ідентичні)."""
    orig_processed_match = original.shape != processed.shape
    if orig_processed_match:
        a = _to_gray_if_needed(original).astype(np.float64)
        b = _to_gray_if_needed(processed).astype(np.float64)
    else:
        a = original.astype(np.float64)
        b = processed.astype(np.float64)

    return float(np.mean((a - b) ** 2))


def psnr(original, processed, max_value=255.0):
    """пікове відношення сигнал/шум у дБ (більше = краще, inf = ідентичні)."""
    error = mse(original, processed)
    if error == 0:
        return float("inf")
    return float(10 * np.log10(max_value ** 2 / error))


def _gaussian_kernel(size=11, sigma=1.5):
    """1D гаусове ядро для SSIM (розділюване)."""
    coords = np.arange(size) - size // 2
    kernel = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def ssim(original, processed, max_value=255.0):
    """індекс структурної подібності (1.0 = ідентичні, менше = гірше)."""
    # pylint: disable=too-many-locals
    # Уніфікуємо розмірності та канали
    if original.shape != processed.shape:
        a = _to_gray_if_needed(original).astype(np.float64)
        b = _to_gray_if_needed(processed).astype(np.float64)
    else:
        a = original.astype(np.float64)
        b = processed.astype(np.float64)

    # Якщо кольорове (обидва триканальні) — усереднюємо SSIM по каналах
    if a.ndim == 3 and a.shape[2] == 3:
        return float(np.mean([ssim(a[:, :, c], b[:, :, c], max_value) for c in range(3)]))

    h, w = a.shape
    min_side = min(h, w)

    # Динамічний підбір розміру ядра для дуже маленьких зображень (наприклад, тести 3x3)
    kernel_size = min(11, min_side if min_side % 2 != 0 else min_side - 1)
    if kernel_size < 3:
        kernel_size = 3 if min_side >= 3 else min_side
        if kernel_size < 1:
            return 1.0  # Захист для вироджених випадків

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    kernel = _gaussian_kernel(kernel_size, sigma=1.5)
    pad = len(kernel) // 2

    def blur(x):
        # Якщо зображення менше за паддінг, захищаємося від помилок згортки
        if x.shape[0] <= pad or x.shape[1] <= pad:
            return x
        xp = np.pad(x, pad, mode="edge")
        rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, xp)
        cols = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, rows)
        return cols

    mu_a = blur(a)
    mu_b = blur(b)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a2 = blur(a * a) - mu_a2
    sigma_b2 = blur(b * b) - mu_b2
    sigma_ab = blur(a * b) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)

    valid_mask = denominator != 0
    if not np.any(valid_mask):
        return 1.0

    result = np.mean(numerator[valid_mask] / denominator[valid_mask])
    return float(np.clip(result, -1.0, 1.0))


CSV_FIELDS = ["image", "pipeline", "step", "mse", "psnr", "ssim"]


def write_metrics_csv(path, rows):
    """записує список словників з метриками у CSV (створює теку за потреби)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def evaluate_pipeline(original, pipeline_steps, pipeline_name="pipeline", image_name=""):
    """проганяє кроки пайплайну і повертає метрики після кожного кроку.

    pipeline_steps: список (ім'я_кроку, функція) або (ім'я_кроку, функція, params).
    """
    rows = []
    current = original
    for item in pipeline_steps:
        if len(item) == 3:
            name, func, params = item
        else:
            name, func = item
            params = {}
        current = func(current, params)
        rows.append({
            "image": image_name,
            "pipeline": pipeline_name,
            "step": name,
            "mse": mse(original, current),
            "psnr": psnr(original, current),
            "ssim": ssim(original, current),
        })
    return rows, current


def add_gaussian_noise(image, params):
    """додає гаусовий шум (sigma у рівнях яскравості 0..255)."""
    sigma = params.get("sigma", 20.0)
    seed = params.get("seed", 0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(np.round(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, params):
    """замінює частину пікселів на чорні/білі (amount = частка 0..1)."""
    amount = params.get("amount", 0.05)
    seed = params.get("seed", 0)
    rng = np.random.default_rng(seed)
    result = image.copy()
    mask = rng.random(image.shape[:2])
    salt = mask < amount / 2
    pepper = (mask >= amount / 2) & (mask < amount)
    result[salt] = 255
    result[pepper] = 0
    return result


def gaussian_blur(image, params):
    """розмиває зображення гаусовим ядром (sigma, size)."""
    sigma = params.get("sigma", 2.0)
    size = params.get("size", 5)
    kernel = _gaussian_kernel(size, sigma)
    pad = size // 2
    img = image.astype(np.float64)

    def blur_channel(x):
        xp = np.pad(x, pad, mode="edge")
        rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, xp)
        return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, rows)

    if img.ndim == 3:
        out = np.stack([blur_channel(img[:, :, c]) for c in range(img.shape[2])], axis=2)
    else:
        out = blur_channel(img)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def change_brightness(image, params):
    """множить яскравість на factor (>1 світліше, <1 темніше)."""
    factor = params.get("factor", 1.3)
    return np.clip(np.round(image.astype(np.float64) * factor), 0, 255).astype(np.uint8)


def add_contrast_reduction(image, params):
    """зменшує контраст: тягне яскравості до сірого 128."""
    factor = params.get("factor", 0.5)
    img = image.astype(np.float64)
    return np.clip(np.round(128 + (img - 128) * factor), 0, 255).astype(np.uint8)


def random_crop_shift(image, params):
    """зсуває зображення (імітує тремтіння камери)."""
    shift_y = params.get("shift_y", 3)
    shift_x = params.get("shift_x", 3)
    return np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))


DEGRADATIONS = {
    "gaussian_noise": add_gaussian_noise,
    "salt_pepper": add_salt_pepper_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": change_brightness,
    "contrast": add_contrast_reduction,
    "crop_shift": random_crop_shift,
}
