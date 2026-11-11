"""пайплайни Віки."""
import numpy as np


def gamma_correction(image, params):
    """гамма-корекція: темні ділянки змінюються сильніше, ніж світлі."""
    gamma = params["gamma"] # з config.json, менше 1 освітлює, більше 1 затемнює
    normalized = image.astype(np.float64) / 255 # переводимо яскравість у 0..1
    return 255 * normalized ** gamma # формула: 255 * (p / 255) ^ gamma


def equalize_histogram(image, _params):
    """вирівнювання гістограми: яскравості розподіляються рівномірно на весь діапазон 0..255."""
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2] # в сіру
    gray = np.clip(np.round(gray), 0, 255).astype(np.int64) # цілі рівні яскравості 0..255
    hist = np.bincount(gray.ravel(), minlength=256) # скільки пікселів кожної яскравості
    cdf = np.cumsum(hist) # скільки пікселів мають яскравість не більшу за дану
    cdf_min = cdf[cdf > 0][0] # перше ненульове значення
    if cdf[-1] == cdf_min: # вся картинка одного кольору, вирівнювати нічого
        return gray
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255) # нова яскравість для кожного рівня
    return lut[gray] # замінюємо кожен піксель за таблицею


def otsu_threshold(image, _params):
    """сегментація оцу: сам шукає поріг і ділить картинку на темне і світле."""
    gray = np.clip(np.round(image), 0, 255).astype(np.int64) # цілі рівні яскравості 0..255
    hist = np.bincount(gray.ravel(), minlength=256) # скільки пікселів кожної яскравості
    levels = np.arange(256) # самі рівні 0, 1, ..., 255
    best_t, best_var = 0, -1
    for t in range(256): # пробуємо кожен поріг
        w0 = hist[:t + 1].sum() # пікселів у темному класі
        w1 = hist[t + 1:].sum() # пікселів у світлому класі
        if w0 == 0 or w1 == 0: # один клас порожній, такий поріг не ділить картинку
            continue
        m0 = (levels[:t + 1] * hist[:t + 1]).sum() / w0 # середня яскравість темного класу
        m1 = (levels[t + 1:] * hist[t + 1:]).sum() / w1 # середня яскравість світлого класу
        var = w0 * w1 * (m0 - m1) ** 2 # міжкласова дисперсія, чим більша, тим краще розділено
        if var > best_var:
            best_t, best_var = t, var
    return np.where(gray > best_t, 255, 0) # світле стає білим, темне чорним


def scale_nearest(image, params):
    """масштабування найближчим сусідом: кожен новий піксель копіює найближчий старий."""
    scale = params["scale"] # з config.json, більше 1 збільшує, менше 1 зменшує
    height, width = image.shape[:2]
    new_height = max(1, round(height * scale)) # новий розмір, щонайменше 1 піксель
    new_width = max(1, round(width * scale))
    rows = np.minimum((np.arange(new_height) / scale).astype(np.int64), height - 1) # старий рядок
    cols = np.minimum((np.arange(new_width) / scale).astype(np.int64), width - 1) # старий стовпець
    return image[rows][:, cols] # збираємо нову картинку зі старих пікселів


def kuwahara_filter(image, params): # pylint: disable=too-many-locals
    """фільтр кувахари: згладжує шум, але зберігає межі областей."""
    radius = params["radius"] # з config.json, кожна з 4 областей має сторону radius + 1
    size = radius + 1
    img = image.astype(np.float64) # дробові числа, щоб рахувати середнє
    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge") # копіюємо краї
    gray = 0.299 * padded[:, :, 0] + 0.587 * padded[:, :, 1] + 0.114 * padded[:, :, 2] # яскравість
    gray_windows = np.lib.stride_tricks.sliding_window_view(gray, (size, size)) # усі квадрати
    color_windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(0, 1))
    variances = gray_windows.var(axis=(2, 3)) # дисперсія яскравості кожного квадрата
    means = color_windows.mean(axis=(3, 4)) # середній колір кожного квадрата
    height, width = img.shape[:2]
    corners = [(0, 0), (0, radius), (radius, 0), (radius, radius)] # зсуви 4 областей
    area_vars = np.stack([variances[y:y + height, x:x + width] for y, x in corners]) # 4 дисперсії
    area_means = np.stack([means[y:y + height, x:x + width] for y, x in corners]) # 4 середні
    best = np.argmin(area_vars, axis=0) # область, де найменша дисперсія, тобто найрівніша
    rows, cols = np.indices((height, width))
    return area_means[best, rows, cols] # беремо середній колір найрівнішої області


def rgb_to_hsv(image):
    """переводить rgb у hsv: відтінок 0..360, насиченість і яскравість 0..1."""
    img = image.astype(np.float64) / 255 # переводимо канали у 0..1
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    value = img.max(axis=2) # яскравість, найбільший канал
    delta = value - img.min(axis=2) # різниця між найбільшим і найменшим каналом
    saturation = np.divide(delta, value, out=np.zeros_like(delta), where=value > 0) # у чорного 0
    safe_delta = np.where(delta > 0, delta, 1) # у сірих пікселів ділити ні на що, беремо 1
    hue = np.select(
        [delta == 0, value == red, value == green], # сірий, найбільший r, найбільший g, інакше b
        [0, ((green - blue) / safe_delta) % 6, (blue - red) / safe_delta + 2],
        (red - green) / safe_delta + 4,
    ) * 60 # переводимо у градуси
    return np.stack([hue, saturation, value], axis=2)


def hsv_to_rgb(hsv):
    """переводить hsv назад у rgb 0..255."""
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    chroma = value * saturation # сила кольору
    sector = hue / 60 # у якому з 6 секторів кольорового кола відтінок
    second = chroma * (1 - np.abs(sector % 2 - 1)) # другий за величиною канал
    zero = np.zeros_like(hue)
    index = np.minimum(sector.astype(np.int64), 5) # номер сектора 0..5
    table = np.stack([ # r, g, b для кожного сектора
        np.stack([chroma, second, zero], axis=2),
        np.stack([second, chroma, zero], axis=2),
        np.stack([zero, chroma, second], axis=2),
        np.stack([zero, second, chroma], axis=2),
        np.stack([second, zero, chroma], axis=2),
        np.stack([chroma, zero, second], axis=2),
    ])
    rows, cols = np.indices(hue.shape)
    return 255 * (table[index, rows, cols] + (value - chroma)[:, :, np.newaxis]) # додаємо мінімум


def change_saturation(image, params):
    """переводить у hsv, змінює насиченість і повертає в rgb."""
    hsv = rgb_to_hsv(image)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["saturation"], 0, 1) # множимо s
    return hsv_to_rgb(hsv)


PIPELINES = {
    "vika1": [
        ("gamma", gamma_correction),
        ("equalize", equalize_histogram),
        ("otsu", otsu_threshold),
    ],
    "vika2": [
        ("scale", scale_nearest),
        ("kuwahara", kuwahara_filter),
        ("saturation", change_saturation),
    ],
}