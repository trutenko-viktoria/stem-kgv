"""пайплайни Віки."""
import numpy as np


def gamma_correction(image, params):
    """гамма-корекція: темні ділянки змінюються сильніше, ніж світлі."""
    gamma = params["gamma"] # з config.json, менше 1 освітлює, більше 1 затемнює
    normalized = image.astype(np.float64) / 255 # переводимо яскравість у 0..1
    return 255 * normalized ** gamma # формула: 255 * (p / 255) ^ gamma


def equalize_histogram(image, params):
    """вирівнювання гістограми: яскравості розподіляються рівномірно на весь діапазон 0..255."""
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2] # переводимо в сіру
    gray = np.clip(np.round(gray), 0, 255).astype(np.int64) # цілі рівні яскравості 0..255
    hist = np.bincount(gray.ravel(), minlength=256) # скільки пікселів кожної яскравості
    cdf = np.cumsum(hist) # скільки пікселів мають яскравість не більшу за дану
    cdf_min = cdf[cdf > 0][0] # перше ненульове значення
    if cdf[-1] == cdf_min: # вся картинка одного кольору, вирівнювати нічого
        return gray
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255) # нова яскравість для кожного рівня
    return lut[gray] # замінюємо кожен піксель за таблицею


def otsu_threshold(image, params):
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


PIPELINES = {
    "vika1": [("gamma", gamma_correction), ("equalize", equalize_histogram), ("otsu", otsu_threshold)],
}
