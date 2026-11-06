"""пайплайни та алгоритми обробки зображень для Даші."""
import numpy as np


def gray_world_white_balance(image, params):
    img = image.astype(np.float64)
    means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    gains = np.divide(gray, means, out=np.ones_like(means), where=means > 0)
    corrected = img * gains
    return np.clip(np.round(corrected), 0, 255).astype(np.uint8)


def bilateral_filter(image, params):
    radius = params.get("radius", params.get("window", 5) // 2)
    sigma_space = params.get("sigma_space", 3.0)
    sigma_color = params.get("sigma_color", 30.0)

    img = image.astype(np.float64)
    height, width = img.shape[:2]
    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    space_weights = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma_space ** 2))
    rows, cols = np.indices((height, width))
    total_weight = np.zeros((height, width))
    result = np.zeros_like(img)
    center = img

    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            neighbor = padded[rows + oy + radius, cols + ox + radius]
            diff = neighbor - center
            color_dist = np.sum(diff ** 2, axis=2)
            color_weights = np.exp(-color_dist / (2 * sigma_color ** 2))
            w = space_weights[oy + radius, ox + radius] * color_weights
            result += neighbor * w[:, :, np.newaxis]
            total_weight += w

    return np.clip(np.round(result / total_weight[:, :, np.newaxis]), 0, 255).astype(np.uint8)


def region_growing(image, params):
    threshold = params.get("threshold", 20.0)
    seeds = params.get("seeds", [])

    img = image.astype(np.float64)
    height, width = img.shape[:2]
    labels = np.zeros((height, width), dtype=np.int64)

    for seed_y, seed_x in seeds:
        if labels[seed_y, seed_x] != 0:
            continue
        region_sum = img[seed_y, seed_x].copy()
        region_count = 1
        mean_color = region_sum / region_count
        queue = [(seed_y, seed_x)]
        labels[seed_y, seed_x] = 1

        while queue:
            y, x = queue.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                if labels[ny, nx] != 0:
                    continue
                diff = img[ny, nx] - mean_color
                if np.sqrt(np.sum(diff ** 2)) > threshold:
                    continue
                labels[ny, nx] = 1
                region_sum += img[ny, nx]
                region_count += 1
                mean_color = region_sum / region_count
                queue.append((ny, nx))

    return np.where(labels[:, :, np.newaxis] == 1, image, 0).astype(np.uint8)


def rgb_to_ycbcr(image, params):
    img = image.astype(np.float64)
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return np.clip(np.round(np.stack([y, cb, cr], axis=2)), 0, 255).astype(np.uint8)


def linear_contrast_stretch(image, params):
    low_percent = params.get("low_percent", 2.0)
    high_percent = params.get("high_percent", 98.0)

    img = image.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    low = np.percentile(gray, low_percent)
    high = np.percentile(gray, high_percent)
    if high <= low:
        return image.copy()
    stretched = (img - low) / (high - low) * 255
    return np.clip(np.round(stretched), 0, 255).astype(np.uint8)


def lbp(image, params):
    radius = params.get("radius", params.get("window", 3) // 2)

    img = image.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    height, width = gray.shape
    padded = np.pad(gray, radius, mode="edge")
    rows, cols = np.indices((height, width))
    center = padded[rows + radius, cols + radius]
    angles = np.arange(8) * (2 * np.pi / 8)
    offsets_y = np.round(-radius * np.sin(angles)).astype(np.int64)
    offsets_x = np.round(radius * np.cos(angles)).astype(np.int64)
    codes = np.zeros((height, width), dtype=np.int64)

    for bit, (oy, ox) in enumerate(zip(offsets_y, offsets_x)):
        neighbor = padded[rows + oy + radius, cols + ox + radius]
        codes += (neighbor >= center).astype(np.int64) << bit

    return codes.astype(np.uint8)


PIPELINES = {
    "dasha1": [
        ("white_balance", gray_world_white_balance),
        ("bilateral", bilateral_filter),
        ("region_growing", region_growing),
    ],
    "dasha2": [
        ("ycbcr", rgb_to_ycbcr),
        ("contrast_stretch", linear_contrast_stretch),
        ("lbp", lbp),
    ],
}