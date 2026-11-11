"""запуск програми: python main.py --pipeline vika1"""
import argparse
import json
import os

import numpy as np

from src import dasha, karolina, vika
from src.image_io import list_images, load_image, save_image
from src.metrics import mse, psnr, ssim, write_metrics_csv

# збираємо пайплайни всіх трьох в один словник: назва -> список кроків
PIPELINES = {}
PIPELINES.update(vika.PIPELINES)
PIPELINES.update(dasha.PIPELINES)
PIPELINES.update(karolina.PIPELINES)


def run_pipeline(name, image, params, out_dir):
    """проганяє картинку через усі кроки пайплайну і повертає результат кожного кроку."""
    stage_params = {**params, "out_dir": out_dir} # крок може зберегти свої файли в папку знімка
    results = []
    for stage_name, stage in PIPELINES[name]: # кроки йдуть по черзі
        image = stage(image, stage_params) # результат кроку йде на вхід наступному
        results.append((stage_name, image))
    return results


def match_size(original, result):
    """підганяє оригінал під розмір результату методом найближчого сусіда."""
    height, width = result.shape[:2]
    old_height, old_width = original.shape[:2]
    if (height, width) == (old_height, old_width): # розмір не змінився, підганяти не треба
        return original
    rows = np.arange(height) * old_height // height # який рядок оригіналу брати для кожного нового
    cols = np.arange(width) * old_width // width # який стовпець оригіналу брати для кожного нового
    return original[rows][:, cols]


def main():
    """читає аргументи й конфіг, обробляє всі знімки і зберігає результати."""
    # аргументи з командного рядка
    parser = argparse.ArgumentParser(description="обробка супутникових знімків пайплайнами")
    parser.add_argument("--pipeline", help="назва пайплайну, наприклад vika1")
    parser.add_argument("--config", default="config.json", help="шлях до файлу з параметрами")
    parser.add_argument("--list", action="store_true", help="показати, які пайплайни є")
    args = parser.parse_args()

    # якщо пайплайн не вказали, просто показуємо список
    if args.list or not args.pipeline:
        print("доступні пайплайни:", ", ".join(PIPELINES) or "поки немає")
        return
    if args.pipeline not in PIPELINES:
        print(f"пайплайн «{args.pipeline}» не знайдено, подивись список: python main.py --list")
        return

    # читаємо налаштування з config.json
    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)

    np.random.seed(config["seed"]) # щоб випадкові речі щоразу були однакові
    params = config["pipelines"].get(args.pipeline, {}) # параметри саме цього пайплайну

    metrics_rows = [] # сюди збираємо метрики всіх знімків для csv

    # обробляємо кожен знімок з папки
    for path in list_images(config["images_dir"]):
        image = load_image(path)
        if image is None: # битий файл просто пропускаємо
            print(f"пропущено, не вдалося прочитати: {path}")
            continue

        name = os.path.splitext(os.path.basename(path))[0] # "river26.png" -> "river26"
        out_dir = os.path.join(config["results_dir"], args.pipeline, name) # results/vika1/river26/

        # зберігаємо оригінал і картинку після кожного кроку
        save_image(os.path.join(out_dir, "0_original.png"), image)
        steps = run_pipeline(args.pipeline, image, params, out_dir)
        for number, (stage_name, result) in enumerate(steps, start=1):
            save_image(os.path.join(out_dir, f"{number}_{stage_name}.png"), result) # 1_gamma.png, 2_...

            # порівнюємо результат кроку з оригіналом
            result = np.clip(result, 0, 255) # так само, як при збереженні картинки
            reference = match_size(image, result) # оригінал того ж розміру, що й результат
            metrics_rows.append({
                "image": name,
                "pipeline": args.pipeline,
                "step": stage_name,
                "mse": mse(reference, result),
                "psnr": psnr(reference, result),
                "ssim": ssim(reference, result),
            })
        print(f"{name}: готово")

    # записуємо метрики всіх знімків в один файл, наприклад results/vika1/metrics.csv
    csv_path = os.path.join(config["results_dir"], args.pipeline, "metrics.csv")
    write_metrics_csv(csv_path, metrics_rows)
    print(f"метрики записано: {csv_path}")


if __name__ == "__main__":
    main()
