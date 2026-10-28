"""запуск програми: python main.py --pipeline vika1"""
import argparse
import json
import os

import numpy as np

from src import dasha, karolina, vika
from src.image_io import list_images, load_image, save_image

# збираємо пайплайни всіх трьох в один словник: назва -> список кроків
PIPELINES = {}
PIPELINES.update(vika.PIPELINES)
PIPELINES.update(dasha.PIPELINES)
PIPELINES.update(karolina.PIPELINES)


def run_pipeline(name, image, params):
    """проганяє картинку через усі кроки пайплайну і повертає результат кожного кроку."""
    results = []
    for stage_name, stage in PIPELINES[name]: # кроки йдуть по черзі
        image = stage(image, params) # результат кроку йде на вхід наступному
        results.append((stage_name, image))
    return results


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
        steps = run_pipeline(args.pipeline, image, params)
        for number, (stage_name, result) in enumerate(steps, start=1):
            save_image(os.path.join(out_dir, f"{number}_{stage_name}.png"), result) # 1_gamma.png, 2_...

        # тут пізніше будуть метрики (psnr, mse, ssim) і запис у csv
        print(f"{name}: готово")


if __name__ == "__main__":
    main()
