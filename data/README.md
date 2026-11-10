# Вибірка зображень

**Джерело:** UC Merced Land Use Dataset — 21 клас землекористування, по 100 аерофотознімків у класі, 256×256 пікселів, роздільна здатність 1 фут/піксель. Знімки вирізано з колекції USGS National Map Urban Area Imagery (суспільне надбання).

- Оригінальний сайт: http://weegee.vision.ucmerced.edu/datasets/landuse.html
- Завантажено з дзеркала: https://huggingface.co/datasets/torchgeo/ucmerced
- Цитування: Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.

## Як відібрано

- 105 зображень: по 5 випадкових з кожного з 21 класу (`random.seed(0)`).
- Збережено у форматі PNG (без втрат) замість оригінального TIF, щоб їх було видно на GitHub.
- 4 зображення мають розмір, відмінний від 256×256 (так в оригінальному датасеті): `baseballdiamond45.png` (250×256), `golfcourse07.png` (256×251), `overpass62.png` (255×256), `overpass65.png` (247×247) — корисно для перевірки крайових випадків.

## Еталонні зображення

| Файл | Що на ньому | Для чого |
|---|---|---|
| `river26.png` | річка серед полів | відокремлення води від суші |
| `forest40.png` | ліс | текстури, шум |
| `buildings18.png` | забудова | чіткі межі, контури |
| `freeway56.png` | автотраса | прямі лінії |
| `beach12.png` | пляж і море | дві великі однорідні області |
