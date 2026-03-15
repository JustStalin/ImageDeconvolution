# Исследование восстановления размытого изображения с известной ФРТ

## Что было сделано

- Выбраны 3 изображения из USC SIPI: Mandrill, Sailboat on lake, Peppers.
- Все изображения переведены в градации серого `float32` в диапазоне `[0, 1]` и приведены к размеру `256x256`.
- Сгенерированы 4 ФРТ через `pyolimp/olimp`: 2 гауссовы и 2 оптические.
- Для каждой пары «изображение-ФРТ» выполнено размытие через `olimp.processing.fft_conv`.
- К каждому размытому изображению добавлен гауссов шум со стандартным отклонением `0.01`, `0.05`, `0.1`.
- Получено `36` сгруппированных троек `оригинал + ФРТ + размытое шумное изображение`.

## Методы восстановления

1. `Wiener` — частотная деконволюция с параметром баланса.
2. `Richardson-Lucy` — итеративная деконволюция по известной ФРТ.
3. `TV (Montalto/FISTA)` — задача `0.5||Hx-y||^2 + λ TV(x)` с FISTA-подобным алгоритмом

## Подбор параметров

Параметры подбирались отдельно для каждого уровня шума по среднему `SSIM` на всём наборе SIPI-троек.

| method              | noise_sigma | best_param |
| ------------------- | ----------- | ---------- |
| Richardson-Lucy     | 0.0100      | 30.0000    |
| Richardson-Lucy     | 0.0500      | 16.0000    |
| Richardson-Lucy     | 0.1000      | 12.0000    |
| TV (Montalto/FISTA) | 0.0100      | 0.0050     |
| TV (Montalto/FISTA) | 0.0500      | 0.0200     |
| TV (Montalto/FISTA) | 0.1000      | 0.0400     |
| Wiener              | 0.0100      | 0.0100     |
| Wiener              | 0.0500      | 0.1000     |
| Wiener              | 0.1000      | 0.2000     |

## Сводная таблица по всем случаям

| method              | psnr    | ssim   | mae    | delta_psnr_vs_blurred | delta_ssim_vs_blurred |
| ------------------- | ------- | ------ | ------ | --------------------- | --------------------- |
| TV (Montalto/FISTA) | 21.8398 | 0.5410 | 0.0576 | 2.9553                | 0.2399                |
| Wiener              | 20.1435 | 0.5138 | 0.0809 | 1.2590                | 0.2127                |
| Richardson-Lucy     | 19.8467 | 0.3624 | 0.0810 | 0.9622                | 0.0613                |
| Blurred+noise       | 18.8845 | 0.3011 | 0.0894 | 0.0000                | 0.0000                |

## Средние метрики по уровням шума

| noise_sigma | method              | psnr    | ssim   | delta_psnr_vs_blurred | delta_ssim_vs_blurred |
| ----------- | ------------------- | ------- | ------ | --------------------- | --------------------- |
| 0.0100      | Wiener              | 22.8421 | 0.5978 | 2.4940                | 0.1297                |
| 0.0100      | TV (Montalto/FISTA) | 22.7308 | 0.5906 | 2.3827                | 0.1225                |
| 0.0100      | Richardson-Lucy     | 22.2559 | 0.5618 | 1.9078                | 0.0937                |
| 0.0100      | Blurred+noise       | 20.3481 | 0.4681 | 0.0000                | 0.0000                |
| 0.0500      | TV (Montalto/FISTA) | 21.6670 | 0.5324 | 2.4131                | 0.2530                |
| 0.0500      | Wiener              | 19.9089 | 0.4970 | 0.6550                | 0.2176                |
| 0.0500      | Richardson-Lucy     | 20.0537 | 0.3328 | 0.7998                | 0.0534                |
| 0.0500      | Blurred+noise       | 19.2539 | 0.2794 | 0.0000                | 0.0000                |
| 0.1000      | TV (Montalto/FISTA) | 21.1214 | 0.4999 | 4.0699                | 0.3441                |
| 0.1000      | Wiener              | 17.6795 | 0.4465 | 0.6280                | 0.2907                |
| 0.1000      | Richardson-Lucy     | 17.2304 | 0.1926 | 0.1789                | 0.0368                |
| 0.1000      | Blurred+noise       | 17.0515 | 0.1558 | 0.0000                | 0.0000                |

## Число побед по кейсам

Победа считалась по максимуму `SSIM` и отдельно по максимуму `PSNR` среди трёх алгоритмов восстановления.

| method              | ssim_wins | psnr_wins |
| ------------------- | --------- | --------- |
| TV (Montalto/FISTA) | 29        | 30        |
| Wiener              | 7         | 6         |
| Richardson-Lucy     | 0         | 0         |

## Интернет-примеры, где каждый алгоритм выглядит лучше остальных

- Wiener: `Stars constellations` (https://commons.wikimedia.org/wiki/File:Stars_constellations.jpg), PSF `sca_strong`, sigma `0.10`, SSIM `0.8469`, отрыв от следующего метода `0.0097`.
- Richardson-Lucy: `Regular glass facade` (https://commons.wikimedia.org/wiki/File:Regular_glass_facade_(Unsplash).jpg), PSF `sca_mild`, sigma `0.01`, SSIM `0.5857`, отрыв от следующего метода `0.0014`.
- TV (Montalto/FISTA): `Landscape` (https://commons.wikimedia.org/wiki/File:Landscape_(Unsplash).jpg), PSF `gauss_iso_mild`, sigma `0.01`, SSIM `0.9268`, отрыв от следующего метода `0.0983`.

Фигуры сохранены в каталоге `outputs/figures`:

- `internet_example_wiener.png`
- `internet_example_richardson_lucy.png`
- `internet_example_tv_montalto.png`

## Структура результатов

- `outputs/prepared/triples` — все `36` сгруппированных троек с `npy/png` и `metadata.json`.
- `outputs/results/restored` — восстановленные изображения по каждому методу.
- `outputs/tables/study_tables.xlsx` — итоговые таблицы в Excel.
- `outputs/tables/*.csv` — те же таблицы в CSV.
- `outputs/figures` — обзорные фигуры и интернет-примеры.

## Источники

- SIPI database (официальный каталог): https://sipi.usc.edu/database/database.php?volume=misc
- SIPI checksums / прямые TIFF-файлы: https://sipi.usc.edu/database/checksums.php
- Pyolimp / olimp (официальный репозиторий): https://github.com/pyolimp/pyolimp
- `skimage.restoration` API: https://scikit-image.org/docs/stable/api/skimage.restoration.html
- Пример Richardson-Lucy из официальной документации scikit-image: https://scikit-image.org/docs/stable/auto_examples/filters/plot_deconvolution.html
- Интернет-фото для отдельных иллюстраций:
  - https://commons.wikimedia.org/wiki/File:Landscape_(Unsplash).jpg
  - https://commons.wikimedia.org/wiki/File:Regular_glass_facade_(Unsplash).jpg
  - https://commons.wikimedia.org/wiki/File:Stars_constellations.jpg
