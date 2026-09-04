# 09. Ландшафт open-source: модели, пакеты, источники данных, стек

**Дата проверки: 2026-09-04.** Все версии, звёзды и даты получены прямыми запросами в этот день:
PyPI JSON (`https://pypi.org/pypi/<pkg>/json`, включая теги wheel-файлов), GitHub REST API (`/repos/<owner>/<repo>`, `/search/repositories`),
Hugging Face API (`/api/models/<id>`), npm registry, STAC-эндпоинты (`/collections`), а также `uv pip compile --python-version 3.14`
и `uv pip install --dry-run` в venv с локальным **CPython 3.14.6**. Там, где проверить не удалось — так и написано: **не проверено**.

Условные обозначения совместимости с Python 3.14:
- **да (wheel)** — на PyPI есть `cp314`/`abi3`/`py3-none-<platform>` wheel для win_amd64 и manylinux x86_64;
- **да (pure)** — чистый Python, резолв под 3.14 прошёл;
- **да (резолв)** — резолв зависимостей под 3.14 прошёл (`uv pip install --dry-run`), но классификатора 3.14 нет;
- **нет** — `requires-python` исключает 3.14 или нет wheel и есть C-расширение.

---

## 1. Предобученные / foundation-модели

### 1.1. EO-модели для спутниковых временных рядов

| Модель | Вход | Пиксель / тайл | Пропуски / маскирование | Размер | Лицензия | GitHub ★ / последний push | pip / веса | Реалистично дообучить за хакатон? |
|---|---|---|---|---|---|---|---|---|
| **Presto** ([nasaharvest/presto](https://github.com/nasaharvest/presto)) | Пиксельный ряд, 15 динамических каналов на шаг: S1 VV/VH, S2 (10 полос), ERA5 (осадки, T2m), NDVI, Dynamic World; статические: SRTM высота/уклон, координаты. Шаг — месяц, окно 12 мес. | **пиксель** | Предобучение MAE с 4 стратегиями структурного маскирования (случайные каналы/шаги, группы каналов, непрерывные интервалы, целые шаги). В статье показана устойчивость к подмножеству месяцев и отсутствию модальностей | **0.40M** параметров | MIT | 279 ★ / 2026-06-10 | Нет на PyPI (`presto` на PyPI — чужой AGPL-пакет!). Установка из репозитория. Веса: HF [nasaharvest/presto](https://huggingface.co/nasaharvest/presto) (MIT) | **Да, на CPU.** Самое точное совпадение с нашими данными (S2+ERA5 пиксельный ряд с пропусками). Дообучение регрессионной головы на маскированный NDVI |
| **Galileo** ([nasaharvest/galileo](https://github.com/nasaharvest/galileo)) | S1 VV/VH, S2 (все полосы кроме B1/B9/B10), NDVI (пространство-время); SRTM, Dynamic World, WorldCereal (пространство); ERA5 осадки/T, TerraClimate (дефицит влаги, влажность почвы, ET), VIIRS (время); LandScan, lat/lon (статика). Предобучение: 24 месячных шага, 96×96 px, 10 м | тайл **и пиксель** (1×1 явно поддержан: CropHarvest, Breizhcrops) | Structured «space/time masking» на предобучении; в коде 3-значная маска (0 — видит энкодер, 1 — игнор, 2 — декодируется) | Nano **0.8M**, Tiny **5.3M**, Base **85M** | MIT | 200 ★ / 2026-04-21 | Нет на PyPI; `uv sync` из репо; веса `hf download nasaharvest/galileo` (HF обновлён 2025-02-13) | **Nano/Tiny — да, на CPU/одной GPU.** Преемник Presto; больше модальностей, чем у нас есть |
| **TerraMind 1.0** ([IBM/terramind](https://github.com/IBM/terramind)) | Одномоментные тайлы: S2 L2A, S2 L1C, S1 GRD, S1 RTC, DEM, RGB (+ генерируемые модальности LULC/NDVI/подписи). **Не временной ряд** | тайл | Any-to-any генерация через диффузию («mental images», не реконструкция ряда); Thinking-in-Modalities | tiny/small/base/large (параметры на карточке не указаны) | Apache-2.0 | 318 ★ / 2026-08-31; GitHub-релизов нет | Через `terratorch` (`BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=True)`); HF [ibm-esa-geospatial/TerraMind-1.0-base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) — 15 764 загрузки/30 дн., обновлён 2025-11-03 | **Не для нашей задачи** (тайловый, одномоментный). Только GPU |
| **Prithvi-EO-2.0** ([NASA-IMPACT/Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)) | HLS v2, 6 полос (Blue, Green, Red, NIR-narrow, SWIR1, SWIR2), 30 м, мультивременные тайлы (3D patch embedding). Варианты **TL** = + эмбеддинги даты (год, DOY) и координат | тайл | MAE-предобучение; dropout на метаданных; про облака/пропуски явно не заявлено | 5M (tiny-TL) … **300M**, **600M** | Apache-2.0 (веса), MIT (репо) | 305 ★ / 2025-02-13 | `terratorch` 1.2.13; HF [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) 18 340 загрузок/30 дн. | Только GPU, тайлы 224×224 — **тяжело для хакатона**, не восстанавливает ряды |
| **Clay v1.5** ([Clay-foundation/model](https://github.com/Clay-foundation/model)) | Одномоментные чипы любого сенсора (полосы задаются длинами волн) + время/lat-lon | тайл | — | — (не проверено) | Apache-2.0 | 613 ★ / 2026-05-11 | PyPI `claymodel` **1.5.0** (2025-05-28, requires >=3.11); HF [made-with-clay/Clay](https://huggingface.co/made-with-clay/Clay) (обновлён 2026-04-05) | Эмбеддинги чипов, **не ряды** |
| **AnySat** ([gastruc/AnySat](https://github.com/gastruc/AnySat)) | S2 и S1 **временные ряды** + VHR (aerial/SPOT/NAIP), 11 сенсоров, 0.2–250 м; для рядов нужен тензор `_dates` (DOY) → произвольные даты | патч (`tile`/`patch`/`dense`) | Про пропуски явно не заявлено | не указан | MIT | 210 ★ / 2025-10-16 | `torch.hub.load('gastruc/anysat', 'anysat', pretrained=True)`; HF [g-astruc/AnySat](https://huggingface.co/g-astruc/AnySat) | Возможен эксперимент, но патчевый и без гарантии по пропускам |
| **DOFA** ([zhu-xlab/DOFA](https://github.com/zhu-xlab/DOFA)) | Одиночное изображение, полосы задаются длинами волн | тайл | — | — | MIT (код), CC-BY-4.0 (веса на HF) | 210 ★ / 2026-07-22 | Встроен в `torchgeo` **0.10.0** (2026-08-14, classifier 3.14) | **Не ряды** |

**Новое в 2025–2026, что стоит знать:**

| Что | Суть | Статус / доступ | Применимость |
|---|---|---|---|
| **TESSERA / TESSERA v2** ([ucam-eo/tessera](https://github.com/ucam-eo/tessera), CVPR 2026; v2: [arXiv 2607.03949](https://arxiv.org/abs/2607.03949)) | Попиксельная FM по рядам S1+S2 → 128-мерные эмбеддинги (Matryoshka 16/32/64/128) на 10 м. Студенты 1.07M–43.83M параметров + учитель 2B | 726 ★ / 2026-08-17, код MIT, **веса CC0** (HF org `geotessera`). PyPI **`geotessera` 0.10.2** (requires >=3.12; резолв под 3.14 ОК) — скачивание **готовых эмбеддингов**: глобально за 2024, США/Европа 2017–2025. Своя инференция очень тяжёлая (≥128 ГБ RAM) | **Не восстанавливает ряды** — только эмбеддинги. Годится как признак для бустинга/аномалий, если наш регион покрыт (Россия за 2024 — вероятно да, глобальное покрытие; **не проверено**) |
| **AlphaEarth Foundations** (Google DeepMind, 2025) | 64-мерные эмбеддинги на 10 м, распространяются как датасет в GEE | Только через GEE (**не проверено** имя ассета) | Признак; требует GEE |
| **Chronos-2** (2025-10), **TimesFM-3** (2026-08-31), **Moirai 2.0**, **TiRex-2** ([arXiv 2607.01204](https://arxiv.org/abs/2607.01204)) | Новое поколение TS-FM — см. 1.3 | | |
| **Delineate Anything v2** ([Lavreniuk/Delineate-Anything](https://github.com/Lavreniuk/Delineate-Anything), ECCV 2026) | Resolution-agnostic FM для границ полей | 150 ★ / 2026-08-10, **AGPL-3.0** | Границы полей (лицензия копилефт) |
| **Global FTW field boundaries 2024/2025** ([arXiv 2605.11055](https://arxiv.org/abs/2605.11055)) | 3.17 млрд полигонов полей, 241 страна, 10 м | CC BY 4.0, GeoParquet + PMTiles, схема fiboa/vecorel, [source.coop/ftw/global-data](https://source.coop/ftw/global-data) | Готовые контуры полей — см. раздел 4 |

### 1.2. Библиотека импутации PyPOTS

| Параметр | Значение |
|---|---|
| Версия | **PyPOTS 1.5** (PyPI 2026-05-05), requires-python >=3.8, BSD-3-Clause |
| GitHub | [WenjieDu/PyPOTS](https://github.com/WenjieDu/PyPOTS) — **2 057 ★**, push 2026-08-31 |
| Python 3.14 | **да (резолв)**: под CPython 3.14.6 ставится с torch 2.14.0, pandas 3.0.5, scikit-learn 1.9.0 |
| Формат входа | 3D-массив `(n_samples, n_steps, n_features)`, пропуски = `NaN` |
| Модели импутации (40+) | **SAITS**, **ImputeFormer**, **CSDI**, **BRITS**, **TimesNet**, iTransformer, PatchTST, DLinear, TimeMixer(++), SegRNN, ModernTCN, Crossformer, FEDformer, Autoformer, Informer, Pyraformer, Nonstationary Transformer, FiLM, FreTS, Koopa, TiDE, SCINet, MICN, TSLANet, FITS, TEFN, HELIX, GP-VAE, US-GAN, M-RNN, GRU-D, CSAI, TCN, Transformer, StemGNN, TRMF; адаптеры к FM: **MOMENT**, Time-LLM, GPT4TS; наивные: Lerp, LOCF/NOCB, Mean, Median |
| CPU | Да (маленькие SAITS/BRITS на рядах длиной ~100–300 точек обучаются на CPU за минуты) |
| Родственные | `pygrinder` 0.7 (генерация масок пропусков MCAR/MAR/MNAR — полезно для честной валидации), `tsdb` 0.x, [WenjieDu/Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation) 426 ★ |

### 1.3. Foundation-модели временных рядов: умеют ли импутацию?

| Модель | PyPI (версия, дата) | Python 3.14 | Веса / лицензия | Размер | Импутация? | Комментарий |
|---|---|---|---|---|---|---|
| **Chronos-2** ([amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting), 5 805 ★) | `chronos-forecasting` **2.3.1** (2026-07-02), >=3.10 | да (резолв) | HF [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) Apache-2.0, 25.1 млн загрузок/30 дн. | 120M, контекст 8192, горизонт 1024; мультивариантный + ковариаты | **Нет API импутации.** `NaN` в контексте допустимы (маска, заполнение нулями) — [discussion #443](https://github.com/amazon-science/chronos-forecasting/discussions/443). Заполнение дыр только трюком «прогноз вперёд + назад» | Лучший по GIFT-Eval среди открытых; хорош как zero-shot базлайн |
| **TimesFM-3** ([google-research/timesfm](https://github.com/google-research/timesfm), 30 942 ★) | `timesfm` **3.0.1** (2026-09-02), >=3.10 | да (резолв) | Код Apache-2.0; **веса TimesFM-3 (330M) — только non-commercial/non-production**; HF-репо 3.0 закрыт (401). TimesFM-2.5-200M — Apache-2.0 | 330M (3.0), 200M (2.5) | Нет (только прогноз) | Для хакатона допустимо, для продукта — нет |
| **Moirai 2.0** ([SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts), 1 586 ★) | `uni2ts` **2.0.0** (2025-11-04) | **нет**: под 3.14 резолвер откатывается на `uni2ts` 1.1.1 с numpy 1.26 / pandas 2.1 (несовместимо с нашим окружением) | HF [Salesforce/moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) — **CC-BY-NC-4.0** | small | Нет (decoder-only) | Пропускаем |
| **TabPFN 3 / 2.5 / 2.6** ([PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN), 7 885 ★) | `tabpfn` **8.5.0** (2026-08-27), >=3.10, classifier 3.14 | да (wheel pure) | Код: Prior Labs License (Apache-2.0 + attribution). **Веса TabPFN-2.5/2.6/3 — non-commercial**; загрузка весов требует логина PriorLabs / `TABPFN_TOKEN` | TabPFN-3: до 1M×200 строк×признаков (CPU — до 5 000 строк) | **Косвенно да**: скрытые точки = тестовые строки табличной регрессии по признакам (DOY, соседи, климатология, погода) | Сильный zero-shot вариант для малых данных |
| **TabPFN-TS** ([PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series), 454 ★, Apache-2.0) | `tabpfn-time-series` **1.2.0** (2026-06-17) | да (резолв) | те же веса TabPFN | — | **Да, естественно**: прогноз = регрессия по календарным признакам, можно предсказывать любые пропущенные t | Использует `tabpfn` или `tabpfn-client` **0.5.1** (облачный API) |
| **Sundial** ([thuml/Sundial](https://github.com/thuml/Sundial), 226 ★) | На PyPI `sundial` — чужой пакет 2019 г. Только HF `transformers` + `trust_remote_code` | зависит от transformers 5.16.1 (classifier 3.14) | HF [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m) Apache-2.0, 38 698 загрузок/30 дн. | 128M | Нет | Flow-matching, вероятностный прогноз |
| **TiRex / TiRex-2** ([NX-AI/tirex](https://github.com/NX-AI/tirex), 299 ★, push 2026-09-04) | `tirex-ts` **1.4.2** (2026-06-09), >=3.10 | да (резолв) | **NXAI Community License**: бесплатно, в т.ч. коммерчески, если выручка < €100M; обязательна атрибуция «Built with technology from NXAI» | 35M (xLSTM), CPU ОК | Нет | TiRex-2 — мультивариантный + ковариаты |
| **Toto** ([DataDog/toto](https://github.com/DataDog/toto), 541 ★) | `toto-ts` **0.2.0** (2026-02-26) | **нет**: жёсткий пин `torch==2.7.0`, у которого нет cp314-wheel | HF [Datadog/Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0) Apache-2.0 | — | Нет | Ориентирован на observability |
| **MOMENT** ([moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment), 835 ★, MIT) | через PyPOTS-адаптер | — | HF | — | **Да** (masked reconstruction, imputation, anomaly) | Единственная TS-FM с нативной реконструкцией |
| **AutoGluon-TimeSeries** ([autogluon/autogluon](https://github.com/autogluon/autogluon), 10 633 ★) | `autogluon.timeseries` **1.6.1** (2026-08-06), **requires-python <3.14** | **нет** для 1.6.1 (под 3.14 uv подбирает старую версию с pandas 2.3.3) | Apache-2.0 | — | Нет | Только в venv 3.13 |

**Вывод по 1.3:** нативную импутацию из FM даёт только MOMENT (через PyPOTS). Chronos-2/TiRex/TimesFM годятся как zero-shot «прогноз с двух сторон», TabPFN-TS — как табличный zero-shot регрессор по признакам.

---

## 2. Классические методы реконструкции NDVI и готовые пакеты

| Метод / пакет | Версия (дата) | requires-python / wheels | Python 3.14 | Лицензия | Комментарий |
|---|---|---|---|---|---|
| **Whittaker–Eilers** — `whittaker-eilers` ([AnBowell/whittaker-eilers](https://github.com/AnBowell/whittaker-eilers), 45 ★, push 2026-01-08) | **0.2.0** (2024-11-28) | >=3.7; **abi3-wheels** `cp37-abi3-win_amd64`, `manylinux_2_17_x86_64`, `aarch64`, macOS | **да (wheel abi3)**, dry-run ОК | MIT | Rust-ядро на разреженных матрицах; веса наблюдений (0 = пропуск → интерполяция), подбор λ по кросс-валидации, произвольная сетка x. **Базлайн №1** |
| **HANTS** | PyPI-пакета **нет** (`pyhants`, `hants` — 404). Референс: [gespinoza/hants](https://github.com/gespinoza/hants) 53 ★, Apache-2.0, последний push 2018 | — | — | — | Гармонический МНК (≈40 строк на `numpy.linalg.lstsq` с итеративным отбросом выбросов ниже кривой). Писать самим |
| **Savitzky–Golay** — `scipy.signal.savgol_filter` | scipy **1.18.1** (2026-08-21) | >=3.12; cp314 wheels win/linux/mac | **да (wheel)** | BSD-3 | Нужна регулярная сетка → сначала линейная интерполяция, потом итеративный SG (Chen et al., 2004) |
| **STL / сезонная декомпозиция** — `statsmodels` | **0.15.0** (2026-08-30) | >=3.10; cp314 wheels | **да (wheel)** | BSD-3 | Для аномалий: остатки STL относительно сезонной компоненты |
| **Сглаживающие сплайны** — `csaps` | 1.3.3 (2025-09-07) | >=3.10, pure | да (резолв, не проверялся отдельно) | MIT | Альтернатива Whittaker |
| **TIMESAT** — `timesat` | 4.4.1 (2026-04-30) | wheels только cp310–cp312; лицензия проприетарная (LicenseRef-Proprietary) | **нет** | проприетарная | Пропускаем |
| **PhenoloPy** ([lewistrotter/PhenoloPy](https://github.com/lewistrotter/PhenoloPy)) | GitHub-only, 100 ★, push 2025-08-12; PyPI `phenolopy` — 404 | xarray-based | — | Apache-2.0 | Фенометрики (SOS/EOS/POS/амплитуда) из xarray-рядов NDVI — идея для объяснения аномалий |
| **modape** (Whittaker + V-curve для MODIS) | 1.0.3 (2023-04-01) | C-расширение, cp314-wheel нет | **нет** без сборки | не указана | Устарел |
| **pyet** | 1.5.0 (2026-05-26) | >=3.10, pure | да (резолв не проверялся) | MIT | **Эвапотранспирация (PET, Penman–Monteith и др.)**, не реконструкция NDVI. Полезен для погодных признаков из ERA5 |
| **spyndex** | 0.12.0 (2026-07-23) | >=3.8, pure | да | не указана на PyPI | Каталог спектральных индексов (NDVI, EVI, NDWI, kNDVI…) |
| **stumpy** (matrix profile) | 1.14.1 (2026-02-08) | >=3.10, classifier 3.14 | **да** | BSD-3 | Аномалии/discords в рядах |
| **pyod** | 3.6.5 (2026-08-17) | >=3.9 | да (резолв) | BSD-2 | IsolationForest, ECOD и др. для табличных аномалий |
| **ruptures** (changepoint) | 1.1.10 (2025-09-10) | **requires-python <3.14** | **нет** | BSD-2 | Только venv 3.13 |
| **tsfresh** | 0.21.2 (2026-05-31) | >=3.9 | да (резолв) | MIT | Автопризнаки рядов |
| `xarray.interpolate_na`, `pandas.interpolate` | xarray 2026.7.0, pandas 3.0.5 | — | да | Apache/BSD | Линейная/сплайн-интерполяция как самый простой базлайн |

---

## 3. Табличные модели

| Пакет | Версия (дата) | requires-python | Wheels под 3.14 (Windows / Linux) | Лицензия | Примечание |
|---|---|---|---|---|---|
| **LightGBM** | **4.7.0** (2026-07-18) | >=3.10, classifier 3.14 | wheels не привязаны к ABI: `py3-none-win_amd64`, `py3-none-manylinux_2_27_x86_64` → **да**; dry-run ОК | MIT (на PyPI не указана) | Быстрый выбор для регрессии пропусков |
| **CatBoost** | **1.2.10** (2026-02-18) | не указан | **`cp314-cp314-win_amd64`, `cp314-manylinux2014_x86_64`, aarch64, macOS universal2** → да | Apache-2.0 | Хорош на категориальных (культура, регион) |
| **XGBoost** | **3.4.1** (2026-08-15) | >=3.12, classifier 3.14 | `py3-none-win_amd64`, `py3-none-manylinux_2_28_x86_64`, `win_arm64` → **да** | Apache-2.0 | |
| **scikit-learn** | **1.9.0** (2026-06-02) | >=3.11, classifier 3.14 | cp314 + cp314t для win/linux/mac → **да** | BSD-3 | |
| **shap** | 0.52.0 (2026-05-28) | >=3.12, classifier 3.14 | abi3 (cp312+) + cp314t; резолв под Windows/Linux 3.14 прошёл | MIT | Объяснение вкладов признаков в аномалию/восстановление |
| **optuna** | 4.9.0 (2026-06-01) | >=3.9, classifier 3.14 | pure → да | не указана на PyPI | Подбор гиперпараметров |
| **TabPFN** | 8.5.0 (2026-08-27) | >=3.10, classifier 3.14 | pure → да | код Apache-2.0+attribution; **веса non-commercial**, нужен токен | См. 1.3 |
| **AutoGluon(-TimeSeries)** | 1.6.1 (2026-08-06) | **<3.14** | **нет** | Apache-2.0 | Только Python 3.13 |
| **scikit-learn-intelex** | 2026.1.0 (2026-06-10) | >=3.7 | cp314-wheel нет | Apache-2.0 | Не нужен |

---

## 4. Источники данных и клиенты «без ручной загрузки»

### 4.1. Спутниковые данные

| Источник | Клиент (версия) | Бесплатно? | Ключ / регистрация | Лимиты | Задержка | Python 3.14 | Комментарии |
|---|---|---|---|---|---|---|---|
| **AWS Earth Search v1** (Element84), `https://earth-search.aws.element84.com/v1` | `pystac-client` **0.9.0** (2025-07-18), `odc-stac` **0.5.3** (2026-07-30), `stackstac` **0.5.1** (2024-08-10, не обновляется, но работает), `rasterio` **1.5.1** (cp314 wheels) | **Да** | **Нет ключа, нет аккаунта AWS** | Не задокументированы; COG-чтение по HTTP из публичного бакета us-west-2 | S2 L2A появляется в течение часов | все — да (pystac-client/odc-stac/stackstac pure; rasterio wheel) | Коллекции (проверено 2026-09-04): `sentinel-2-l2a`, `sentinel-2-c1-l2a`, `sentinel-2-l1c`, `sentinel-2-pre-c1-l2a`, `sentinel-1-grd`, `landsat-c2-l2` (1982–…; **ассеты в s3://usgs-landsat — requester-pays!**), `cop-dem-glo-30/90`, `naip`. Earth Search v0 объявлен deprecated. **Основной источник S2 для нас** |
| **Microsoft Planetary Computer**, `https://planetarycomputer.microsoft.com/api/stac/v1` | `planetary-computer` **1.0.0** (2023-07-05) для подписи SAS + `pystac-client`/`odc-stac` | **Да** (публичный каталог «free for anyone»; Pro — отдельный платный Azure-продукт GeoCatalog) | Нет (анонимные SAS-токены; при наличии ключа подписки лимиты выше — **лимиты не проверены**) | Не задокументированы | — | да (pure) | 136 коллекций (проверено): `sentinel-2-l2a`, **`landsat-c2-l2`** (1982–…, бесплатно, без requester-pays), **`modis-13Q1-061`**, `modis-13A1-061`, **`hls2-s30`/`hls2-l30`** (HLS v2, 2020–…), `sentinel-1-rtc`, `sentinel-1-grd`, `esa-worldcover` (2020, 2021), `io-lulc-annual-v02` (2017–2024), `era5-pds` (1979–…). Хостинг Azure West Europe. **Единая точка для Landsat + MODIS + HLS** |
| **Google Earth Engine** | `earthengine-api` **1.7.42** (2026-08-31, classifier 3.14, Apache-2.0, 3 410 ★); `geemap` **0.38.5** (>=3.12); `xee` 0.1.2 (xarray-бэкенд); `eemont` 2025.7.1; `wxee` 0.5.0 | Да для noncommercial | **Обязателен Google Cloud-проект** (для Python-клиента — с 2025-01-28; незарегистрированные проекты отключены с 2024-06-17), регистрация проекта как noncommercial + анкета верификации (проекты до 2025-04-15 обязаны были верифицироваться до 2025-09-26). Биллинг для noncommercial не нужен | Квоты noncommercial-тира ([Noncommercial Tiers](https://developers.google.com/earth-engine/guides/noncommercial_tiers)) | — | да (pure) | Всё в одном (S2, Landsat, MODIS, HLS, ERA5-Land, WorldCereal 2021 `ESA/WorldCereal/2021/MODELS/v100`, WorldCover), серверные вычисления. **Риск для РФ** — см. 4.4 |
| **openEO @ Copernicus Data Space Ecosystem** | `openeo` **0.51.0** (2026-07-16, classifier 3.14, Apache-2.0, 213 ★); `openeo-gfmap` 0.4.8 | Да, free tier | Регистрация CDSE (открытая, «simple pre-registration»); OAuth | **10 000 openEO-кредитов/мес** (CDSE) + 5 000 (Terrascope); с 2026-03-02 биллинг пересчитан (≈−25 % потребления, очередь бесплатна) | S2 — часы | да (pure) | Есть готовые процессы/UDP для **WorldCereal** ([worldcereal-classification](https://github.com/WorldCereal/worldcereal-classification), 98 ★, MIT, push 2026-09-04 — не на PyPI, установка из GitHub) |
| **Sentinel Hub** (через CDSE) | `sentinelhub` **3.11.5** (2026-03-10, MIT, 910 ★) | Да, free tier CDSE | OAuth-клиент CDSE | **10 000 PU/мес** для бесплатного аккаунта по FAQ CDSE (в другом источнике — 40 000; расхождение не разрешено) | — | да (pure) | Удобные Statistical API для NDVI по полигону |
| **Landsat C2 через USGS** | `landsatxplore` 0.15.0 (2023-04-11; репо push 2024-11-30; issue #126 — endpoint `login` **устарел с февраля 2025**, нужен `login-token`) | Да | Аккаунт USGS + **заявка на доступ к M2M (одобрение занимает дни)** + application token | — | — | pure | **Не рекомендуется**: проще PC `landsat-c2-l2` или GEE. S3 `usgs-landsat` — requester-pays (нужен AWS-аккаунт с биллингом) |
| **MODIS MOD13Q1 v061** | `earthaccess` **0.19.0** (2026-09-04, >=3.12, MIT, 639 ★); AppEEARS API | Да | **Бесплатный Earthdata Login** (для earthaccess/AppEEARS); PC — без логина | AppEEARS — очередь заданий | 16-дневные композиты, лаг ~1–2 нед. | да (pure) | Проще всего — PC `modis-13Q1-061`; альтернатива GEE |
| **HLS v2 (HLSL30/HLSS30)** | `earthaccess` (Earthdata Cloud) или PC `hls2-s30`/`hls2-l30` | Да | Earthdata Login (earthaccess) / нет (PC) | — | 2–3 дня | да | Гармонизированный 30 м ряд Landsat+S2 — отличный источник для плотного ряда |
| **eodag** (унифицированный клиент: CDSE, PC, Earth Search, USGS…) | **4.7.2** (2026-08-28, classifier 3.14, Apache-2.0) | — | по провайдеру | — | — | да | Если хочется один интерфейс поверх нескольких провайдеров |

### 4.2. Погода (ERA5 / реанализ)

| Источник | Клиент | Бесплатно / ключ | Лимиты | Данные | Задержка | Python 3.14 |
|---|---|---|---|---|---|---|
| **Open-Meteo Historical Weather API** `/v1/archive` | `openmeteo-requests` **1.7.5** (2026-01-19, classifier 3.14); сервер [open-meteo/open-meteo](https://github.com/open-meteo/open-meteo) 6 127 ★ **AGPL-3.0** (можно self-host) | **Бесплатно, без ключа** (non-commercial); данные CC BY 4.0 | **<10 000 запросов/день, 5 000/час, 600/мин** ([terms](https://open-meteo.com/en/terms)) | **ERA5** 25 км (1940–), **ERA5-Land** 11 км (1950–), ECMWF IFS 9 км (2017–), CERRA 5 км (Европа, 1985–2021); суточные агрегаты: Tmax/Tmin, осадки, **ET₀**, ветер, солнечное сияние | **~5 дней** (ERA5/ERA5-Land); IFS — без задержки | да |
| **CDS API (ERA5/ERA5-Land, ECMWF)** | `cdsapi` **0.7.7** (2025-09-30, Apache-2.0, 321 ★) | Бесплатно, **нужен аккаунт CDS + Personal Access Token + принятие лицензий датасетов** | Очередь заданий (минуты–часы) | Полный ERA5/ERA5-Land, почасово, NetCDF/GRIB | ~5 дней | да (pure) | Нужен только если нужны сырые почасовые поля/грид |
| **NASA POWER** `/api/temporal/daily/point` | `requests`/`httpx` | **Бесплатно, без ключа** | Не задокументированы; блокировка при «долблении» одной точки; ≤20 параметров/запрос | Суточные 1981–NRT, сетка **0.5°**, community `AG` (агроклиматология) | NRT | — |

### 4.3. Границы полей и маски

| Источник | Что даёт | Доступ | Лицензия | Клиент / Python 3.14 | Покрытие РФ |
|---|---|---|---|---|---|
| **OSM Overpass** | Полигоны `landuse=farmland` (в РФ — грубые, часто «поле-массив») | Публичный Overpass API, без ключа, fair-use | ODbL | `osmnx` **2.1.1** (2026-07-21, MIT, classifier 3.14: `features_from_polygon(..., tags={"landuse": "farmland"})`); `overpy` 0.7 (2023-12); `OSMPythonTools` 0.3.6 (**GPL-3**) | частичное |
| **Global FTW field boundaries 2024 & 2025** ([source.coop/ftw/global-data](https://source.coop/ftw/global-data)) | **3.17 млрд полигонов полей**, 10 м, confidence 0–100 на полигон, 241 страна | Публичный S3-совместимый бакет Source Cooperative; GeoParquet + PMTiles; схема fiboa/vecorel | **CC BY 4.0** | `geopandas` 1.1.4 + `pyarrow` 25.0.1 (читать по bbox из GeoParquet); `fiboa-cli` **0.21.0** (2026-02-16, classifier 3.14) | «241 страна и территория» — вероятно да, **явно не проверено** |
| **Fields of the World (FTW) benchmark** ([ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines), 160 ★, MIT) | Датасет + U-Net базлайны для делинеации | Source Cooperative | CC BY (часть) | `ftw-tools` 1.4.3 — **requires-python <3.13 → нет** | — |
| **ESA WorldCereal** | Маска temporary cropland / crop type 10 м (2021 глобально; сезонные продукты через openEO) | GEE `ESA/WorldCereal/2021/MODELS/v100`, openEO CDSE, Terrascope, Zenodo (community `worldcereal-rdm` — reference data) | CC BY 4.0 | `openeo` / `earthengine-api` | да |
| **ESA WorldCover 2020/2021** | LULC 10 м (класс cropland) | PC `esa-worldcover` (проверено); публичный AWS-бакет `esa-worldcover` (**не перепроверялся сегодня**); GEE | CC BY 4.0 | `odc-stac` | да |
| **EuroCrops** ([maja601/EuroCrops](https://github.com/maja601/EuroCrops), 227 ★) | Полигоны полей с культурами, ЕС | Zenodo | CC-BY-SA-4.0 | geopandas | **нет** |
| **AI4Boundaries** (JRC) | Обучающий датасет границ, ЕС | JRC FTP/Zenodo (не проверено) | CC BY | — | **нет** |
| **Delineate Anything v2** / **agribound** ([montimaj/agribound](https://github.com/montimaj/agribound), 87 ★, Apache-2.0) | Модели делинеации по S2 | GitHub | AGPL-3.0 / Apache-2.0 | torch | — |

### 4.4. Доступность из России (санкции / блокировки) — **практически не проверено**, только известные факты

| Сервис | Известное | Оценка риска |
|---|---|---|
| **Google Earth Engine** | Google ограничил создание новых аккаунтов из РФ (сент. 2024, [Moscow Times](https://www.themoscowtimes.com/2024/09/26/google-restricts-account-creation-in-russia-digital-ministry-says-a86486)); Google Play billing из РФ не работает с 2024-12-26; обсуждается поэтапное ограничение «Google for Business» (дек. 2025). Для noncommercial GEE биллинг не нужен, но нужен Google-аккаунт и Cloud-проект + анкета верификации. YouTube полностью заблокирован с 2026-02-12 — Google-сервисы в целом под давлением | **Высокий**: работоспособно только с уже существующим аккаунтом; не делать GEE единственной зависимостью |
| **CDSE (openEO / Sentinel Hub / STAC)** | Регистрация открыта для всех; ограничения по странам в T&C только для CCM-данных (EU/участники Copernicus). Хостинг CloudFerro (Польша) | **Средний**: явных блокировок не найдено; риск сетевых ограничений РКН |
| **AWS Earth Search / публичные S3-бакеты** | Аккаунт не нужен. Но в 2025 при ограничениях РКН «зарубежные платформы на CDN Cloudflare и Amazon переставали загружаться» ([Wikipedia: 2025 internet restrictions in Russia](https://en.wikipedia.org/wiki/2025_internet_restrictions_in_Russia)); новые AWS-аккаунты из РФ не открыть → requester-pays Landsat недоступен | **Средний**: сам HTTP-доступ к S3 обычно работает, но возможны деградации |
| **Planetary Computer (Azure West Europe)** | Аккаунт не нужен; блокировок не найдено | **Низкий–средний** |
| **Open-Meteo / NASA POWER / Earthdata** | Без ключа (Open-Meteo, POWER); Earthdata — открытая регистрация | **Низкий**; Open-Meteo можно поднять локально (AGPL) |

**Практическая рекомендация:** цепочка фолбэков `Planetary Computer → Earth Search → GEE`, кэш всех скачанных рядов в Parquet/DuckDB, погода — Open-Meteo с фолбэком на NASA POWER; никакой источник не должен быть единственным.

---

## 5. Стек агента и сервиса (коротко)

| Компонент | Версия (дата) | Python 3.14 | Лицензия | Заметки |
|---|---|---|---|---|
| **pydantic-ai** ([pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai), 19 723 ★) | **2.39.0** (2026-09-04), classifier 3.14 | да | MIT | Провайдеры (по `infer_provider_class`): `anthropic`, `openai`, **`ollama`** (нативный `OllamaProvider(base_url='http://localhost:11434/v1')`), `google`, `bedrock`, `groq`, `mistral`, `huggingface`, `openrouter`, `litellm`… Ставить `pydantic-ai-slim[anthropic,openai]` |
| **anthropic SDK** ([anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python), 3 881 ★) | **1.3.0** (2026-09-01), classifier 3.14 | да | MIT | 1.x построен на **`httpx2`**, не `httpx` (готча при собственных HTTP-клиентах). Актуальные модели: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`; adaptive thinking; структурированный вывод через `output_config.format` |
| **MCP Python SDK** ([modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk), 24 205 ★) | **2.1.1** (2026-08-25), classifier 3.14 | да | MIT | Инструменты анализа (NDVI-ряд, погода, аномалии) как MCP-сервер |
| **LangGraph** ([langchain-ai/langgraph](https://github.com/langchain-ai/langgraph), 41 054 ★) | 1.2.11 (2026-08-11) | да (резолв) | MIT | Опционально; pydantic-ai достаточно |
| **openai** | 3.8.0 (2026-09-03), classifier 3.14 | да | Apache-2.0 | Также для OpenAI-совместимых локальных моделей |
| **ollama** | 0.6.2 (2026-04-29) | да (pure) | MIT | Локальные модели (оффлайн-демо) |
| **FastAPI / uvicorn / pydantic** | **0.141.1** (2026-07-29) / **0.52.4** (2026-08-19) / **2.13.5** (2026-08-28) — все с classifier 3.14 | да | MIT / BSD-3 / MIT | |
| **httpx** | 0.28.1 (2024-12-06) | да | BSD-3 | Давно не обновлялся; anthropic использует `httpx2` |
| **Leaflet** ([Leaflet/Leaflet](https://github.com/Leaflet/Leaflet), 45 571 ★) | stable **1.9.4** (2023-05-18); **2.0.0-alpha.1** (2025-08-16) — по-прежнему альфа, ESM-only, без IE, Pointer Events; дата стабильного релиза не объявлена | — | BSD-2 | Для хакатона брать 1.9.4 (плагины) |
| **MapLibre GL JS** ([maplibre/maplibre-gl-js](https://github.com/maplibre/maplibre-gl-js), 11 536 ★) | **6.7.0** (2026-09-02) | — | BSD-3 | Векторные тайлы, WebGL; хорошо для PMTiles границ полей |
| Рисование полигонов | `@geoman-io/leaflet-geoman-free` **2.20.0** (2026-06-23, MIT, 2 428 ★); `@geoman-io/maplibre-geoman-free` 0.9.1 (2026-08-17); `@mapbox/mapbox-gl-draw` **1.5.1** (2025-11-03, ISC, 1 083 ★; работает с MapLibre); **`terra-draw` 1.33.0** (2026-09-01, MIT; адаптеры Leaflet/MapLibre/OpenLayers/Google); `leaflet-draw` 1.0.4 (2018, мёртв) | — | | Рекомендация: terra-draw или leaflet-geoman |
| Графики | **plotly.js-dist-min 4.0.0** (2026-08-24, MIT; новый major, python `plotly` 7.0.0); **echarts 6.1.0** (2026-05-19, Apache-2.0) | — | | |
| Быстрый UI-фолбэк | `streamlit` 1.63.0 (classifier 3.14), `gradio` 6.26.0, `folium` 0.20.0, `ipyleaflet` 0.20.0 | да | | Если фронт не успеваем |

---

## 6. Похожие open-source проекты (≥100 ★, кроме отмеченных)

| Репозиторий | ★ | Лицензия | Push | Что переиспользовать |
|---|---|---|---|---|
| [WenjieDu/PyPOTS](https://github.com/WenjieDu/PyPOTS) | 2 057 | BSD-3 | 2026-08-31 | Импутация (SAITS/ImputeFormer/CSDI), генерация масок (`pygrinder`), протокол оценки на скрытых точках |
| [sentinel-hub/eo-learn](https://github.com/sentinel-hub/eo-learn) | 1 248 | MIT | 2026-01-15 | EOPatch-пайплайн, задачи интерполяции/gap-filling рядов, маски облаков (s2cloudless). PyPI 1.5.7 (2024-09-27) |
| [microsoft/farmvibes-ai](https://github.com/microsoft/farmvibes-ai) | 895 | MIT | 2026-08-27 | Готовые агро-воркфлоу: NDVI-сводки по полю, **SpaceEye** (DL-восстановление безоблачных S2), делинеация, погода. Тяжёлая инфраструктура (K8s) — брать идеи и код модулей |
| [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) | 835 | MIT | 2026-02-10 | TS-FM с нативной реконструкцией/импутацией и детекцией аномалий |
| [ucam-eo/tessera](https://github.com/ucam-eo/tessera) | 726 | MIT (веса CC0) | 2026-08-17 | Попиксельные эмбеддинги S1/S2-рядов как признак (`geotessera`) |
| [WenjieDu/SAITS](https://github.com/WenjieDu/SAITS) / [ermongroup/CSDI](https://github.com/ermongroup/CSDI) | 513 / 461 | MIT / MIT | 2026-08-25 / 2024-03-14 | Референсные реализации импутации |
| [nasaharvest/presto](https://github.com/nasaharvest/presto) + [galileo](https://github.com/nasaharvest/galileo) | 279 + 200 | MIT | 2026-06-10 / 2026-04-21 | Пиксельные FM на S1/S2/ERA5 с маскированием — уровень 3 нашего пайплайна |
| [nasaharvest/cropharvest](https://github.com/nasaharvest/cropharvest) | 237 | CC-BY-SA-4.0 | 2024-04-30 | Глобальный датасет пиксельных рядов S1/S2/ERA5/SRTM (те же признаки, что у Presto) — для предобучения/валидации |
| [MarcCoru/crop-type-mapping](https://github.com/MarcCoru/crop-type-mapping) | 168 | MIT | 2024-07-25 | Self-attention на сырых S2-рядах (BreizhCrops) — архитектура для ряда с пропусками |
| [sentinel-hub/field-delineation](https://github.com/sentinel-hub/field-delineation) / [Lavreniuk/Delineate-Anything](https://github.com/Lavreniuk/Delineate-Anything) / [fieldsoftheworld/ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines) | 165 / 150 / 160 | MIT / AGPL-3.0 / MIT | 2023-08-10 / 2026-08-10 / 2026-08-19 | Делинеация полей по S2 (если пользователь не рисует полигон) |
| [lewistrotter/PhenoloPy](https://github.com/lewistrotter/PhenoloPy) | 100 | Apache-2.0 | 2025-08-12 | Фенометрики SOS/EOS/POS из xarray NDVI — для объяснения аномалий («сдвиг начала сезона на N дней») |
| [gee-community/geemap](https://github.com/gee-community/geemap) | 4 022 | MIT | 2026-09-03 | Если используем GEE: извлечение рядов по полигонам, тайлы на карту |
| [Agri-Hub/Deep-Learning-for-Cloud-Gap-Filling-on-NDVI](https://github.com/Agri-Hub/Deep-Learning-for-Cloud-Gap-Filling-on-Normalized-Difference-Vegetation-Index) (<100 ★: 49) | 49 | MIT | 2024-06-14 | CNN-RNN gap-filling NDVI по S1+S2 — единственный найденный репозиторий точно по нашей теме |
| [WorldCereal/worldcereal-classification](https://github.com/WorldCereal/worldcereal-classification) (<100 ★: 98) | 98 | MIT | 2026-09-04 | Маска пашни/культур через openEO CDSE |

GitHub-поиск по фразам «NDVI gap filling» (6 репо всего), «NDVI anomaly detection» (24), «satellite time series imputation» (3), «crop monitoring dashboard satellite» (27) не даёт репозиториев ≥100 ★ — ниша открыта.

---

## 7. Итог: рекомендуемый shortlist

### 7.1. Пайплайн в три уровня

| Уровень | Что | Пакеты | Почему |
|---|---|---|---|
| **1. Базлайн-классика** (первый день) | (a) Whittaker–Eilers с весами (0 = пропуск), λ по CV; (b) итеративный Savitzky–Golay поверх линейной интерполяции; (c) HANTS (своя реализация на numpy); (d) **климатология по DOY** (многолетняя медианная кривая для пикселя/поля) + интерполяция остатков; (e) STL-остатки для аномалий | `whittaker-eilers`, `scipy`, `statsmodels`, `numpy`, `xarray` | Быстро, объяснимо, CPU. Ожидаемо бьёт «среднее соседей» (RMSE≈0.10) — величину выигрыша нужно измерить на скрытых точках |
| **2. Бустинг с признаками** (основной submit) | LightGBM/CatBoost-регрессия скрытого NDVI по признакам: DOY (sin/cos), климатология в этой точке, значения/расстояние до k соседних наблюдений, длина дыры, агрегаты погоды Open-Meteo (ERA5-Land: T, осадки, ET₀, GDD, сумма осадков за 10/30 дн.), S1 VV/VH (если есть), кропмаска (WorldCereal/WorldCover), lat/lon, статистики поля. Обучение на **синтетических масках**, повторяющих распределение скрытых точек (`pygrinder`). Zero-shot альтернатива — TabPFN-TS | `lightgbm`, `catboost`, `xgboost`, `scikit-learn`, `optuna`, `shap`, `tabpfn-time-series` (опц.) | Лучшее соотношение качество/время; SHAP даёт объяснение аномалии для агента |
| **3. Foundation-модель для дообучения** (если остаётся время) | **Presto** (0.4M, MIT) или **Galileo-Nano/Tiny** (0.8M/5.3M, MIT): вход — пиксельный ряд S2-полос + NDVI + ERA5, регрессионная голова на маскированный NDVI; либо **PyPOTS SAITS/ImputeFormer** с нуля на наших рядах (многомерно: NDVI + полосы + погода); для сравнения — **Chronos-2** zero-shot «вперёд+назад» | `torch` (CPU), `pypots`, `chronos-forecasting`, код Presto/Galileo из GitHub | Прямое совпадение модальностей и маскирования с нашей задачей; дообучение на CPU реально. **Не брать**: TerraMind/Prithvi/Clay/DOFA (тайловые, не ряды), TimesFM-3 и Moirai-2 (non-commercial веса; Moirai ещё и без 3.14), Toto (нет 3.14) |

**Аномалии:** Z-score относительно климатологии по DOY (окно ±8–15 дн., многолетнее μ/σ) → STL-остатки → matrix profile (`stumpy`) для формы; объяснение через погодные признаки (тепловой стресс, засуха, заморозки, переувлажнение), фенометрики (сдвиг SOS/EOS), кропмаску и confidence границ.

### 7.2. Источники данных

| Данные | Основной | Фолбэк |
|---|---|---|
| Sentinel-2 L2A | Earth Search v1 (`sentinel-2-l2a`/`sentinel-2-c1-l2a`, без ключа) через `pystac-client` + `odc-stac` | Planetary Computer `sentinel-2-l2a`; GEE |
| Landsat C2 L2, MODIS MOD13Q1, HLS v2 | Planetary Computer (`landsat-c2-l2`, `modis-13Q1-061`, `hls2-s30`/`hls2-l30`, без ключа) | GEE; `earthaccess` (Earthdata Login) |
| ERA5 / ERA5-Land | Open-Meteo `/v1/archive` (без ключа, ET₀ из коробки, лаг 5 дн.) | NASA POWER; `cdsapi` |
| Границы полей | Полигон пользователя; Global FTW boundaries 2024/2025 (GeoParquet/PMTiles, CC BY 4.0); OSM farmland (`osmnx`) | WorldCover/WorldCereal как маска пашни |

### 7.3. Стек агента

`pydantic-ai-slim[anthropic,openai]` 2.39.0 (провайдер `anthropic` → `claude-sonnet-5`; `ollama` для оффлайн-демо) + `mcp` 2.1.1 (аналитические инструменты как MCP-сервер) + FastAPI 0.141.1/uvicorn 0.52.4; фронт — Leaflet 1.9.4 + terra-draw 1.33.0 (или MapLibre 6.7.0 + PMTiles границ) + Plotly.js 4.0.0.

### 7.4. Пакеты для `pyproject.toml` (точные последние версии на 2026-09-04)

Все пакеты ниже прошли `uv pip compile --python-version 3.14` для Windows и Linux, ключевые — `uv pip install --dry-run` под CPython 3.14.6.

```toml
[project]
requires-python = ">=3.14"

[dependency-groups]
eda = [
  "numpy==2.5.2",          # да (cp314 wheel)
  "pandas==3.0.5",         # да (cp314 wheel)
  "pyarrow==25.0.1",       # да (cp314 wheel)
  "scipy==1.18.1",         # да (cp314 wheel)
  "matplotlib==3.11.1",    # да (cp314 wheel)
  "seaborn==0.13.2",       # да (pure; релиз 2024-01, живёт)
  "plotly==7.0.0",         # да (pure)
  "statsmodels==0.15.0",   # да (cp314 wheel)
  "polars==1.44.1",        # да (резолв)
  "duckdb==1.5.5",         # да (cp314 wheel)
]
ml = [
  "scikit-learn==1.9.0",   # да (cp314 wheel)
  "lightgbm==4.7.0",       # да (py3-none-win_amd64 / manylinux)
  "catboost==1.2.10",      # да (cp314 wheel)
  "xgboost==3.4.1",        # да (py3-none wheel, >=3.12)
  "whittaker-eilers==0.2.0", # да (abi3 wheel)
  "shap==0.52.0",          # да (abi3)
  "optuna==4.9.0",         # да (pure)
  "stumpy==1.14.1",        # да (pure)
  "pyod==3.6.5",           # да (резолв)
  "tsfresh==0.21.2",       # да (резолв)
  "spyndex==0.12.0",       # да (pure)
  "torch==2.14.0",         # да (cp314 wheel; CPU-индекс через [[tool.uv.index]])
  "pypots==1.5",           # да (резолв)
  "pygrinder==0.7",        # да (pure) — маски пропусков
  "chronos-forecasting==2.3.1",  # да (резолв) — zero-shot сравнение
]
ml-extra = [               # опционально, по времени
  "tabpfn==8.5.0",               # да; веса non-commercial, нужен TABPFN_TOKEN
  "tabpfn-time-series==1.2.0",   # да (резолв)
  "tirex-ts==1.4.2",             # да (резолв); NXAI Community License
  "geotessera==0.10.2",          # да (резолв, >=3.12) — эмбеддинги TESSERA
  "terratorch==1.2.13",          # да (резолв) — только если пробуем Prithvi/TerraMind
]
geo = [
  "pystac-client==0.9.0",  # да (pure)
  "pystac==1.15.2",        # да (pure)
  "odc-stac==0.5.3",       # да (pure)
  "stackstac==0.5.1",      # да (pure; не обновляется с 2024-08)
  "rasterio==1.5.1",       # да (cp314 wheel)
  "rioxarray==0.23.0",     # да (pure, >=3.12)
  "xarray==2026.7.0",      # да (pure)
  "dask==2026.8.0",        # да (pure)
  "zarr==3.3.0",           # да (pure, >=3.12)
  "geopandas==1.1.4",      # да (pure)
  "shapely==2.1.2",        # да (cp314 wheel)
  "pyproj==3.7.2",         # да (cp314 wheel)
  "pyogrio==0.13.0",       # да (abi3 wheel)
  "planetary-computer==1.0.0",  # да (pure; 2023, но работает)
  "earthaccess==0.19.0",   # да (pure, >=3.12)
  "openmeteo-requests==1.7.5",  # да (pure)
  "osmnx==2.1.1",          # да (pure)
  "overpy==0.7",           # да (pure)
  "fiboa-cli==0.21.0",     # да (pure) — валидация/чтение границ FTW
  "eodag==4.7.2",          # да (pure) — опционально
]
geo-optional = [
  "earthengine-api==1.7.42",  # да (pure) — только при наличии Google-аккаунта/проекта
  "geemap==0.38.5",           # да (pure, >=3.12)
  "openeo==0.51.0",           # да (pure) — CDSE/WorldCereal
  "sentinelhub==3.11.5",      # да (pure) — CDSE Sentinel Hub
  "cdsapi==0.7.7",            # да (pure) — сырые ERA5 через CDS
]
service = [
  "fastapi==0.141.1",      # да
  "uvicorn==0.52.4",       # да
  "pydantic==2.13.5",      # да
  "httpx==0.28.1",         # да
]
agent = [
  "pydantic-ai-slim[anthropic,openai]==2.39.0",  # да
  "anthropic==1.3.0",      # да (httpx2!)
  "mcp==2.1.1",            # да
  "openai==3.8.0",         # да
  "ollama==0.6.2",         # да (pure)
  # "langgraph==1.2.11",   # да (резолв) — только если понадобится граф
]
dev = [
  "pytest==9.1.1",         # да
  "ruff==0.16.6",          # да
]
```

**Не совместимо с Python 3.14 (и что делать):**

| Пакет | Причина | Решение |
|---|---|---|
| `autogluon.timeseries` 1.6.1 | `requires-python <3.14` | отдельный venv `uv venv --python 3.13` в подпапке `experiments/` |
| `uni2ts` 2.0.0 (Moirai 2) | под 3.14 откатывается к 1.1.1/numpy 1.26 | venv 3.13; или пропустить (веса NC) |
| `toto-ts` 0.2.0 | пин `torch==2.7.0` без cp314 | venv 3.13; или пропустить |
| `timesat` 4.4.1 | wheels только cp310–312, проприетарный | пропустить, Whittaker покрывает |
| `ftw-tools` 1.4.3 | `requires-python <3.13` | venv 3.12 только для делинеации |
| `ruptures` 1.1.10 | `requires-python <3.14` | venv 3.13 или сборка из sdist (нужен компилятор) |
| `modape` 1.0.3 | C-расширение без wheel, 2023 | пропустить |
| Presto / Galileo / AnySat / WorldCereal | нет на PyPI | `uv pip install git+https://github.com/...` или `uv sync` в клоне; веса с HF |

Общее правило: если wheel под 3.14 нет — (1) проверить `uv pip install --dry-run`, (2) попробовать sdist (на Windows нужен MSVC Build Tools, на Linux — gcc), (3) изолировать в venv 3.13 с отдельным `pyproject`, (4) в крайнем случае — Docker/Colab для одной модели.

---

## Приложение: методика проверки

- PyPI: `curl -s https://pypi.org/pypi/<pkg>/json` → `info.version`, `info.requires_python`, classifiers `Programming Language :: Python :: 3.14`, имена файлов последнего релиза (теги `cp314`, `abi3`, `py3-none-<platform>`), дата загрузки.
- GitHub: `GET /repos/<owner>/<repo>` → `stargazers_count`, `license.spdx_id`, `pushed_at`; `GET /search/repositories?q=...&sort=stars` для раздела 6 (без токена: 60 запросов/час core, 10/мин search — лимиты были исчерпаны на нескольких запросах, часть поисков не выполнена).
- Hugging Face: `GET /api/models/<id>` → `downloads` (30 дней), `likes`, `lastModified`, тег `license:*`.
- npm: `GET https://registry.npmjs.org/<pkg>` → `dist-tags`, `time`.
- STAC: `GET .../collections` на Earth Search v1 и Planetary Computer — фактический список коллекций на дату проверки.
- Совместимость с 3.14: `uv pip compile --python-version 3.14 --python-platform {windows,linux}` для каждого пакета отдельно + `uv pip install --dry-run` в `uv venv --python 3.14` (CPython 3.14.6, Windows x86_64).
- Документация: Context7 (`/pydantic/pydantic-ai`), карточки моделей на HF, README/статьи по ссылкам в тексте.
