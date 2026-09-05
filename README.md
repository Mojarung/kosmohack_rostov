# Космохакатон (Ростов): мониторинг вегетационной динамики

Веб-сервис для мониторинга состояния сельскохозяйственных полей по спутниковым данным: выбор региона и полигона на карте, автоматический сбор спутниковых и метеоданных, восстановление пропусков во временном ряде NDVI и детекция аномальных периодов вегетации.

**Запуск всего продукта одной командой:** `docker compose up --build` → http://localhost:8000/

**Стек.** Бэкенд: Python 3.14, FastAPI, LightGBM, PyTorch, odc-stac/rasterio, pandas. Интерфейс: React 19,
Vite, TypeScript, MUI X Charts, AntV L7 (карта на тайлах OpenStreetMap), GSAP, TanStack Query.
Окружение и зависимости: uv (`pyproject.toml` + `uv.lock`) и npm (`web/package-lock.json`), образ — Docker.

## Документация

Полное ТЗ, критерии оценки, чек-лист сдачи и отчёты — в папке [`docs/`](docs/README.md). Рабочие заметки для команды и Claude Code — в [`CLAUDE.md`](CLAUDE.md).

- [Постановка задачи](docs/01-task-statement.md)
- [Теория и глоссарий](docs/02-theory-and-glossary.md)
- [Данные и формат submission](docs/03-data.md)
- [Метрика оценки](docs/04-metric.md)
- [Технические требования](docs/05-technical-requirements.md)
- [Критерии оценки](docs/06-evaluation-criteria.md)
- [Чек-лист сдачи](docs/07-submission-checklist.md)
- [Отчёт EDA](docs/08-eda-report.md), [обзор open-source](docs/09-open-source-landscape.md), [сравнение двух EDA](docs/10-branch-comparison.md), [заметки первого EDA](docs/11-eda-v1-notes.md)
- [Модель восстановления пропусков](docs/12-gapfill-model.md), [детекция аномалий](docs/13-anomaly-detection.md), [исследовательский отчёт](docs/14-research-report.md), [вопросы к экспертам](docs/15-consultation-questions.md)
- [Улучшенная модель: эксперименты, проверка и отдельные веса](docs/16-model-improvement.md)

## Разведочный анализ (EDA)

```bash
uv sync                         # Python 3.14 + зависимости из pyproject.toml / uv.lock
uv run python -m eda.run_all    # графики в reports/eda/figures/, числа в reports/eda/summary.json
```

Выводы и графики — в [docs/08-eda-report.md](docs/08-eda-report.md). Скрипты первого прохода анализа и сборка дашборда — в [`eda/v1/`](eda/v1/README.md).

Интерактивный дашборд «NDVI-атлас полей»: [`reports/dashboard/dashboard.html`](reports/dashboard/dashboard.html) (самодостаточный файл, открывается двойным кликом; пересборка `uv run python eda/v1/build_dashboard_data.py`).

## Восстановление пропусков `primary_ndvi` (задача 1)

Пакет [`gapfill/`](gapfill/): признаки по ряду полигона, циклам съёмки и «шуму дня» других полигонов,
LightGBM + нейросеть по сезонной сетке (SeasonNet), смесь 0.5/0.5. Метод, валидация и результаты —
в [docs/12-gapfill-model.md](docs/12-gapfill-model.md), журнал экспериментов — в [`experiments/`](experiments/README.md).

**Входные данные.** `data/test_features_new.csv` — вторая версия `private_features.csv` (обновление организаторов
2026-09-05: 20 полигонов, история 2010–2024, 2 323 контрольные точки); по ней считается метрика. Первая версия
`data/test_dataset.csv` больше не оценивается и используется только как дополнительные известные точки
(параметр `--extra`; чтобы отключить — `--extra` без значений). Сравнение версий — [docs/03](docs/03-data.md).
Выход — `submission.csv` (`anon_polygon_id,date,primary_ndvi_pred`, только строки `is_synthetic_gap = True`).

```bash
uv sync --group ml --group dl                                                # LightGBM, CatBoost, torch (CUDA 13.0)
uv run python -m gapfill.train --n-masks 20 --clip -0.1 1.0 --out lgb_v4     # валидация LightGBM (RMSE 0.055)
uv run python -m gapfill.nn_model --epochs 600 --dropout 0.25 --out nn_v4    # валидация SeasonNet (RMSE 0.059, GPU)
uv run python -m gapfill.ensemble lgb_v4 nn_v4                               # смесь на валидации (RMSE 0.054)
uv run python -m gapfill.predict --input data/test_features_new.csv --output submission.csv --n-masks 30 --rounds 5500 --seeds 0 1 2 --out final_lgb   # обучение LightGBM + предсказание
for s in 0 1 2 3 4; do uv run python -m gapfill.nn_model --final --epochs 600 --dropout 0.25 --seed $s --out final_nn; done
uv run python -m gapfill.make_submission final_lgb:0.5 final_nn:0.5          # → submission.csv (2 323 строки)
# batch-инференс без обучения, из сохранённых моделей models/ (LightGBM .txt.gz + SeasonNet .pt, ~60 МБ, в репозитории):
uv run python -m gapfill.predict_saved --input data/test_features_new.csv --output submission.csv --models models
uv run pytest tests -q
```

Готовый [`submission.csv`](submission.csv) лежит в корне (для первой версии test —
[`reports/gapfill/submission_v1_test_dataset.csv`](reports/gapfill/submission_v1_test_dataset.csv)). На валидации,
имитирующей контрольные точки (15 % известных точек всех файлов, 9 199 точек), смесь даёт RMSE 0.054, на страте
состава test (новые полигоны с историей) 0.056 → GapScore ≈ 13, против 0.093 у baseline «среднее соседей» (exp-007).

### Улучшенная модель — exp-008

Отдельный пакет `models/improved/` содержит LightGBM с 427 признаками и ResidualSeasonNet.
Исходные `models/` и `submission.csv` сохранены; веб-сервис продолжает использовать прежние артефакты.
Разбор обучения, контрольных масок и ограничений — [отчёт](docs/16-model-improvement.md).

```bash
uv sync --frozen --no-default-groups --group infer --group torch
uv run --no-sync python -m gapfill.predict_improved --output submission_improved.csv --model-only-output submission_model.csv
# Только обычный ML, без опубликованных исторических агрегатов:
uv run --no-sync python -m gapfill.predict_improved --no-calibration --output submission_model.csv
```

**Важно:** `submission_improved.csv` использует mean/std из первой версии `data/test_dataset.csv`,
которые содержат информацию о скрытых значениях новой версии. Это специфичная для данного набора
калибровка, не переносимая на новые поля. Перед конкурсной отправкой её допустимость нужно подтвердить
у организаторов. `submission_model.csv` — тот же ансамбль **без этой калибровки**. Результат закрытой
платформы неизвестен; локальные метрики и контрольные прогнозы — в
[`reports/gapfill/improvement/`](reports/gapfill/improvement/).

На macOS, если LightGBM не находит `libomp.dylib`, перед запуском:

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/torch/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

## Детекция и интерпретация аномалий (задача 2)

Пакет [`anomaly/`](anomaly/): гармонизированная кривая сезона, норма по истории полигона (или по культуре),
эпизоды «устойчивого и/или сильного» отклонения (Z < −1), причина по правилам с погодой ERA5, фенологией и
региональным контекстом, текст на русском по правилам. Интерфейс показывает эпизоды и объяснения по правилам.
Метод и проверка — в [docs/13-anomaly-detection.md](docs/13-anomaly-detection.md), результаты — в
[`reports/anomalies/`](reports/anomalies/) (`episodes.csv`, `seasons.csv`, `figures/`).

```bash
uv run python -m anomaly.run                                 # все полигоны → reports/anomalies/
uv run python -m anomaly.evaluate reports/anomalies          # прокси-метрики детектора
uv run python -m anomaly.plots AOI-0065:2024 AOI-0043:2019   # графики сезонов с эпизодами
```

## Веб-сервис (обязательная точка запуска по ТЗ)

### Запуск всего продукта одной командой

```bash
docker compose up --build      # соберёт интерфейс и бэкенд, поднимет http://localhost:8000/
```

Образ собирается в два этапа: `node:24-alpine` собирает интерфейс (`web/`), затем образ `uv` с Python 3.14
ставит зависимости (`infer`, `geo`, `service` плюс CPU-сборка torch) и получает готовую статику. Поля,
добавленные пользователем, лежат в именованном томе и переживают перезапуск. Нужен только интернет для внешних
каталогов данных; ключи и регистрация не требуются.

Batch-инференс в том же контейнере:

```bash
docker compose exec app uv run python -m gapfill.predict_saved   --input data/test_features_new.csv --output submission.csv
```

### Пересчёт метрики задачи 1

```bash
uv run --no-sync python -m gapfill.metrics
```

Считает RMSE и GapScore по сохранённым предсказаниям отложенной выборки
(`reports/gapfill/improvement/validation_*.csv`) — без обучения и инференса, за доли секунды.
Обновляет `reports/gapfill/validation.json`, откуда числа берёт интерфейс.
Заголовочная цифра — модель с калибровкой историческими агрегатами, рядом всегда показывается
результат чистого ансамбля. Это отложенная выборка, а не приватный лидерборд.

### Запуск без Docker

```bash
uv sync --group ml --group geo --group service --group torch   # FastAPI, STAC-клиенты, rasterio, Open-Meteo, torch (CPU)
uv run python -m anomaly.run                                # один раз: эпизоды для полигонов кейса
cd web && npm ci && npm run build && cd ..                  # интерфейс (Node 20+); без этого шага откроется резервный HTML
uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Открыть http://127.0.0.1:8000/. Для разработки интерфейса отдельно: `cd web && npm run dev` (порт 5173,
запросы `/api` проксируются на 8000). Если сборки `web/dist` нет, сервис отдаёт простой резервный интерфейс
на одном HTML-файле (он же всегда доступен по адресу `/legacy`).

### Интерфейс

React 19 + Vite, графики — MUI X Charts, карта — AntV L7 (WebGL) на тайлах OpenStreetMap, анимации — GSAP.
Иконочных шрифтов нет: все иллюстрации нарисованы в проекте (`web/src/components/art/`). Экраны:

| Экран | Что показывает |
|---|---|
| Обзор | Метрики решения, причины угнетения по всем полям, список полей кейса с поиском и фильтром |
| Поле | Сезон по годам: наблюдения по сенсорам, восстановленная кривая, норма ±1σ, восстановленные контрольные точки, Z-score, погода ERA5, эпизоды с объяснением |
| Новая территория | Карта, поиск контуров OSM, рисование полигона, сбор данных, набор «Мои поля» |
| Аномалии | Все эпизоды с фильтрами по году, причине и тяжести |
| Как это работает | Пайплайн, метрики обеих задач, источники данных |

Альтернативный запуск бэкенда — `uv run --locked --group service python -m service`: тот же сервер плюс
проверка зависимостей автосбора при старте, готовность — `GET /api/health`. Повторный запуск без синхронизации —
`uv run --no-sync python -m service`. Эпизоды полигонов кейса уже включены в репозиторий; пересчёт при
необходимости — `uv run --no-sync python -m anomaly.run`.

Резервный интерфейс берёт графики из установленного пакета Plotly через `/vendor/plotly.min.js`.
Поиск контуров OSM не импортирует спутниковый сборщик. Для Sentinel-2 выбираются публичные HTTPS COG;
JP2-дубликаты и сцены, доступные только через S3 с авторизацией, исключаются до чтения.
При отказе Overpass поиск переключается между публичными серверами VK Maps, Private.coffee и overpass-api.de;
успешные ответы по рамке карты кэшируются до пяти минут. При слишком широком обзоре кнопка поиска
сама приближает карту вокруг её центра до допустимого размера области.

Два сценария из ТЗ:

1. **Готовый полигон**: в «Новая территория» кнопка «Найти поля OSM в видимой области» запрашивает `landuse=farmland` из
   OpenStreetMap (Overpass API), клик по контуру выбирает поле.
2. **Произвольный полигон**: контур рисуется на карте (`@antv/l7-draw`).

После выбора «Собрать данные и проанализировать» сервис сам получает Sentinel-2 L2A (Earth Search STAC,
маска облаков по SCL), Landsat 8/9 C2 L2 и MODIS MOD13Q1 (Planetary Computer STAC), ERA5 (Open-Meteo,
по центроиду поля), строит ряд `primary_ndvi` по приоритету S2 → Landsat → MODIS, восстановленную кривую,
норму по собственной истории поля, находит эпизоды угнетения и объясняет их. Семь сезонов для поля ~1.5 км²
собираются примерно за 3 минуты; если источник недоступен, ряд строится по остальным, а в интерфейсе
выводится, что именно собрано.

Проанализированные поля попадают в набор **«Мои поля»** (`artifacts/service/polygons/*.json`, не в git): их можно
открыть без повторного сбора данных и удалить; на карте контур раскрашен по состоянию последнего сезона
(зелёный — норма, оранжевый — умеренное угнетение, красный — критическое), клик по контуру открывает поле.
Полигоны кейса (78 анонимных `AOI-xxxx`, координат нет) показываются списком: исходные наблюдения по сенсорам,
восстановленные контрольные точки из `submission.csv` (и из `reports/gapfill/submission_*.csv` для первой версии test),
кривая, норма ±1σ, Z-score, погода ERA5 и эпизоды с причинами.

API: `GET /api/health`, `GET /api/polygons`, `GET /api/polygon/{pid}`, `GET /api/episodes?year=&cause=&severity=`, `GET /api/summary`,
`GET /api/meta`, `GET /api/fields?bbox=юг,запад,север,восток`, `POST /api/analyze` (`{"geometry": <GeoJSON Polygon>,
"name": "...", "start_year": 2019, "end_year": 2025}`), набор пользователя `GET /api/user-polygons`,
`GET|DELETE /api/user-polygons/{uid}`. Объяснение эпизодов языковой моделью включается переменной окружения
`ANTHROPIC_API_KEY` (`uv sync --group agent`); без неё текст формируется по правилам.

### Дополнительные погодные графики

На экране поля слева — графики NDVI и погоды с переключателем, справа — эпизоды угнетения.
Сверху — последнее наблюдение NDVI, осадки, накопленное тепло и сухой период.
Панели прокручиваются внутри экрана; карта Sentinel-2 раскрывается под NDVI. Выбор контура и список
сохранённых полей доступны на вкладке «Новая территория», включая поля прежнего интерфейса.
Наведение связывает даты двух графиков и показывает значения в карточках. Кнопка в эпизоде приближает
соответствующий период на обоих графиках; «Весь сезон» возвращает исходный масштаб.

Режимы погодного графика:

- **Осадки за 30 дней**, мм, со сравнением по прошлым годам. Под графиком — самый длинный сухой период сезона.
- **Накопленное тепло**: сумма `max(Tср − 10 °C, 0)` от 1 апреля, °C·дни. Это общий показатель тепла сезона,
  без автоматического определения культуры или даты посева. Форма не требуется.

Сухой период — дни подряд с осадками менее 1 мм; пропуски разрывают серию.
Если собрана ET₀, первый график можно переключить на **осадки минус испарение за 30 дней**.

Серая линия — среднее предыдущих лет (до 30), диапазон — 10–90-й процентили.
Текущий год исключён, сравнение идёт по календарным датам. Показатели не изменяют детектор аномалий и модель NDVI.
ET₀ учитывает температуру, радиацию, влажность и ветер; водный баланс не равен влажности почвы или потребности в поливе.

Источник: [ERA5 через Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), полные годы,
средняя/минимальная/максимальная температура, осадки и ET₀. Пропуски остаются неизвестными.
Для анонимных AOI используются имеющиеся средняя температура и осадки. ET₀ и Tmin/Tmax не выдумываются.
Если погоды нет вообще, погодная часть скрывается. Z-score, исходные спутниковые точки, контрольные
восстановления модели и температура/осадки по дням доступны в «Данные и расчёт». Основные наблюдения NDVI
приведены к шкале Sentinel-2; исходные значения доступны при наведении и в подробностях.
Объяснения эпизодов свёрнуты и формулируются как гипотезы. Формулы и сам детектор не изменены.
Собранные поля доступны в прежнем списке по названию; отчёты и параметры графиков сохранены в
`artifacts/fields/`, погодный кэш — в `artifacts/weather/`.

API графиков: `GET /api/polygon/{pid}/agro?year=2025`, `POST /api/polygon/{pid}/agro`
(`year`, `profile`, `sowing_date`, `base`), `POST /api/polygon/{pid}/weather-refresh`.
Формулы и границы изменений — в [плане дополнения графиков](docs/18-farmer-report-implementation.md).
E2E-проверки через Playwright: [tests/e2e/README.md](tests/e2e/README.md).

## Зависимости

Базовые зависимости ставятся `uv sync`. Остальное разбито на группы в `pyproject.toml` (все версии актуальны на сентябрь 2026 и имеют wheels под Python 3.14):

| Группа | Что внутри | Команда |
|---|---|---|
| `ml` | scikit-learn, LightGBM, CatBoost, XGBoost, statsmodels, whittaker-eilers, optuna, shap | `uv sync --group ml` |
| `infer` | только LightGBM и scikit-learn — минимум для инференса и детекции, эту группу берёт образ Docker | `uv sync --group infer` |
| `dl` | torch, PyPOTS, pygrinder, chronos-forecasting (эксперименты) | `uv sync --group dl` |
| `torch` | только torch — хватает для инференса SeasonNet из `models/` | `uv sync --group torch` |

Индекс сборки torch задаётся переменной `UV_TORCH_BACKEND`: `UV_TORCH_BACKEND=cu130 uv sync --group dl` — сборка
под CUDA 13.0 (RTX 5070, для обучения нейросети), `UV_TORCH_BACKEND=cpu uv sync --group torch` — CPU. В образе
Docker torch ставится из индекса CPU отдельной строкой, чтобы в контейнер не попадали пакеты `nvidia-*`.
| `geo` | pystac-client, odc-stac, stackstac, planetary-computer, rasterio, rioxarray, xarray, geopandas, shapely, earthengine-api, openmeteo-requests, osmnx, overpy | `uv sync --group geo` |
| `openeo` | клиент Copernicus Data Space (конфликтует с `geo` по xarray) | `uv sync --group openeo` |
| `service` | FastAPI, uvicorn, pydantic, httpx, Plotly + группы `geo` и `ml` | `uv sync --group service` |
| `agent` | pydantic-ai, anthropic, mcp | `uv sync --group agent` |
| `dev` | ruff, pytest, pytest-cov, mypy | ставится по умолчанию |

Обзор моделей, источников данных и обоснование выбора — в [docs/09-open-source-landscape.md](docs/09-open-source-landscape.md).

Системные зависимости: только `uv` (сам ставит Python 3.14 по `.python-version`); GDAL/PROJ приходят внутри wheels
`rasterio`/`pyproj`, отдельная установка не нужна. GPU не обязателен: инференс `gapfill.predict_saved`, сервис и
детекция аномалий работают на CPU; обучение SeasonNet на GPU (CUDA 13.0) занимает ~8 минут на seed, на CPU — дольше.
Внешние сервисы (Earth Search, Planetary Computer, Open-Meteo, Overpass) — без ключей; единственная необязательная
переменная окружения — `ANTHROPIC_API_KEY`.

## Структура репозитория

```
.
├── README.md
├── CLAUDE.md               # рабочие заметки по проекту, окружению и данным
├── data/
│   ├── train_dataset.csv        # обучающий датасет (99 955 строк)
│   ├── test_features_new.csv    # тестовый датасет = private_features.csv, вторая версия (49 190 строк, 2 323 контрольные точки)
│   └── test_dataset.csv         # первая версия test (57 185 строк): только дополнительные известные точки
├── docs/                   # ТЗ, критерии, чек-лист, отчёты 08–15, исходные PDF в source/
├── eda/                    # модули разведочного анализа (uv run python -m eda.run_all)
│   └── v1/                 # скрипты первого прохода EDA и сборка дашборда
├── gapfill/                # восстановление primary_ndvi: признаки, LightGBM, SeasonNet, смесь, submission
├── anomaly/                # детекция и интерпретация аномалий: кривые, нормы, эпизоды, погода, причины
├── service/                # бэкенд: FastAPI (app.py), данные (data.py), сбор (collect.py), набор полигонов (polygons.py), сводка (meta.py), резервный UI (static/)
├── web/                    # интерфейс: React 19 + Vite, MUI X Charts, AntV L7, GSAP (src/pages, src/components)
├── experiments/            # журнал экспериментов (exp-000…007 задача 1, exp-100…104 задача 2 и сервис)
├── tests/                  # pytest: ядро gapfill, детектор аномалий, набор полигонов сервиса
├── models/                 # готовые веса: 3 LightGBM (gzip) + 5 SeasonNet (.pt) для инференса без обучения
├── Dockerfile              # образ всего продукта: сборка интерфейса + сервис (uv, CPU)
├── docker-compose.yml      # запуск одной командой: docker compose up --build
├── submission.csv          # предсказания контрольных точек test (задача 1, вторая версия test)
├── reports/
│   ├── eda/                # графики и summary.json, генерируются EDA
│   ├── dashboard/          # template.html + собранный dashboard.html
│   ├── gapfill/            # submission для первой версии test (сверка с ответами организаторов)
│   └── anomalies/          # эпизоды, фенометрики сезонов, графики (генерирует anomaly.run)
├── artifacts/              # модели и кэш признаков (не в git)
├── pyproject.toml          # зависимости по группам (uv)
└── uv.lock
```

Структура и статистика датасетов описаны в [docs/03-data.md](docs/03-data.md).
