# Космохакатон (Ростов): мониторинг вегетационной динамики

Веб-сервис для мониторинга состояния сельскохозяйственных полей по спутниковым данным: выбор региона и полигона на карте, автоматический сбор спутниковых и метеоданных, восстановление пропусков во временном ряде NDVI и детекция аномальных периодов вегетации.

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
- [Модель восстановления пропусков](docs/12-gapfill-model.md), [детекция аномалий](docs/13-anomaly-detection.md), [исследовательский отчёт](docs/14-research-report.md)

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

```bash
uv sync --group ml --group dl                                                # LightGBM, CatBoost, torch (CUDA 13.0)
uv run python -m gapfill.train --n-masks 20 --clip -0.1 1.0 --out lgb_v3     # валидация LightGBM (RMSE 0.057)
uv run python -m gapfill.nn_model --epochs 600 --dropout 0.25 --out nn_v2    # валидация SeasonNet (RMSE 0.060, GPU)
uv run python -m gapfill.ensemble lgb_v3 nn_v2                               # смесь на валидации (RMSE 0.055)
uv run python -m gapfill.predict --input data/test_dataset.csv --output submission.csv --n-masks 30 --rounds 5500 --seeds 0 1 2 --out final_lgb   # batch-инференс
for s in 0 1 2 3 4; do uv run python -m gapfill.nn_model --final --epochs 600 --dropout 0.25 --seed $s --out final_nn; done
uv run python -m gapfill.make_submission final_lgb:0.5 final_nn:0.5          # → submission.csv (3 112 строк)
# без обучения, из сохранённых моделей models/ (LightGBM .txt.gz + SeasonNet .pt, ~60 МБ, в репозитории):
uv run python -m gapfill.predict_saved --input data/test_dataset.csv --output submission.csv --models models
uv run pytest tests -q
```

Готовый [`submission.csv`](submission.csv) лежит в корне. На валидации, имитирующей контрольные точки
(15 % известных точек train + test), смесь даёт RMSE 0.053–0.055 на двух масках (GapScore 11–14 в зависимости
от доли выбросов среди скрытых ответов) против 0.092 у baseline «среднее соседей».

## Детекция и интерпретация аномалий (задача 2)

Пакет [`anomaly/`](anomaly/): гармонизированная кривая сезона, норма по истории полигона (или по культуре),
эпизоды «устойчивого и/или сильного» отклонения (Z < −1), причина по правилам с погодой ERA5, фенологией и
региональным контекстом, текст на русском (при `ANTHROPIC_API_KEY` — связное объяснение от Claude).
Метод и проверка — в [docs/13-anomaly-detection.md](docs/13-anomaly-detection.md), результаты — в
[`reports/anomalies/`](reports/anomalies/) (`episodes.csv`, `seasons.csv`, `figures/`).

```bash
uv run python -m anomaly.run                                 # все полигоны → reports/anomalies/
uv run python -m anomaly.evaluate reports/anomalies          # прокси-метрики детектора
uv run python -m anomaly.plots AOI-0065:2024 AOI-0043:2019   # графики сезонов с эпизодами
```

## Веб-сервис (обязательная точка запуска по ТЗ)

```bash
uv sync --group ml --group geo --group service        # FastAPI, STAC-клиенты, rasterio, Open-Meteo
uv run python -m anomaly.run                         # один раз: эпизоды для полигонов кейса
uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Открыть http://127.0.0.1:8000/. Два сценария из ТЗ:

1. **Готовый полигон**: кнопка «Найти поля OSM в видимой области» запрашивает контуры `landuse=farmland` из
   OpenStreetMap (Overpass API), клик по контуру выбирает поле.
2. **Произвольный полигон**: контур рисуется на карте (Leaflet.draw).

После выбора «Собрать данные и проанализировать» сервис сам получает Sentinel-2 L2A (Earth Search STAC,
маска облаков по SCL), Landsat 8/9 C2 L2 и MODIS MOD13Q1 (Planetary Computer STAC), ERA5 (Open-Meteo,
по центроиду поля), строит ряд `primary_ndvi` по приоритету S2 → Landsat → MODIS, восстановленную кривую,
норму по собственной истории поля, находит эпизоды угнетения и объясняет их. Ключи и регистрация не нужны;
нужен доступ в интернет. Семь сезонов (2019–2025) для поля ~1.5 км² собираются примерно за 3 минуты; если источник
недоступен, ряд строится по остальным, а в интерфейсе выводится, что именно собрано. Полигоны кейса (78 анонимных `AOI-xxxx`, координат нет) показываются списком слева:
исходные наблюдения по сенсорам, восстановленные контрольные точки из `submission.csv`, кривая, норма ±1σ,
Z-score, погода ERA5 и эпизоды с причинами.

API: `GET /api/polygons`, `GET /api/polygon/{pid}`, `GET /api/episodes?year=&cause=&severity=`, `GET /api/summary`,
`GET /api/fields?bbox=юг,запад,север,восток`, `POST /api/analyze` (`{"geometry": <GeoJSON Polygon>, "name": "...",
"start_year": 2019, "end_year": 2025}`). Объяснение эпизодов языковой моделью включается переменной окружения
`ANTHROPIC_API_KEY` (`uv sync --group agent`); без неё текст формируется по правилам.

Контейнер: `docker build -t kosmohack . && docker run -p 8000:8000 kosmohack` (`Dockerfile` на образе
`ghcr.io/astral-sh/uv:python3.14-bookworm-slim`; на машине разработки демон Docker не был запущен, сборка не проверена).

## Зависимости

Базовые зависимости ставятся `uv sync`. Остальное разбито на группы в `pyproject.toml` (все версии актуальны на сентябрь 2026 и имеют wheels под Python 3.14):

| Группа | Что внутри | Команда |
|---|---|---|
| `ml` | scikit-learn, LightGBM, CatBoost, XGBoost, statsmodels, whittaker-eilers, optuna, shap | `uv sync --group ml` |
| `dl` | torch, PyPOTS, pygrinder, chronos-forecasting | `uv sync --group dl` |
| `geo` | pystac-client, odc-stac, stackstac, planetary-computer, rasterio, rioxarray, xarray, geopandas, shapely, earthengine-api, openmeteo-requests, osmnx, overpy | `uv sync --group geo` |
| `openeo` | клиент Copernicus Data Space (конфликтует с `geo` по xarray) | `uv sync --group openeo` |
| `service` | FastAPI, uvicorn, pydantic, httpx | `uv sync --group service` |
| `agent` | pydantic-ai, anthropic, mcp | `uv sync --group agent` |
| `dev` | ruff, pytest, pytest-cov, mypy | ставится по умолчанию |

Обзор моделей, источников данных и обоснование выбора — в [docs/09-open-source-landscape.md](docs/09-open-source-landscape.md).

## Структура репозитория

```
.
├── README.md
├── CLAUDE.md               # рабочие заметки по проекту, окружению и данным
├── data/
│   ├── train_dataset.csv   # обучающий датасет (99 955 строк)
│   └── test_dataset.csv    # тестовый датасет = private_features.csv из ТЗ (57 185 строк)
├── docs/                   # ТЗ, критерии, чек-лист, отчёты 08–11, исходные PDF в source/
├── eda/                    # модули разведочного анализа (uv run python -m eda.run_all)
│   └── v1/                 # скрипты первого прохода EDA и сборка дашборда
├── gapfill/                # восстановление primary_ndvi: признаки, LightGBM, SeasonNet, смесь, submission
├── anomaly/                # детекция и интерпретация аномалий: кривые, нормы, эпизоды, погода, причины
├── service/                # веб-сервис: FastAPI (app.py), данные (data.py), сбор для новых полигонов (collect.py), UI (static/)
├── experiments/            # журнал экспериментов по задаче 1 (exp-000…005)
├── tests/                  # pytest для ядра gapfill
├── models/                 # готовые веса: 3 LightGBM (gzip) + 5 SeasonNet (.pt) для инференса без обучения
├── Dockerfile              # образ сервиса и batch-инференса (uv, CPU)
├── submission.csv          # предсказания контрольных точек test (задача 1)
├── reports/
│   ├── eda/                # графики и summary.json, генерируются EDA
│   ├── dashboard/          # template.html + собранный dashboard.html
│   └── anomalies/          # эпизоды, фенометрики сезонов, графики (генерирует anomaly.run)
├── artifacts/              # модели и кэш признаков (не в git)
├── pyproject.toml          # зависимости по группам (uv)
└── uv.lock
```

Структура и статистика датасетов описаны в [docs/03-data.md](docs/03-data.md).
