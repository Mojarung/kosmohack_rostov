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
региональным контекстом, текст на русском по правилам. Интерфейс показывает эпизоды и объяснения по правилам.
Метод и проверка — в [docs/13-anomaly-detection.md](docs/13-anomaly-detection.md), результаты — в
[`reports/anomalies/`](reports/anomalies/) (`episodes.csv`, `seasons.csv`, `figures/`).

```bash
uv run python -m anomaly.run                                 # все полигоны → reports/anomalies/
uv run python -m anomaly.evaluate reports/anomalies          # прокси-метрики детектора
uv run python -m anomaly.plots AOI-0065:2024 AOI-0043:2019   # графики сезонов с эпизодами
```

## Веб-сервис (обязательная точка запуска по ТЗ)

```bash
uv run --locked --group service python -m service
```

Команда устанавливает все зависимости сервиса из `uv.lock` в `.venv` и запускает API на порту 8000.
Группа `service` включает `geo` и `ml`, поэтому отдельная установка `planetary_computer` не требуется.
Подходит для PowerShell и Linux/macOS; другой порт — `--port 8001`, внешний доступ — `--host 0.0.0.0`.
При старте проверяются зависимости автосбора, готовность — `GET /api/health`.
Эпизоды полигонов кейса уже включены в репозиторий; пересчёт при необходимости:
`uv run --no-sync python -m anomaly.run`. После установки повторный запуск без синхронизации:
`uv run --no-sync python -m service`.

Графики используют JS из установленного пакета Plotly через `/vendor/plotly.min.js`.
Поиск контуров OSM не импортирует спутниковый сборщик. Для Sentinel-2 выбираются публичные HTTPS COG;
JP2-дубликаты и сцены, доступные только через S3 с авторизацией, исключаются до чтения.
При отказе Overpass поиск переключается между публичными серверами VK Maps, Private.coffee и overpass-api.de;
успешные ответы по рамке карты кэшируются до пяти минут. При слишком широком обзоре кнопка поиска
сама приближает карту вокруг её центра до допустимого размера области.

Открыть http://127.0.0.1:8000/. Два сценария из ТЗ:

1. **Готовый полигон**: в «Новая территория» кнопка «Найти поля OSM в видимой области» запрашивает `landuse=farmland` из
   OpenStreetMap (Overpass API), клик по контуру выбирает поле.
2. **Произвольный полигон**: контур рисуется на карте (Leaflet.draw).

После выбора «Собрать данные и проанализировать» сервис сам получает Sentinel-2 L2A (Earth Search STAC,
маска облаков по SCL), Landsat 8/9 C2 L2 и MODIS MOD13Q1 (Planetary Computer STAC), ERA5 (Open-Meteo,
по центроиду поля), строит ряд `primary_ndvi` по приоритету S2 → Landsat → MODIS, восстановленную кривую,
норму по собственной истории поля, находит эпизоды угнетения и объясняет их. Ключи и регистрация не нужны;
нужен доступ в интернет. Семь сезонов (2019–2025) для поля ~1.5 км² в проверке на Windows собраны за 225 секунд; если источник
недоступен, ряд строится по остальным, а в интерфейсе выводится, что именно собрано. Полигоны кейса (78 анонимных `AOI-xxxx`, координат нет) показываются списком слева:
исходные наблюдения по сенсорам, восстановленные контрольные точки из `submission.csv`, кривая, норма ±1σ,
Z-score, погода ERA5 и эпизоды с причинами.

API: `GET /api/health`, `GET /api/polygons`, `GET /api/polygon/{pid}`, `GET /api/episodes?year=&cause=&severity=`, `GET /api/summary`,
`GET /api/fields?bbox=юг,запад,север,восток`, `POST /api/analyze` (`{"geometry": <GeoJSON Polygon>, "name": "...",
"start_year": 2019, "end_year": 2025}`). Функция LLM в репозитории не подключена к основному маршруту анализа;
интерфейс использует проверяемые численные факты и формулировки по правилам.

### Дополнительные погодные графики

Сохранён исходный экран: карта и список слева, NDVI, Z-score, температура/осадки и эпизоды справа.
Под ними добавлены только два погодных графика:

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
Если погоды нет вообще, дополнительная панель скрывается. Подробные пояснения свёрнуты в «Как считается».
Собранные поля доступны в прежнем списке по названию; отчёты и параметры графиков сохранены в
`artifacts/fields/`, погодный кэш — в `artifacts/weather/`.

API графиков: `GET /api/polygon/{pid}/agro?year=2025`, `POST /api/polygon/{pid}/agro`
(`year`, `profile`, `sowing_date`, `base`), `POST /api/polygon/{pid}/weather-refresh`.
Формулы и границы изменений — в [плане дополнения графиков](docs/18-farmer-report-implementation.md).
E2E-проверки через Playwright: [tests/e2e/README.md](tests/e2e/README.md).

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
| `service` | FastAPI, uvicorn, pydantic, httpx, Plotly + группы `geo` и `ml` | `uv sync --group service` |
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
