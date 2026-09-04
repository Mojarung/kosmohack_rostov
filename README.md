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

## Разведочный анализ (EDA)

```bash
uv sync                         # Python 3.14 + зависимости из pyproject.toml / uv.lock
uv run python -m eda.run_all    # графики в reports/eda/figures/, числа в reports/eda/summary.json
```

Выводы и графики — в [docs/08-eda-report.md](docs/08-eda-report.md). Скрипты первого прохода анализа и сборка дашборда — в [`eda/v1/`](eda/v1/README.md).

Интерактивный дашборд «NDVI-атлас полей»: [`reports/dashboard/dashboard.html`](reports/dashboard/dashboard.html) (самодостаточный файл, открывается двойным кликом; пересборка `uv run python eda/v1/build_dashboard_data.py`).

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
├── reports/
│   ├── eda/                # графики и summary.json, генерируются EDA
│   └── dashboard/          # template.html + собранный dashboard.html
├── pyproject.toml          # зависимости по группам (uv)
└── uv.lock
```

Структура и статистика датасетов описаны в [docs/03-data.md](docs/03-data.md).
