# Космохакатон (Ростов): мониторинг вегетационной динамики

Веб-сервис для мониторинга состояния сельскохозяйственных полей по спутниковым данным: выбор региона и полигона на карте, автоматический сбор спутниковых и метеоданных, восстановление пропусков во временном ряде NDVI и детекция аномальных периодов вегетации.

## Документация

Полное ТЗ, критерии оценки и чек-лист сдачи — в папке [`docs/`](docs/README.md):

- [Постановка задачи](docs/01-task-statement.md)
- [Теория и глоссарий](docs/02-theory-and-glossary.md)
- [Данные и формат submission](docs/03-data.md)
- [Метрика оценки](docs/04-metric.md)
- [Технические требования](docs/05-technical-requirements.md)
- [Критерии оценки](docs/06-evaluation-criteria.md)
- [Чек-лист сдачи](docs/07-submission-checklist.md)

## Разведочный анализ (EDA)

```bash
uv sync                         # Python 3.14 + зависимости из pyproject.toml / uv.lock
uv run python -m eda.run_all    # графики в reports/eda/figures/, числа в reports/eda/summary.json
```

Выводы и графики — в [docs/08-eda-report.md](docs/08-eda-report.md).

## Структура репозитория

```
.
├── README.md
├── data/
│   ├── train_dataset.csv   # обучающий датасет (99 955 строк)
│   └── test_dataset.csv    # тестовый датасет = private_features.csv из ТЗ (57 185 строк)
├── docs/                   # ТЗ, критерии, чек-лист, отчёт EDA, исходные PDF
├── eda/                    # модули разведочного анализа (uv run python -m eda.run_all)
├── reports/eda/            # графики и summary.json, генерируются EDA
├── pyproject.toml          # зависимости (uv)
└── uv.lock
    └── source/
```

Структура и статистика датасетов описаны в [docs/03-data.md](docs/03-data.md).
