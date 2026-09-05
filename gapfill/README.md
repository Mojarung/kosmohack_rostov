# Пакет `gapfill` — восстановление пропусков `primary_ndvi` (задача 1)

Карта модулей, чтобы не искать логику по файлам. Подробности метода — в
[docs/12](../docs/12-gapfill-model.md) и [docs/16](../docs/16-model-improvement.md), журнал — в
[`experiments/`](../experiments/README.md).

## Путь конкурсного предсказания

`private_features.csv` → признаки → LightGBM + сезонная нейросеть → смесь → калибровка → `submission.csv`.

| Модуль | Роль |
|---|---|
| `config.py` | пути и константы пайплайна (единственное место с путями) |
| `data.py` | загрузка train/test в общий формат: наблюдения, ежедневная сетка, контрольные точки |
| `features*.py` | признаки: по ряду поля (`features_series`), по циклам съёмки (`features_cycles`), по соседним полям и «шуму дня» (`features_context`), сборка матрицы (`features`) |
| `dataset.py`, `smooth.py` | кэш обучающих примеров и локальная регрессия по времени |
| `nn_data.py`, `nn_model.py` | сезонная сетка (полигон-год × 213 дней) и SeasonNet: дилатированные свёртки + трансформер |
| `train.py`, `train_cb.py`, `tune.py` | обучение LightGBM, CatBoost и подбор гиперпараметров |
| `ensemble.py`, `make_submission.py` | смесь предсказаний и запись `submission.csv` с проверками формата |
| **`predict_improved.py`** | **финальный batch-инференс из готовых весов `models/improved/` (exp-008)** |
| `predict.py`, `predict_saved.py` | инференс с обучением и инференс из прежних весов `models/` (exp-007) |
| `metrics.py` | пересчёт RMSE и GapScore по сохранённым holdout-предсказаниям за доли секунды |

```bash
uv run --no-sync python -m gapfill.predict_improved --output submission.csv --device cpu
uv run --no-sync python -m gapfill.metrics
```

## Модули `research/`

Исследовательская часть вынесена в отдельный подпакет: в корне `gapfill/` остались только модули
конкурсного пути. Здесь — изолированный кэш признаков, схемы маскирования, калибровка по опубликованным
историческим агрегатам и проверки альтернатив. Четыре модуля участвуют и в конкурсном инференсе:

- `research/data.py` — признаки и честное маскирование для экспериментов (импортируется `predict_improved`);
- `research/nn.py` — `ResidualSeasonNet`, сеть на остатке дерева (импортируется `predict_improved`);
- `research/calibrate.py`, `research/constraints.py` — калибровка по историческим агрегатам (импортируются `predict_improved`);
- `research/final.py`, `research/package.py` — финальное обучение и сборка пакета весов;
- остальные (`research.kriging`, `research.mcmc`, `research.robust_linear`, `research.truncated`,
  `research.variational`, `research.uncertainty`, `research.blend`, `research.mix_trees`, `research.analog`,
  `research.refine`, `research.audit`, `research.results`, `research.features`, `research.train`,
  `research.calibration_sweep`, `chronos_member`, `calibration_posterior`) — проверенные гипотезы,
  оставлены ради воспроизводимости журнала экспериментов; в инференсе не используются.

У каждого модуля в первой строке docstring на русском с назначением.
