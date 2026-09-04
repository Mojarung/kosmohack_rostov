# EDA v1: скрипты первого прохода

Линейные скрипты из ветки `eda/data-analysis` (первый проход анализа) и сборка интерактивного дашборда «NDVI-атлас полей». Выводы описаны в [`docs/11-eda-v1-notes.md`](../../docs/11-eda-v1-notes.md), сверка с итоговым анализом — в [`docs/10-branch-comparison.md`](../../docs/10-branch-comparison.md).

Основной, модульный EDA живёт уровнем выше (`eda/*.py`, запуск `uv run python -m eda.run_all`) и покрывает все находки этих скриптов. Скрипты сохранены для воспроизводимости чисел из заметок и для сборки дашборда.

## Запуск

Из корня репозитория, в окружении проекта (Python 3.14, pandas 3.0.5 — проверено, предупреждений нет):

```bash
uv run python eda/v1/eda_01_overview.py        # размеры, NaN, полигоны
uv run python eda/v1/eda_02_deep.py            # сенсоры, климатология, структура test
uv run python eda/v1/eda_03_gaps.py            # ERA5, расписания сенсоров, baseline, соседи
uv run python eda/v1/eda_04_signal.py          # сенсорно-осведомлённый baseline, графики -> eda/v1/figures/
uv run python eda/v1/eda_05_crop_check.py      # пик сезона по культурам
uv run python eda/v1/build_dashboard_data.py   # reports/dashboard/dashboard_data.json + dashboard.html
```

Скрипты читают `data/*.csv` по относительным путям, поэтому запускать только из корня. Результаты печатаются в stdout; числа для отчёта ищутся по тексту вывода.

## Дашборд

`reports/dashboard/template.html` — шаблон с плейсхолдером `__DATA__`; `build_dashboard_data.py` подставляет JSON и пишет самодостаточный `reports/dashboard/dashboard.html` (около 2 МБ, открывается через `file://`, внешних скриптов нет, только шрифты Google Fonts с системным фолбэком). Промежуточный `dashboard_data.json` в git не хранится.
