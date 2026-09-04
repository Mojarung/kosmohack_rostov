"""Точка входа EDA: запускает модули анализа, сохраняет графики и сводку чисел.

Запуск:  uv run python -m eda.run_all [--only coverage,seasonality]
Выход:   reports/eda/figures/*.png и reports/eda/summary.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np

from eda.config import SUMMARY_PATH
from eda.load import load_test, load_train
from eda.plotting import setup_style

MODULES = ["coverage", "seasonality", "sensors", "cycles", "anomalies", "gaps", "baseline"]


def _to_builtin(value):
    """numpy-типы -> встроенные, чтобы сводка сериализовалась в JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Не сериализуется: {type(value)}")


def _load_existing() -> dict:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Разведочный анализ данных Космохакатона")
    parser.add_argument("--only", default="", help="Список модулей через запятую (по умолчанию все)")
    args = parser.parse_args()
    selected = [m for m in args.only.split(",") if m] or MODULES

    setup_style()
    train, test = load_train(), load_test()
    print(f"train: {len(train)} строк, test: {len(test)} строк, контрольных точек: {int(test['is_gap'].sum())}")

    summary = _load_existing() if args.only else {}
    for name in selected:
        module = importlib.import_module(f"eda.{name}")
        started = time.perf_counter()
        summary[name] = module.run(train, test)
        print(f"[{name}] готово за {time.perf_counter() - started:.1f} с")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_to_builtin), encoding="utf-8")
    print(f"Сводка: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
