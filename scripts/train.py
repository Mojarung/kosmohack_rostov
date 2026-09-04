"""Обучение финальной модели восстановления пропусков и сохранение артефактов.

Обучающие примеры собираются двумя способами:
1. многократное маскирование train (несколько сидов) — основной объём;
2. маскирование известных наблюдений test — примеры на новых полигонах и в сезоне 2025.

Второй источник — не утечка: целевые значения скрытых точек test нам недоступны,
мы прячем и восстанавливаем те наблюдения, которые организаторы оставили открытыми.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from ndvi.data import load_test, load_train
from ndvi.pipeline import MODEL_PATH, build_training_table, fit
from ndvi.sensors import estimate_offsets
from ndvi.validation import rmse

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024, 31, 99],
                help="сиды маскирования train; больше сидов = больше примеров (на валидации второй сид дал -0.002 RMSE)")
ap.add_argument("--cold-frac", type=float, default=0.25, help="доля полигонов без истории в обучении")
ap.add_argument("--use-test", action="store_true", default=True,
                help="добавлять примеры из открытой части test")
ap.add_argument("--loss", default="absolute_error")
ap.add_argument("--single", action="store_true", help="одна модель вместо ансамбля из трёх")
args = ap.parse_args()

t0 = time.time()
train = load_train()
tables = [build_training_table(train, seeds=args.seeds, cold_start_frac=0.0)]
if args.cold_frac > 0:
    tables.append(build_training_table(train, seeds=[s + 1000 for s in args.seeds],
                                       cold_start_frac=args.cold_frac))
if args.use_test:
    test = load_test()
    # у test своя специфика: новые полигоны и сезон 2025 — добавляем их в обучение
    tables.append(build_training_table(test, seeds=[s + 500 for s in args.seeds]))

table = pd.concat(tables, ignore_index=True)
print(f"обучающих примеров: {len(table)} (из {len(tables)} источников)")

offsets = estimate_offsets(train[train.primary_ndvi.notna()])
art = fit(table, params={"loss": args.loss}, offsets=offsets, ensemble=not args.single,
          meta={"seeds": args.seeds, "cold_frac": args.cold_frac, "use_test": args.use_test,
                "n_train_rows": len(table), "loss": args.loss, "ensemble": not args.single})
art.save()
print(f"in-sample RMSE: {rmse(table.y_true, art.model.predict(table)):.4f}")
print(f"модель сохранена: {MODEL_PATH} за {time.time() - t0:.1f} с")
