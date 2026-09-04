"""Прогон экспериментов по восстановлению пропусков и печать сравнительной таблицы."""

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ndvi.experiment import build_validation_table, cv_predict, holdout_year_predict
from ndvi.models.baselines import BASELINES
from ndvi.models.gbm import GapModel
from ndvi.validation import gap_score, rmse, segment_report

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)

ap = argparse.ArgumentParser()
ap.add_argument("--cold", type=float, default=0.0, help="доля полигонов без истории")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no-cache", action="store_true")
args = ap.parse_args()

t0 = time.time()
df = build_validation_table(seed=args.seed, cold_start_frac=args.cold, use_cache=not args.no_cache)
print(f"валидационная таблица: {df.shape}, холодный старт: {int(df.is_cold_start.sum())} точек")

preds = {}
for name, fn in BASELINES.items():
    preds[name] = fn(df)

variants = {
    "gbm_mse": lambda: GapModel(),
    "gbm_huber": lambda: GapModel({"loss": "absolute_error"}),
    "gbm_deep": lambda: GapModel({"max_leaf_nodes": 63, "min_samples_leaf": 20, "max_iter": 700}),
    "gbm_on_smooth": lambda: GapModel(base_col="base_smooth"),
}
for name, make in variants.items():
    t = time.time()
    preds[name] = cv_predict(df, make, n_splits=5, seed=args.seed)
    print(f"  {name} обучен за {time.time() - t:.1f} с")

# блендинг лучшего бустинга с сенсорно-осведомлённой интерполяцией
preds["blend_gbm"] = 0.7 * preds["gbm_mse"] + 0.3 * preds["sensor_aware"]

for k, v in preds.items():
    df[k] = v

names = list(preds)
print("\nRMSE (group-CV по полигонам) и GapScore:")
res = []
for n in names:
    r = rmse(df.y_true, df[n])
    res.append((n, r, gap_score(r)))
for n, r, g in sorted(res, key=lambda x: x[1]):
    print(f"  {n:16s} RMSE={r:.4f}  GapScore={g:5.2f}")

print("\nразрезы:")
cols = [n for n, _, _ in sorted(res, key=lambda x: x[1])[:5]]
print(segment_report(df, cols).to_string(index=False))

if df.is_cold_start.any():
    print("\nхолодный старт (полигоны без истории):")
    sub = df[df.is_cold_start]
    for n in cols:
        print(f"  {n:16s} RMSE={rmse(sub.y_true, sub[n]):.4f}  (n={len(sub)})")

print("\nhold-out сезона 2024 (обучение на 2010-2023):")
best = cols[0]
va, p = holdout_year_predict(df, variants.get(best, lambda: GapModel()), year=2024)
print(f"  {best:16s} RMSE={rmse(df.loc[va, 'y_true'], p):.4f}  (n={int(va.sum())})")
for n in ("sensor_aware", "mean2"):
    print(f"  {n:16s} RMSE={rmse(df.loc[va, 'y_true'], df.loc[va, n]):.4f}")

print(f"\nвсего {time.time() - t0:.1f} с")
