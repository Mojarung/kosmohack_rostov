"""Прогон baseline-методов по протоколу валидации. Точка отсчёта для всех экспериментов."""

import time

import _bootstrap  # noqa: F401
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.data import load_train
from ndvi.features import build_features
from ndvi.models.baselines import BASELINES
from ndvi.sensors import estimate_offsets
from ndvi.validation import gap_score, make_masked, rmse, segment_report

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

t0 = time.time()
train = load_train()
split = make_masked(train, frac=0.15, seed=42)
observed = split.context[split.context.primary_ndvi.notna()]
offsets = estimate_offsets(observed)
print("смещения сенсоров к S2:", {k: round(v, 4) for k, v in offsets.items()})

clim = Climatology().fit(observed)
feats = build_features(split.context, split.targets, offsets, clim)
print(f"признаки собраны: {feats.shape} за {time.time() - t0:.1f} с")

df = feats.merge(split.truth, on=["anon_polygon_id", "date"], how="left")
df["y_true"] = df.primary_ndvi
print("точность угадывания сенсора:", round((df.hidden_src == df.src_true).mean(), 3))
print(pd.crosstab(df.src_true, df.hidden_src))

for name, fn in BASELINES.items():
    df[name] = fn(df)

names = list(BASELINES)
print("\nRMSE и GapScore:")
for n in names:
    r = rmse(df.y_true, df[n])
    print(f"  {n:14s} RMSE={r:.4f}  GapScore={gap_score(r):5.2f}")

print("\nразрезы:")
print(segment_report(df, names).to_string(index=False))
print(f"\nготово за {time.time() - t0:.1f} с")
