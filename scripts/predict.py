"""Batch-инференс: test_dataset.csv (он же private_features.csv) -> submission.csv."""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.data import load_test, load_train
from ndvi.paths import ROOT
from ndvi.pipeline import Artifacts, predict_gaps
from ndvi.sensors import estimate_offsets

ap = argparse.ArgumentParser()
ap.add_argument("--test", default=None, help="путь к private_features.csv")
ap.add_argument("--out", default=str(ROOT / "submission.csv"))
ap.add_argument("--no-train-context", action="store_true",
                help="не использовать историю train для общих полигонов")
args = ap.parse_args()

test = load_test(args.test)
gaps = test[test.is_synthetic_gap] if "is_synthetic_gap" in test else test[test.primary_ndvi.isna()]
print(f"строк в test: {len(test)}, гэпов: {len(gaps)}")

# контекст = сам test плюс история train по общим полигонам: она легальна и даёт норму
context = test
if not args.no_train_context:
    train = load_train()
    common = set(test.anon_polygon_id) & set(train.anon_polygon_id)
    extra = train[train.anon_polygon_id.isin(common)].copy()
    extra["is_synthetic_gap"] = False
    context = pd.concat([test, extra], ignore_index=True).sort_values(
        ["anon_polygon_id", "date"]).reset_index(drop=True)
    print(f"контекст расширен историей train по {len(common)} общим полигонам: {len(context)} строк")

art = Artifacts.load()
observed = context[context.primary_ndvi.notna()]
offsets = estimate_offsets(observed)
clim = Climatology().fit(observed)
targets = gaps[["anon_polygon_id", "date", "year", "doy", "crop_type"]].copy()
feats = predict_gaps(context, targets, art, clim=clim)

sub = pd.DataFrame({
    "anon_polygon_id": feats.anon_polygon_id.values,
    "date": pd.to_datetime(feats.date).dt.strftime("%Y-%m-%d"),
    "primary_ndvi_pred": np.round(feats.primary_ndvi_pred.astype(float).values, 6),
})

# --- валидатор файла ---------------------------------------------------------
problems = []
if list(sub.columns) != ["anon_polygon_id", "date", "primary_ndvi_pred"]:
    problems.append("неверный набор колонок")
if len(sub) != len(gaps):
    problems.append(f"строк {len(sub)}, ожидалось {len(gaps)}")
if sub.duplicated(["anon_polygon_id", "date"]).any():
    problems.append("есть дубликаты пары polygon+date")
if sub.primary_ndvi_pred.isna().any():
    problems.append("есть NaN в предсказаниях")
if not sub.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
    problems.append("неверный формат даты")
out_of_range = ((sub.primary_ndvi_pred < -0.2) | (sub.primary_ndvi_pred > 1.0)).sum()
if out_of_range:
    problems.append(f"{out_of_range} значений вне [-0.2, 1]")

if problems:
    raise SystemExit("submission не прошёл проверку: " + "; ".join(problems))

sub.to_csv(args.out, index=False, encoding="utf-8")
print(f"submission сохранён: {args.out}")
print(sub.primary_ndvi_pred.describe().round(4).to_string())
print("\nпервые строки:\n" + sub.head().to_string(index=False))
