"""Каркас экспериментов: сборка валидационной таблицы с кэшем и прогон вариантов модели."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.data import load_train
from ndvi.features import build_features
from ndvi.paths import ARTIFACTS_DIR
from ndvi.sensors import estimate_offsets
from ndvi.validation import group_folds, make_masked, rmse

CACHE = ARTIFACTS_DIR / "val_features.pkl"


def build_validation_table(seed: int = 42, frac: float = 0.15,
                           cold_start_frac: float = 0.0, use_cache: bool = True) -> pd.DataFrame:
    """Маскирует часть train и собирает признаки для скрытых точек.

    ``cold_start_frac`` — доля полигонов, у которых дополнительно вырезается вся история
    кроме сезона гэпа. Так эмулируется самый трудный сегмент test: новые поля без прошлого.
    """
    key = CACHE.with_name(f"val_features_s{seed}_f{frac}_c{cold_start_frac}.pkl")
    if use_cache and key.exists():
        return pickle.loads(key.read_bytes())

    t0 = time.time()
    train = load_train()
    split = make_masked(train, frac=frac, seed=seed)
    ctx = split.context

    cold = np.array([], dtype=object)
    if cold_start_frac > 0:
        rng = np.random.default_rng(seed + 1)
        pids = np.array(sorted(ctx.anon_polygon_id.unique()))
        cold = rng.choice(pids, size=max(1, int(cold_start_frac * pids.size)), replace=False)
        # у «холодных» полигонов оставляем только сезон, в котором стоит гэп
        gap_years = split.targets.groupby("anon_polygon_id").year.agg(set)
        drop_mask = np.zeros(len(ctx), dtype=bool)
        for pid in cold:
            years = gap_years.get(pid, set())
            drop_mask |= (ctx.anon_polygon_id == pid).to_numpy() & (~ctx.year.isin(years)).to_numpy()
        ctx = ctx[~drop_mask].copy()

    observed = ctx[ctx.primary_ndvi.notna()]
    offsets = estimate_offsets(observed)
    clim = Climatology().fit(observed)
    feats = build_features(ctx, split.targets, offsets, clim)
    df = feats.merge(split.truth, on=["anon_polygon_id", "date"], how="left")
    df["y_true"] = df.primary_ndvi
    df["is_cold_start"] = df.anon_polygon_id.isin(cold)
    df.attrs["offsets"] = offsets
    df.attrs["build_seconds"] = round(time.time() - t0, 1)
    key.write_bytes(pickle.dumps(df))
    return df


def cv_predict(df: pd.DataFrame, make_model, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Кросс-валидация с разбиением по полигонам: предсказание для каждой строки вне обучения."""
    pred = np.full(len(df), np.nan)
    for tr_mask, va_mask in group_folds(df.anon_polygon_id, n_splits=n_splits, seed=seed):
        model = make_model()
        model.fit(df[tr_mask], df.loc[tr_mask, "y_true"].to_numpy())
        pred[va_mask] = model.predict(df[va_mask])
    return pred


def holdout_year_predict(df: pd.DataFrame, make_model, year: int = 2024) -> tuple[np.ndarray, np.ndarray]:
    """Обучение на всех годах кроме ``year``, проверка на нём. Возвращает (маска, прогноз)."""
    va = (df.year == year).to_numpy()
    model = make_model()
    model.fit(df[~va], df.loc[~va, "y_true"].to_numpy())
    return va, model.predict(df[va])


def score(df: pd.DataFrame, pred: np.ndarray, mask=None) -> float:
    if mask is None:
        return rmse(df.y_true, pred)
    return rmse(df.loc[mask, "y_true"], pred)
