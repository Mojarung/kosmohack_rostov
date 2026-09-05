"""Изолированный кэш экспериментов с честным маскированием погоды и сенсорных колонок."""

import hashlib
import time

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, INDEX_COLS, TARGET, WEATHER_COLS
from gapfill.data import data_tag, make_mask, polygon_kinds
from gapfill.dataset import META
from gapfill.features import build_features, sensor_prior_features
from gapfill.research_features import full_sensor_features, spatial_features

CACHE = ARTIFACTS_DIR / "research" / "features"
VERSION = 1


def hidden_grid(grid, hidden):
    """Убирает динамические данные контрольных строк, как в настоящем private_features."""
    keys = pd.MultiIndex.from_frame(hidden[["pid", "date"]])
    mask = pd.MultiIndex.from_frame(grid[["pid", "date"]]).isin(keys)
    clean = grid.copy()
    clean.loc[mask, [TARGET, *INDEX_COLS, *WEATHER_COLS]] = np.nan
    return clean


def features(targets, context, grid, version="all"):
    """Базовые и дополнительные признаки из одной и той же видимой части данных."""
    base = sensor_prior_features(build_features(targets, context, grid))
    if version == "baseline":
        return base.reset_index(drop=True).astype("float32")
    raw = full_sensor_features(targets, context)
    spatial = spatial_features(targets, context)
    parts = [base, raw, spatial]
    if version in ("analog", "kriging"):
        from gapfill.research_analog import analog_features
        parts.append(analog_features(targets, context))
    if version == "kriging":
        from gapfill.research_kriging import kriging_features
        parts.append(kriging_features(targets, context))
    return pd.concat(parts, axis=1).reset_index(drop=True).astype("float32")


def _cached(targets, context, grid, key, analog=False, kriging=False):
    """Сохраняет рассчитанные признаки и позволяет перезапускать прерванную серию."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.parquet"
    if path.exists():
        x = pd.read_parquet(path)
    else:
        start = time.monotonic()
        x = features(targets, context, grid)
        x.to_parquet(path)
        print(f"Признаки {key}: {x.shape}, {time.monotonic() - start:.1f} с", flush=True)
    if analog:
        from gapfill.research_analog import analog_features
        extra_path = CACHE / f"analog1_{key}.parquet"
        if extra_path.exists():
            extra = pd.read_parquet(extra_path)
        else:
            extra = analog_features(targets, context).reset_index(drop=True).astype("float32")
            extra.to_parquet(extra_path)
            print(f"Аналоги {key}: {extra.shape}", flush=True)
        x = pd.concat([x, extra], axis=1)
    if kriging:
        from gapfill.research_kriging import kriging_features
        kg_path = CACHE / f"kriging1_{key}.parquet"
        if kg_path.exists():
            kg = pd.read_parquet(kg_path)
        else:
            kg = kriging_features(targets, context).reset_index(drop=True).astype("float32")
            kg.to_parquet(kg_path)
            print(f"Ковариации {key}: {kg.shape}", flush=True)
        x = pd.concat([x, kg], axis=1)
    return x


def examples(obs, grid, val_seed=777, n_masks=12, share=.15, final=False, analog=False, kriging=False):
    """Постоянный внешний holdout и повторные маски только внутри обучающего контекста."""
    tag = f"v{VERSION}_{data_tag(obs)}_val{'none' if final else val_seed}_share{share}"
    held = np.zeros(len(obs), dtype=bool) if final else make_mask(obs, seed=val_seed)
    base = obs.loc[~held].reset_index(drop=True)
    base_grid = hidden_grid(grid, obs.loc[held])
    kind = polygon_kinds(obs)
    xs, metas = [], []
    for k in range(n_masks):
        mask = make_mask(base, share=share, seed=k)
        target, context = base.loc[mask], base.loc[~mask]
        clean = hidden_grid(base_grid, target)
        xs.append(_cached(target, context, clean, f"{tag}_m{k}", analog, kriging))
        metas.append(target[META].assign(mask_id=k).reset_index(drop=True))
    meta = pd.concat(metas, ignore_index=True)
    meta = meta.assign(poly_kind=meta.pid.map(kind), is_2025=meta.year.eq(2025))
    if final:
        return pd.concat(xs, ignore_index=True), meta, None, None
    val = obs.loc[held]
    xv = _cached(val, base, base_grid, f"{tag}_val", analog, kriging)
    mv = val[META].reset_index(drop=True).assign(poly_kind=val.pid.map(kind).to_numpy(),
                                               is_2025=val.year.eq(2025).to_numpy())
    return pd.concat(xs, ignore_index=True), meta, xv, mv
