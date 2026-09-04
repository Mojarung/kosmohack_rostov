"""Кэшированные обучающие и валидационные примеры.

Признаки детерминированы по (версия признаков, seed валидационной маски, seed и номер обучающей маски),
поэтому считаются один раз и складываются в artifacts/features/*.parquet. При изменении признаков
увеличить FEATURE_VERSION.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import make_mask, polygon_kinds
from gapfill.features import build_features, sensor_prior_features

FEATURE_VERSION = 2
CACHE_DIR = ARTIFACTS_DIR / "features"
META = ["pid", "date", "day_num", "year", "doy", "split", "sensor", TARGET]


def _key(val_seed: int | None, seed: int | None = None, k: int | None = None) -> str:
    part = f"v{FEATURE_VERSION}_val{'none' if val_seed is None else val_seed}"
    return part if seed is None else f"{part}_s{seed}_m{k}"


def _load(key: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    fx, fm = CACHE_DIR / f"{key}_X.parquet", CACHE_DIR / f"{key}_meta.parquet"
    if fx.exists() and fm.exists():
        return pd.read_parquet(fx), pd.read_parquet(fm)
    return None


def _save(key: str, X: pd.DataFrame, meta: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(CACHE_DIR / f"{key}_X.parquet")
    meta.to_parquet(CACHE_DIR / f"{key}_meta.parquet")


def _with_kind(meta: pd.DataFrame, kinds: pd.Series) -> pd.DataFrame:
    return meta.assign(poly_kind=meta["pid"].map(kinds).astype(str), is_2025=meta["year"].eq(2025))


def _examples(targets: pd.DataFrame, context: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = sensor_prior_features(build_features(targets, context, grid)).reset_index(drop=True)
    return X, targets[META].reset_index(drop=True)


def val_examples(obs: pd.DataFrame, grid: pd.DataFrame, val_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Валидационные точки (маска val_seed) с признаками по остальным известным точкам."""
    key = _key(val_seed)
    cached = _load(key)
    if cached is None:
        val_mask = make_mask(obs, seed=val_seed)
        cached = _examples(obs.loc[val_mask], obs.loc[~val_mask], grid)
        _save(key, *cached)
    X, meta = cached
    return X, _with_kind(meta, polygon_kinds(obs))


def train_examples(obs: pd.DataFrame, grid: pd.DataFrame, val_seed: int | None, n_masks: int,
                   seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обучающие примеры: n_masks масок по контексту (obs без валидационной маски, если val_seed задан)."""
    base = obs if val_seed is None else obs.loc[~make_mask(obs, seed=val_seed)]
    base = base.reset_index(drop=True)
    xs, metas = [], []
    for k in range(n_masks):
        key = _key(val_seed, seed, k)
        cached = _load(key)
        if cached is None:
            t0 = time.time()
            m = make_mask(base, seed=seed * 1000 + k)
            cached = _examples(base.loc[m], base.loc[~m], grid)
            _save(key, *cached)
            print(f"  маска {k + 1}/{n_masks}: {len(cached[0])} примеров, {time.time() - t0:.1f} с", flush=True)
        xs.append(cached[0])
        metas.append(cached[1].assign(mask_id=k))
    X = pd.concat(xs, ignore_index=True)
    meta = _with_kind(pd.concat(metas, ignore_index=True), polygon_kinds(obs))
    return X, meta


def groups_of(meta: pd.DataFrame) -> np.ndarray:
    """Группы полигон-год для отложенной выборки ранней остановки и OOF."""
    return (meta["pid"].astype(str) + "_" + meta["year"].astype(str)).to_numpy()
