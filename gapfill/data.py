"""Загрузка train/test в единый формат: наблюдения (obs), ежедневная сетка (grid), контрольные точки (gaps).

Все функции возвращают новые DataFrame и не изменяют входные объекты.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gapfill.config import (CROP_CODES, EPOCH, GAP_SHARE, INDEX_COLS, SENSOR_CODE, SENSOR_NDVI,
                            TARGET, TEST_PATH, TRAIN_PATH, WEATHER_COLS)

KEEP = ["pid", "date", "day_num", "year", "doy", "crop", "split", "is_gap", TARGET, *INDEX_COLS, *WEATHER_COLS]


def _read(path, split: str) -> pd.DataFrame:
    """Читает CSV и добавляет календарные колонки, вычисленные из даты (в test они замаскированы)."""
    raw = pd.read_csv(path, low_memory=False)
    date = pd.to_datetime(raw["date"], format="%Y-%m-%d")
    is_gap = raw["is_synthetic_gap"].astype(str).str.lower().eq("true") if "is_synthetic_gap" in raw else False
    df = raw.assign(
        pid=raw["anon_polygon_id"],
        date=date,
        day_num=(date - pd.Timestamp(EPOCH)).dt.days.astype("int32"),
        year=date.dt.year.astype("int16"),
        doy=date.dt.dayofyear.astype("int16"),
        crop=raw["crop_type"].map(CROP_CODES).fillna(-1).astype("int8"),
        split=split,
        is_gap=np.asarray(is_gap, dtype=bool),
    )
    # Вспомогательные индексы грязные (EVI до 1e12, NDWI до ±18) — обрезаем до физического диапазона
    for col in INDEX_COLS:
        df[col] = df[col].clip(-1.0, 1.0)
    return df[KEEP]


def sensor_of(df: pd.DataFrame) -> np.ndarray:
    """Код сенсора-источника primary_ndvi по приоритету s2 > landsat > modis (-1, если нет ни одного)."""
    conds = [df[SENSOR_NDVI[s]].notna().to_numpy() for s in SENSOR_CODE]
    return np.select(conds, list(SENSOR_CODE.values()), default=-1).astype("int8")


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (obs, grid, gaps).

    obs  — все строки с известным primary_ndvi из train и test (без контрольных точек), с кодом сенсора;
    grid — все строки обоих наборов (нужна ежедневная погода ERA5);
    gaps — контрольные точки test, которые нужно предсказать.
    """
    grid = pd.concat([_read(TRAIN_PATH, "train"), _read(TEST_PATH, "test")], ignore_index=True)
    grid = grid.sort_values(["pid", "date"]).reset_index(drop=True)
    obs = grid.loc[grid[TARGET].notna() & ~grid["is_gap"]].copy()
    obs["sensor"] = sensor_of(obs)
    obs = obs.reset_index(drop=True)
    gaps = grid.loc[grid["is_gap"]].reset_index(drop=True)
    return obs, grid, gaps


def make_mask(obs: pd.DataFrame, share: float = GAP_SHARE, seed: int = 0) -> np.ndarray:
    """Случайная маска синтетических пропусков: доля share известных точек, равномерно по всем строкам.

    В test контрольные точки составляют ~15 % исходно известных в каждом году, поэтому
    равномерная выборка без стратификации воспроизводит их плотность.
    """
    rng = np.random.default_rng(seed)
    n = len(obs)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=int(round(n * share)), replace=False)] = True
    return mask


def gap_score(rmse: float) -> float:
    """GapScore из ТЗ: 30 при RMSE 0, ноль при RMSE >= 0.10."""
    return round(30.0 * max(0.0, 1.0 - rmse / 0.10), 2)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Корень из средней квадратичной ошибки."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def polygon_kinds(obs: pd.DataFrame) -> pd.Series:
    """Тип полигона: old (есть в train), new_hist (только test, но с историей лет), new_2025only."""
    train_polys = set(obs.loc[obs["split"] == "train", "pid"])
    n_years = obs.groupby("pid")["year"].nunique()
    kinds = {pid: ("old" if pid in train_polys else "new_2025only" if n <= 1 else "new_hist")
             for pid, n in n_years.items()}
    return pd.Series(kinds, name="poly_kind")


# Состав контрольных точек test по стратам (тип полигона, год 2025?) — для взвешивания валидации
TEST_STRATA = {("new_2025only", True): 228, ("new_hist", False): 2187, ("new_hist", True): 233, ("old", True): 464}


def testlike_rmse(err: np.ndarray, kind: np.ndarray, is_2025: np.ndarray) -> float:
    """RMSE, взвешенный под состав контрольных точек test (страты полигон × 2025)."""
    mse, weight = 0.0, 0.0
    for (k, y), n in TEST_STRATA.items():
        sel = (kind == k) & (is_2025 == y)
        if sel.any():
            mse += n * float(np.mean(err[sel] ** 2))
            weight += n
    return float(np.sqrt(mse / weight)) if weight else float("nan")
