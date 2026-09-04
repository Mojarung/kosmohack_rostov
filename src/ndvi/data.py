"""Загрузка и очистка датасетов.

Ключевые факты из EDA, которые здесь зашиты:
* ``primary_ndvi`` = coalesce(s2 -> landsat -> modis), проверено на 100 % строк train;
* в строке-гэпе test затёрты ВСЕ динамические колонки, включая ``year``/``doy``,
  погоду и климатологию — их восстанавливаем сами;
* в данных есть мусор: ``s2_evi`` порядка 1e11, отрицательные осадки, NDVI вне [-0.2, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.paths import DATA_DIR

SENSORS = ("s2", "landsat", "modis")
NDVI_COLS = ["s2_ndvi", "landsat_ndvi", "modis_ndvi"]

#: колонки, которые организаторы затирают в строке-гэпе (мы делаем так же при валидации)
DYNAMIC_COLS = [
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm",
    "primary_ndvi",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
]

NDVI_MIN, NDVI_MAX = -0.2, 1.0


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Достраивает ``year``/``doy`` из даты и колонку сенсора ``src``."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # year/doy в гэпах затёрты, но однозначно восстанавливаются из даты
    df["year"] = df["date"].dt.year.astype("int16")
    df["doy"] = df["date"].dt.dayofyear.astype("int16")
    df["src"] = np.select(
        [df.s2_ndvi.notna(), df.landsat_ndvi.notna(), df.modis_ndvi.notna()],
        list(SENSORS),
        default="none",
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Чистит заведомо битые значения. Целевой ``primary_ndvi`` не трогаем."""
    df = df.copy()
    # EVI встречается порядка 1e11 — это переполнение формулы при знаменателе около нуля
    for col in ("s2_evi", "landsat_evi", "modis_evi"):
        if col in df:
            df.loc[df[col].abs() > 5, col] = np.nan
    # отрицательные осадки (~ -1e-5) — артефакт реанализа ERA5
    if "era5_precip_mm" in df:
        df.loc[df.era5_precip_mm < 0, "era5_precip_mm"] = 0.0
    # NDVI физически лежит в [-1, 1]; всё за пределами [-0.2, 1] — облако/тень/вода
    for col in NDVI_COLS:
        bad = df[col].notna() & ((df[col] < -1) | (df[col] > 1))
        df.loc[bad, col] = np.nan
    return df


def load_train(path=None) -> pd.DataFrame:
    """Обучающий набор: 39 полигонов, 2010-2024, ``primary_ndvi`` известен везде, где есть съёмка."""
    df = pd.read_csv(path or DATA_DIR / "train_dataset.csv", low_memory=False)
    return add_derived(clean(df)).sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)


def load_test(path=None) -> pd.DataFrame:
    """Тестовый набор (в ТЗ ``private_features.csv``): 3 112 строк с ``is_synthetic_gap``."""
    df = pd.read_csv(path or DATA_DIR / "test_dataset.csv", low_memory=False)
    df = add_derived(clean(df))
    if "is_synthetic_gap" in df:
        df["is_synthetic_gap"] = df["is_synthetic_gap"].astype(str).str.lower().eq("true")
    return df.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)


def neighbor_value(df: pd.DataFrame) -> pd.Series:
    """Значение NDVI, пригодное как опора для соседа: клиппинг вместо выброса.

    Выброс -0.47 посреди июля не выкидываем (сосед всё равно нужен), но ограничиваем,
    чтобы он не утаскивал интерполяцию.
    """
    return df["primary_ndvi"].clip(NDVI_MIN, NDVI_MAX)
