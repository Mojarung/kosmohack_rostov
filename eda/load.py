"""Загрузка и первичная типизация датасетов.

Все функции возвращают новые DataFrame и не изменяют входные объекты.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda.config import TARGET, TEST_PATH, TRAIN_PATH


def _read(path) -> pd.DataFrame:
    """Читает CSV с явными типами для ключевых колонок."""
    df = pd.read_csv(path, low_memory=False)
    return df.assign(
        date=pd.to_datetime(df["date"], format="%Y-%m-%d"),
        crop_type=df["crop_type"].astype("category"),
    )


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные признаки, вычисленные из даты.

    В контрольных строках test колонки year/doy замаскированы, поэтому
    для анализа всегда используем производные cal_year / cal_doy.
    """
    return df.assign(
        cal_year=df["date"].dt.year.astype("int32"),
        cal_doy=df["date"].dt.dayofyear.astype("int32"),
        is_known=df[TARGET].notna(),
    )


def load_train() -> pd.DataFrame:
    """Обучающий датасет с календарными признаками и флагом известного target."""
    df = add_calendar(_read(TRAIN_PATH))
    # status как упорядоченная категория для единообразия графиков
    return df.assign(status=df["status"].astype("category"))


def load_test() -> pd.DataFrame:
    """Тестовый датасет; is_gap — контрольные точки автометрики."""
    df = add_calendar(_read(TEST_PATH))
    is_gap = df["is_synthetic_gap"].astype(str).str.lower().eq("true")
    return df.assign(is_gap=is_gap)


def sort_by_polygon_date(df: pd.DataFrame) -> pd.DataFrame:
    """Сортировка по полигону и дате — базовая для всех временных операций."""
    return df.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)


def known_series(df: pd.DataFrame) -> pd.DataFrame:
    """Только строки с известным primary_ndvi, отсортированные по полигону и дате."""
    return sort_by_polygon_date(df.loc[df["is_known"]])


def days_between(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Разница между датами в днях (float, NaN там, где нет пары)."""
    return (a - b).dt.days.to_numpy(dtype="float64", na_value=np.nan)
