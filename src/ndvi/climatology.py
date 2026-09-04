"""Собственная климатическая норма NDVI по полигону.

Точную формулу организаторов воспроизвести не удалось; ближе всего (корреляция 0.99)
среднее ``primary_ndvi`` того же полигона в окне +-8 дней по doy по всем годам.
Мы считаем норму **без текущего года** — иначе при валидации это утечка ответа.

Для полигонов без истории (в test есть такие: только сезон 2025) предусмотрен откат
на норму по культуре, а затем на общую норму по всем полигонам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOW_DOY = 8


@dataclass
class Climatology:
    """Нормы трёх уровней: по полигону, по культуре, общая."""

    window: int = WINDOW_DOY
    _poly: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    _crop: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    _all: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, obs: pd.DataFrame) -> "Climatology":
        """``obs`` — только строки с известным ``primary_ndvi`` (видимая часть данных)."""
        o = obs[obs.primary_ndvi.notna()]
        for pid, d in o.groupby("anon_polygon_id"):
            self._poly[pid] = (
                d.doy.to_numpy(np.int16), d.year.to_numpy(np.int16), d.primary_ndvi.to_numpy(float),
            )
        for crop, d in o.groupby("crop_type"):
            self._crop[crop] = (d.doy.to_numpy(np.int16), d.primary_ndvi.to_numpy(float))
        self._all = (o.doy.to_numpy(np.int16), o.primary_ndvi.to_numpy(float))
        return self

    # --- внутренние помощники -------------------------------------------------

    def _window_stats(self, doys, values, doy):
        m = np.abs(doys - doy) <= self.window
        v = values[m]
        if v.size == 0:
            return np.nan, np.nan, 0
        return float(v.mean()), float(v.std(ddof=0)), int(v.size)

    # --- публичный интерфейс --------------------------------------------------

    def polygon_norm(self, pid: str, doy: int, year: int | None = None):
        """Норма по самому полигону, исключая указанный год. Возвращает (mean, std, n_лет)."""
        rec = self._poly.get(pid)
        if rec is None:
            return np.nan, np.nan, 0
        doys, years, values = rec
        m = np.abs(doys - doy) <= self.window
        if year is not None:
            m &= years != year
        v, y = values[m], years[m]
        if v.size == 0:
            return np.nan, np.nan, 0
        return float(v.mean()), float(v.std(ddof=0)), int(np.unique(y).size)

    def crop_norm(self, crop: str, doy: int):
        """Откат: норма по культуре, усреднённая по всем полигонам."""
        rec = self._crop.get(crop)
        if rec is None:
            return self.global_norm(doy)
        mean, std, n = self._window_stats(rec[0], rec[1], doy)
        return (mean, std, n) if n else self.global_norm(doy)

    def global_norm(self, doy: int):
        """Последний откат: общая сезонная кривая по всем данным."""
        if self._all is None:
            return np.nan, np.nan, 0
        return self._window_stats(self._all[0], self._all[1], doy)

    def lookup(self, pid: str, crop: str, doy: int, year: int | None = None, min_years: int = 3):
        """Норма с каскадом откатов. Возвращает (mean, std, n_лет, уровень)."""
        mean, std, n = self.polygon_norm(pid, doy, year)
        if n >= min_years and np.isfinite(mean):
            return mean, std, n, 0  # 0 = собственная история полигона
        cmean, cstd, cn = self.crop_norm(crop, doy)
        if np.isfinite(cmean):
            # если своя история есть, но короткая — смешиваем её с нормой по культуре
            if n > 0 and np.isfinite(mean):
                w = n / (n + min_years)
                return w * mean + (1 - w) * cmean, std if np.isfinite(std) else cstd, n, 1
            return cmean, cstd, cn, 2  # 2 = холодный старт, только культура
        gmean, gstd, gn = self.global_norm(doy)
        return gmean, gstd, gn, 3

    def lookup_frame(self, df: pd.DataFrame, exclude_current_year: bool = True) -> pd.DataFrame:
        """Норма для каждой строки таблицы. Колонки: clim_mean/clim_std/clim_n/clim_level."""
        years = df.year.values if exclude_current_year else [None] * len(df)
        rows = [
            self.lookup(pid, crop, int(doy), yr)
            for pid, crop, doy, yr in zip(df.anon_polygon_id.values, df.crop_type.values, df.doy.values, years)
        ]
        return pd.DataFrame(rows, columns=["clim_mean", "clim_std", "clim_n", "clim_level"], index=df.index)
