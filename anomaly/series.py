"""Ряд полигона: гармонизация сенсоров, отбраковка артефактов, гладкая ежедневная кривая сезона.

Все функции возвращают новые объекты и не изменяют входные DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import ARTIFACT_ABS, ARTIFACT_MAD_K, CURVE_BW, SEASON_DOY
from gapfill.config import SENSOR_CODE, SENSOR_OFFSET
from gapfill.smooth import local_linear

SHRINK_PAIRS = 10       # сжатие смещения полигона к глобальному (в парах наблюдений)
CURVE_RANGE = (-0.1, 1.0)  # физический диапазон NDVI для кривой
MIN_SLOPE_POINTS = 2.0     # наклон кривой оценивается только при ≥ 2 эффективных точках в окне ядра
EPOCH = pd.Timestamp("2000-01-01")


def polygon_offsets(rows: pd.DataFrame) -> dict[str, float]:
    """Смещения Landsat и MODIS относительно S2 для полигона по совместным наблюдениям в один день,
    сжатые к глобальным значениям из EDA при малом числе пар."""
    out = {"s2": 0.0}
    for name, col in (("landsat", "landsat_ndvi"), ("modis", "modis_ndvi")):
        pair = rows.dropna(subset=[col, "s2_ndvi"])
        n = len(pair)
        local = float((pair[col] - pair["s2_ndvi"]).mean()) if n else SENSOR_OFFSET[name]
        out[name] = (n * local + SHRINK_PAIRS * SENSOR_OFFSET[name]) / (n + SHRINK_PAIRS)
    return out


def harmonized_series(rows: pd.DataFrame) -> pd.DataFrame:
    """Известные наблюдения полигона в шкале S2 с LOO-остатком и флагом артефакта.

    rows — строки obs одного полигона (из gapfill.data.load_all) с колонкой sensor.
    """
    s = rows.sort_values("day_num").reset_index(drop=True)
    offsets = polygon_offsets(s)
    codes = {v: k for k, v in SENSOR_CODE.items()}
    off = np.array([offsets[codes[int(c)]] for c in s["sensor"]])
    h = s["primary_ndvi"].to_numpy(dtype=float) - off
    day = s["day_num"].to_numpy(dtype=float)
    fit, _, w = local_linear(day, h, day, CURVE_BW, loo=True)
    res = np.where(w >= 0.5, h - fit, np.nan)
    mad = float(np.nanmedian(np.abs(res - np.nanmedian(res)))) * 1.4826 if np.isfinite(res).any() else 0.05
    artifact = np.abs(np.nan_to_num(res)) > max(ARTIFACT_ABS, ARTIFACT_MAD_K * mad)
    return s.assign(h=h, res=res, artifact=artifact, offset=off)


def season_days(year: int) -> pd.DataFrame:
    """Ежедневная сетка сезона (1 апреля — 30 октября) с day_num и днём года."""
    dates = pd.date_range(f"{year}-04-01", f"{year}-10-30", freq="D")
    return pd.DataFrame({"date": dates, "day_num": (dates - EPOCH).days, "doy": dates.dayofyear, "year": year})


def daily_curve(series: pd.DataFrame, year: int, bw: float = CURVE_BW) -> pd.DataFrame:
    """Гладкая кривая сезона на ежедневной сетке по гармонизированным наблюдениям без артефактов.

    Возвращает day_num, date, doy, value, slope, weight (сумма весов ядра — надёжность), n_obs сезона.
    """
    grid = season_days(year)
    sel = (series["year"] == year) & ~series["artifact"]
    day = series.loc[sel, "day_num"].to_numpy(dtype=float)
    h = series.loc[sel, "h"].to_numpy(dtype=float)
    value, slope, weight = local_linear(day, h, grid["day_num"].to_numpy(dtype=float), bw, min_points=MIN_SLOPE_POINTS)
    # на всякий случай удерживаем кривую в физическом диапазоне NDVI
    value = np.clip(value, CURVE_RANGE[0], CURVE_RANGE[1])
    return grid.assign(value=value, slope=slope, weight=weight, n_obs=int(sel.sum()))


def curves_by_year(series: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Кривые всех сезонов полигона, у которых есть хотя бы 5 наблюдений без артефактов."""
    out = {}
    for year, g in series.groupby("year"):
        if int((~g["artifact"]).sum()) >= 5:
            out[int(year)] = daily_curve(series, int(year))
    return out


def in_season(doy: np.ndarray) -> np.ndarray:
    return (doy >= SEASON_DOY[0]) & (doy <= SEASON_DOY[1])
